// SPDX-License-Identifier: GPL-2.0+
/*
 * In-kernel run of the cc_xarray order > 0 contract, as stated in
 * Documentation/core-api/cc-xarray.rst ("Order and split interfaces").
 *
 * The cases follow tools/testing/radix-tree/cc-multi-order.c.  That test
 * decides where a second writer runs.  Here the kernel's own cc_xarray.o runs
 * under kthreads on real CPUs, and mm/huge_memory.c splits real folios of a
 * mapping whose i_pages is a ccXArray.
 *
 * Runs once at boot.  The test_cc_xarray.stress* parameters give the number
 * of attempts of each threaded case; 0 skips it.
 */
#define pr_fmt(fmt) "test_cc_xarray: " fmt

#include <linux/mm.h>
#include <linux/cc_xarray.h>
#include <linux/completion.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/kthread.h>
#include <linux/memcontrol.h>
#include <linux/module.h>
#include <linux/pagemap.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/slab.h>

#define BASE	4UL	/* order-2 entry covers BASE..BASE + 3 */
#define ORDER	2U
#define NR	(1UL << ORDER)

/* Entries of a leaf or more fill the leaves of their range. */
#define FAR		(4UL * CC_XA_CHUNK_SIZE)
#define FAR_ORDER	7U
#define FAR_NR		(1UL << FAR_ORDER)
#define WALK_ORDER	8U
#define WALK_NR		(1UL << WALK_ORDER)

#define SHADOW	cc_xa_mk_value(7)

static struct xarray tree;
static struct cc_xarray *XA;

static int anchor, big, other;
static unsigned int failures;
/* Lookups are running: they hold references on the nodes they pass. */
static bool readers;

#define EXPECT(expr) do { \
	if (!(expr)) { \
		pr_err("violated at line %d: %s\n", __LINE__, #expr); \
		failures++; \
	} \
} while (0)

/* Insert as the CC copy of __filemap_add_folio() does: expect an empty slot. */
static int insert(unsigned long index, unsigned int order, void *entry)
{
	CC_XA_STATE_ORDER(xas, XA, index, order);
	int err;

	do {
		cc_xas_store_expected(&xas, NULL, entry);
	} while (cc_xas_nomem(&xas, GFP_KERNEL));
	err = cc_xas_error(&xas);
	cc_xas_clear_xa_node(&xas);
	return err;
}

/* Delete as page_cache_delete() does: the caller owns the entry. */
static int erase(unsigned long index, unsigned int order)
{
	CC_XA_STATE_ORDER(xas, XA, index, order);
	int err;

	cc_xas_store(&xas, NULL);
	err = cc_xas_error(&xas);
	cc_xas_clear_xa_node(&xas);
	return err;
}

static int store_order(unsigned long index, unsigned int order, void *entry)
{
	CC_XA_STATE_ORDER(xas, XA, index, order);
	int err;

	do {
		cc_xas_store(&xas, entry);
	} while (cc_xas_nomem(&xas, GFP_KERNEL));
	err = cc_xas_error(&xas);
	cc_xas_clear_xa_node(&xas);
	return err;
}

static struct cc_xa_node *leaf_of(unsigned long index)
{
	CC_XA_STATE(xas, XA, index);
	struct cc_xa_node *node;

	cc_xas_load(&xas, false);
	node = xas.xa_node;
	cc_xas_rewind_refcnt(&xas);
	cc_xas_clear_xa_node(&xas);
	return node;
}

static unsigned int occupied(struct cc_xa_node *node)
{
	unsigned int i, nr = 0;

	for (i = 0; i < CC_XA_CHUNK_SIZE; i++)
		nr += !!node->slots[i];
	return nr;
}

static unsigned int value_slots(struct cc_xa_node *node)
{
	unsigned int i, nr = 0;

	for (i = 0; i < CC_XA_CHUNK_SIZE; i++) {
		void *entry = node->slots[i];

		if (cc_xa_is_sibling(entry))
			entry = node->slots[cc_xa_to_sibling(entry)];
		nr += cc_xa_is_value(entry);
	}
	return nr;
}

static void counted(struct cc_xa_node *node)
{
	EXPECT(node->count == occupied(node));
	EXPECT(node->nr_values == value_slots(node));
	EXPECT(readers || !node->refcnt);
}

/* Every leaf of the range agrees with its slots. */
static void far_counted(unsigned long nr)
{
	unsigned long i;

	for (i = 0; i < nr; i += CC_XA_CHUNK_SIZE) {
		struct cc_xa_node *node = leaf_of(FAR + i);

		/* A leaf without entries is deleted and no longer reached. */
		if (!node || node->shift)
			continue;
		counted(node);
	}
}

static struct cc_xa_node *begin(void)
{
	XA = CC_XARRAY(&tree);
	cc_xa_init(XA);
	/* A neighbour keeps the leaf allocated and linked for the whole case. */
	EXPECT(!insert(1, 0, &anchor));
	return leaf_of(BASE);
}

static void end(void)
{
	cc_xa_destroy(XA);
	rcu_barrier();
}

/* One lookup as a reader makes it. */
static void *peek(unsigned long index)
{
	CC_XA_STATE(xas, XA, index);
	void *entry = cc_xas_load(&xas, true);

	cc_xas_clear_xa_node(&xas);
	return entry;
}

static unsigned long holds(unsigned long base, unsigned long nr, void *entry)
{
	unsigned long i, n = 0;

	for (i = 0; i < nr; i++)
		n += cc_xa_load(XA, base + i) == entry;
	return n;
}

static int split_shadow(void)
{
	CC_XA_STATE(xas, XA, BASE);
	int err;

	do {
		cc_xas_try_split(&xas, SHADOW, ORDER);
	} while (cc_xas_nomem(&xas, GFP_KERNEL));
	err = cc_xas_error(&xas);
	cc_xas_clear_xa_node(&xas);
	return err;
}

static int replace_shadow(unsigned long index, void *entry)
{
	CC_XA_STATE(xas, XA, index);
	int err;

	cc_xas_store_expected(&xas, SHADOW, entry);
	err = cc_xas_error(&xas);
	cc_xas_clear_xa_node(&xas);
	return err;
}

static void alone(void)
{
	struct cc_xa_node *node = begin();
	unsigned long i;

	EXPECT(!insert(BASE, ORDER, &big));
	EXPECT(holds(BASE, NR, &big) == NR);
	EXPECT(insert(BASE + 1, 0, &other) == -EEXIST);
	counted(node);
	EXPECT(!erase(BASE, ORDER));
	EXPECT(holds(BASE, NR, NULL) == NR);
	counted(node);

	/* A shadow is never multi-index: a range of it is one per slot. */
	EXPECT(!insert(BASE, ORDER, SHADOW));
	counted(node);
	EXPECT(!split_shadow());
	for (i = 0; i < NR; i++)
		EXPECT(node->slots[BASE + i] == SHADOW);
	EXPECT(!replace_shadow(BASE + 1, &other));
	EXPECT(cc_xa_load(XA, BASE + 1) == &other);
	counted(node);
	end();
}

static void far_alone(void)
{
	unsigned int order;

	for (order = CC_XA_CHUNK_SHIFT; order <= 8; order++) {
		CC_XA_STATE_ORDER(xas, CC_XARRAY(&tree), FAR, order);
		unsigned long nr = 1UL << order;

		begin();
		/* The leaves exist, deleted: an entry was here before. */
		EXPECT(!insert(FAR + 70, 0, &other));
		EXPECT(!erase(FAR + 70, 0));
		EXPECT(!insert(FAR + 5, 0, &other));
		EXPECT(!erase(FAR + 5, 0));

		EXPECT(!insert(FAR, order, &big));
		EXPECT(holds(FAR, nr, &big) == nr);
		EXPECT(!cc_xa_load(XA, FAR - 1) && !cc_xa_load(XA, FAR + nr));
		far_counted(nr);
		/* An occupied range refuses a second entry and keeps the first. */
		EXPECT(insert(FAR, order, &other) == -EEXIST);
		EXPECT(insert(FAR + nr - 1, 0, &other) == -EEXIST);
		EXPECT(holds(FAR, nr, &big) == nr);

		/* Delete with the order, as page_cache_delete() does. */
		EXPECT(!erase(FAR, order));
		EXPECT(holds(FAR, nr, NULL) == nr);

		/* Delete from the first index alone, as a batch delete does. */
		EXPECT(!insert(FAR, order, &big));
		EXPECT(!erase(FAR, 0));
		EXPECT(holds(FAR, nr, NULL) == nr);

		/* One index in the way refuses the whole entry. */
		EXPECT(!insert(FAR + nr - 1, 0, &other));
		EXPECT(insert(FAR, order, &big) == -EEXIST);
		EXPECT(holds(FAR, nr, &big) == 0);
		EXPECT(cc_xa_load(XA, FAR + nr - 1) == &other);
		far_counted(nr);
		EXPECT(!erase(FAR + nr - 1, 0));

		/* Eviction, then a refault of the same range over the shadows. */
		EXPECT(!insert(FAR, order, &big));
		EXPECT(!store_order(FAR, order, SHADOW));
		EXPECT(holds(FAR, nr, SHADOW) == nr);
		far_counted(nr);
		cc_xas_store_expected(&xas, SHADOW, &other);
		EXPECT(!cc_xas_error(&xas));
		cc_xas_clear_xa_node(&xas);
		EXPECT(holds(FAR, nr, &other) == nr);
		far_counted(nr);
		EXPECT(!erase(FAR, order));
		EXPECT(holds(FAR, nr, NULL) == nr);
		end();
	}
}

/* The owner splits, one step at a time as mm/huge_memory.c does. */
static void far_split(void)
{
	CC_XA_STATE(xas, CC_XARRAY(&tree), FAR + 70);
	unsigned int order = 8, next;
	unsigned long i;

	begin();
	EXPECT(!insert(FAR, order, &big));
	while (order) {
		next = cc_xas_try_split_min_order(order);
		cc_xas_set_order(&xas, FAR + 70, next);
		cc_xas_try_split(&xas, &big, order);
		EXPECT(!cc_xas_error(&xas));
		cc_xas_clear_xa_node(&xas);
		order = next;
	}
	EXPECT(holds(FAR, 256, &big) == 256);
	EXPECT(cc_xa_get_order(XA, FAR + 70) == 0);
	/* The other leaves were not touched: orders 6 and 7 remain. */
	EXPECT(cc_xa_get_order(XA, FAR) == CC_XA_CHUNK_SHIFT);
	EXPECT(cc_xa_get_order(XA, FAR + 200) == CC_XA_CHUNK_SHIFT + 1);
	far_counted(256);
	/* Each index of the split leaf can now go on its own. */
	EXPECT(!erase(FAR + 70, 0));
	EXPECT(!cc_xa_load(XA, FAR + 70));
	EXPECT(cc_xa_load(XA, FAR + 71) == &big);
	for (i = 64; i < 128; i++)
		if (i != 70)
			EXPECT(!erase(FAR + i, 0));

	/* A uniform split reaches every leaf. */
	cc_xas_set_order(&xas, FAR, 0);
	cc_xas_split(&xas, &big, 6);
	EXPECT(!cc_xas_error(&xas));
	cc_xas_clear_xa_node(&xas);
	EXPECT(cc_xa_get_order(XA, FAR + 1) == 0);
	EXPECT(holds(FAR, 64, &big) == 64);
	cc_xas_set_order(&xas, FAR + 128, 2);
	cc_xas_split(&xas, &big, 7);
	EXPECT(!cc_xas_error(&xas));
	cc_xas_clear_xa_node(&xas);
	EXPECT(cc_xa_get_order(XA, FAR + 128) == 2);
	EXPECT(cc_xa_get_order(XA, FAR + 252) == 2);
	EXPECT(holds(FAR + 128, 128, &big) == 128);
	far_counted(256);
	end();
}

static int before, after, piece[3];

/* What a walk over present entries returns from @start on. */
static unsigned int walk(unsigned long start, void **seen,
			 unsigned long *at, int *order, bool pausing)
{
	CC_XA_STATE(xas, XA, start);
	unsigned int n = 0;
	void *entry;

	cc_xas_for_each(&xas, entry, ULONG_MAX) {
		if (n == 8)
			break;
		seen[n] = entry;
		at[n] = xas.xa_index;
		order[n++] = cc_xas_get_order(&xas);
		if (pausing)
			cc_xas_pause(&xas);
	}
	cc_xas_clear_xa_node(&xas);
	return n;
}

static unsigned int walk_marked(unsigned long start, void **seen)
{
	CC_XA_STATE(xas, XA, start);
	unsigned int n = 0;
	void *entry;

	cc_xas_for_each_marked(&xas, entry, ULONG_MAX, CC_XA_MARK_0) {
		if (n == 8)
			break;
		seen[n++] = entry;
	}
	cc_xas_clear_xa_node(&xas);
	return n;
}

static unsigned int entries(void)
{
	CC_XA_STATE(xas, XA, 0);
	unsigned int n = 0;
	void *entry;

	cc_xas_for_each(&xas, entry, ULONG_MAX)
		n++;
	cc_xas_clear_xa_node(&xas);
	return n;
}

static void mark(unsigned long index, bool set)
{
	CC_XA_STATE(xas, XA, index);

	cc_xas_load(&xas, true);
	if (set)
		cc_xas_set_mark(&xas, CC_XA_MARK_0);
	else
		cc_xas_clear_mark(&xas, CC_XA_MARK_0);
	cc_xas_clear_xa_node(&xas);
}

static bool marked(unsigned long index)
{
	CC_XA_STATE(xas, XA, index);
	bool ret;

	cc_xas_load(&xas, true);
	ret = cc_xas_get_mark(&xas, CC_XA_MARK_0);
	cc_xas_clear_xa_node(&xas);
	return ret;
}

/*
 * An entry of several leaves looks like one entry to the callers: a walk
 * returns it once, its order is its real order, and its marks are found
 * from any index inside it.
 */
static void far_walk(void)
{
	void *seen[8];
	unsigned long at[8], index;
	int order[8];
	unsigned int n, i;
	void *entry;

	begin();
	EXPECT(!insert(FAR - 1, 0, &before));
	EXPECT(!insert(FAR, WALK_ORDER, &big));
	EXPECT(!insert(FAR + WALK_NR, 0, &after));

	for (i = 0; i < 2; i++) {
		n = walk(FAR - 1, seen, at, order, i);
		EXPECT(n == 3 && seen[0] == &before && seen[1] == &big &&
		       seen[2] == &after);
		EXPECT(at[1] == FAR && order[1] == WALK_ORDER);
		EXPECT(at[2] == FAR + WALK_NR && order[2] == 0);

		/* A walk that starts inside the entry. */
		n = walk(FAR + 130, seen, at, order, i);
		EXPECT(n == 2 && seen[0] == &big && seen[1] == &after);
		EXPECT(at[0] >= FAR + 130 && at[0] < FAR + WALK_NR);
		EXPECT(order[0] == WALK_ORDER);
	}
	EXPECT(cc_xa_get_order(XA, FAR + 200) == WALK_ORDER);

	/* The index based walk. */
	n = 0;
	cc_xa_for_each_start(XA, index, entry, FAR - 1)
		n += entry == &big;
	EXPECT(n == 1);

	/* Skip by the size of the entry, as the read path does. */
	{
		CC_XA_STATE(xas, XA, FAR);

		EXPECT(cc_xas_load(&xas, true) == &big);
		cc_xas_advance(&xas, FAR + WALK_NR - 1);
		EXPECT(cc_xas_next(&xas) == &after);
		EXPECT(xas.xa_index == FAR + WALK_NR);
		cc_xas_clear_xa_node(&xas);
	}

	/* Marks. */
	EXPECT(!walk_marked(0, seen));
	mark(FAR, true);
	for (index = FAR; index < FAR + WALK_NR; index += 50)
		EXPECT(marked(index));
	EXPECT(!marked(FAR - 1) && !marked(FAR + WALK_NR));
	EXPECT(walk_marked(0, seen) == 1 && seen[0] == &big);
	EXPECT(walk_marked(FAR + 130, seen) == 1 && seen[0] == &big);
	EXPECT(walk_marked(FAR + 192, seen) == 1 && seen[0] == &big);
	index = FAR + 200;
	EXPECT(cc_xa_find(XA, &index, ULONG_MAX, CC_XA_MARK_0) == &big);
	EXPECT(!cc_xa_find_after(XA, &index, ULONG_MAX, CC_XA_MARK_0));
	mark(FAR, false);
	EXPECT(!walk_marked(0, seen));
	EXPECT(!cc_xa_marked(XA, CC_XA_MARK_0));

	/*
	 * A folio split: the pieces inherit the marks.  Each later leaf then
	 * gets a folio of its own.
	 */
	mark(FAR, true);
	{
		CC_XA_STATE_ORDER(xas, XA, FAR, CC_XA_CHUNK_SHIFT);

		cc_xas_split(&xas, &big, WALK_ORDER);
		EXPECT(!cc_xas_error(&xas));
		cc_xas_clear_xa_node(&xas);
	}
	for (i = 0; i < 3; i++)
		EXPECT(!store_order(FAR + (i + 1) * CC_XA_CHUNK_SIZE,
				    CC_XA_CHUNK_SHIFT, &piece[i]));
	n = walk(FAR, seen, at, order, false);
	EXPECT(n == 5 && seen[0] == &big && seen[1] == &piece[0] &&
	       seen[2] == &piece[1] && seen[3] == &piece[2] &&
	       seen[4] == &after);
	for (i = 0; i < 4; i++)
		EXPECT(order[i] == CC_XA_CHUNK_SHIFT);
	n = walk_marked(0, seen);
	EXPECT(n == 4 && seen[0] == &big && seen[1] == &piece[0] &&
	       seen[2] == &piece[1] && seen[3] == &piece[2]);
	mark(FAR + 2 * CC_XA_CHUNK_SIZE, false);
	n = walk_marked(0, seen);
	EXPECT(n == 3 && seen[0] == &big && seen[1] == &piece[0] &&
	       seen[2] == &piece[2]);
	for (i = 0; i < 4; i++)
		EXPECT(!erase(FAR + i * CC_XA_CHUNK_SIZE, CC_XA_CHUNK_SHIFT));
	EXPECT(!walk_marked(0, seen));

	/* Delete and evict leave no mark behind. */
	EXPECT(!insert(FAR, WALK_ORDER, &big));
	mark(FAR, true);
	EXPECT(!erase(FAR, WALK_ORDER));
	EXPECT(!walk_marked(0, seen));
	EXPECT(!insert(FAR, WALK_ORDER, &big));
	EXPECT(!marked(FAR + 100));
	mark(FAR, true);
	EXPECT(!store_order(FAR, WALK_ORDER, SHADOW));
	EXPECT(!walk_marked(0, seen));
	EXPECT(holds(FAR, WALK_NR, SHADOW) == WALK_NR);
	EXPECT(walk(FAR, seen, at, order, false) == 8);
	end();
}

/*
 * Threads.  One writer inserts and deletes the large entry, one writer per
 * index does the same with an order-0 entry of its own, and readers check
 * that an index never returns another index's entry.
 */
static int small[NR];
static bool stop;
static unsigned long attempts;
static atomic_long_t stress_wrong, store_hidden, erase_missed, gave_up;
static atomic_long_t walk_seen, walk_moved;

#define BREATHE(n) do { if (!((n) & 1023)) cond_resched(); } while (0)

struct cc_worker {
	struct completion exited;
	unsigned long index;
	unsigned long done, lost, loads;
};

static struct cc_worker big_w, small_w[NR], read_w[4];

static void start(struct cc_worker *w, int (*fn)(void *), unsigned long index)
{
	memset(w, 0, sizeof(*w));
	init_completion(&w->exited);
	w->index = index;
	if (IS_ERR(kthread_run(fn, w, "test_cc_xarray"))) {
		EXPECT(0);
		complete(&w->exited);
	}
}

/* A writer met an error that the contract does not allow. */
static int give_up(struct cc_worker *w)
{
	atomic_long_inc(&gave_up);
	complete(&w->exited);
	return 0;
}

static int big_writer(void *arg)
{
	struct cc_worker *w = arg;

	while (w->done + w->lost < attempts) {
		BREATHE(w->done + w->lost);
		if (insert(BASE, ORDER, &big)) {
			w->lost++;
			continue;
		}
		if (erase(BASE, ORDER))
			return give_up(w);
		w->done++;
	}
	complete(&w->exited);
	return 0;
}

static int small_writer(void *arg)
{
	struct cc_worker *w = arg;

	while (!READ_ONCE(stop)) {
		BREATHE(w->done + w->lost);
		if (insert(BASE + w->index, 0, &small[w->index])) {
			w->lost++;
			continue;
		}
		if (erase(BASE + w->index, 0))
			return give_up(w);
		WRITE_ONCE(w->done, w->done + 1);
	}
	complete(&w->exited);
	return 0;
}

/*
 * The order-2 writer turns its entry into a shadow before it removes it, as
 * eviction and reclaim do.
 */
static int shadow_writer(void *arg)
{
	struct cc_worker *w = arg;

	while (w->done + w->lost < attempts) {
		CC_XA_STATE_ORDER(xas, XA, BASE, ORDER);
		int err;

		BREATHE(w->done + w->lost);
		if (insert(BASE, ORDER, &big)) {
			w->lost++;
			continue;
		}
		cc_xas_store(&xas, SHADOW);
		err = cc_xas_error(&xas);
		cc_xas_clear_xa_node(&xas);
		if (err)
			return give_up(w);
		cc_xas_set_order(&xas, BASE, ORDER);
		cc_xas_store_expected(&xas, SHADOW, NULL);
		err = cc_xas_error(&xas);
		cc_xas_clear_xa_node(&xas);
		if (err && err != -EEXIST)
			return give_up(w);
		w->done++;
	}
	complete(&w->exited);
	return 0;
}

/* The order-0 writers split that shadow and replace their own index. */
static int splitting_writer(void *arg)
{
	struct cc_worker *w = arg;
	unsigned long index = BASE + w->index;

	while (!READ_ONCE(stop)) {
		int err;

		BREATHE(w->done + w->lost);
		err = split_shadow();
		if (err && err != -EEXIST)
			return give_up(w);
		if (replace_shadow(index, &small[w->index]) &&
		    insert(index, 0, &small[w->index])) {
			w->lost++;
			continue;
		}
		if (erase(index, 0))
			return give_up(w);
		WRITE_ONCE(w->done, w->done + 1);
	}
	complete(&w->exited);
	return 0;
}

static int reader(void *arg)
{
	struct cc_worker *w = arg;
	unsigned long i;

	while (!READ_ONCE(stop)) {
		BREATHE(w->loads >> 2);
		for (i = 0; i < NR; i++) {
			void *entry = peek(BASE + i);

			w->loads++;
			if (entry && entry != &big && entry != SHADOW &&
			    entry != &small[i])
				atomic_long_inc(&stress_wrong);
		}
	}
	complete(&w->exited);
	return 0;
}

static void reset_counters(void)
{
	WRITE_ONCE(stop, false);
	atomic_long_set(&stress_wrong, 0);
	atomic_long_set(&store_hidden, 0);
	atomic_long_set(&erase_missed, 0);
	atomic_long_set(&gave_up, 0);
	atomic_long_set(&walk_seen, 0);
	atomic_long_set(&walk_moved, 0);
}

/* Wait for the large writer, stop the others, add up what they did. */
static void finish(unsigned int readers, unsigned long *small_done,
		   unsigned long *small_lost, unsigned long *loads)
{
	unsigned long i;

	*small_done = *small_lost = *loads = 0;
	wait_for_completion(&big_w.exited);
	WRITE_ONCE(stop, true);
	for (i = 0; i < NR; i++) {
		wait_for_completion(&small_w[i].exited);
		*small_done += small_w[i].done;
		*small_lost += small_w[i].lost;
	}
	for (i = 0; i < readers; i++) {
		wait_for_completion(&read_w[i].exited);
		*loads += read_w[i].loads;
	}
}

static void stress(unsigned long nr, bool shadow)
{
	struct cc_xa_node *node = begin();
	unsigned long i, small_done, small_lost, loads;

	reset_counters();
	attempts = nr;
	start(&big_w, shadow ? shadow_writer : big_writer, 0);
	for (i = 0; i < NR; i++)
		start(&small_w[i], shadow ? splitting_writer : small_writer, i);
	for (i = 0; i < 2; i++)
		start(&read_w[i], reader, i);
	finish(2, &small_done, &small_lost, &loads);

	pr_info("  order-2 stored %lu, refused %lu\n", big_w.done, big_w.lost);
	pr_info("  order-0 stored %lu, refused %lu\n", small_done, small_lost);
	pr_info("  lookups %lu, wrong entry %ld\n", loads,
		atomic_long_read(&stress_wrong));
	EXPECT(!atomic_long_read(&stress_wrong));
	EXPECT(!atomic_long_read(&gave_up));
	EXPECT(big_w.done && small_done);
	for (i = 0; i < NR; i++) {
		void *entry = cc_xa_load(XA, BASE + i);

		/* A shadow split off before its delete may stay behind. */
		EXPECT(!entry || (shadow && entry == SHADOW));
	}
	counted(node);
	end();
}

static const unsigned long far_at[NR] = { 0, 63, 64, 127 };
static bool far_idle;

static void drop_shadows(void)
{
	unsigned long i;

	for (i = 0; i < FAR_NR; i++) {
		CC_XA_STATE(xas, XA, FAR + i);

		cc_xas_store_expected(&xas, SHADOW, NULL);
		cc_xas_clear_xa_node(&xas);
	}
}

static unsigned long small_total(void)
{
	unsigned long i, n = 0;

	for (i = 0; i < NR; i++)
		n += READ_ONCE(small_w[i].done);
	return n;
}

/*
 * The order-7 entry fills two leaves.  Its writer stores it, turns it into
 * shadows, stores a second entry over the shadows and deletes that.  When
 * idle, it only waits: nothing else keeps the leaves of the order-0 writers
 * in use, so those leaves are deleted and reused all the time.
 */
static int far_writer(void *arg)
{
	struct cc_worker *w = arg;

	while (far_idle && small_total() < attempts &&
	       !atomic_long_read(&gave_up))
		schedule_timeout_uninterruptible(1);
	while (!far_idle && w->done + w->lost < attempts) {
		CC_XA_STATE_ORDER(xas, XA, FAR, FAR_ORDER);

		BREATHE(w->done + w->lost);
		if (insert(FAR, FAR_ORDER, &big)) {
			w->lost++;
			continue;
		}
		if (holds(FAR, FAR_NR, &big) != FAR_NR)
			atomic_long_inc(&stress_wrong);
		if (w->done & 1) {
			if (erase(FAR, FAR_ORDER))
				return give_up(w);
			w->done++;
			continue;
		}
		if (store_order(FAR, FAR_ORDER, SHADOW))
			return give_up(w);
		do {
			cc_xas_store_expected(&xas, SHADOW, &other);
		} while (cc_xas_nomem(&xas, GFP_KERNEL));
		if (!cc_xas_error(&xas)) {
			cc_xas_clear_xa_node(&xas);
			if (holds(FAR, FAR_NR, &other) != FAR_NR)
				atomic_long_inc(&stress_wrong);
			if (erase(FAR, FAR_ORDER))
				return give_up(w);
		} else {
			cc_xas_clear_xa_node(&xas);
			drop_shadows();
		}
		w->done++;
	}
	complete(&w->exited);
	return 0;
}

static int far_small_writer(void *arg)
{
	struct cc_worker *w = arg;
	unsigned long index = FAR + far_at[w->index];

	while (!READ_ONCE(stop)) {
		BREATHE(w->done + w->lost);
		if (replace_shadow(index, &small[w->index]) &&
		    insert(index, 0, &small[w->index])) {
			w->lost++;
			continue;
		}
		/* Nobody else writes this index while it holds our entry. */
		if (peek(index) != &small[w->index])
			atomic_long_inc(&store_hidden);
		if (erase(index, 0))
			return give_up(w);
		if (peek(index) == &small[w->index])
			atomic_long_inc(&erase_missed);
		WRITE_ONCE(w->done, w->done + 1);
	}
	complete(&w->exited);
	return 0;
}

static int far_reader(void *arg)
{
	struct cc_worker *w = arg;
	unsigned long i, j;

	while (!READ_ONCE(stop)) {
		CC_XA_STATE(xas, XA, FAR + (w->loads & 127));
		void *entry;

		cond_resched();
		for (i = 0; i < FAR_NR; i++) {
			entry = peek(FAR + i);

			w->loads++;
			if (!entry || entry == &big || entry == &other ||
			    entry == SHADOW)
				continue;
			for (j = 0; j < NR; j++)
				if (far_at[j] == i && entry == &small[j])
					break;
			if (j == NR)
				atomic_long_inc(&stress_wrong);
		}
		/* A walk meets only entries of the range, at their index. */
		rcu_read_lock();
		cc_xas_for_each(&xas, entry, FAR + FAR_NR - 1) {
			if (cc_xas_retry(&xas, entry))
				continue;
			if (xas.xa_index < FAR ||
			    xas.xa_index >= FAR + FAR_NR ||
			    peek(xas.xa_index) != entry)
				atomic_long_inc(&walk_moved);
			cc_xas_get_order(&xas);
			atomic_long_inc(&walk_seen);
		}
		rcu_read_unlock();
		cc_xas_clear_xa_node(&xas);
	}
	complete(&w->exited);
	return 0;
}

static void stress_far(unsigned long nr, bool idle)
{
	unsigned long i, small_done, small_lost, loads;

	begin();
	reset_counters();
	far_idle = idle;
	attempts = nr;
	for (i = 0; i < NR; i++)
		start(&small_w[i], far_small_writer, i);
	start(&big_w, far_writer, 0);
	for (i = 0; i < 2; i++)
		start(&read_w[i], far_reader, i);
	finish(2, &small_done, &small_lost, &loads);
	drop_shadows();

	if (!idle)
		pr_info("  order-7 stored %lu, refused %lu\n", big_w.done,
			big_w.lost);
	pr_info("  order-0 stored %lu, refused %lu\n", small_done, small_lost);
	pr_info("  lookups %lu, wrong entry %ld\n", loads,
		atomic_long_read(&stress_wrong));
	pr_info("  stores that a lookup did not find %ld, deletes that left the entry %ld\n",
		atomic_long_read(&store_hidden),
		atomic_long_read(&erase_missed));
	pr_info("  entries walked %ld, no longer there at a second look %ld\n",
		atomic_long_read(&walk_seen), atomic_long_read(&walk_moved));
	EXPECT(!atomic_long_read(&stress_wrong));
	EXPECT(!atomic_long_read(&gave_up));
	EXPECT(!atomic_long_read(&store_hidden));
	EXPECT(!atomic_long_read(&erase_missed));
	EXPECT((idle || big_w.done) && small_done);
	EXPECT(holds(FAR, FAR_NR, NULL) == FAR_NR);
	far_counted(FAR_NR);
	end();
}

#ifdef CONFIG_TRANSPARENT_HUGEPAGE
/*
 * Real folios in a mapping whose i_pages is a ccXArray, split by
 * mm/huge_memory.c.  No filesystem makes such a mapping yet, so the inode is
 * made here and holds what the split path reads: the size, the gfp mask, the
 * order range and the tree.
 */
static const struct address_space_operations cc_test_aops = { };
static struct inode *cc_inode;

static bool cc_mapping_begin(void)
{
	struct address_space *mapping;

	cc_inode = kzalloc_obj(*cc_inode, GFP_KERNEL);
	if (!cc_inode)
		return false;
	inode_init_once(cc_inode);
	mapping = &cc_inode->i_data;
	cc_inode->i_mapping = mapping;
	mapping->host = cc_inode;
	mapping->a_ops = &cc_test_aops;
	mapping_set_gfp_mask(mapping, GFP_KERNEL);
	i_size_write(cc_inode, (loff_t)(FAR + WALK_NR) << PAGE_SHIFT);

	/* The filesystem asks for every order; the tree decides the limit. */
	mapping_set_large_folios(mapping);
	EXPECT(mapping_max_folio_order(mapping) == MAX_PAGECACHE_ORDER);
	XA = CC_XARRAY(&mapping->i_pages);
	cc_xa_init(XA);
	EXPECT(mapping_is_cc(mapping));
	EXPECT(mapping_max_folio_order(mapping) == MAX_CC_PAGECACHE_ORDER);
	EXPECT(mapping_max_folio_order(mapping) < HPAGE_PMD_ORDER);
	return true;
}

static void cc_mapping_end(void)
{
	end();
	kfree(cc_inode);
	cc_inode = NULL;
}

/* A locked order-8 folio in the page cache, as filemap_add_folio() leaves it. */
static struct folio *cc_folio_add(void)
{
	struct folio *folio = folio_alloc(GFP_KERNEL, WALK_ORDER);

	if (!folio)
		return NULL;
	if (mem_cgroup_charge(folio, NULL, GFP_KERNEL)) {
		folio_put(folio);
		return NULL;
	}
	__folio_set_locked(folio);
	folio->mapping = cc_inode->i_mapping;
	folio->index = FAR;
	folio_ref_add(folio, WALK_NR);
	if (insert(FAR, WALK_ORDER, folio)) {
		folio->mapping = NULL;
		__folio_clear_locked(folio);
		folio_put_refs(folio, WALK_NR + 1);
		return NULL;
	}
	return folio;
}

/*
 * Split, check every after-split folio against the tree, then remove them.
 * Returns the error of the split.
 */
static int cc_folio_split(struct folio *head, unsigned int new_order,
			  unsigned long at, bool uniform, bool quiet)
{
	struct page *first = &head->page, *page;
	unsigned int n = 0;
	LIST_HEAD(list);
	int ret;

	mark(FAR, true);
	if (uniform)
		ret = split_huge_page_to_list_to_order(first, &list, new_order);
	else
		ret = folio_split(head, new_order, first + at, &list);

	for (page = first; page < first + WALK_NR; n++) {
		struct folio *folio = page_folio(page);
		unsigned long nr = folio_nr_pages(folio);
		unsigned long off = page - first;
		unsigned int order = folio_order(folio);

		EXPECT(&folio->page == page);
		EXPECT(folio->index == FAR + off);
		EXPECT(folio->mapping == cc_inode->i_mapping);
		EXPECT(folio_test_locked(folio) == (folio == head));
		if (quiet)
			EXPECT(folio_ref_count(folio) == nr + 1);
		EXPECT(holds(FAR + off, nr, folio) == nr);
		EXPECT(cc_xa_get_order(XA, FAR + off) == order);
		EXPECT(cc_xa_get_order(XA, FAR + off + nr - 1) == order);
		EXPECT(marked(FAR + off) && marked(FAR + off + nr - 1));
		if (!ret && (uniform || (off <= at && at < off + nr)))
			EXPECT(order == new_order);
		page += nr;
	}
	EXPECT(entries() == n);
	EXPECT(ret || n > 1);
	far_counted(WALK_NR);

	/* Each after-split folio leaves the page cache on its own. */
	for (page = first; page < first + WALK_NR;) {
		struct folio *folio = page_folio(page);
		unsigned long nr = folio_nr_pages(folio);

		EXPECT(!erase(folio->index, folio_order(folio)));
		if (folio == head)
			folio_unlock(folio);
		else
			list_del(&folio->lru);
		folio->mapping = NULL;
		folio_put_refs(folio, nr + 1);
		page += nr;
	}
	EXPECT(holds(FAR, WALK_NR, NULL) == WALK_NR);
	EXPECT(!entries());
	return ret;
}

static void folio_split_alone(void)
{
	static const unsigned long at[] = { 0, 70, 128, 255 };
	static const unsigned int orders[] = { 0, 2, 5, 6, 7 };
	unsigned int i, j;

	if (!cc_mapping_begin())
		return;
	for (i = 0; i < ARRAY_SIZE(at) + 1; i++) {
		for (j = 0; j < ARRAY_SIZE(orders); j++) {
			bool uniform = i == ARRAY_SIZE(at);
			struct folio *folio = cc_folio_add();

			if (!folio) {
				EXPECT(0);
				continue;
			}
			EXPECT(!cc_folio_split(folio, orders[j],
					       uniform ? 0 : at[i], uniform,
					       true));
		}
	}
	cc_mapping_end();
}

/* A lookup as filemap_get_entry() makes it. */
static int folio_reader(void *arg)
{
	struct cc_worker *w = arg;
	unsigned long i;

	while (!READ_ONCE(stop)) {
		cond_resched();
		for (i = 0; i < WALK_NR; i++) {
			unsigned long index = FAR + (i * 37 + w->index) % WALK_NR;
			struct folio *folio;

			/* Leave the splitter a chance to find no reference. */
			udelay(2);
			rcu_read_lock();
			folio = peek(index);
			if (!folio || !folio_try_get(folio)) {
				rcu_read_unlock();
				continue;
			}
			rcu_read_unlock();
			/* Still the entry of @index: it is not being split. */
			if (peek(index) == folio) {
				w->loads++;
				if (index < folio->index ||
				    index >= folio->index + folio_nr_pages(folio))
					atomic_long_inc(&stress_wrong);
			}
			folio_put(folio);
		}
	}
	complete(&w->exited);
	return 0;
}

/* Splits while readers take references on the folios. */
static void stress_folio(unsigned long nr)
{
	static const unsigned int orders[] = { 0, 2, 5, 6, 7 };
	unsigned long i, done = 0, busy = 0, loads = 0;

	if (!cc_mapping_begin())
		return;
	reset_counters();
	readers = true;
	for (i = 0; i < ARRAY_SIZE(read_w); i++)
		start(&read_w[i], folio_reader, i);
	for (i = 0; i < nr; i++) {
		struct folio *folio = cc_folio_add();
		int ret;

		if (!folio) {
			EXPECT(0);
			break;
		}
		ret = cc_folio_split(folio, orders[i % ARRAY_SIZE(orders)],
				     (i * 31) % WALK_NR, i & 1, false);
		/* A reader's reference makes the split back off. */
		EXPECT(!ret || ret == -EAGAIN);
		done += !ret;
		busy += !!ret;
		cond_resched();
	}
	WRITE_ONCE(stop, true);
	for (i = 0; i < ARRAY_SIZE(read_w); i++) {
		wait_for_completion(&read_w[i].exited);
		loads += read_w[i].loads;
	}
	readers = false;
	pr_info("  folios split %lu, busy %lu\n", done, busy);
	pr_info("  lookups that got the folio %lu, folio of another index %ld\n",
		loads, atomic_long_read(&stress_wrong));
	EXPECT(done && loads);
	EXPECT(!atomic_long_read(&stress_wrong));
	cc_mapping_end();
}
#else
static void folio_split_alone(void) { }
static void stress_folio(unsigned long nr) { }
#endif

static unsigned long stress_attempts = 1000000;
module_param_named(stress, stress_attempts, ulong, 0444);
static unsigned long stress_split_attempts = 300000;
module_param_named(stress_split, stress_split_attempts, ulong, 0444);
static unsigned long stress_far_attempts = 20000;
module_param_named(stress_far, stress_far_attempts, ulong, 0444);
static unsigned long stress_nodes_attempts = 200000;
module_param_named(stress_nodes, stress_nodes_attempts, ulong, 0444);
static unsigned long stress_folio_attempts = 20000;
module_param_named(stress_folio, stress_folio_attempts, ulong, 0444);

static unsigned int failed;

static void ran(const char *name)
{
	pr_info("%s %s\n", failures ? "FAIL" : "PASS", name);
	failed += !!failures;
	failures = 0;
}

static int __init test_cc_xarray_init(void)
{
	alone();
	ran("alone");
	far_alone();
	ran("far-alone");
	far_split();
	ran("far-split");
	far_walk();
	ran("far-walk");
	folio_split_alone();
	ran("folio-split");
	if (stress_attempts) {
		stress(stress_attempts, false);
		ran("stress");
	}
	if (stress_split_attempts) {
		stress(stress_split_attempts, true);
		ran("stress-split");
	}
	if (stress_far_attempts) {
		stress_far(stress_far_attempts, false);
		ran("stress-far");
	}
	if (stress_nodes_attempts) {
		stress_far(stress_nodes_attempts, true);
		ran("stress-nodes");
	}
	if (stress_folio_attempts) {
		stress_folio(stress_folio_attempts);
		ran("stress-folio");
	}
	pr_info("%s: %u failed\n", failed ? "FAILED" : "all passed", failed);
	return 0;
}
late_initcall(test_cc_xarray_init);
