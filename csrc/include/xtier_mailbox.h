#pragma once
// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments
// renamed only, so the kernels compile to the same code.
#ifdef __cplusplus
extern "C" {
#endif

// The bridge retains plan. stream is cudaStream_t, represented as void*.
// All slots must be prepared before capture; prepare is also performed lazily
// during eager enqueue. Replay of one captured slot must not overlap on streams.
void *xtier_mailbox_create(void *plan);
int xtier_mailbox_prepare(void *bridge, void *stream, int m, int k, int shadow);
// x/ids/routing/output are device pointers with the CPU API's element types.
// shadow=1 reads output as the existing reference and does not change it.
int xtier_mailbox_enqueue(void *bridge, void *stream, int m, int k, const void *x,
                       const void *ids, const void *routing, void *output, int shadow);
// Zero-copy path (mailbox bridge only). slot returns device-visible addresses of the
// non-shadow slot's pinned x, ids, routing and output buffers for (stream,m,k); the
// caller's kernels fill the inputs, then notify enqueues only the mailbox handshake,
// then the caller's kernels read the output. Same serial-use rules as enqueue.
int xtier_mailbox_slot(void *bridge, void *stream, int m, int k, void **pointers);
int xtier_mailbox_notify(void *bridge, void *stream, int m, int k);
// XTier cache expert pool (XTIER_CACHE=1). slot2 also returns the slot's original-route buffer (fifth pointer): the packing
// kernel stores the routes as routed there, and -1 in ids for routes the GPU computes. signal enqueues
// half of the mailbox handshake: part 1 = request the CPU, part 2 = wait for it (GPU work goes between).
// pool_enable makes the dispatcher read the original routes (statistics, XTIER_CACHE_LOG, route trace);
// pool_stats returns three doubles: calls, distinct experts, distinct experts served by the GPU.
int xtier_mailbox_slot2(void *bridge, void *stream, int m, int k, void **pointers);
int xtier_mailbox_signal(void *bridge, void *stream, int m, int k, int part);
int xtier_mailbox_pool_enable(void *bridge, int on);
int xtier_mailbox_pool_stats(void *bridge, double *statistics);
// Adaptive residency: pool_ring returns the device addresses of the mapped command ring and its sequence
// words (posted at +0, applied at +64 B, landed at +128 B; commands are int32 {layer, in, out, slot}).
// pool_attach registers one layer (arena: device w13, w2, s13, s2 slots; resident: S experts, -1 = empty);
// pool_start launches the policy thread (every windows, budget swaps per window, decay, minimum gain).
int xtier_mailbox_pool_ring(void **pointers);
int xtier_mailbox_pool_attach(void *bridge, int layer, int slots, int experts, void **arena, const int *resident);
int xtier_mailbox_pool_start(int every, int budget, float decay, float min_gain);
// Second GPU as a hot expert pool (cache/second.py), all with that GPU current on the calling thread. gpu2_init returns the
// device addresses of the host control block and output; gpu2_attach registers a pool_attached layer's slots there and
// loads its residents; a decode shape's graph is captured between gpu2_capture_begin and _end (or _abort); gpu2_start
// launches the policy thread (swaps per round, minimum gain, minimum route count).
int xtier_mailbox_gpu2_init(void *stream, int H, void **pointers);
int xtier_mailbox_gpu2_attach(void *bridge, int slots, void **arena, const int *resident);
int xtier_mailbox_gpu2_capture_begin(void *stream);
int xtier_mailbox_gpu2_capture_end(void *bridge, void *stream, int m, int k);
int xtier_mailbox_gpu2_capture_abort(void *stream);
int xtier_mailbox_gpu2_start(int budget, float min_gain, float min_use);
// Seven doubles: calls, elements, squared error, squared reference norm,
// max absolute error, max per-call relative L2 error, nonfinite element count.
// Error statistics describe shadow calls; counts include both modes.
int xtier_mailbox_stats(void *bridge, double *statistics);
// Sixteen doubles, for M=1..4 in order: calls, callback wall-clock us, gate+barrier us,
// down+barrier us (sums). Mailbox bridge only; the host-callback bridge returns -1.
int xtier_mailbox_timing(void *bridge, double *timing);
const char *xtier_mailbox_error(void);
// Destroy all graphs using this bridge first. Destroy synchronizes the device
// before releasing host buffers. A failure leaves the bridge allocated.
int xtier_mailbox_destroy(void *bridge);

#ifdef __cplusplus
}
#endif
