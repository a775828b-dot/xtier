#pragma once
// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments
// renamed only, so the kernels compile to the same code.
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Linux x86 with AVX512 BF16/VBMI or AVX2/FMA: one library holds both kernel builds and picks
// one per process on first use (XTIER_ISA=avx512|avx2 forces it; see xtier_isa). Worker nodes come from sysfs: every NUMA
// node with CPUs (any NPS setting, one or two sockets); each expert's Gate/Up and Down rows
// are split evenly over them, so H and I must be multiples of 16 * nodes. Workers bind to the
// node's physical cores first, then their SMT siblings (XTIER_NODE_CORES). Without sysfs:
// the original fixed host (4 nodes, CPUs 16v..16v+15, siblings 64+16v..). Outputs do not
// depend on the node count (same operations in the same order for every row).
// One generic kernel: hidden size H and expert intermediate size I are per plan
// (multiples of 64); NVFP4 E2M1 weights, low nibble first, E4M3 block scales per 16.
// All arrays contiguous, row-major:
// w13: [experts,2*I,H/2] uint8, gate rows then up rows.
// w2:  [experts,H,I/2] uint8.
// s13: [experts,2*I,H/16] positive finite E4M3 bytes; s2: [experts,H,I/16].
// g13: [experts,2] float32, independent gate/up multipliers; g2: [experts].
// swiglu_limit > 0 clamps up to [-L,L] and gate to (-inf,L] before SiLU (DeepSeek V4);
// 0 disables the clamp. create owns an explicitly NUMA-bound, sharded copy (overrides
// inherited interleave policy for its own mappings only). The input buffers can be
// released after create returns. experts=10..512; threads<=128, divisible by the node count.
void *xtier_create2(int hidden, int inter, int experts, int threads, float swiglu_limit,
                    const uint8_t *w13, const uint8_t *w2, const uint8_t *s13, const uint8_t *s2,
                    const float *g13, const float *g2);
// scale_format: 0=NVFP4 (E4M3/16), 1=MXFP4 (UE8M0/32).
// MXFP4 globals must be arrays of 1. Shard scales retain compact 32-wide UE8M0
// blocks; packed weights are unchanged. NVFP4 retains its original layout.
void *xtier_create3(int hidden,int inter,int experts,int threads,float limit,int scale_format,
                    const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                    const float *g13,const float *g2);
// RAM slots (cold experts on NVMe, xtier/cache/cold.py): like create3, but the NUMA shards hold the weights of
// `slots` experts (1..experts), slot j holding resident[j] (distinct experts). A route to an expert that is in no slot
// fails the forward ("Expert N is not in RAM"); it never runs on another expert's weights. create3 = create4 with
// slots == experts and resident[j] = j.
// create3 / create4 with w13, w2, s13 and s2 all null allocate the shards without filling them: the caller writes
// every slot (xtier_shard / xtier_slot_pieces, e.g. read straight from the checkpoint or the cold store) and checks
// its block scales, before the first forward.
void *xtier_create4(int hidden,int inter,int experts,int slots,int threads,float limit,int scale_format,
                    const uint8_t *w13,const uint8_t *w2,const uint8_t *s13,const uint8_t *s2,
                    const float *g13,const float *g2,const int32_t *resident);
// Mark `expert` as held by `slot` after its weights were written there (the slot's previous expert leaves RAM; an
// expert already in another slot leaves that one). From the plan's one caller, between forwards. Returns the expert
// the slot held or -1; -2 invalid arguments, -3 a forward in progress.
int xtier_place(void *plan,int expert,int slot);
// Cold experts handled by the forward: files[e*nodes+n] / offsets[e*nodes+n] locate expert e's node part n in the
// store (cold/store.cpp; the six pieces of xtier_slot_pieces back to back), files[e*nodes] = -1 for an expert that
// never leaves RAM (it must be in a slot). Before a forward runs, the experts its routes name that are not in RAM are read into the
// slots of the experts with the lowest decayed use counts (initial counts `scores`, may be NULL; halved every
// `half_life` forwards) that the call does not use and nobody pinned. With XTIER_COLD_OVERLAP=1 (default) the forward
// computes the routes to the experts in RAM while those reads run, then the others (two FP32 sums, added: not bitwise
// the one-pass order); 0: it waits for the reads first. Before the first forward. 0 or -1.
int xtier_cold_attach(void *plan,void *store,const int32_t *files,const long long *offsets,const float *scores,float half_life);
// For readers on other threads (the GPU pool's copies): the slot holding `expert`, kept until xtier_slot_unpin;
// -1 if it is not in RAM.
int xtier_slot_pin(void *plan,int expert);
void xtier_slot_unpin(void *plan,int slot);
// out[6]: forwards, experts read from NVMe, seconds the forwards waited for reads; the same for the forwards of at
// most 8 tokens (decode steps).
void xtier_cold_plan_stats(void *plan,double *out);
// table[0..experts): the slot of every expert, -1 if in none. Returns the slot count, or -1 (-3: a forward in progress).
int xtier_slot_table(void *plan,int32_t *table);
// One slot's six pieces on worker node `node`, ordered as in xtier_shard (gate, up, gate scales, up scales, down,
// down scales): where a cold expert's weights are written. 0 or -1.
int xtier_slot_pieces(void *plan,int slot,int node,void **pointers,size_t *bytes);
// GPT-OSS SwiGLU (MiniMax M3): gate*sigmoid(alpha*gate)*(up+beta) after the clamp. The defaults 1 / 0 keep the
// plain SiLU path; set before the first forward. 0 on success.
int xtier_set_swiglu(void *plan, float alpha, float beta);
// Four ints: hidden, inter, experts, largest decode batch per CUDA slot (8).
int xtier_shape(void *plan, int *shape);

// x: BF16 bits [m,H], ids: int32 [m,k], routing: float32 [m,k].
// y: float32 [m,H]. m=1..2048, k=1..10. Buffers must not alias.
// FP32 intermediate and accumulation; no BF16 rounding after SiLU.
// One caller per plan; different plans serialize on a process-wide shared pool.
// All active plans must request the same thread count. Create after worker spawn;
// do not fork a process holding live plans. Returns expert microgroups (equal to
// unique experts for m<=4), or -1 on error. CPU synchronous calls support batches;
// the CUDA bridge accepts m<=8 per slot.
// Bench only: groups and splits like xtier_forward, but streams the routed weight and
// scale bytes without computing (y is not written). Memory ceiling for the kernels.
int xtier_read_probe(void *plan, int m, int k, const uint16_t *x, const int32_t *ids,
                      const float *routing, float *y);
// Move weight-shard pages that were placed on another NUMA node back to their own node
// (once, after loading, before pinning). Returns ranges not fully moved, or -1.
int xtier_localize(void *plan);
int xtier_forward(void *plan, int m, int k, const uint16_t *x, const int32_t *ids,
                   const float *routing, float *y);
// Query only from the sequential caller after a completed forward, not concurrently.
// Two wall-clock durations in microseconds: gate+SiLU barrier, down+reduce barrier.
void xtier_last_timings(void *plan, double *times);
// Read-only view of worker node's immutable weight shard (valid until the last handle is
// destroyed), with N = xtier_nodes: gate [E,I/N,H/2], up [E,I/N,H/2], gate scales [E,I/N,H/16],
// up scales [E,I/N,H/16], down [E,H/N,I/2], down scales [E,H/N,I/16]; MXFP4 uses /32
// instead of /16 for all scale widths; rows node*I/N..
// (gate/up) and node*H/N.. (down) of the original tensors. Returns 0 or -1.
int xtier_shard(void *plan, int node, void **pointers, size_t *bytes);
// Number of worker nodes (weight shards) of the plan, or -1.
int xtier_nodes(void *plan);
// CPUs the expert workers bind to for `threads` (same rule as create, current topology and
// XTIER_NODE_CORES); fills cpus[0..threads) and returns threads, or -1. No threads started.
int xtier_worker_cpus(int threads, int *cpus, int capacity);
// Independent handle retaining the same immutable weights and per-plan scratch.
void *xtier_retain(void *plan);
const char *xtier_last_error(void);
// Kernel build in use: "avx512", "avx2", or "none" (unsupported CPU / XTIER_ISA).
const char *xtier_isa(void);
// Call only after all forward calls have completed.
void xtier_destroy(void *plan);

// Cold experts on NVMe (cold/store.cpp, xtier/cache/cold.py): expert records in the shard layout, one file per drive,
// read with O_DIRECT straight into RAM slots and written with O_DIRECT through an aligned bounce buffer, so no data
// passes the page cache. Offsets and every piece size are multiples of 512 bytes; read buffers are 512-aligned (RAM
// slots are). `threads` readers per file. NULL on error (xtier_cold_error).
void *xtier_cold_open(const char *const *paths,int count,int threads,int writable);
// One record: the n pieces back to back at `offset` of file `file`. 0 or -1.
int xtier_cold_write(void *store,int file,long long offset,const void *const *pieces,const size_t *sizes,int n);
// `jobs` reads at once, job j: file files[j], offset offsets[j], n pieces into dst[j*n..] of sizes[j*n..] (one
// preadv); the files' readers run them in parallel; returns when every job is done. 0 or -1.
int xtier_cold_read(void *store,int jobs,const int *files,const long long *offsets,void *const *dst,const size_t *sizes,
                    int n);
// Asynchronous reads (a GPU prefill's staging): like xtier_cold_read, but returns at once with a ticket (NULL on
// error); job j counts toward group groups[j] (0 <= groups[j] < group_count). xtier_cold_read's jobs run first.
void *xtier_cold_submit(void *store,int jobs,const int *files,const long long *offsets,void *const *dst,
                        const size_t *sizes,int n,const int *groups,int group_count);
// Returns once every job of groups [g0, g1) is done. 0 or -1 (a job of the ticket failed).
int xtier_cold_wait(void *store,void *ticket,int g0,int g1);
// Waits for all of the ticket's jobs, counts them in xtier_cold_stats and frees the ticket. 0 or -1.
int xtier_cold_finish(void *store,void *ticket);
// out[4]: bytes read, bytes written, read jobs, seconds from queueing a read to its last job done (summed).
void xtier_cold_stats(void *store,double *out);
void xtier_cold_close(void *store);
const char *xtier_cold_error(void);

#ifdef __cplusplus
}
#endif
