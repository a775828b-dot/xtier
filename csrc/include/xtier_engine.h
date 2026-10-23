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

#ifdef __cplusplus
}
#endif
