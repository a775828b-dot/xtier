"""Grouped GPU prefill of a MoE layer whose experts live in host memory (the CPU engine's NUMA shards).
Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_prefill.py: names changed only.

Compressed weights and FP8 block scales are unchanged. Gate/up use exact BF16
operands with FP32 accumulation; activation, down and route reduction stay FP32.
Calls are synchronous, serialized, and release temporary device weights before
returning to attention. Retaining the workspace is an explicit experimental opt-in.
"""
import ctypes as C
import gc
import os
import threading
import weakref
import torch
import triton as tr
import triton.language as tl


class Pointer:
    def __init__(self, address, dtype):
        self.address = address
        self.dtype = dtype

    def data_ptr(self):
        return self.address


@tr.jit
def _copy_ids(S, D, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(D + i, tl.load(S + i, i < N, 0), i < N)


@tr.jit
def _weight(W, S, e, row, col, ROWS: tl.constexpr, COLS: tl.constexpr, MX: tl.constexpr = False):
    packed = tl.load(W + e * ROWS * (COLS // 2) + row * (COLS // 2) + col // 2)
    q = (packed >> ((col % 2) * 4)) & 15
    a = q & 7
    v = tl.where(a < 4, a * 0.5, tl.where(a == 4, 2., tl.where(a == 5, 3., tl.where(a == 6, 4., 6.))))
    v = tl.where(q >= 8, -v, v)
    s = tl.load(S + e * ROWS * (COLS // (32 if MX else 16)) + row * (COLS // (32 if MX else 16)) + col // (32 if MX else 16))
    if MX:
        # UE8M0 byte 0 represents 2^-127 (a float32 subnormal).
        scale = tl.where(s == 0, 2.0 ** -127, (s.to(tl.uint32) << 23).to(tl.float32, bitcast=True))
    else:
        scale = s.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return v * scale


@tr.jit
def _unpack_all(W, S, D, ROWS: tl.constexpr, COLS: tl.constexpr, B: tl.constexpr, MX: tl.constexpr = False):
    e = tl.program_id(0).to(tl.int64)  # E*rows*cols exceeds 2**31 for DeepSeek V4.1 layers
    i = tl.program_id(1)*B + tl.arange(0,B)
    value = _weight(W,S,e,i//COLS,i%COLS,ROWS,COLS,MX)
    tl.store(D+e*ROWS*COLS+i,value)


@tr.jit
def _gate(X, W, S, G, ORDER, TILE_E, TILE_START, TILE_END, MID, E0, R0,
          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, UNPACKED: tl.constexpr,
          HID: tl.constexpr, INT: tl.constexpr, LIMIT: tl.constexpr, MX: tl.constexpr = False,
          ALPHA: tl.constexpr = 1.0, BETA: tl.constexpr = 0.0):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)  # expert offsets exceed 2**31 for DeepSeek V4.1
    ew = e - E0  # index inside the staged expert chunk; global scales use e
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    end = tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, r < end, 0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for base in range(0, HID, BK):
        c = base + kk
        x = tl.load(X + (route // K)[:, None] * HID + c[None, :], (r < end)[:, None], 0)
        if UNPACKED:
            w0 = tl.load(W + ew*(2*INT)*HID + n[None,:]*HID + c[:,None])
            w1 = tl.load(W + ew*(2*INT)*HID + (n+INT)[None,:]*HID + c[:,None])
        else:
            w0 = _weight(W, S, ew, n[None, :], c[:, None], 2*INT, HID, MX).to(tl.bfloat16)
            w1 = _weight(W, S, ew, (n + INT)[None, :], c[:, None], 2*INT, HID, MX).to(tl.bfloat16)
        gate = tl.dot(x, w0, gate)
        up = tl.dot(x, w1, up)
    gate *= tl.load(G + e * 2)
    up *= tl.load(G + e * 2 + 1)
    if LIMIT > 0:
        # Clamped SwiGLU (DeepSeek V4): up to [-L,L], gate to (-inf,L].
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        gate = tl.minimum(gate, LIMIT)
    if ALPHA != 1.0 or BETA != 0.0:   # GPT-OSS SwiGLU (MiniMax M3)
        value = gate / (1. + tl.exp(-ALPHA * gate)) * (up + BETA)
    else:
        value = gate / (1. + tl.exp(-gate)) * up
    tl.store(MID + (r - R0)[:, None] * INT + n[None, :], value, (r < end)[:, None])


@tr.jit
def _down(MID, W, S, G, ORDER, TILE_E, TILE_START, TILE_END, ROUTE_OUT, E0, R0,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PRECISION: tl.constexpr, UNPACKED: tl.constexpr,
          HID: tl.constexpr, INT: tl.constexpr, BY_ROW: tl.constexpr = False, MX: tl.constexpr = False):
    tile = tl.program_id(0)
    e = tl.load(TILE_E + tile).to(tl.int64)  # expert offsets exceed 2**31 for DeepSeek V4.1
    ew = e - E0  # index inside the staged expert chunk; global scales use e
    r = tl.load(TILE_START + tile) + tl.arange(0, BM)
    end = tl.load(TILE_END + tile)
    route = tl.load(ORDER + r, r < end, 0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, INT, BK):
        c = base + kk
        x = tl.load(MID + (r - R0)[:, None] * INT + c[None, :], (r < end)[:, None], 0)
        if UNPACKED:
            w = tl.load(W + ew*HID*INT + n[None,:]*INT + c[:,None]).to(tl.float32)
        else:
            w = _weight(W, S, ew, n[None, :], c[:, None], HID, INT, MX)
        if PRECISION == 'bf16':
            acc = tl.dot(x.to(tl.bfloat16), w.to(tl.bfloat16), acc)
        elif PRECISION == 'split-bf16':
            hi = x.to(tl.bfloat16)
            rem = x - hi.to(tl.float32)
            lo = rem.to(tl.bfloat16)
            tail = (rem - lo.to(tl.float32)).to(tl.bfloat16)
            wb = w.to(tl.bfloat16)
            acc = tl.dot(tail, wb, acc)
            acc = tl.dot(lo, wb, acc)
            acc = tl.dot(hi, wb, acc)
        else:
            acc = tl.dot(x, w, acc, input_precision=PRECISION)
    acc *= tl.load(G + e)
    if BY_ROW:  # chunk-local rows (XTIER_PREFILL_ROUTE_ACCUM); summed into the token by _accum
        tl.store(ROUTE_OUT + (r - R0)[:, None] * HID + n[None, :], acc, (r < end)[:, None])
    else:
        tl.store(ROUTE_OUT + route[:, None] * HID + n[None, :], acc, (r < end)[:, None])


@tr.jit
def _reduce(R, WEIGHTS, OUT, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr, HID: tl.constexpr):
    t = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0., tl.float32)
    for k in range(K):
        value = tl.load(R + (t * K + k) * HID + n, n < HID, 0)
        weight = tl.load(WEIGHTS + t * K + k)
        acc += value * weight
    tl.store(OUT + t * HID + n, acc, n < HID)


@tr.jit
def _accum(R, ORDER, INV, WEIGHTS, ACC, R0, ROWS, K: tl.constexpr, B: tl.constexpr, HID: tl.constexpr):
    """Add one expert chunk's routes (rows R0..R0+ROWS of the sorted order, stored chunk-locally in R)
    into the FP32 per-token accumulator. The token's lowest route slot inside the chunk adds all of its
    in-chunk slots in slot order, so each token is written by one program: no atomics, deterministic."""
    i = tl.program_id(0)
    n = tl.program_id(1) * B + tl.arange(0, B)
    route = tl.load(ORDER + R0 + i)
    t = route // K
    s = route % K
    lead = True
    for q in tl.static_range(K):
        p = tl.load(INV + t * K + q)
        lead = lead & ((q >= s) | (p < R0) | (p >= R0 + ROWS))
    if lead:
        acc = tl.load(ACC + t.to(tl.int64) * HID + n, n < HID, 0)
        for q in tl.static_range(K):
            p = tl.load(INV + t * K + q)
            inside = (p >= R0) & (p < R0 + ROWS)
            value = tl.load(R + (p - R0).to(tl.int64) * HID + n, (n < HID) & inside, 0)
            weight = tl.where(inside, tl.load(WEIGHTS + t * K + q), 0.)
            acc += value * weight
        tl.store(ACC + t.to(tl.int64) * HID + n, acc, n < HID)


@tr.jit
def _emit(ACC, OUT, B: tl.constexpr, HID: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1) * B + tl.arange(0, B)
    tl.store(OUT + t * HID + n, tl.load(ACC + t * HID + n, n < HID, 0), n < HID)


_lock = threading.Lock()
_slots = None      # packed staging slots, each [w13, w2, s13, s2] for one expert chunk
_unpacked = None   # BF16 weights of the chunk being computed (XTIER_PREFILL_UNPACK=1)
_pending = None    # (plan, upload-done event, slot) of a cross-layer prefetch
_copy_stream = None
_last_plan = None
_unpack_stream = None  # XTIER_PREFILL_OVERLAP: FP4->BF16 unpack stream
_unpacked2 = None      # XTIER_PREFILL_OVERLAP: two alternating BF16 weight buffers


def _pending_chunks(pending):
    """[(chunk, upload-done event, slot), ...] of a cross-layer prefetch. Formats: (plan, event, slot)
    from _run; (plan, event, slot, (event, slot)) or (plan, event, slot, [(chunk, event, slot), ...])
    from _run_overlap (XTIER_PREFILL_PREFETCH_CHUNKS)."""
    chunks = [(0, pending[1], pending[2])]
    extra = pending[3] if len(pending) > 3 else None
    if isinstance(extra, list):
        chunks += extra
    elif extra is not None:
        chunks.append((1, extra[0], extra[1]))
    return chunks


def drain_prefetch():
    """Release any in-flight source ownership before model teardown or tests."""
    global _pending
    with _lock:
        if _pending is not None:
            for _, done, _ in _pending_chunks(_pending):
                done.synchronize()
            _pending = None


def link_prefill_ring(plans):
    """Explicit model-order ring; never infer ownership from call timing."""
    if not plans or len({id(p) for p in plans}) != len(plans):
        raise ValueError('Expected unique prefill plans')
    if len({(p.experts, p.hidden, p.inter) for p in plans}) != 1:
        raise ValueError('Prefetch ring requires equal weight shapes')
    drain_prefetch()
    for index, plan in enumerate(plans):
        if not plan.prefetch_enabled:
            raise ValueError('Prefetch must be explicitly enabled on every plan')
        plan.next_prefill = weakref.ref(plans[(index + 1) % len(plans)])


class _DeviceArray:
    """A raw device address as a torch tensor (no copy) through __cuda_array_interface__."""
    def __init__(self, address, shape, typestr):
        self.__cuda_array_interface__ = dict(shape=tuple(shape), typestr=typestr, data=(int(address), False),
                                             version=3, strides=None)


def _device_tensor(address, shape, dtype):
    if dtype == torch.bfloat16:
        return torch.as_tensor(_DeviceArray(address, shape, '<i2'), device='cuda').view(torch.bfloat16)
    typestr = {torch.int32: '<i4', torch.float32: '<f4'}[dtype]
    return torch.as_tensor(_DeviceArray(address, shape, typestr), device='cuda')


class _Fp4Prefetch:
    """Owner of a cross-layer prefetch from _run_fp4: the slots hold the CUTLASS layout ([up; gate],
    interleaved scales), so the BF16 paths (which compare the owner with `is self`) never reuse them."""
    def __init__(self, plan):
        self.plan = plan


_cudart = None


def _runtime():
    global _cudart
    if _cudart is None:
        lib = C.CDLL('libcudart.so.13')
        lib.cudaHostRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
        lib.cudaHostUnregister.argtypes = [C.c_void_p]
        lib.cudaMemcpy2DAsync.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t,
                                          C.c_size_t, C.c_size_t, C.c_int, C.c_void_p]
        _cudart = lib
    return _cudart


def _check(rc, what):
    if rc != 0:
        raise RuntimeError(f'{what} failed with CUDA error {rc}')


def _shard_layout(H, I, nodes=4, group=16):
    # Placement of each node shard inside the original tensors: (staging index, first
    # row of node 0, rows per node, total rows, bytes per row). Order matches xtier_shard.
    IN, HN = I // nodes, H // nodes
    return [(0, 0, IN, 2 * I, H // 2), (0, I, IN, 2 * I, H // 2), (2, 0, IN, 2 * I, H // group),
            (2, I, IN, 2 * I, H // group), (1, 0, HN, H, I // 2), (3, 0, HN, H, I // group)]


class Prefill:
    """Grouped GPU prefill of one MoE layer whose weights live in host memory.

    Weights are uploaded in expert chunks (XTIER_PREFILL_EXPERT_CHUNK experts, default
    all): two packed staging slots alternate so the copy engine uploads chunk j+1 while
    chunk j is unpacked and computed. With one chunk this is the original single-staging
    path (Qwen3.8); DeepSeek V4.1's 7.6 GB layers use chunks to bound device memory.
    Results do not depend on the chunk size: every route is computed by the same kernels
    with the same operands and reduced in the same order (with XTIER_PREFILL_ROUTE_ACCUM=1
    the FP32 route sum runs in chunk order instead, see __init__).
    """

    def __init__(self, experts, pointers, native=None, hidden=2560, inter=640, swiglu_limit=0.0, swiglu_alpha=1.0,
                 swiglu_beta=0.0):
        self.mxfp4 = getattr(native, 'scale_format', 'nvfp4') == 'mxfp4'
        if self.mxfp4 and os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') != '1':
            raise ValueError('MXFP4 requires native scale shards')
        self.scale_group = 32 if self.mxfp4 else 16
        self.experts = experts
        self.hidden, self.inter, self.swiglu_limit = int(hidden), int(inter), float(swiglu_limit)
        # GPT-OSS SwiGLU (MiniMax M3): gate * sigmoid(alpha * gate) * (up + beta); 1 / 0 = plain SiLU
        self.swiglu_alpha, self.swiglu_beta = float(swiglu_alpha), float(swiglu_beta)
        H, I = self.hidden, self.inter
        self.layout = _shard_layout(H, I, group=self.scale_group)
        # Experts per uploaded chunk. Default: the whole layer while its unpacked BF16
        # weights fit in 6 GiB (Qwen3.8: ~5 GB, unchanged single staging); otherwise
        # ~1.6 GiB of BF16 per chunk (DeepSeek V4.1: 24 of 384 experts), which also lets
        # the upload of one chunk overlap the computation of the previous one.
        chunk = int(os.environ.get('XTIER_PREFILL_EXPERT_CHUNK', '0') or 0)
        per_expert = 3 * H * I * 2
        if chunk <= 0:
            chunk = experts if experts * per_expert <= 6 << 30 else max(1, (1638 << 20) // per_expert)
        self.chunk = min(chunk, experts)
        # XTIER_PREFILL_PREFETCH=1: while this layer computes, the copy engine uploads
        # the next layer's first weight chunk into a free staging slot. The successor is
        # learned from the observed call order (a mismatch only costs a normal upload).
        self.learned_ring = os.environ.get('XTIER_PREFILL_PREFETCH') == '1'
        self.prefetch_enabled = self.learned_ring or os.environ.get('XTIER_EXPERIMENTAL_PREFILL_PREFETCH') == '1'
        # Keeping all unpacked expert weights alive across layers adds ~6 GiB
        # to the next attention call. Return them to the allocator by default.
        self.release_workspace = os.environ.get('XTIER_PREFILL_RELEASE_WORKSPACE', '1') == '1'
        if self.release_workspace and self.prefetch_enabled and not self.learned_ring:
            raise ValueError('Workspace release cannot retain a next-layer prefetch')
        self.workspace_releases = self.workspace_gc_collections = 0
        self.next_prefill = None
        self.order = 0  # creation (layer) order, set by the standalone adapter
        self.prefetch_hits = self.prefetch_submissions = self.prefetch_uploads = 0
        self.chunk_uploads = self.chunks_skipped = 0
        if self.prefetch_enabled and os.environ.get('XTIER_PREFILL_UNPACK') != '1':
            raise ValueError('Prefetch requires unpacked weights')
        # XTIER_PREFILL_ROUTE_ACCUM=1: instead of one FP32 row per route for the whole call
        # ([m*k, H], 4.0 GB for DeepSeek V4.1 at 32K tokens), each expert chunk's routes go to a
        # chunk-sized buffer and are added into an FP32 [m, H] accumulator right away. Same kernels
        # and operands; only the order of the FP32 additions changes (chunk order, then slot order),
        # which is bitwise identical with one chunk. Meant for several chunks.
        self.route_accum = os.environ.get('XTIER_PREFILL_ROUTE_ACCUM') == '1'
        self.down_tile128 = os.environ.get('XTIER_EXPERIMENTAL_PREFILL_TILE128') == '1'
        if self.down_tile128 and os.environ.get('XTIER_PREFILL_UNPACK') != '1':
            raise ValueError('Experimental down tile requires validated unpacked BF16 weights')
        if self.down_tile128 and os.environ.get('XTIER_DOWN_PRECISION','tf32x3') != 'tf32x3':
            raise ValueError('Experimental down tile requires the original tf32x3 precision')
        shapes = [(experts, 2 * I, H // 2), (experts, H, I // 2),
                  (experts, 2 * I, H // self.scale_group), (experts, H, I // self.scale_group), (experts, 2), (experts,)]
        self.shapes = shapes
        self.slot_shapes = [(self.chunk,) + s[1:] for s in shapes[:4]]
        self.device_globals = None
        self.weights = []
        self.shards = None
        self.native = native
        self.registered = False
        if native is not None and os.environ.get('XTIER_PREFILL_FROM_SHARDS', '1') == '1':
            # Upload straight from the NUMA-bound CPU shards instead of keeping a second
            # full pinned copy of every CPU layer (~1.4 GB each). Bytes are identical.
            shards = native.shards()
            if shards is not None:
                # Pinned lazily (first upload): pages must first be moved to their NUMA node,
                # and pinned pages cannot migrate.
                self.shards = shards
                self.layout = _shard_layout(H, I, len(shards), self.scale_group)
                for i in (4, 5):
                    w = torch.empty(shapes[i], dtype=torch.float32, pin_memory=True)
                    C.memmove(w.data_ptr(), pointers[i], w.numel() * w.element_size())
                    self.weights.append(w)
                return
        for i, (shape, ptr) in enumerate(zip(shapes, pointers)):
            w = torch.empty(shape, dtype=torch.uint8 if i < 4 else torch.float32, pin_memory=True)
            C.memmove(w.data_ptr(), ptr, w.numel() * w.element_size())
            self.weights.append(w)

    def ensure_registered(self):
        if self.shards is None or self.registered:
            return
        if self.native is not None and os.environ.get('XTIER_LOCALIZE', '1') != '0':
            self.native.localize()
        cudart = _runtime(); registered = []
        try:
            for row in self.shards:
                for address, size in row:
                    _check(cudart.cudaHostRegister(address, size, 0), 'cudaHostRegister')
                    registered.append(address)
        except BaseException:
            for address in registered: cudart.cudaHostUnregister(address)
            raise
        self.registered = True

    def close(self):
        if self.shards is not None:
            if self.registered:
                torch.cuda.synchronize()
                for row in self.shards:
                    for address, _ in row: _runtime().cudaHostUnregister(address)
            self.shards = None

    def _globals(self):
        # Per-projection global scales stay on the device (a few KiB per layer), so a
        # prefetch into the shared staging never races with this layer's kernels.
        if self.device_globals is None:
            self.device_globals = tuple(w.cuda() for w in self.weights[-2:])
        return self.device_globals

    def _upload_chunk(self, slot, e0, e1, stream, up_first=False):
        """Copy experts [e0, e1) into a staging slot on `stream` (bytes unchanged). up_first: W13 and
        its scales as [up; gate] (FlashInfer CUTLASS) instead of the engine's [gate; up]."""
        self.ensure_registered()
        n = e1 - e0
        I = self.inter
        if self.shards is None:
            with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
                for i, (dst, src) in enumerate(zip(slot, self.weights[:4])):
                    if up_first and i in (0, 2):  # per expert: contiguous halves (async from pinned memory)
                        for e in range(n):
                            dst[e, :I].copy_(src[e0 + e, I:], non_blocking=True)
                            dst[e, I:].copy_(src[e0 + e, :I], non_blocking=True)
                    else:
                        dst[:n].copy_(src[e0:e1], non_blocking=True)
            return
        layout = self.layout
        if up_first:  # gate rows (first 0) go to I.., up rows (first I) to 0..
            layout = [(index, (I - first) if index in (0, 2) else first, rows, total, width)
                      for index, first, rows, total, width in self.layout]
        cudart = _runtime()
        for node, row in enumerate(self.shards):
            for (address, size), (index, first, rows, total, width) in zip(row, layout):
                dst = slot[index].data_ptr() + (first + node * rows) * width
                _check(cudart.cudaMemcpy2DAsync(dst, total * width, address + e0 * rows * width, rows * width,
                                                rows * width, n, 1, stream), 'cudaMemcpy2DAsync')

    def run(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16):
        global _last_plan
        if self.learned_ring:
            previous = _last_plan() if _last_plan is not None else None
            # Forward links only (layer order): no prefetch from the last layer back to
            # the first, so the staging buffer is released between requests.
            if previous is not None and previous is not self and previous.order < self.order:
                previous.next_prefill = weakref.ref(self)
            _last_plan = weakref.ref(self)
        retired = self._run(x, out, ids, routing, m, k, stream, output_dtype)
        if retired:
            self.workspace_releases += 1
            # A first Triton specialization can leave cyclic compilation frames
            # holding the previous global workspace list. Run GC only when weak
            # references prove the buffers survived the completed call frame.
            if any(ref() is not None for ref in retired):
                gc.collect()
                self.workspace_gc_collections += 1

    def _run(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16):
        global _slots, _unpacked, _pending, _copy_stream
        if m < 1 or not 1 <= k <= 10:
            raise ValueError('Invalid prefill shape')
        H, I, E, CH = self.hidden, self.inter, self.experts, self.chunk
        n_chunks = (E + CH - 1) // CH
        n_slots = 1 if n_chunks == 1 else 2
        unpacked = os.environ.get('XTIER_PREFILL_UNPACK', '0') == '1'
        if os.environ.get('XTIER_PREFILL_FP4') == '1' and self._fp4_ready(output_dtype):
            return self._run_fp4(x, out, ids, routing, m, k, stream)
        if (os.environ.get('XTIER_PREFILL_OVERLAP') == '1' and unpacked and n_chunks > 1
                and self.route_accum and not self.down_tile128 and output_dtype == torch.bfloat16):
            # (hybrid CPU/GPU calls pass FP32 output and use few chunks: keep the chunk-skipping path)
            return self._run_overlap(x, out, ids, routing, m, k, stream, output_dtype)
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            events=[]
            def stamp(name):
                if os.environ.get('XTIER_PREFILL_PROFILE')=='1':
                    event=torch.cuda.Event(enable_timing=True);event.record();events.append((name,event))
            stamp('start')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            if _copy_stream is None:
                _copy_stream = torch.cuda.Stream()
            ready = {}  # chunk index -> (slot, upload-done event)
            if _pending is not None:
                target, done, slot = _pending[:3]
                for _, more, _ in _pending_chunks(_pending)[1:]:
                    # further prefetched chunks of XTIER_PREFILL_OVERLAP: not used on this path
                    more.synchronize()
                _pending = None
                if self.prefetch_enabled and target is self and _slots is not None \
                        and len(_slots) == n_slots and _slots[slot] is not None \
                        and [tuple(t.shape) for t in _slots[slot]] == self.slot_shapes:
                    ready[0] = (slot, done)
                else:
                    # CPU rendezvous also retains pinned source ownership until DMA
                    # completes. Out-of-order calls wait and upload their own weights.
                    done.synchronize()
            if _slots is None or len(_slots) != n_slots:
                _slots = [None] * n_slots
            for j in range(n_slots):
                if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                    _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
            if ready:
                self.prefetch_hits += 1
            else:
                self.prefetch_uploads += 1
            g13, g2 = self._globals()
            freed = [None] * n_slots  # event: slot contents no longer needed by compute

            def upload(j, slot):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                _copy_stream.wait_stream(compute) if freed[slot] is None else _copy_stream.wait_event(freed[slot])
                self._upload_chunk(_slots[slot], e0, e1, _copy_stream.cuda_stream)
                done = torch.cuda.Event(); done.record(_copy_stream)
                ready[j] = (slot, done)
                self.chunk_uploads += 1

            # Routing metadata (small host rendezvous). One chunk: token blocks of
            # XTIER_GPU_PREFILL_CHUNK as before. Several chunks: one block, so each weight
            # chunk is uploaded once per call.
            block = int(os.environ.get('XTIER_GPU_PREFILL_CHUNK', '4096'))
            if block not in (4096, 8192, 16384):raise ValueError('Unsupported GPU prefill chunk')
            if n_chunks > 1:
                block = m
            state = dict(current=None)
            # Tile shapes (defaults are the validated Qwen3.8 kernels): routes per tile,
            # gate BN/BK, down BN/BK, and optional warps/stages as "w,s".
            def env_int(name, default):
                return int(os.environ.get(name, default))
            tm, gbn, gbk = env_int('XTIER_PREFILL_TILE_M', 32), env_int('XTIER_GATE_BN', 64), env_int('XTIER_GATE_BK', 64)
            dbn, dbk = env_int('XTIER_DOWN_BN', 64), env_int('XTIER_DOWN_BK', 32)
            def launch_opts(name):
                v = os.environ.get(name)
                return {} if not v else dict(zip(('num_warps', 'num_stages'), map(int, v.split(','))))
            gate_opts, down_opts = launch_opts('XTIER_GATE_LAUNCH'), launch_opts('XTIER_DOWN_LAUNCH')

            def acquire(j, active):
                """Weights of chunk j ready for compute; also queue the next active chunk."""
                global _unpacked
                if state['current'] == j:
                    return state['weights']
                if j not in ready:
                    upload(j, j % n_slots if n_slots > 1 else 0)
                slot, done = ready.pop(j)
                compute.wait_event(done)
                w = _slots[slot]
                position = active.index(j)
                if position + 1 < len(active) and active[position + 1] not in ready:
                    nxt = active[position + 1]
                    nslot = (slot + 1) % n_slots
                    if n_slots > 1:
                        upload(nxt, nslot)
                if unpacked:
                    shape = (CH, 2 * I, H)
                    if _unpacked is None or tuple(_unpacked[0].shape) != shape:
                        _unpacked = [torch.empty(shape, dtype=torch.bfloat16, device='cuda'),
                                     torch.empty((CH, H, I), dtype=torch.bfloat16, device='cuda')]
                    count = min(E, (j + 1) * CH) - j * CH
                    _unpack_all[(count, 2 * I * H // 1024)](w[0], w[2], _unpacked[0], 2 * I, H, 1024, MX=self.mxfp4)
                    _unpack_all[(count, H * I // 1024)](w[1], w[3], _unpacked[1], H, I, 1024, MX=self.mxfp4)
                    free = torch.cuda.Event(); free.record(compute); freed[slot] = free
                    weights = (_unpacked[0], _unpacked[1], w[2], w[3], slot)
                    stamp('unpack')
                else:
                    weights = (w[0], w[1], w[2], w[3], slot)
                state.update(current=j, weights=weights)
                return weights

            stamp('upload')
            for start in range(0, m, block):
                size = min(block, m - start)
                di = torch.empty(size * k, dtype=torch.int32, device='cuda')
                _copy_ids[(tr.cdiv(size*k, 256),)](Pointer(ids+start*k*4, torch.int32), di, size*k, 256)
                # The small host rendezvous validates routes and builds bounded
                # tile metadata. No assumptions about balanced expert routing.
                host_ids = di.cpu()
                # -1 marks a route computed elsewhere (hybrid CPU/GPU prefill): it sorts
                # first, gets no tile, and its routing weight must be 0.
                skipped = int((host_ids == -1).sum())
                if int(host_ids.min()) < -1 or int(host_ids.max()) >= E:
                    raise ValueError('Invalid expert id')
                counts = torch.bincount(host_ids[host_ids >= 0] if skipped else host_ids, minlength=E).tolist()
                order = torch.argsort(di, stable=True).to(torch.int32)
                tiles = []; offset = skipped; spans = []  # per chunk: (tile begin, tile end, row begin, rows)
                for j in range(n_chunks):
                    t0, r0 = len(tiles), offset
                    for e in range(j * CH, min(E, (j + 1) * CH)):
                        count = counts[e]
                        for q in range(0, count, tm):
                            tiles.append((e, offset + q, offset + count))
                        offset += count
                    spans.append((t0, len(tiles), r0, offset - r0))
                meta = torch.tensor(tiles, dtype=torch.int32).T.contiguous().cuda()
                active = [j for j in range(n_chunks) if spans[j][3] > 0]
                self.chunks_skipped += n_chunks - len(active)
                rows_max = max([s[3] for s in spans] + [1])
                mid = torch.empty((rows_max, I), dtype=torch.float32, device='cuda')
                accum = self.route_accum
                if accum:
                    inv = torch.empty_like(order)
                    inv[order.long()] = torch.arange(size * k, dtype=torch.int32, device='cuda')
                    acc = torch.zeros((size, H), dtype=torch.float32, device='cuda')
                    routes = torch.empty((rows_max, H), dtype=torch.float32, device='cuda')
                    weights_ptr = Pointer(routing + start * k * 4, torch.float32)
                else:
                    routes = (torch.zeros if skipped else torch.empty)((size*k, H), dtype=torch.float32, device='cuda')
                stamp('route')
                for j in active:
                    t0, t1, r0, rows = spans[j]
                    w13, w2, s13, s2, slot = acquire(j, active)
                    e0 = j * CH
                    _gate[(t1 - t0, I // gbn)](Pointer(x+start*H*2, torch.bfloat16), w13, s13, g13,
                                             order, meta[0, t0:t1], meta[1, t0:t1], meta[2, t0:t1], mid, e0, r0,
                                             k, tm, gbn, gbk, unpacked, H, I, self.swiglu_limit, MX=self.mxfp4,
                                             ALPHA=self.swiglu_alpha, BETA=self.swiglu_beta, **gate_opts)
                    stamp('gate')
                    if self.down_tile128:
                        _down[(t1 - t0, H // 128)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                                   meta[2, t0:t1], routes, e0, r0,
                                                   32, 128, 32, os.environ.get('XTIER_DOWN_PRECISION','tf32x3'),
                                                   unpacked, H, I, accum, MX=self.mxfp4, num_warps=8, num_stages=3)
                    else:
                        _down[(t1 - t0, H // dbn)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                                  meta[2, t0:t1], routes, e0, r0,
                                                  tm, dbn, dbk, os.environ.get('XTIER_DOWN_PRECISION','tf32x3'),
                                                  unpacked, H, I, accum, MX=self.mxfp4, **down_opts)
                    if not unpacked:
                        free = torch.cuda.Event(); free.record(compute); freed[slot] = free
                    stamp('down')
                    if accum:
                        _accum[(rows, tr.cdiv(H, 256))](routes, order, inv, weights_ptr, acc, r0, rows, k, 256, H)
                        stamp('accum')
                dst = Pointer(out+start*H*(2 if output_dtype==torch.bfloat16 else 4), output_dtype)
                if accum:
                    _emit[(size, tr.cdiv(H, 256))](acc, dst, 256, H)
                else:
                    _reduce[(size, tr.cdiv(H, 256))](routes, Pointer(routing+start*k*4, torch.float32), dst,
                                        size, k, 256, H)
                stamp('reduce')
            # Cross-layer prefetch: the next layer's first chunk into a slot whose
            # contents this call no longer needs (the only slot, once unpacked).
            target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
            if target is not None and unpacked and target.slot_shapes == self.slot_shapes:
                last = state['weights'][4] if state['current'] is not None else 0
                slot = (last + 1) % n_slots
                try:
                    _copy_stream.wait_stream(compute) if freed[slot] is None else _copy_stream.wait_event(freed[slot])
                    target._upload_chunk(_slots[slot], 0, min(target.experts, target.chunk), _copy_stream.cuda_stream)
                    for dst in _slots[slot]:
                        dst.record_stream(_copy_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    _pending = (target, done, slot)
                    self.prefetch_submissions += 1
                except BaseException:
                    _copy_stream.synchronize()
                    raise
            # Ensures callers can free pointer-only inputs immediately and the
            # next layer/stream can safely overwrite the shared staging buffers.
            torch.cuda.current_stream().synchronize()
            self.last_profile=[dict(phase=events[i][0],ms=events[i-1][1].elapsed_time(events[i][1])) for i in range(1,len(events))]
            if self.release_workspace:
                # Kernels have completed. Return temporary weights to the CUDA
                # allocator so the next attention layer can reuse this memory.
                # No empty_cache(), arithmetic changes or CPU weight eviction.
                # A pending next-layer prefetch keeps only its compressed staging
                # slot; the unpacked workspace is always released.
                keep = _pending[2] if _pending is not None else None
                retired = []
                for j in range(len(_slots)):
                    if j != keep and _slots[j] is not None:
                        retired += [weakref.ref(t) for t in _slots[j]]
                        _slots[j] = None
                if keep is None:
                    _slots = None
                retired += [weakref.ref(t) for t in (_unpacked or [])]
                _unpacked = None
                return tuple(retired)

    def _fp4_ready(self, output_dtype):
        """XTIER_PREFILL_FP4 applies to MXFP4 layers split into several equal expert chunks whose
        global scales are all 1 (CUTLASS takes one fc1 global per expert); BF16 output only."""
        ok = getattr(self, '_fp4_ok', None)
        if ok is None:
            g13, g2 = self.weights[-2], self.weights[-1]
            ok = (self.mxfp4 and self.experts % self.chunk == 0 and self.experts // self.chunk > 1
                  and bool((g13 == 1).all()) and bool((g2 == 1).all()))
            self._fp4_ok = ok
        return ok and output_dtype == torch.bfloat16

    def _upload_chunk_fp4(self, slot, e0, e1, stream):
        """Upload in the CUTLASS layout: [up; gate] rows, then the UE8M0 scales interleaved in place.
        Experts resident in the hot pool (self._fp4_residency, set per call) are copied from its arena."""
        from flashinfer import block_scale_interleave
        res = getattr(self, '_fp4_residency', None)
        pool = getattr(self.native, 'pool', None) if res is not None else None
        if pool is None:
            self._upload_chunk(slot, e0, e1, stream, up_first=True)
        else:
            e = e0
            while e < e1:
                if res[e] >= 0:
                    self._copy_from_pool(slot, e - e0, pool, res[e], stream)
                    self.pool_reused = getattr(self, 'pool_reused', 0) + 1
                    e += 1
                else:
                    f = e
                    while f < e1 and res[f] < 0:
                        f += 1
                    self._upload_chunk([t[e - e0:] for t in slot], e, f, stream, up_first=True)
                    e = f
        n = e1 - e0
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for i in (2, 3):
                scales = slot[i][:n]
                scales.copy_(block_scale_interleave(scales).reshape_as(scales))

    def _copy_from_pool(self, slot, i, pool, r, stream):
        """Expert in pool slot r -> staging expert i, [gate; up] -> [up; gate] (bytes unchanged)."""
        I = self.inter
        a = pool.arena
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            for t in (0, 2):
                slot[t][i, :I].copy_(a[t][r, I:], non_blocking=True)
                slot[t][i, I:].copy_(a[t][r, :I], non_blocking=True)
            for t in (1, 3):
                slot[t][i].copy_(a[t][r], non_blocking=True)

    def _run_fp4(self, x, out, ids, routing, m, k, stream):
        """XTIER_PREFILL_FP4=1 (MXFP4 layers): the experts on the SM120 FP4 tensor cores through
        FlashInfer's CUTLASS fused MoE (MXFP4 weights x MXFP8 activations, block-32 UE8M0, the
        official reference's activation precision), as sglang's DSv4 path runs GPU-resident experts.

        - no FP4->BF16 unpack and no BF16 weight buffers: the staging slots (same shapes as the packed
          ones) hold the CUTLASS layout, uploaded as [up; gate] with the scales interleaved on the copy
          stream; the freed memory buys more slots (XTIER_PREFILL_FP4_SLOTS, default 12);
        - per expert chunk, one fused call over only the tokens routed into it (row lists from one
          sorted unique over the routes and a single host sync of the counts), ep_size = chunks and
          ep_rank = chunk so the other routes are skipped, BF16 output added into an FP32 accumulator
          (rows are unique per chunk: deterministic);
        - the slot ring and the cross-layer prefetch work as in _run_overlap (owner _Fp4Prefetch).
        """
        global _slots, _pending, _copy_stream
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
        from flashinfer.fused_moe.core import ActivationType
        H, E, CH = self.hidden, self.experts, self.chunk
        n_chunks = E // CH
        n_slots = min(n_chunks, max(2, int(os.environ.get('XTIER_PREFILL_FP4_SLOTS', '12'))))
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            if _copy_stream is None:
                _copy_stream = torch.cuda.Stream()
            ready = {}  # chunk -> (slot, upload-done event)
            if _pending is not None:
                owner, chunks = _pending[0], _pending_chunks(_pending)
                _pending = None
                if (isinstance(owner, _Fp4Prefetch) and owner.plan is self and _slots is not None
                        and len(_slots) == n_slots
                        and all(c < n_chunks and _slots[s] is not None
                                and [tuple(t.shape) for t in _slots[s]] == self.slot_shapes for c, _, s in chunks)):
                    for c, done, s in chunks:
                        ready[c] = (s, done)
                else:
                    for _, done, _ in chunks:
                        done.synchronize()
            if _slots is None or len(_slots) != n_slots:
                _slots = [None] * n_slots
            for j in range(n_slots):
                if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                    _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
            target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
            if target is not None and (target.slot_shapes != self.slot_shapes or target.chunk != CH
                                       or not target._fp4_ready(torch.bfloat16)):
                target = None
            n_cross = min(n_slots, target.experts // target.chunk) if target is not None else 0
            cross = []  # [(next-layer chunk, upload-done event, slot), ...]
            if ready:
                self.prefetch_hits += 1
            else:
                self.prefetch_uploads += 1
            freed = [None] * n_slots  # event on the compute stream: slot no longer read
            state = dict(next=0, cross=0)
            issued = set(ready)  # chunks of this layer already uploaded (computed ones leave `ready`)

            def upload(j, slot):
                if freed[slot] is None:
                    _copy_stream.wait_stream(compute)
                else:
                    _copy_stream.wait_event(freed[slot])
                self._upload_chunk_fp4(_slots[slot], j * CH, (j + 1) * CH, _copy_stream.cuda_stream)
                done = torch.cuda.Event(); done.record(_copy_stream)
                ready[j] = (slot, done); issued.add(j)
                self.chunk_uploads += 1

            def refill(slot):
                while state['next'] < n_chunks and state['next'] in issued:
                    state['next'] += 1
                if state['next'] < n_chunks:
                    upload(state['next'], slot); state['next'] += 1
                elif state['cross'] < n_cross:
                    c = state['cross']; state['cross'] += 1
                    if freed[slot] is None:
                        _copy_stream.wait_stream(compute)
                    else:
                        _copy_stream.wait_event(freed[slot])
                    target._upload_chunk_fp4(_slots[slot], c * CH, (c + 1) * CH, _copy_stream.cuda_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    cross.append((c, done, slot))

            # Start the weight pipeline before the host rendezvous on the routes.
            used = {s for s, _ in ready.values()}
            for slot in range(n_slots):
                if slot not in used:
                    refill(slot)

            x_t = _device_tensor(x, (m, H), torch.bfloat16)
            ids_t = _device_tensor(ids, (m, k), torch.int32)
            w_t = _device_tensor(routing, (m, k), torch.float32)
            ids64 = ids_t.long()
            token = torch.arange(m, device='cuda')
            chunk_of = torch.where(ids64 >= 0, ids64 // CH, n_chunks)
            keys = torch.unique(chunk_of * m + token[:, None])  # sorted: chunk-major, then token
            counts = torch.bincount(keys // m, minlength=n_chunks + 1)
            pools = [getattr(p.native, 'pool', None) if p is not None else None for p in (self, target)]
            tables = [p.table.long() for p in pools if p is not None]
            host = torch.cat([counts[:n_chunks], ((ids64 < -1) | (ids64 >= E)).sum()[None],
                              (ids64 == -1).sum()[None]] + tables).cpu().tolist()
            if host[n_chunks]:
                raise ValueError('Invalid expert id')
            rows = host[n_chunks + 2:]
            for p, pool in zip((self, target), pools):
                if pool is not None:
                    p._fp4_residency, rows = rows[:E], rows[E:]
            has_skipped = host[n_chunks + 1] > 0
            rows_all = keys % m
            offsets = [0]
            for c in host[:n_chunks]:
                offsets.append(offsets[-1] + int(c))
            self.chunks_skipped += sum(1 for c in host[:n_chunks] if c == 0)
            acc = torch.zeros((m, H), dtype=torch.float32, device='cuda')
            ones = torch.ones(CH, dtype=torch.float32, device='cuda')
            limit = (torch.full((CH,), self.swiglu_limit, dtype=torch.float32, device='cuda')
                     if self.swiglu_limit > 0 else None)
            # At most `cap` tokens per call, one workspace of that size for every call of the layer.
            cap = max(256, int(os.environ.get('XTIER_PREFILL_FP4_ROWS', '4096')))
            ws_key = (cap, H, self.inter, E, k, n_chunks)
            if getattr(self, '_fp4_ws_key', None) != ws_key:
                self._fp4_ws_bytes = cutlass_fused_moe_workspace_size(
                    cap, H, self.inter, E, k, x_dtype=torch.float8_e4m3fn, weight_dtype=torch.int64,
                    use_mxfp8_act_scaling=True, ep_size=n_chunks, ep_rank=0)
                self._fp4_ws_key = ws_key
            workspace = torch.empty(self._fp4_ws_bytes, dtype=torch.uint8, device='cuda')
            calls = []
            for j in range(n_chunks):
                for a in range(offsets[j], offsets[j + 1], cap):
                    calls.append((j, a, min(a + cap, offsets[j + 1])))
            self.fp4_calls = getattr(self, 'fp4_calls', 0) + len(calls)
            by_chunk = {}
            for j, a, b in calls:
                by_chunk.setdefault(j, []).append((a, b))
            for j in range(n_chunks):
                slot, done = ready.pop(j)
                compute.wait_event(done)
                for r0, r1 in by_chunk.get(j, []):
                    rows = rows_all[r0:r1]
                    ids_j, w_j = ids_t.index_select(0, rows), w_t.index_select(0, rows)
                    if has_skipped:  # route computed elsewhere: an id outside this chunk, weight 0
                        skip = ids_j < 0
                        ids_j = torch.where(skip, ((j + 1) % n_chunks) * CH, ids_j)
                        w_j = torch.where(skip, 0.0, w_j)
                    xq, xsf = mxfp8_quantize(x_t.index_select(0, rows), is_sf_swizzled_layout=True, alignment=32)
                    w = _slots[slot]
                    part = torch.empty((r1 - r0, H), dtype=torch.bfloat16, device='cuda')
                    cutlass_fused_moe(input=xq, token_selected_experts=ids_j, token_final_scales=w_j,
                                      fc1_expert_weights=w[0].view(torch.int64), fc2_expert_weights=w[1].view(torch.int64),
                                      output_dtype=torch.bfloat16,
                                      quant_scales=[w[2].view(torch.int32), ones, w[3].view(torch.int32), ones],
                                      input_sf=xsf, swiglu_limit=limit, use_w4_group_scaling=False,
                                      use_mxfp8_act_scaling=True, activation_type=ActivationType.Swiglu,
                                      ep_size=n_chunks, ep_rank=j, tune_max_num_tokens=cap, output=part,
                                      workspace_buffer=workspace)
                    acc.index_add_(0, rows, part.float())
                ev = torch.cuda.Event(); ev.record(compute); freed[slot] = ev
                refill(slot)
            _emit[(m, tr.cdiv(H, 256))](acc, Pointer(out, torch.bfloat16), 256, H)
            if cross:
                cross.sort(key=lambda t: t[0])
                _pending = (_Fp4Prefetch(target), cross[0][1], cross[0][2], cross[1:])
                self.prefetch_submissions += 1
            torch.cuda.current_stream().synchronize()
            for p in (self, target):  # copies already enqueued keep their decisions
                if p is not None:
                    p._fp4_residency = None
            self.last_profile = []
            if self.release_workspace:
                keep = {s for _, _, s in _pending_chunks(_pending)} if _pending is not None else set()
                retired = []
                for j in range(len(_slots)):
                    if j not in keep and _slots[j] is not None:
                        retired += [weakref.ref(t) for t in _slots[j]]
                        _slots[j] = None
                if not keep:
                    _slots = None
                return tuple(retired)

    def _run_overlap(self, x, out, ids, routing, m, k, stream, output_dtype=torch.bfloat16):
        """XTIER_PREFILL_OVERLAP=1 (unpacked weights, several expert chunks, route accumulation).

        Same kernels, tiles and FP32 accumulation order as _run, so outputs are bitwise identical;
        only the scheduling differs:
        - FP4->BF16 unpack runs on its own stream into two alternating BF16 buffers, so chunk j+1
          is unpacked while the gate/down kernels of chunk j run (unpack is bandwidth bound, the
          GEMMs are compute bound);
        - N packed staging slots (XTIER_PREFILL_PREFETCH_CHUNKS, default 2) form a ring: as soon as
          chunk j is unpacked, its slot takes the upload of chunk j+N, and in the last N chunks of the
          layer it takes chunk j+N-n_chunks of the next layer (cross-layer prefetch), which then
          uploads during this layer's tail and the next attention layer (N=16 at GLM-5.3 shapes:
          16 x 216 MiB, 256 of 288 experts);
        - the first upload/unpack is issued before the host rendezvous on the routes, so the GPU is
          busy while the host builds tile metadata (built with numpy, same order as the loop);
        - every chunk is treated as active (an empty chunk is unpacked but gets no tiles).
        """
        global _slots, _unpacked, _pending, _copy_stream, _unpack_stream, _unpacked2
        import numpy as np
        H, I, E, CH = self.hidden, self.inter, self.experts, self.chunk
        n_chunks = (E + CH - 1) // CH
        n_slots = min(n_chunks, max(2, int(os.environ.get('XTIER_PREFILL_PREFETCH_CHUNKS', '2'))))
        with _lock, torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            compute = torch.cuda.current_stream()
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GPU prefill capture is not supported; use native CPU decode')
            if _copy_stream is None:
                _copy_stream = torch.cuda.Stream()
            if _unpack_stream is None:
                _unpack_stream = torch.cuda.Stream()
            ready = {}  # chunk -> (slot, upload-done event)
            if _pending is not None:
                target, chunks = _pending[0], _pending_chunks(_pending)
                _pending = None
                if self.prefetch_enabled and target is self and _slots is not None and len(_slots) == n_slots \
                        and all(c < n_chunks and _slots[s] is not None
                                and [tuple(t.shape) for t in _slots[s]] == self.slot_shapes for c, _, s in chunks):
                    for c, done, s in chunks:
                        ready[c] = (s, done)
                else:
                    for _, done, _ in chunks:
                        done.synchronize()
            if _slots is None or len(_slots) != n_slots:
                _slots = [None] * n_slots
            for j in range(n_slots):
                if _slots[j] is None or [tuple(t.shape) for t in _slots[j]] != self.slot_shapes:
                    _slots[j] = [torch.empty(s, dtype=torch.uint8, device='cuda') for s in self.slot_shapes]
            # cross-layer prefetch target (next layer of the learned ring, same staging shapes)
            target = self.next_prefill() if self.prefetch_enabled and self.next_prefill else None
            if target is not None and target.slot_shapes != self.slot_shapes:
                target = None
            n_cross = min(n_slots, (target.experts + target.chunk - 1) // target.chunk) if target is not None else 0
            cross = []  # [(next-layer chunk, upload-done event, slot), ...]
            if ready:
                self.prefetch_hits += 1
            else:
                self.prefetch_uploads += 1
            shape13, shape2 = (CH, 2 * I, H), (CH, H, I)
            if _unpacked2 is None or tuple(_unpacked2[0][0].shape) != shape13:
                _unpacked2 = [[torch.empty(shape13, dtype=torch.bfloat16, device='cuda'),
                               torch.empty(shape2, dtype=torch.bfloat16, device='cuda')] for _ in range(2)]
            g13, g2 = self._globals()
            slot_free = [None] * n_slots   # event on the unpack stream: slot contents consumed
            buf_free = [None, None]    # event on the compute stream: BF16 buffer no longer read
            unpacked_done = {}

            def upload(j, slot):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                if slot_free[slot] is None:
                    _copy_stream.wait_stream(compute)
                else:
                    _copy_stream.wait_event(slot_free[slot])
                self._upload_chunk(_slots[slot], e0, e1, _copy_stream.cuda_stream)
                done = torch.cuda.Event(); done.record(_copy_stream)
                ready[j] = (slot, done)
                self.chunk_uploads += 1

            def unpack(j):
                b = j % 2
                slot, done = ready.pop(j)
                _unpack_stream.wait_event(done)
                if buf_free[b] is None:
                    _unpack_stream.wait_stream(compute)
                else:
                    _unpack_stream.wait_event(buf_free[b])
                w = _slots[slot]; dst = _unpacked2[b]
                count = min(E, (j + 1) * CH) - j * CH
                with torch.cuda.stream(_unpack_stream):
                    _unpack_all[(count, 2 * I * H // 1024)](w[0], w[2], dst[0], 2 * I, H, 1024, MX=self.mxfp4)
                    _unpack_all[(count, H * I // 1024)](w[1], w[3], dst[1], H, I, 1024, MX=self.mxfp4)
                    ev = torch.cuda.Event(); ev.record(_unpack_stream)
                slot_free[slot] = ev
                unpacked_done[j] = ev
                if j + n_slots < n_chunks:
                    upload(j + n_slots, slot)
                elif j + n_slots - n_chunks < n_cross:
                    # Slot is free for the rest of this layer: next layer's chunk. Only the unpack
                    # reads staging slots (the GEMMs get the BF16 buffers), so the unpack event suffices.
                    c = j + n_slots - n_chunks
                    _copy_stream.wait_event(ev)
                    target._upload_chunk(_slots[slot], c * target.chunk,
                                         min(target.experts, (c + 1) * target.chunk), _copy_stream.cuda_stream)
                    done = torch.cuda.Event(); done.record(_copy_stream)
                    cross.append((c, done, slot))

            # Start the weight pipeline before the host rendezvous on the routes.
            used = {s for s, _ in ready.values()}
            free = [s for s in range(n_slots) if s not in used]
            for j in range(min(n_slots, n_chunks)):
                if j not in ready:
                    upload(j, free.pop(0))
            unpack(0)

            def env_int(name, default):
                return int(os.environ.get(name, default))
            tm, gbn, gbk = env_int('XTIER_PREFILL_TILE_M', 32), env_int('XTIER_GATE_BN', 64), env_int('XTIER_GATE_BK', 64)
            dbn, dbk = env_int('XTIER_DOWN_BN', 64), env_int('XTIER_DOWN_BK', 32)
            def launch_opts(name):
                v = os.environ.get(name)
                return {} if not v else dict(zip(('num_warps', 'num_stages'), map(int, v.split(','))))
            gate_opts, down_opts = launch_opts('XTIER_GATE_LAUNCH'), launch_opts('XTIER_DOWN_LAUNCH')
            precision = os.environ.get('XTIER_DOWN_PRECISION', 'tf32x3')

            size = m
            di = torch.empty(size * k, dtype=torch.int32, device='cuda')
            _copy_ids[(tr.cdiv(size * k, 256),)](Pointer(ids, torch.int32), di, size * k, 256)
            order = torch.argsort(di, stable=True).to(torch.int32)
            inv = torch.empty_like(order)
            inv[order.long()] = torch.arange(size * k, dtype=torch.int32, device='cuda')
            host_ids = di.cpu().numpy()
            skipped = int((host_ids == -1).sum())
            if int(host_ids.min()) < -1 or int(host_ids.max()) >= E:
                raise ValueError('Invalid expert id')
            counts = np.bincount(host_ids[host_ids >= 0], minlength=E).astype(np.int64)
            ends = skipped + np.cumsum(counts)          # row end of each expert
            starts = ends - counts
            ntile = (counts + tm - 1) // tm
            tile_e = np.repeat(np.arange(E, dtype=np.int64), ntile)
            first = np.cumsum(ntile) - ntile                # first tile index of each expert
            q = np.arange(tile_e.size, dtype=np.int64) - first[tile_e]
            meta_np = np.ascontiguousarray(np.stack([tile_e, starts[tile_e] + q * tm, ends[tile_e]]).astype(np.int32))
            meta = torch.from_numpy(meta_np).cuda()
            tile_bounds = np.concatenate([[0], np.cumsum(ntile)])
            spans = []
            for j in range(n_chunks):
                e0, e1 = j * CH, min(E, (j + 1) * CH)
                spans.append((int(tile_bounds[e0]), int(tile_bounds[e1]), int(starts[e0]),
                              int(ends[e1 - 1] - starts[e0])))
            self.chunks_skipped += sum(1 for s in spans if s[3] == 0)
            rows_max = max([s[3] for s in spans] + [1])
            mid = torch.empty((rows_max, I), dtype=torch.float32, device='cuda')
            acc = torch.zeros((size, H), dtype=torch.float32, device='cuda')
            routes = torch.empty((rows_max, H), dtype=torch.float32, device='cuda')
            weights_ptr = Pointer(routing, torch.float32)
            for j in range(n_chunks):
                if j + 1 < n_chunks:
                    unpack(j + 1)
                t0, t1, r0, rows = spans[j]
                b = j % 2
                compute.wait_event(unpacked_done.pop(j))
                if rows:
                    w13, w2 = _unpacked2[b]
                    s13, s2 = _slots[0][2], _slots[0][3]  # not read with unpacked weights
                    _gate[(t1 - t0, I // gbn)](Pointer(x, torch.bfloat16), w13, s13, g13,
                                             order, meta[0, t0:t1], meta[1, t0:t1], meta[2, t0:t1], mid, j * CH, r0,
                                             k, tm, gbn, gbk, True, H, I, self.swiglu_limit, MX=self.mxfp4,
                                             ALPHA=self.swiglu_alpha, BETA=self.swiglu_beta, **gate_opts)
                    _down[(t1 - t0, H // dbn)](mid, w2, s2, g2, order, meta[0, t0:t1], meta[1, t0:t1],
                                              meta[2, t0:t1], routes, j * CH, r0,
                                              tm, dbn, dbk, precision, True, H, I, True, MX=self.mxfp4, **down_opts)
                ev = torch.cuda.Event(); ev.record(compute); buf_free[b] = ev
                if rows:
                    _accum[(rows, tr.cdiv(H, 256))](routes, order, inv, weights_ptr, acc, r0, rows, k, 256, H)
            _emit[(size, tr.cdiv(H, 256))](acc, Pointer(out, output_dtype), 256, H)
            # Cross-layer prefetch: the next layer's first n_cross chunks were queued on the copy
            # stream as this layer's last slots were unpacked; they upload while the next
            # attention layer computes.
            if cross:
                cross.sort(key=lambda t: t[0])
                _pending = (target, cross[0][1], cross[0][2], cross[1:])
                self.prefetch_submissions += 1
            compute.wait_stream(_unpack_stream)
            torch.cuda.current_stream().synchronize()
            self.last_profile = []
            if self.release_workspace:
                keep = {s for _, _, s in _pending_chunks(_pending)} if _pending is not None else set()
                retired = []
                for j in range(len(_slots)):
                    if j not in keep and _slots[j] is not None:
                        retired += [weakref.ref(t) for t in _slots[j]]
                        _slots[j] = None
                if not keep:
                    _slots = None
                for pair in (_unpacked2 or []):
                    retired += [weakref.ref(t) for t in pair]
                _unpacked2 = None
                return tuple(retired)
