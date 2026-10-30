"""Python binding of the CPU expert engine built from csrc/.

``CpuPlan``: one MoE layer's experts as NUMA-sharded host copies, computed on the CPU worker pools (synchronous
batches of up to 2048 tokens); with ``slots`` (cold experts, xtier/cache/cold.py) the copies hold only that many
experts, the others are read from the ``ColdStore`` on NVMe when a forward needs them. ``Mailbox``: connects a plan to CUDA streams through mapped host buffers that GPU
kernels fill and read and stream memory operations that hand a request to the CPU dispatcher and wait for it
(CUDA-graph capturable; slots of up to 8 tokens), plus the expert-pool command ring.
The libraries are looked up in ``xtier/engine/lib`` unless ``XTIER_ENGINE_LIB`` names another directory.
"""

import ctypes as C
import os
from pathlib import Path

_P, _I = C.c_void_p, C.c_int
_engine = None
_mailbox = None
SCALE_FORMATS = {'nvfp4': 0, 'mxfp4': 1}


def lib_dir() -> Path:
    return Path(os.environ.get('XTIER_ENGINE_LIB') or Path(__file__).with_name('lib'))


def engine() -> C.CDLL:
    """libxtier_engine.so (it dlopens the AVX512 or AVX2 kernel build next to it on first use)."""
    global _engine
    if _engine is None:
        lib = C.CDLL(str(lib_dir() / 'libxtier_engine.so'))
        lib.xtier_create3.argtypes = [_I, _I, _I, _I, C.c_float, _I] + [_P] * 6
        lib.xtier_create3.restype = _P
        if hasattr(lib, 'xtier_set_swiglu'):   # builds before 2026-10-24 have the plain SiLU only
            lib.xtier_set_swiglu.argtypes = [_P, C.c_float, C.c_float]
            lib.xtier_set_swiglu.restype = _I
        lib.xtier_forward.argtypes = [_P, _I, _I] + [_P] * 4
        lib.xtier_forward.restype = _I
        lib.xtier_destroy.argtypes = [_P]
        lib.xtier_last_error.restype = C.c_char_p
        lib.xtier_isa.restype = C.c_char_p
        lib.xtier_localize.argtypes = [_P]
        lib.xtier_localize.restype = _I
        lib.xtier_shard.argtypes = [_P, _I, _P, _P]
        lib.xtier_shard.restype = _I
        lib.xtier_nodes.argtypes = [_P]
        lib.xtier_nodes.restype = _I
        lib.xtier_worker_cpus.argtypes = [_I, _P, _I]
        lib.xtier_worker_cpus.restype = _I
        if hasattr(lib, 'xtier_create4'):   # RAM slots and the cold store (2026-10-26)
            lib.xtier_create4.argtypes = [_I, _I, _I, _I, _I, C.c_float, _I] + [_P] * 7
            lib.xtier_create4.restype = _P
            for name, args, res in (('place', [_P, _I, _I], _I), ('slot_table', [_P, _P], _I),
                                    ('slot_pieces', [_P, _I, _I, _P, _P], _I),
                                    ('cold_attach', [_P, _P, _P, _P, _P, C.c_float], _I),
                                    ('slot_pin', [_P, _I], _I), ('slot_unpin', [_P, _I], None),
                                    ('cold_plan_stats', [_P, _P], None),
                                    ('cold_open', [_P, _I, _I, _I], _P), ('cold_write', [_P, _I, C.c_longlong, _P, _P, _I], _I),
                                    ('cold_read', [_P, _I, _P, _P, _P, _P, _I], _I), ('cold_stats', [_P, _P], None),
                                    ('cold_submit', [_P, _I, _P, _P, _P, _P, _I, _P, _I], _P),
                                    ('cold_wait', [_P, _P, _I, _I], _I), ('cold_finish', [_P, _P], _I),
                                    ('cold_close', [_P], None), ('cold_error', [], C.c_char_p)):
                fn = getattr(lib, 'xtier_' + name)
                fn.argtypes, fn.restype = args, res
        _engine = lib
    return _engine


def isa() -> str:
    """Kernel build in use: "avx512", "avx2", or "none" (unsupported CPU / XTIER_ISA)."""
    return engine().xtier_isa().decode()


def worker_cpus(threads: int) -> list[int]:
    """CPUs the expert workers bind to for ``threads`` (current NUMA topology; no threads are started)."""
    cpus = (C.c_int * threads)()
    if engine().xtier_worker_cpus(threads, C.cast(cpus, _P), threads) < 0:
        raise RuntimeError(engine().xtier_last_error().decode())
    return list(cpus)


class CpuPlan:
    """weight_pointers: addresses of w13 [E, 2I, H/2], w2 [E, H, I/2], s13 [E, 2I, H/g], s2 [E, H, I/g] (uint8; g = 16
    E4M3 for NVFP4, 32 UE8M0 for MXFP4) and g13 [E, 2], g2 [E] (float32; all ones for MXFP4). The engine keeps its own
    sharded copy: the inputs can be released after construction. With the first four None the shards are allocated
    and left to the caller to fill (shards / slot_pieces) before the first forward."""

    def __init__(self, experts, threads, weight_pointers, hidden, inter, swiglu_limit=0.0, scale_format='nvfp4',
                 swiglu_alpha=1.0, swiglu_beta=0.0, resident=None):
        if scale_format not in SCALE_FORMATS:
            raise ValueError(f'Unknown scale format {scale_format!r}')
        self.lib = engine()
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        self.swiglu_limit, self.scale_format = float(swiglu_limit), scale_format
        self.slots = self.experts if resident is None else len(resident)   # resident: the experts of slots 0..
        self.dynamic = None   # cold experts: the RAM slots experts can leave (cold.attach)
        if resident is None:
            self.plan = self.lib.xtier_create3(self.hidden, self.inter, self.experts, int(threads), self.swiglu_limit,
                                               SCALE_FORMATS[scale_format], *weight_pointers)
        else:
            if not hasattr(self.lib, 'xtier_create4'):
                raise RuntimeError('xtier: this engine build has no RAM slots; rebuild it (tools/build_engine.sh)')
            res = (C.c_int32 * self.slots)(*[int(e) for e in resident])
            self.plan = self.lib.xtier_create4(self.hidden, self.inter, self.experts, self.slots, int(threads),
                                               self.swiglu_limit, SCALE_FORMATS[scale_format], *weight_pointers,
                                               C.cast(res, _P))
        if not self.plan:
            raise RuntimeError(self.lib.xtier_last_error().decode())
        # GPT-OSS SwiGLU (MiniMax M3): gate * sigmoid(alpha * gate) * (up + beta); 1 / 0 keep the plain SiLU path
        if (swiglu_alpha, swiglu_beta) != (1.0, 0.0):
            if not hasattr(self.lib, 'xtier_set_swiglu'):
                raise RuntimeError('xtier: this engine build has no GPT-OSS SwiGLU; rebuild it (tools/build_engine.sh)')
            if self.lib.xtier_set_swiglu(self.plan, swiglu_alpha, swiglu_beta):
                raise RuntimeError(self.lib.xtier_last_error().decode())
        self._localized = False

    def forward(self, m, k, x, ids, routing, out):
        """x: BF16 [m, H], ids: int32 [m, k], routing: float32 [m, k] -> out: float32 [m, H] (host addresses here)."""
        if not self.plan:
            raise RuntimeError('Plan is closed')
        if self.dynamic is not None and m * k > self.dynamic // 2:
            return self._forward_in_passes(m, k, x, ids, routing, out)
        return self._forward_once(m, k, x, ids, routing, out)

    def _forward_in_passes(self, m, k, x, ids, routing, out):
        """Cold experts: every expert of a forward must be in RAM at once, and the ones in RAM it uses cannot make room
        for the others. A call whose experts outside RAM, with its experts in RAM, need more than half the RAM slots
        runs in passes: first the experts in RAM, then the others, half those slots at a time, each read once; the
        outputs add up in pass order (float32)."""
        import numpy as np
        routes = np.ctypeslib.as_array((C.c_int32 * (m * k)).from_address(ids)).copy()
        table = np.asarray(self.slot_table(), dtype=np.int64)
        valid = routes >= 0
        safe = np.where(valid, routes, 0)
        in_ram = valid & (table[safe] >= 0)
        cold = np.unique(routes[valid & ~in_ram])
        step = max(1, self.dynamic // 2)
        if len(cold) == 0 or len(cold) + len(np.unique(routes[in_ram])) <= step:
            return self._forward_once(m, k, x, ids, routing, out)
        result = np.ctypeslib.as_array((C.c_float * (m * self.hidden)).from_address(out))
        part = np.empty(m * self.hidden, np.float32)
        masked = np.empty_like(routes)
        rc = 0
        if in_ram.any():
            np.copyto(masked, np.where(in_ram, routes, -1))
            rc += self._forward_once(m, k, x, masked.ctypes.data, routing, out)
        else:
            result[:] = 0
        for i in range(0, len(cold), step):
            np.copyto(masked, np.where(np.isin(routes, cold[i:i + step]), routes, -1))
            rc += self._forward_once(m, k, x, masked.ctypes.data, routing, part.ctypes.data)
            result += part
        return rc

    def _forward_once(self, m, k, x, ids, routing, out):
        rc = self.lib.xtier_forward(self.plan, m, k, x, ids, routing, out)
        if rc < 0:
            raise RuntimeError(self.lib.xtier_last_error().decode())
        return rc

    def localize(self):
        """Move shard pages placed on another NUMA node back to their node (once, before pinning)."""
        if self._localized:
            return 0
        self._localized = True
        return self.lib.xtier_localize(self.plan)

    def nodes(self):
        """Worker nodes: every expert's rows are split evenly over this many weight shards."""
        return self.lib.xtier_nodes(self.plan)

    def shards(self):
        """Per worker node: six (address, bytes) views of the immutable weight shard."""
        rows = []
        for node in range(self.nodes()):
            pointers, sizes = (C.c_void_p * 6)(), (C.c_size_t * 6)()
            if self.lib.xtier_shard(self.plan, node, C.cast(pointers, _P), C.cast(sizes, _P)) < 0:
                raise RuntimeError(self.lib.xtier_last_error().decode())
            rows.append([(pointers[j], sizes[j]) for j in range(6)])
        return rows

    def slot_table(self):
        """Per expert, its RAM slot or -1 (between forwards)."""
        table = (C.c_int32 * self.experts)()
        if self.lib.xtier_slot_table(self.plan, C.cast(table, _P)) < 0:
            raise RuntimeError(self.lib.xtier_last_error().decode())
        return list(table)

    def slot_pieces(self, slot, node):
        """One slot's six (address, bytes) pieces on a worker node, in xtier_shard order."""
        pointers, sizes = (C.c_void_p * 6)(), (C.c_size_t * 6)()
        if self.lib.xtier_slot_pieces(self.plan, slot, node, C.cast(pointers, _P), C.cast(sizes, _P)) < 0:
            raise RuntimeError(self.lib.xtier_last_error().decode())
        return [(pointers[j], sizes[j]) for j in range(6)]

    def attach_cold(self, store, files, offsets, scores, half_life):
        """Cold experts handled by the forwards: files / offsets of every expert's node parts ([e * nodes + n];
        -1: never leaves RAM), initial decayed use counts, their half-life in forwards (xtier_cold_attach)."""
        if len(files) != self.experts * self.nodes() or len(offsets) != len(files):
            raise ValueError('xtier: cold files / offsets need one entry per expert and node part')
        f = (C.c_int32 * len(files))(*files)
        o = (C.c_longlong * len(offsets))(*offsets)
        sc = (C.c_float * self.experts)(*scores)
        if self.lib.xtier_cold_attach(self.plan, store.handle, C.cast(f, _P), C.cast(o, _P), C.cast(sc, _P),
                                      float(half_life)):
            raise RuntimeError(self.lib.xtier_last_error().decode())

    def cold_stats(self):
        """(forwards, experts read from NVMe, seconds spent reading; the same for forwards of at most 8 tokens)."""
        out = (C.c_double * 6)()
        self.lib.xtier_cold_plan_stats(self.plan, C.cast(out, _P))
        return tuple(out)

    def close(self):
        if self.plan:
            self.lib.xtier_destroy(self.plan)
            self.plan = None


class ColdStore:
    """Expert records on NVMe (csrc/cold/store.cpp): one file per drive, O_DIRECT both ways (no page cache)."""

    def __init__(self, paths, threads=8, writable=True):
        self.lib = engine()
        if not hasattr(self.lib, 'xtier_cold_open'):
            raise RuntimeError('xtier: this engine build has no cold store; rebuild it (tools/build_engine.sh)')
        self.paths = [str(p) for p in paths]
        names = (C.c_char_p * len(self.paths))(*[p.encode() for p in self.paths])
        self.handle = self.lib.xtier_cold_open(C.cast(names, _P), len(self.paths), int(threads), int(bool(writable)))
        if not self.handle:
            raise RuntimeError(self.lib.xtier_cold_error().decode())

    def write(self, file, offset, pieces):
        """One record: pieces [(address, bytes)] back to back at `offset` of file `file`."""
        ptrs = (C.c_void_p * len(pieces))(*[a for a, _ in pieces])
        sizes = (C.c_size_t * len(pieces))(*[b for _, b in pieces])
        if self.lib.xtier_cold_write(self.handle, int(file), int(offset), C.cast(ptrs, _P), C.cast(sizes, _P),
                                     len(pieces)):
            raise RuntimeError(self.lib.xtier_cold_error().decode())

    @staticmethod
    def _jobs(jobs):
        n = len(jobs[0][2])
        files = (C.c_int * len(jobs))(*[f for f, _, _ in jobs])
        offsets = (C.c_longlong * len(jobs))(*[o for _, o, _ in jobs])
        dst = (C.c_void_p * (len(jobs) * n))(*[a for _, _, ps in jobs for a, _ in ps])
        sizes = (C.c_size_t * (len(jobs) * n))(*[b for _, _, ps in jobs for _, b in ps])
        return n, [C.cast(a, _P) for a in (files, offsets, dst, sizes)]

    def read(self, jobs):
        """jobs: [(file, offset, [(address, bytes)] x n)], all at once; returns when every one is done."""
        if not jobs:
            return
        n, args = self._jobs(jobs)
        if self.lib.xtier_cold_read(self.handle, len(jobs), *args, n):
            raise RuntimeError(self.lib.xtier_cold_error().decode())

    def submit(self, jobs, groups, count):
        """Asynchronous read: job j counts toward group groups[j] (0 <= groups[j] < count). Returns a ticket for
        wait() and finish(); the destinations must stay valid until finish()."""
        n, args = self._jobs(jobs) if jobs else (1, [None] * 4)
        g = (C.c_int * max(1, len(groups)))(*groups)
        ticket = self.lib.xtier_cold_submit(self.handle, len(jobs), *args, n, C.cast(g, _P), int(count))
        if not ticket:
            raise RuntimeError(self.lib.xtier_cold_error().decode())
        return ticket

    def wait(self, ticket, g0, g1):
        """Every job of groups [g0, g1) is done."""
        if self.lib.xtier_cold_wait(self.handle, ticket, int(g0), int(g1)):
            raise RuntimeError(self.lib.xtier_cold_error().decode())

    def finish(self, ticket):
        """Waits for all of the ticket's jobs and frees it."""
        if self.lib.xtier_cold_finish(self.handle, ticket):
            raise RuntimeError(self.lib.xtier_cold_error().decode())

    def stats(self):
        """(bytes read, bytes written, read jobs, seconds in reads)."""
        out = (C.c_double * 4)()
        self.lib.xtier_cold_stats(self.handle, C.cast(out, _P))
        return tuple(out)

    def close(self):
        if self.handle:
            self.lib.xtier_cold_close(self.handle)
            self.handle = None


def mailbox_lib() -> C.CDLL:
    """libxtier_mailbox.so: needs the CUDA runtime and driver."""
    global _mailbox
    if _mailbox is None:
        c = C.CDLL(str(lib_dir() / 'libxtier_mailbox.so'))
        signatures = {
            'create': ([_P], _P), 'prepare': ([_P, _P, _I, _I, _I], _I), 'enqueue': ([_P, _P, _I, _I, _P, _P, _P, _P, _I], _I),
            'slot': ([_P, _P, _I, _I, _P], _I), 'notify': ([_P, _P, _I, _I], _I), 'slot2': ([_P, _P, _I, _I, _P], _I),
            'signal': ([_P, _P, _I, _I, _I], _I), 'pool_enable': ([_P, _I], _I), 'pool_stats': ([_P, _P], _I),
            'pool_ring': ([_P], _I), 'pool_attach': ([_P, _I, _I, _I, _I, _P, _P], _I),
            'pool_pause': ([_I], _I), 'pool_quiesce': ([_I], _I), 'pool_shrink': ([_I], _I), 'pool_resume': ([], _I),
            'pool_grow': ([_I], _I), 'pool_active': ([], _I), 'vmm_reserve': ([C.c_size_t, _P, _P], _I),
            'vmm_map': ([_P, C.c_size_t, C.c_size_t], _I), 'vmm_unmap': ([_P, C.c_size_t], _I),
            'vmm_mapped': ([_P], C.c_longlong), 'vmm_granularity': ([], C.c_size_t),
            'pool_start': ([_I, _I, C.c_float, C.c_float], _I), 'pool_prompt': ([_P, _P, _I, _I], _I),
            'pool_warm': ([C.c_float, C.c_float, _I, _I], _I), 'stats': ([_P, _P], _I), 'timing': ([_P, _P], _I),
            'gpu2_init': ([_P, _I, _P], _I), 'gpu2_attach': ([_P, _I, _P, _P], _I),
            'gpu2_capture_begin': ([_P], _I), 'gpu2_capture_end': ([_P, _P, _I, _I], _I), 'gpu2_capture_abort': ([_P], _I),
            'gpu2_start': ([_I, C.c_float, C.c_float], _I),
            'destroy': ([_P], _I),
        }
        for name, (args, res) in signatures.items():
            fn = getattr(c, f'xtier_mailbox_{name}')
            fn.argtypes, fn.restype = args, res
        c.xtier_mailbox_error.restype = C.c_char_p
        _mailbox = c
    return _mailbox


class Mailbox:
    """CPU-GPU mailbox of one plan. Slots must be prepared before graph capture; one slot's replays must not overlap."""

    def __init__(self, plan: CpuPlan):
        self.lib = mailbox_lib()
        self.bridge = self.lib.xtier_mailbox_create(plan.plan)
        if not self.bridge:
            raise RuntimeError(self.error())
        self.plan = plan
        self._slots = {}

    def error(self):
        return self.lib.xtier_mailbox_error().decode()

    def _check(self, rc):
        if rc < 0:
            raise RuntimeError(self.error())
        return rc

    def slot(self, stream, m, k, with_routes=False):
        """Device-visible addresses of the slot's pinned x, ids, routing, output (and original routes with
        with_routes, for the expert pool)."""
        key = (stream, m, k, with_routes)
        if key not in self._slots:
            n = 5 if with_routes else 4
            pointers = (C.c_void_p * n)()
            fn = self.lib.xtier_mailbox_slot2 if with_routes else self.lib.xtier_mailbox_slot
            self._check(fn(self.bridge, stream, m, k, C.cast(pointers, _P)))
            self._slots[key] = tuple(pointers)
        return self._slots[key]

    def notify(self, stream, m, k):
        """Request the CPU and wait for it, on the stream."""
        self._check(self.lib.xtier_mailbox_notify(self.bridge, stream, m, k))

    def signal(self, stream, m, k, part):
        """Half of the handshake: part 1 requests the CPU, part 2 waits for it (GPU work goes between)."""
        self._check(self.lib.xtier_mailbox_signal(self.bridge, stream, m, k, part))

    def close(self):
        if self.bridge:
            self._check(self.lib.xtier_mailbox_destroy(self.bridge))
            self.bridge = None
