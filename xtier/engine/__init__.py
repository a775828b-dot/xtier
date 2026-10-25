"""Python binding of the CPU expert engine built from csrc/.

``CpuPlan``: one MoE layer's experts as NUMA-sharded host copies, computed on the CPU worker pools (synchronous
batches of up to 2048 tokens). ``Mailbox``: connects a plan to CUDA streams through mapped host buffers that GPU
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
    sharded copy: the inputs can be released after construction."""

    def __init__(self, experts, threads, weight_pointers, hidden, inter, swiglu_limit=0.0, scale_format='nvfp4',
                 swiglu_alpha=1.0, swiglu_beta=0.0):
        if scale_format not in SCALE_FORMATS:
            raise ValueError(f'Unknown scale format {scale_format!r}')
        self.lib = engine()
        self.hidden, self.inter, self.experts = int(hidden), int(inter), int(experts)
        self.swiglu_limit, self.scale_format = float(swiglu_limit), scale_format
        self.plan = self.lib.xtier_create3(self.hidden, self.inter, self.experts, int(threads), self.swiglu_limit,
                                           SCALE_FORMATS[scale_format], *weight_pointers)
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
        """x: BF16 [m, H], ids: int32 [m, k], routing: float32 [m, k] -> out: float32 [m, H] (addresses)."""
        if not self.plan:
            raise RuntimeError('Plan is closed')
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

    def close(self):
        if self.plan:
            self.lib.xtier_destroy(self.plan)
            self.plan = None


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
