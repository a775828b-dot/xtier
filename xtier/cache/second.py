"""Second GPU as a hot expert pool (XTIER_GPU2_SLOTS=N|auto, default off; TP = 1, both GPUs visible).

A GPU that holds no part of the model keeps more hot experts of every partly cached layer: for GPUs of different size or
speed, where tensor parallelism would run both at the smaller and slower one's pace. Nothing on the first GPU changes:
its decode graphs still pack, ask the mailbox and add what comes back. The mailbox dispatcher (csrc/mailbox) splits the
routes the first GPU left to the CPU: those this GPU holds get -1 for the CPU and go to a graph on this GPU, captured per
layer and shape here (outside capture, at the shape's first eager call): control block and activations from pinned host
memory, the kernels of xtier.cache.pool, the part and a flag back to host memory (kernels, not the copy engine, which the
swaps' uploads occupy). The dispatcher adds the part to the CPU output before answering. Prefill stays on the first GPU.

Numerics: the kernels of the hot experts (W4A16, FP32 accumulation); only the order of the FP32 sum differs from the first
GPU holding the same experts (tests/second_gpu_check.py: bit-identical when this GPU holds all of a call's GPU routes).
Residency: first the ranking profile after the first GPU's slots, then swaps from the same route counts; the GPUs hold
different experts (g2_policy in csrc/mailbox). Balance: a call ends with the later of this GPU and the CPU, so the number
of experts per call here follows whichever was last (XTIER_GPU2_BALANCE=1, g2_finish in csrc/mailbox).
Not with cold experts. A GPU without FP4 decode next to one with it is refused (xtier.arch.HW_FP4 is decided once).
"""
import ctypes as C
import logging
import math
import os

import torch
import triton
import triton.language as tl

from xtier.arch import HW_FP4
from xtier.backend.kernels import Pointer
from xtier.cache.pool import _combine, _down, _gate_up, CONFIG, MAX_M, MAX_ROUTES, Profile

logger = logging.getLogger(__name__)
CONTROL = 2048   # bytes of the control block before x: rslot, slot list, slot experts, routing (128 x 4 bytes each)


@triton.jit
def _fetch(SRC, DST, N, B: tl.constexpr):
    """Control block and x from host memory, over PCIe by the SMs (always re-read: the dispatcher wrote them)."""
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(DST + i, tl.load(SRC + i, i < N, cache_modifier='.cv'), i < N)


def attach(pools):
    """fill_all, outside capture: slots, residents and buffers on the second GPU for every partly cached layer."""
    value = os.environ.get('XTIER_GPU2_SLOTS', '')
    if value in ('', '0'):
        return
    if any(p.tp_size > 1 for p in pools):
        raise ValueError('xtier cache: XTIER_GPU2_SLOTS is for TP = 1')
    if any(getattr(p.plan, 'slots', p.E) < p.E for p in pools):
        raise ValueError('xtier cache: XTIER_GPU2_SLOTS does not work with cold experts')
    index = int(os.environ.get('XTIER_GPU2_DEVICE', torch.cuda.current_device() + 1))
    if index >= torch.cuda.device_count():
        raise ValueError(f'xtier cache: XTIER_GPU2_SLOTS: no GPU {index} (make both visible: CUDA_VISIBLE_DEVICES)')
    if bool(getattr(HW_FP4, 'value', HW_FP4)) and tuple(torch.cuda.get_device_capability(index)) < (10, 0):
        raise ValueError(f'xtier cache: XTIER_GPU2_SLOTS: GPU {index} has no FP4 decode, the first GPU has: put the older '
                         'GPU first or set XTIER_FP4_DECODE=soft')
    H, I, E = pools[0].H, pools[0].I, pools[0].E
    per_slot = sum(math.prod(s[1:]) for p in pools for s in p.prefill.shapes[:4])
    if value == 'auto':   # all but the context, the buffers and the graphs
        reserve = int(float(os.environ.get('XTIER_GPU2_RESERVE_GIB', '1.5')) * 2**30)
        slots = max(0, torch.cuda.mem_get_info(index)[0] - reserve) // per_slot
    else:
        slots = int(value)
    slots = min(slots, E - min(p.active for p in pools))
    if slots <= 0:
        logger.warning('xtier cache: XTIER_GPU2_SLOTS=%s: no slots on GPU %d', value, index)
        return
    lib = pools[0].mailbox.lib
    for pool in pools:   # on the first GPU (.cuda() in it follows the current device)
        pool.prefill._globals()
    with torch.cuda.device(index):
        stream = torch.cuda.Stream()
        dev = torch.device('cuda', index)
        host = (C.c_void_p * 2)()
        control = torch.empty(CONTROL + 8 * H * 2, dtype=torch.uint8, device=dev)
        if lib.xtier_mailbox_gpu2_init(stream.cuda_stream, H, C.cast(host, C.c_void_p)) < 0:
            raise RuntimeError(lib.xtier_mailbox_error().decode())
        # capture on a stream of its own: the dispatcher may be launching on `stream` while a new shape is captured
        shared = dict(lib=lib, capture=torch.cuda.Stream(), index=index, control=control, dev=dev,
                      host_control=host[0], host_out=host[1],
                      mid=torch.empty((MAX_ROUTES, MAX_M, I), dtype=torch.bfloat16, device=dev),
                      part=torch.empty((MAX_ROUTES, MAX_M, H), dtype=torch.float32, device=dev))
        shared['stream'] = stream   # the graphs run on it
        for pool in pools:
            pool.second = SecondLayer(pool, slots, shared)
        swaps = int(os.environ.get('XTIER_GPU2_SWAPS', '64'))
        gain = float(os.environ.get('XTIER_GPU2_MIN_GAIN', '0.25'))
        use = float(os.environ.get('XTIER_GPU2_MIN_USE', '0.5'))
        if os.environ.get('XTIER_GPU2_ADAPT', '1') != '0':
            if lib.xtier_mailbox_gpu2_start(swaps, C.c_float(gain), C.c_float(use)) < 0:
                raise RuntimeError(lib.xtier_mailbox_error().decode())
    logger.info('xtier cache: second GPU %d (%s): %d slots on %d layers, %.1f GiB', index,
                torch.cuda.get_device_name(index), slots, len(pools), slots * per_slot / 2**30)


class SecondLayer:
    """One layer's slots on the second GPU and its graphs (one per decode shape)."""

    def __init__(self, pool, slots, shared):
        self.pool, self.shared, self.S = pool, shared, slots
        dev = shared['dev']
        self.arena = [torch.empty((slots,) + tuple(s[1:]), dtype=torch.uint8, device=dev) for s in pool.prefill.shapes[:4]]
        self.g13, self.g2 = [g.to(dev) for g in pool.prefill._globals()]
        first = set(pool.resident)
        ranking = [e for e in Profile.ranking(pool.layer_id, pool.E) if e not in first][:slots]
        resident = (C.c_int32 * slots)(*(ranking + [-1] * (slots - len(ranking))))
        arena = (C.c_void_p * 4)(*[a.data_ptr() for a in self.arena])
        if shared['lib'].xtier_mailbox_gpu2_attach(pool.mailbox.bridge, slots, C.cast(arena, C.c_void_p),
                                                   C.cast(resident, C.c_void_p)) < 0:
            raise RuntimeError(pool.mailbox.error())
        self.shapes = set()

    def ensure(self, m, k):
        """Decode of a shape, before its mailbox request; a shape first seen inside capture, or whose graph does not fit
        the GPU's memory, stays with the CPU."""
        if (m, k) in self.shapes or torch.cuda.is_current_stream_capturing():
            return
        self.shapes.add((m, k))
        s, pool, lib, cfg = self.shared, self.pool, self.shared['lib'], CONFIG
        H, I, nk = pool.H, pool.I, m * k
        with torch.cuda.device(s['index']), torch.cuda.stream(s['capture']):
            st = s['capture'].cuda_stream
            base = s['control'].data_ptr()
            if lib.xtier_mailbox_gpu2_capture_begin(st) < 0:
                raise RuntimeError(lib.xtier_mailbox_error().decode())
            try:
                words = (CONTROL + m * H * 2) // 4
                _fetch[(triton.cdiv(words, 256),)](Pointer(s['host_control'], torch.int32), Pointer(base, torch.int32), words, 256)
                rslot, slot_list = Pointer(base, torch.int32), Pointer(base + 512, torch.int32)
                slot_exp, routing = Pointer(base + 1024, torch.int32), Pointer(base + 1536, torch.float32)
                x = Pointer(base + CONTROL, torch.bfloat16)
                _gate_up[(nk, I // cfg['gbn'])](x, slot_list, slot_exp, self.arena[0], self.arena[2], self.g13, s['mid'], m,
                                                H, I, pool.limit, MAX_M, cfg['gbn'], pool.bk_h, MX=pool.prefill.mxfp4,
                                                ALPHA=pool.alpha, BETA=pool.beta, num_warps=cfg['gw'], num_stages=cfg['gs'])
                # XTIER_GPU2_SLOW=n, test only: a slower GPU, n more times the same work (the same result)
                for _ in range(1 + int(os.environ.get('XTIER_GPU2_SLOW', '0'))):
                    _down[(nk, H // cfg['dbn'])](s['mid'], slot_list, slot_exp, self.arena[1], self.arena[3], self.g2,
                                                 s['part'], m, H, I, MAX_M, cfg['dbn'], pool.bk_i, MX=pool.prefill.mxfp4,
                                                 num_warps=cfg['dw'], num_stages=cfg['ds'])
                _combine[(m, triton.cdiv(H, 1024))](s['part'], rslot, routing, Pointer(s['host_out'], torch.float32), H, k,
                                                    MAX_M, 1024)
                if lib.xtier_mailbox_gpu2_capture_end(pool.mailbox.bridge, st, m, k) < 0:
                    raise RuntimeError(lib.xtier_mailbox_error().decode())
            except RuntimeError as e:
                lib.xtier_mailbox_gpu2_capture_abort(st)
                logger.warning('xtier cache: second GPU: no graph for %d tokens x %d routes on layer %d, the CPU computes '
                               'them: %s', m, k, pool.layer_id, e)
