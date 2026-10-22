"""Engram lookup from NVMe rows that CUDA graphs can capture (ported from the 1017 pipeline's engram_backend.py).

Per call, on the current stream: the ids go to pinned host memory, a ``cudaLaunchHostFunc`` node runs the row store's
lookup (``xtier_rows_lookup``: cache, O_DIRECT reads, no CUDA API), the FP8 rows and E8M0 scales go back to the device
and SGLang's ``engram_gather`` dequantizes them to BF16. Rows outside the store's owned range come back as zeros (the
TP all-reduce of ``EngramEmbedding`` adds the shards). Staging buffers are kept per power-of-two capacity and the host
callback argument per token count, so a captured graph replays with the same addresses; both must exist before capture
(``warm``).
"""
import ctypes as C

import torch

from xtier.packs.deepseek_v4.rows import ROW_BYTES, SCALE_BYTES, Work

_cudart = None


def _runtime():
    global _cudart
    if _cudart is None:
        lib = C.CDLL('libcudart.so.13')   # already loaded by torch
        lib.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        lib.cudaLaunchHostFunc.restype = C.c_int
        _cudart = lib
    return _cudart


class NvmeLookup:
    def __init__(self, store, dim=ROW_BYTES, block=ROW_BYTES // SCALE_BYTES):
        if dim != ROW_BYTES or dim // block != SCALE_BYTES:
            raise ValueError(f'xtier: the Engram row store holds rows of {ROW_BYTES} FP8 bytes, not dim {dim}')
        self.store, self.dim, self.block = store, dim, block
        self._staging = {}   # (device, capacity) -> pinned ids / rows / scales, their device copies, 0..capacity
        self._works = {}     # (device, count) -> host callback argument
        self._fn = C.cast(store.lib.xtier_rows_lookup, C.c_void_p)

    def warm(self, count, device):
        """Allocate what a lookup of `count` ids needs (call before graph capture)."""
        self._buffers(count, torch.device(device))

    def _buffers(self, count, device):
        capacity = 1 << max(count - 1, 0).bit_length()
        key = (device.index, capacity)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'xtier: Engram staging for {count} ids must be allocated before graph capture')
            ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            rows = torch.empty((capacity, ROW_BYTES), dtype=torch.uint8, pin_memory=True)
            scales = torch.empty((capacity, SCALE_BYTES), dtype=torch.uint8, pin_memory=True)
            self._staging[key] = (ids, rows, scales, rows.to(device), scales.to(device),
                                  torch.arange(capacity, dtype=torch.int64, device=device))
        staging = self._staging[key]
        wkey = (device.index, count)
        if wkey not in self._works:
            ids, rows, scales = staging[:3]
            self._works[wkey] = Work(self.store.store, ids.data_ptr(), rows.data_ptr(), scales.data_ptr(), count)
        return staging, self._works[wkey]

    def __call__(self, indices):
        """indices: device ids (any shape) -> BF16 rows [*indices.shape, dim] on the current stream."""
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        out = torch.empty(*indices.shape, self.dim, dtype=torch.bfloat16, device=indices.device)
        count = indices.numel()
        if not count:
            return out
        (ids, rows, scales, drows, dscales, sequential), work = self._buffers(count, indices.device)
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        err = _runtime().cudaLaunchHostFunc(C.c_void_p(torch.cuda.current_stream().cuda_stream), self._fn,
                                            C.c_void_p(C.addressof(work)))
        if err:
            raise RuntimeError(f'xtier: cudaLaunchHostFunc failed ({err})')
        drows[:count].copy_(rows[:count], non_blocking=True)
        dscales[:count].copy_(scales[:count], non_blocking=True)
        engram_gather(drows.data_ptr(), dscales.data_ptr(), sequential[:count], out.view(-1, self.dim), self.dim,
                      self.block)
        return out
