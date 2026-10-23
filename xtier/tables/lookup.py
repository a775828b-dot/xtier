"""Row-table lookups that CUDA graphs can capture (from the 1017 pipeline's engram_backend.py).

Per call, on the current stream: the ids go to pinned host memory, a ``cudaLaunchHostFunc`` node runs the store's
lookup (cache, O_DIRECT reads, no CUDA API), and the rows and scales go back to the device, where the model
dequantizes them. A lookup of n ids uses the smallest staging buffer of at least n ids (a new power-of-two one when
none fits) and one host function argument per n, both kept, so a captured graph replays with the same addresses.
Graph capture allocates nothing: ``warm`` provides the staging beforehand. The arguments are bounded by the distinct
id counts (one per token count of a forward).

Every lookup of one table must be ordered on the GPU (one stream, or streams joined in between): the staging is shared.
"""
import ctypes as C

import torch

from xtier.tables import Work

_cudart = None


def _runtime():
    """The CUDA runtime torch already loaded (libcudart.so.13 with cu13 wheels, .12 with cu12)."""
    global _cudart
    if _cudart is None:
        errors = []
        for name in ('libcudart.so.13', 'libcudart.so.12', 'libcudart.so'):
            try:
                lib = C.CDLL(name)
                break
            except OSError as e:
                errors.append(str(e))
        else:
            raise OSError(f'xtier: no CUDA runtime library for cudaLaunchHostFunc: {errors}')
        lib.cudaLaunchHostFunc.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
        lib.cudaLaunchHostFunc.restype = C.c_int
        _cudart = lib
    return _cudart


class GraphLookup:
    def __init__(self, store):
        self.store = store
        self._staging = {}   # (device, capacity) -> pinned ids / rows / scales, device rows / scales, 0..capacity
        self._works = {}     # (device, count) -> (staging, host function argument)
        self._fn = C.cast(store.lib.xtier_rows_lookup, C.c_void_p)

    def warm(self, count, device):
        """Allocate staging for lookups of up to `count` ids (call before graph capture)."""
        device = torch.device(device)
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self._staging_for(count, device)

    def _staging_for(self, count, device):
        fits = [c for d, c in self._staging if d == device.index and c >= count]
        if fits:
            return self._staging[(device.index, min(fits))]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f'xtier: no row-table staging for {count} ids before graph capture (see warm)')
        capacity = 1 << max(count - 1, 0).bit_length()
        rb, sb = self.store.row_bytes, max(self.store.scale_bytes, 1)
        ids = torch.empty(capacity, dtype=torch.int64, device='cpu', pin_memory=True)
        rows = torch.empty((capacity, rb), dtype=torch.uint8, device='cpu', pin_memory=True)
        scales = torch.empty((capacity, sb), dtype=torch.uint8, device='cpu', pin_memory=True)
        staging = (ids, rows, scales, rows.to(device), scales.to(device),
                   torch.arange(capacity, dtype=torch.int64, device=device))
        self._staging[(device.index, capacity)] = staging
        return staging

    def _buffers(self, count, device):
        key = (device.index, count)
        if key not in self._works:
            staging = self._staging_for(count, device)
            ids, rows, scales = staging[:3]
            self._works[key] = staging, Work(self.store.store, ids.data_ptr(), rows.data_ptr(), scales.data_ptr(),
                                             count)
        return self._works[key]

    def fetch(self, indices):
        """indices: device ids (any shape) -> (rows uint8 [n, row_bytes], scales uint8 [n, scale_bytes] or None,
        0..n-1 int64) on the device, written on the current stream (n = indices.numel() > 0)."""
        count = indices.numel()
        (ids, rows, scales, drows, dscales, sequential), work = self._buffers(count, indices.device)
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        err = _runtime().cudaLaunchHostFunc(C.c_void_p(torch.cuda.current_stream().cuda_stream), self._fn,
                                            C.c_void_p(C.addressof(work)))
        if err:
            raise RuntimeError(f'xtier: cudaLaunchHostFunc failed ({err})')
        drows[:count].copy_(rows[:count], non_blocking=True)
        if not self.store.scale_bytes:
            return drows[:count], None, sequential[:count]
        dscales[:count].copy_(scales[:count], non_blocking=True)
        return drows[:count], dscales[:count], sequential[:count]
