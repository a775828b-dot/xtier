"""Engram rows on NVMe: Python binding of csrc/engram/rows.cpp (``libxtier_engram.so`` next to the engine libraries).

One store per Engram layer table: rows of 256 FP8 bytes and their 8 E8M0 scale bytes read from the checkpoint's
safetensors file with O_DIRECT (``XTIER_ENGRAM_MODE=ram``: the whole file mapped and locked instead), a bounded
direct-mapped row cache, the scales of the owned rows preloaded, I/O in AIO rounds on a thread pool
(``XTIER_ENGRAM_IO_THREADS``). A lookup returns zero rows for ids outside the owned range [lo, hi) (TP shards) and
aborts the process on ids outside the table or failed reads: a generation never continues with wrong rows.
The lookup entry point takes one argument and calls no CUDA API, so it can run as a ``cudaLaunchHostFunc`` node of a
captured graph (see docs/design-engram-nvme.md).
"""
import ctypes as C
import json
import struct
from pathlib import Path

import torch

from xtier.engine import lib_dir

_P, _U = C.c_void_p, C.c_uint64
_lib = None
ROW_BYTES, SCALE_BYTES = 256, 8


class Work(C.Structure):
    _fields_ = [('store', _P), ('ids', _P), ('weights', _P), ('scales', _P), ('count', _U)]


def lib():
    global _lib
    if _lib is None:
        c = C.CDLL(str(lib_dir() / 'libxtier_engram.so'))
        c.xtier_rows_open.argtypes = [C.c_char_p, _U, _U, _U, _U]
        c.xtier_rows_open.restype = _P
        c.xtier_rows_range.argtypes = [_P, _U, _U]
        c.xtier_rows_preload_scales.argtypes = [_P]
        c.xtier_rows_preload_scales.restype = C.c_int
        c.xtier_rows_lookup.argtypes = [_P]
        c.xtier_rows_stats.argtypes = [_P, C.POINTER(_U)]
        c.xtier_rows_close.argtypes = [_P]
        _lib = c
    return _lib


def table_location(model_dir, layer_id):
    """(file, rows, weight offset, scale offset) of layer `layer_id`'s Engram table in the checkpoint."""
    root = Path(model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'layers.{layer_id}.engram.embed.'
    if index[prefix + 'weight'] != index[prefix + 'scale']:
        raise ValueError(f'xtier: Engram layer {layer_id}: weight and scale in different files')
    path = root / index[prefix + 'weight']
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
    w, s = header[prefix + 'weight'], header[prefix + 'scale']
    rows = w['shape'][0]
    if w['shape'] != [rows, ROW_BYTES] or w['dtype'] != 'F8_E4M3' or s['shape'] != [rows, SCALE_BYTES] \
            or s['dtype'] != 'F8_E8M0':
        raise ValueError(f'xtier: Engram layer {layer_id}: unexpected table {w["shape"]} {w["dtype"]}, '
                         f'{s["shape"]} {s["dtype"]}')
    return path, rows, 8 + length + w['data_offsets'][0], 8 + length + s['data_offsets'][0]


class RowStore:
    def __init__(self, path, rows, weight_offset, scale_offset, cache_bytes, row_range=None):
        self.lib = lib()
        self.rows = int(rows)
        self.store = self.lib.xtier_rows_open(str(path).encode(), self.rows, int(weight_offset), int(scale_offset),
                                              int(cache_bytes))
        if not self.store:
            raise RuntimeError(f'xtier: cannot open the Engram rows of {path}')
        self.lo, self.hi = row_range or (0, self.rows)
        self.lib.xtier_rows_range(self.store, self.lo, self.hi)

    @classmethod
    def for_layer(cls, model_dir, layer_id, cache_bytes, row_range=None):
        path, rows, woff, soff = table_location(model_dir, layer_id)
        return cls(path, rows, woff, soff, cache_bytes, row_range)

    def preload_scales(self):
        if self.lib.xtier_rows_preload_scales(self.store) < 0:
            raise OSError(C.get_errno(), 'xtier: Engram scale preload failed')

    def lookup(self, ids):
        """ids: int64 host tensor [n] -> (FP8 rows as uint8 [n, 256], E8M0 scales as uint8 [n, 8]), host tensors."""
        ids = ids.to(torch.int64).contiguous()
        n = ids.numel()
        weights = torch.empty((n, ROW_BYTES), dtype=torch.uint8)
        scales = torch.empty((n, SCALE_BYTES), dtype=torch.uint8)
        work = Work(self.store, ids.data_ptr(), weights.data_ptr(), scales.data_ptr(), n)
        self.lib.xtier_rows_lookup(C.addressof(work))
        return weights, scales

    def stats(self):
        out = (_U * 4)()
        self.lib.xtier_rows_stats(self.store, out)
        return dict(hits=out[0], misses=out[1], reads=out[2], cache_bytes=out[3])

    def close(self):
        if self.store:
            self.lib.xtier_rows_close(self.store)
            self.store = None
