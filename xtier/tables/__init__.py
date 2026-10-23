"""Row tables on NVMe: large embedding tables (DeepSeek V4.1 Engram, Qwen3.8 PLE n-grams) read row by row from the
checkpoint files instead of held in device or host memory (csrc/tables/rows.cpp, ``libxtier_tables.so``).

``RowStore``: one table as segments (checkpoint tensors, in global row order) of rows of ``row_bytes`` bytes, with
``scale_bytes`` scale bytes per row in parallel tensors or none; misses are read with O_DIRECT, recent rows kept in a
bounded cache, the owned rows' scales can be preloaded, ids outside the owned range [lo, hi) give zero bytes (TP
shards). ``GraphLookup`` (lookup.py): the lookup of device ids on the current CUDA stream, capturable in CUDA graphs;
it leaves the raw rows and scales on the device for the model's own dequantization.
"""
import ctypes as C
import json
import os
import struct
from pathlib import Path
from typing import NamedTuple

import torch

from xtier.engine import lib_dir

_P, _U = C.c_void_p, C.c_uint64
_lib = None


class Work(C.Structure):
    _fields_ = [('store', _P), ('ids', _P), ('weights', _P), ('scales', _P), ('count', _U)]


def lib():
    global _lib
    if _lib is None:
        c = C.CDLL(str(lib_dir() / 'libxtier_tables.so'), use_errno=True)
        c.xtier_rows_open.argtypes = [_U, C.POINTER(C.c_char_p), C.POINTER(_U), C.POINTER(_U), C.POINTER(_U), _U, _U,
                                      _U]
        c.xtier_rows_open.restype = _P
        c.xtier_rows_range.argtypes = [_P, _U, _U]
        c.xtier_rows_preload_scales.argtypes = [_P]
        c.xtier_rows_preload_scales.restype = C.c_int
        c.xtier_rows_lookup.argtypes = [_P]
        for name in ('xtier_rows_stats', 'xtier_rows_io_stats', 'xtier_rows_timing'):
            getattr(c, name).argtypes = [_P, C.POINTER(_U)]
        c.xtier_rows_close.argtypes = [_P]
        _lib = c
    return _lib


class Segment(NamedTuple):
    path: str
    rows: int
    weight_offset: int     # file offset of the segment's first row
    scale_offset: int = 0  # file offset of its scale bytes (tables with scales)


def tensor_locations(model_dir, names):
    """{name: (file, shape, safetensors dtype, file offset of the data)} from the checkpoint index and file headers."""
    root = Path(model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    headers, out = {}, {}
    for name in names:
        file = index[name]
        if file not in headers:
            with (root / file).open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                headers[file] = 8 + length, json.loads(f.read(length))
        base, header = headers[file]
        t = header[name]
        out[name] = root / file, t['shape'], t['dtype'], base + t['data_offsets'][0]
    return out


class RowStore:
    def __init__(self, segments, row_bytes, scale_bytes=0, cache_bytes=0, row_range=None):
        self.lib = lib()
        self.segments = [Segment(*s) for s in segments]
        self.row_bytes, self.scale_bytes = int(row_bytes), int(scale_bytes)
        self.rows = sum(s.rows for s in self.segments)
        n = len(self.segments)
        paths = (C.c_char_p * n)(*[str(s.path).encode() for s in self.segments])
        self.store = self.lib.xtier_rows_open(
            n, paths, (_U * n)(*[s.rows for s in self.segments]), (_U * n)(*[s.weight_offset for s in self.segments]),
            (_U * n)(*[s.scale_offset for s in self.segments]), self.row_bytes, self.scale_bytes, int(cache_bytes))
        if not self.store:
            err = C.get_errno()
            raise OSError(err, f'xtier: cannot open the row table in {self.describe()}: {os.strerror(err)}')
        self.lo, self.hi = row_range or (0, self.rows)
        self.lib.xtier_rows_range(self.store, self.lo, self.hi)

    def describe(self):
        files = sorted({str(s.path) for s in self.segments})
        return files[0] if len(files) == 1 else f'{len(self.segments)} tensors in {len(files)} files of {Path(files[0]).parent}'

    def preload_scales(self):
        if self.lib.xtier_rows_preload_scales(self.store) < 0:
            err = C.get_errno()
            raise OSError(err, f'xtier: scale preload of {self.describe()} failed: {os.strerror(err)}')

    def lookup(self, ids):
        """ids: int64 host tensor [n] -> (rows uint8 [n, row_bytes], scales uint8 [n, scale_bytes]), host tensors."""
        ids = ids.to(torch.int64).contiguous()
        n = ids.numel()
        weights = torch.empty((n, self.row_bytes), dtype=torch.uint8, device='cpu')
        scales = torch.empty((n, max(self.scale_bytes, 1)), dtype=torch.uint8, device='cpu')
        work = Work(self.store, ids.data_ptr(), weights.data_ptr(), scales.data_ptr(), n)
        self.lib.xtier_rows_lookup(C.addressof(work))
        return weights, scales[:, :self.scale_bytes]

    def _counters(self, name, keys):
        out = (_U * 4)()
        getattr(self.lib, name)(self.store, out)
        return dict(zip(keys, out))

    def stats(self):
        return self._counters('xtier_rows_stats', ('hits', 'misses', 'reads', 'cache_bytes'))

    def io_stats(self):
        return self._counters('xtier_rows_io_stats', ('rounds', 'peak_pages', 'scale_bytes'))

    def timing(self):
        return self._counters('xtier_rows_timing', ('lookup_ns', 'lookups', 'max_ns'))

    def close(self):
        if self.store:
            self.lib.xtier_rows_close(self.store)
            self.store = None
