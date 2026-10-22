"""Host memory helpers.

``pinned_interleaved``: an exact-size page-locked host tensor whose pages are spread over all memory NUMA nodes.
``torch.empty(..., pin_memory=True)`` goes through the caching host allocator, which rounds the request up to the
next power of two, and the driver places the whole block on one node: a 48 GiB table becomes 64 GiB on a 47 GiB
node (NPS4) and the allocation never finishes. Anonymous memory interleaved with mbind and registered with CUDA is
pinned and device-accessible the same way.
"""

import ctypes
import logging
import math
import mmap

import torch

logger = logging.getLogger(__name__)
_SYS_MBIND = 237          # x86_64
_MPOL_INTERLEAVE = 3
_REGISTER_PORTABLE_MAPPED = 0x01 | 0x02
_keep = []                # mappings live as long as the process


def memory_nodes() -> list[int]:
    from xtier.config import cpulist
    try:
        with open('/sys/devices/system/node/has_memory') as f:
            return cpulist(f.read())
    except OSError:
        return [0]


def interleave(address: int, nbytes: int, nodes=None) -> bool:
    """Interleave the pages of [address, address + nbytes) (page-aligned, not yet touched) over ``nodes``."""
    nodes = memory_nodes() if nodes is None else nodes
    mask = ctypes.c_ulong(sum(1 << n for n in nodes if n < 64))
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.syscall(_SYS_MBIND, ctypes.c_void_p(address), ctypes.c_ulong(nbytes), _MPOL_INTERLEAVE,
                      ctypes.byref(mask), ctypes.c_ulong(65), 0)
    return rc == 0


def pinned_interleaved(shape, dtype: torch.dtype) -> torch.Tensor:
    shape = tuple(int(d) for d in shape)
    nbytes = math.prod(shape) * torch.empty(0, dtype=dtype).element_size()
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype, pin_memory=True)
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = torch.frombuffer(buf, dtype=torch.uint8)
    spread = interleave(raw.data_ptr(), nbytes)
    err = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), nbytes, _REGISTER_PORTABLE_MAPPED)
    if int(err) != 0:
        buf.close()
        raise RuntimeError(f'xtier: cudaHostRegister of {nbytes} bytes failed: {err}')
    _keep.append(buf)
    logger.info('xtier: %.2f GiB pinned host memory (%s), %s', nbytes / 2**30, dtype,
                f'interleaved over NUMA nodes {memory_nodes()}' if spread else 'default NUMA placement')
    return raw.view(dtype).view(shape)
