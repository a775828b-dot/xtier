"""Checks of the elastic expert arenas (xtier/cache/elastic.py over csrc/mailbox xtier_mailbox_vmm_*) on the GPU,
without a model: reserved address space wrapped as a torch tensor, granules mapped and released (the device's free
memory follows), the granule a lower slot shares kept, data of the kept slots intact through a release and a new
mapping above them, host-to-device copies into mapped slots, and the time to map / release a GiB.

    python tests/elastic_check.py        (from the repository root, on a machine with a CUDA GPU)
"""
import ctypes as C
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('XTIER_CACHE_ELASTIC', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.cache import elastic  # noqa: E402
from xtier.engine import mailbox_lib  # noqa: E402

MIB = 2 ** 20
lib = mailbox_lib()
dev = torch.device('cuda', 0)
torch.cuda.init()


def free():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def mapped(t):
    return lib.xtier_mailbox_vmm_mapped(C.c_void_p(t.data_ptr()))


# 3 MiB slots: granule boundaries fall inside slots
row = (3, MIB)
stride = 3 * MIB
f0 = free()
a = elastic.arena(lib, 64, row, 16, dev)
assert a.shape == (64, 3, MIB) and a.dtype == torch.uint8 and a.is_cuda
assert mapped(a) == 48 * MIB, mapped(a)                       # 16 slots x 3 MiB = 48 MiB = 24 granules
assert abs((f0 - free()) - 48 * MIB) <= 4 * MIB, f0 - free()
a[:16].copy_(torch.arange(16, dtype=torch.uint8, device=dev).view(16, 1, 1).expand(16, 3, MIB))

# more slots: free memory drops, host-to-device copies land
f1 = free()
assert lib.xtier_mailbox_vmm_map(C.c_void_p(a.data_ptr()), 16 * stride, 48 * stride) == 0
assert mapped(a) == 64 * 3 * MIB
assert abs((f1 - free()) - 48 * 3 * MIB) <= 4 * MIB, f1 - free()
host = torch.full((48, 3, MIB), 200, dtype=torch.uint8).pin_memory()
a[16:].copy_(host, non_blocking=True)
torch.cuda.synchronize()
assert int(a[16:].min()) == 200 and int(a[16:].max()) == 200

# release down to 5 slots (15 MiB: inside granule 7): granules 8.. go, slots 0..4 keep their bytes
f2 = free()
assert lib.xtier_mailbox_vmm_unmap(C.c_void_p(a.data_ptr()), 5 * stride) == 0
assert mapped(a) == 16 * MIB, mapped(a)                       # granules 0..7 (0-16 MiB) stay mapped
assert abs((free() - f2) - (64 * 3 - 16) * MIB) <= 4 * MIB, free() - f2
for s in range(5):
    assert int(a[s].min()) == s and int(a[s].max()) == s, s

# map again above slot 5 and use it; the kept slots stay intact
assert lib.xtier_mailbox_vmm_map(C.c_void_p(a.data_ptr()), 5 * stride, 59 * stride) == 0
a[5:].fill_(7)
torch.cuda.synchronize()
for s in range(5):
    assert int(a[s].min()) == s and int(a[s].max()) == s, s
assert int(a[5:].min()) == 7

# out of range is refused
assert lib.xtier_mailbox_vmm_map(C.c_void_p(a.data_ptr()), 64 * stride, stride) < 0

# speed: 1 GiB in granules
b = elastic.arena(lib, 512, (2, MIB), 1, dev)                # 2 MiB slots, 1 GiB of address space
t0 = time.perf_counter()
assert lib.xtier_mailbox_vmm_map(C.c_void_p(b.data_ptr()), 0, 512 * 2 * MIB) == 0
t_map = time.perf_counter() - t0
t0 = time.perf_counter()
assert lib.xtier_mailbox_vmm_unmap(C.c_void_p(b.data_ptr()), 2 * MIB) == 0
t_unmap = time.perf_counter() - t0
print(f'elastic arenas ok: map {t_map * 1e3:.1f} ms / release {t_unmap * 1e3:.1f} ms per GiB (2 MiB granules)')
