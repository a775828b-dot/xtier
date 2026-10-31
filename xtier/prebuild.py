"""FlashInfer's JIT modules a server would otherwise build in the middle of a request.

FlashInfer compiles a module where it is first used, and again whenever ninja finds it out of date: another FlashInfer,
other flags, another ninja (each version hashes the commands its own way, and a venv's ninja is not the system's). The
fused MoE of XTIER_PREFILL_FP4 is 97 compile jobs, 4,300 CPU seconds, about 4 GiB per job (6 at the peak), and its first
use is the first long prefill of a loaded model. DeepSeek V4.1 leaves about 80 GiB free there: the 17 jobs the scheduler's
CPUs allow took that to the memory guard's line and the guard killed the server, after every restart too, since the
large jobs never ended (2026-10-31). So here, before any weight is loaded, on all the server's CPUs with as many jobs as
the free memory takes. With nothing to build this is one ninja run per module. XTIER_PREBUILD=0 turns it off.
"""

import os
import sys
import threading
import time
from pathlib import Path

KEEP_GIB = 20   # of host memory (the line of tools/mem_guard.sh)
JOB_GIB = 6


def _free_bytes() -> int:
    """What can still be allocated: the host's available memory, or less under a memory cgroup (MemAvailable knows nothing of
    those: a container on a rented GPU, the MemoryMax of a systemd scope)."""
    with open('/proc/meminfo') as f:
        free = next(int(line.split()[1]) for line in f if line.startswith('MemAvailable')) * 1024
    root = Path('/sys/fs/cgroup')
    dirs = [root, root / 'memory']
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, controllers, path = line.split(':', 2)
            if controllers in ('', 'memory'):
                own = (root / 'memory' if controllers else root) / path.lstrip('/')
                dirs += [own, *own.parents]
    except (OSError, ValueError):
        pass
    for d in dirs:
        for limit, used in (('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
            try:
                free = min(free, int((d / limit).read_text()) - int((d / used).read_text()))
            except (OSError, ValueError):   # no such file, or "max"
                pass
    return free


def _jobs(cpus: int) -> int:
    return max(1, min(cpus, int((_free_bytes() / 2**30 - KEEP_GIB) / JOB_GIB)))


def _modules():
    """What the requests of this server would build: {name: function that builds and loads it}."""
    import torch
    from flashinfer import sampling

    modules = {'sampling': sampling.get_sampling_module}
    major, minor = torch.cuda.get_device_capability()
    if os.environ.get('XTIER_PREFILL_FP4') == '1' and major >= 10:   # backend/prefill.py _fp4_ready
        from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
        modules['fused MoE'] = lambda: get_cutlass_fused_moe_module(f'{major * 10 + minor}')
    return modules


def _server_cpus(mine: set[int]) -> set[int]:
    """The process's CPUs with the expert workers': XTier's placement keeps the other threads off those, and the compile
    jobs of a thread inherit its CPUs (ninja starts one job per CPU it sees, whichever it is left with)."""
    from xtier import config
    from xtier.engine import placement, worker_cpus

    if not os.environ.get('XTIER_MAIN_CPUS') or placement.deferred():
        return mine
    return mine | set(worker_cpus(config.threads()))


def run() -> None:
    if os.environ.get('XTIER_PREBUILD', '1') == '0':
        return
    # Everything is caught: SGLang swallows what a plugin raises, and the server would come up without XTier. A module
    # that cannot be built fails again where it is used.
    t0 = time.perf_counter()
    mine = os.sched_getaffinity(0)
    jobs_set = 'MAX_JOBS' not in os.environ   # FlashInfer's limit of ninja's jobs; the user's stays
    note = None
    try:
        modules = _modules()
        cpus = _server_cpus(mine)
        if jobs_set:
            os.environ['MAX_JOBS'] = str(_jobs(len(cpus)))
        # SGLang configures logging only after loading the plugins: stderr
        note = threading.Timer(3.0, print, (f'[xtier] building FlashInfer modules {sorted(modules)}: minutes on the '
                                            f'first start, up to {os.environ["MAX_JOBS"]} compile jobs',),
                               dict(file=sys.stderr, flush=True))
        note.start()
        os.sched_setaffinity(0, cpus)
        for build in modules.values():
            build()
        if time.perf_counter() - t0 > 3.0:
            print(f'[xtier] FlashInfer modules built in {time.perf_counter() - t0:.0f} s', file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'[xtier] FlashInfer modules not built before the load: {e!r}', file=sys.stderr, flush=True)
    finally:
        if note is not None:
            note.cancel()
        os.sched_setaffinity(0, mine)
        if jobs_set:
            os.environ.pop('MAX_JOBS', None)
