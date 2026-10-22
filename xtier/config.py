"""XTier settings, read from the environment once per process.

  XTIER=1                    master switch (off: the plugin registers nothing)
  XTIER_LAYERS=all|0-11,20   model layers whose routed experts XTier runs (default all MoE layers of the target model)
  XTIER_DRAFT=0              1: also take the MoE layers of a speculative draft model
  XTIER_THREADS=64           CPU expert workers (a multiple of the NUMA node count)
  XTIER_MAIN_CPUS=auto|list  CPUs of every other thread of the process (auto: all but the expert workers and the
                             mailbox dispatcher); unset: unchanged
  XTIER_PROBE_DISPATCH_CPU   CPU of the mailbox dispatcher thread (default: a free CPU, see engine/placement.py)
  XTIER_DECODE_MAX=64        calls of up to this many tokens use the mailbox (CUDA-graph capturable); larger ones
                             are prefill
  XTIER_CPU_PREFILL_BATCH=1024  tokens per synchronous CPU engine call in prefill
  XTIER_GPU_PREFILL_MIN=4096 prefill calls of at least this many tokens upload the experts and run on the GPU
                             (0: always on the CPU); switches of the GPU prefill: backend/prefill.py (XTIER_PREFILL_*)
  XTIER_CACHE_SLOTS=0        GPU expert cache slots per layer (xtier.cache; 0: no cache)
  XTIER_CACHE_FULL=list      layers that keep every expert on the GPU (no CPU part)
  XTIER_CACHE_PROFILE=file   ranking profile that fills the slots at start (LKQstrata format)
  XTIER_CACHE_ADAPT=1, XTIER_CACHE_EVERY=4, XTIER_CACHE_SWAPS=16, XTIER_CACHE_DECAY=0.7, XTIER_CACHE_MIN_GAIN=1.5,
  XTIER_CACHE_LOG=N          adaptive residency (see cache/pool.py)
Engine switches (XTIER_DYNAMIC, XTIER_DOWN_BF16, XTIER_SPIN_COUNT, ...) are read by the engine itself, see csrc/.
"""

import os


def _flag(name: str, default: str = '0') -> bool:
    return os.environ.get(name, default) == '1'


def _int(name: str, default: int) -> int:
    value = os.environ.get(name, '')
    return int(value) if value.strip() else default


def cpulist(text: str) -> list[int]:
    """'0-3,8' -> [0, 1, 2, 3, 8]"""
    out = []
    for part in text.replace(' ', '').replace('\n', '').split(','):
        if part:
            lo, _, hi = part.partition('-')
            out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def enabled() -> bool:
    return _flag('XTIER')


def layers() -> set[int] | None:
    """Model layers XTier takes; None = all."""
    spec = os.environ.get('XTIER_LAYERS', 'all').strip()
    return None if spec in ('', 'all') else set(cpulist(spec))


def draft() -> bool:
    return _flag('XTIER_DRAFT')


def threads() -> int:
    return _int('XTIER_THREADS', 64)


def decode_max() -> int:
    return _int('XTIER_DECODE_MAX', 64)


def cpu_prefill_batch() -> int:
    return _int('XTIER_CPU_PREFILL_BATCH', 1024)


def gpu_prefill_min() -> int:
    return _int('XTIER_GPU_PREFILL_MIN', 4096)


def cache_slots() -> int:
    return _int('XTIER_CACHE_SLOTS', 0)


def cache_full() -> set[int]:
    return set(cpulist(os.environ.get('XTIER_CACHE_FULL', '')))
