"""XTier settings, read from the environment once per process.

  XTIER=1                    master switch (off: the plugin registers nothing)
  XTIER_LAYERS=all|0-11,20   model layers whose routed experts XTier runs (default all MoE layers of the target model)
  XTIER_DRAFT=0              1: also take the MoE layers of a speculative draft model
  XTIER_DRAFT_SLOTS=-1       their GPU slots: -1 every expert on the GPU, 0 none (CPU engine only), N hot slots
                             (16 GB cards: the GLM-5.3 NextN layer costs 1.9 GiB per GPU with every expert)
  XTIER_THREADS=64           CPU expert workers (a multiple of the NUMA node count)
  XTIER_MAIN_CPUS=auto|list  CPUs of every other thread of the process (auto: all but the expert workers and the
                             mailbox dispatcher); unset: unchanged
  XTIER_PROBE_DISPATCH_CPU   CPU of the mailbox dispatcher thread (default: a free CPU, see engine/placement.py)
  XTIER_DECODE_MAX=64        calls of up to this many tokens use the mailbox (CUDA-graph capturable); larger ones
                             are prefill
  XTIER_CPU_PREFILL_BATCH=1024  tokens per synchronous CPU engine call in prefill
  XTIER_GPU_PREFILL_MIN=1024 prefill calls of at least this many tokens upload the experts and run on the GPU
                             (0: always on the CPU; Qwen3.8 on PCIe 5: the CPU engine and the upload break even near
                             1000 tokens); switches of the GPU prefill: backend/prefill.py (XTIER_PREFILL_*)
  XTIER_CACHE_SLOTS=0        GPU expert cache slots per layer (xtier.cache; 0: no cache)
  XTIER_CACHE_FULL=list      layers that keep every expert on the GPU (no CPU part)
  XTIER_CACHE_PROFILE=file   ranking profile that fills the slots at start (LKQstrata format)
  XTIER_CACHE_ADAPT=1, XTIER_CACHE_EVERY=4, XTIER_CACHE_SWAPS=16, XTIER_CACHE_DECAY=0.7, XTIER_CACHE_MIN_GAIN=1.5,
  XTIER_CACHE_LOG=N          adaptive residency (see cache/pool.py)
  XTIER_CACHE_PROMPT=0       1: warm the cache from each prompt's routes (cache/pool.py record_prompt;
                             XTIER_CACHE_PROMPT_KEEP=0.7, _WINDOWS=10, _ROUNDS=4, _SWAPS=256)
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


def draft_slots() -> int:
    return _int('XTIER_DRAFT_SLOTS', -1)


def threads() -> int:
    return _int('XTIER_THREADS', 64)


def decode_max() -> int:
    return _int('XTIER_DECODE_MAX', 64)


def cpu_prefill_batch() -> int:
    return _int('XTIER_CPU_PREFILL_BATCH', 1024)


def gpu_prefill_min() -> int:
    return _int('XTIER_GPU_PREFILL_MIN', 1024)


def hybrid_prefill_max() -> int:
    """XTIER_HYBRID_PREFILL_MAX=2560: prefill calls of a partly cached layer below this many tokens run their cached
    routes on the GPU and the others on the CPU at the same time (0: off; above it the GPU prefill uploads the layer)."""
    return _int('XTIER_HYBRID_PREFILL_MAX', 2560)


def split_prefill_max() -> int:
    """XTIER_SPLIT_PREFILL_MAX=0: GPU prefill calls of up to this many tokens on a layer uploaded in expert chunks
    split the experts: the GPU takes the first g (whole chunks) and the CPU engine the others at the same time, g
    balancing the measured upload time per expert against the CPU time per expert and token (lkqmoe 0.5.6's hybrid
    prefill; DeepSeek V4.1 production: 6144). XTIER_SPLIT_CPU_US=0.285: the first CPU estimate in microseconds per
    expert and token at H 5120 x I 2304 (scaled by H x I)."""
    return _int('XTIER_SPLIT_PREFILL_MAX', 0)


def resident_prefill() -> bool:
    """XTIER_CACHE_PREFILL=packed (default): a fully cached layer's prefill runs on the packed slots in place;
    staged: through the GPU prefill's staging and BF16 unpack, like a host layer."""
    return os.environ.get('XTIER_CACHE_PREFILL', 'packed') == 'packed'


def timing() -> bool:
    """XTIER_TIMING=1: log the prefill MoE time per forward and kind of layer (synchronizes around each call)."""
    return _flag('XTIER_TIMING')


def cache_slots() -> int:
    return _int('XTIER_CACHE_SLOTS', 0)


def cache_full() -> set[int]:
    return set(cpulist(os.environ.get('XTIER_CACHE_FULL', '')))


_plan = []


def cache_plan() -> dict[int, int]:
    """XTIER_CACHE_PLAN=file: {"slots": {layer: slots}} from tools/plan_cache.py --write (a layer's experts count as
    full); layers it does not list keep XTIER_CACHE_FULL / XTIER_CACHE_SLOTS."""
    if not _plan:
        path = os.environ.get('XTIER_CACHE_PLAN', '')
        if path:
            import json
            with open(path) as f:
                _plan.append({int(k): int(v) for k, v in json.load(f)['slots'].items()})
        else:
            _plan.append({})
    return _plan[0]
