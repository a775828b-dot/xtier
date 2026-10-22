"""Acceleration pack for SGLang's ``qwen4_exp`` models (Qwen3.8 Flash Next).

- PLE host table: the pinned backend allocates the n-gram embedding table (20M rows x 2560 FP8, 48 GiB) as an
  exact-size, NUMA-interleaved, CUDA-registered mapping (xtier.engine.memory) instead of the caching host
  allocator's power-of-two block on one node, which cannot be placed on an NPS4 host.
"""

import logging

logger = logging.getLogger(__name__)
PLE = 'sglang.srt.models.qwen4_exp_ple_table.allocate_ple_host_table'


def _allocate_ple_host_table(original, shape, dtype, backend='pinned', table_dir=None, tag=None, *args, **kwargs):
    if backend != 'pinned' or args or kwargs:
        return original(shape, dtype, backend, table_dir, tag, *args, **kwargs)
    from xtier.engine.memory import pinned_interleaved
    return pinned_interleaved(shape, dtype)


def install(registry, hook_type):
    registry.register(PLE, _allocate_ple_host_table, hook_type.AROUND)
