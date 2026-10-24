"""DeepSeek V4.1 indexer decode scored from the request table on SM120 (``XTIER_DS_PAGED_INDEXER``, default on;
speed).

``v41_indexer.scoring.decode_scores`` scores a decode step's compressed index keys one of two ways, chosen by
``get_platform().is_sm90`` alone. SM90 scores straight from the request table (``fp4_index_logits_paged``, a
persistent Triton kernel that skips the tiles past each row's length) and selects with the fused top-k v2 kernel
(JIT CUDA; its cluster paths are enabled by an occupancy probe when it is built); the dense-block candidates come
from ``amax_topk_blocks`` and reach the consumer layers as a mask. Every other GPU first gathers each row's slots over
the graph capacity, scores them with the dense Triton kernel (the same ``_fp4_index_logits_tile``, so the same
logits) and selects with ``torch.topk`` plus a sort. Nothing on the paged path is SM90-only and the decode metadata
it reads is built on every GPU. Here ``decode_scores`` sees an SM90 platform on SM120 for the duration of the call;
the module's other readers of ``get_platform`` (none today) keep the real one.
"""

import functools
import os
import sys

DECODE_SCORES = 'sglang.srt.layers.attention.dsv4.v41_indexer.scoring.decode_scores'


class _Hopper:
    """The platform with ``is_sm90`` true, for the paged-path test of ``decode_scores``."""

    def __init__(self, platform):
        self._platform = platform

    def __getattr__(self, name):
        return True if name == 'is_sm90' else getattr(self._platform, name)


@functools.lru_cache
def _sm120():
    from sglang.srt.runtime_context import get_platform
    return bool(get_platform().is_sm120)


def _decode_scores(original, *args, **kwargs):
    if not _sm120():
        return original(*args, **kwargs)
    module = sys.modules[original.__module__]
    real = module.get_platform
    module.get_platform = lambda: _Hopper(real())
    try:
        return original(*args, **kwargs)
    finally:
        module.get_platform = real


def install(registry, hook_type):
    if os.environ.get('XTIER_DS_PAGED_INDEXER', '1') != '0':
        registry.register(DECODE_SCORES, _decode_scores, hook_type.AROUND)
