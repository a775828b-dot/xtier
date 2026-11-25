"""Pack for Thinking Machines Inkling-2 models (Inkling-2-Small: 40 NVFP4 + 8 MXFP8 routed MoE layers with
soft-capped SiTU(4, 25) activation, relative-bias SWA/global attention, short convolutions, attention residuals,
23-layer ViT vision encoder, and 5-layer native DSpark MTP).

- Native XTier MoE backend for ``Inkling2Experts`` (``moe.py``): Inkling-2 implements its own stacked 3D expert
  module rather than ``FusedMoE``. When XTier is enabled on single-GPU (EP=1, TP=1), XTier replaces expert
  allocation with zero-byte placeholders, streams and deinterleaves ``w13`` (``[gate; up]``) one layer at a time on
  CPU (or reads directly from the multi-format NVMe cold store under ``XTIER_FAST_LOAD=1``), and runs hybrid
  CPU+GPU decode/prefill through ``CpuPlan`` (NVFP4 + MXFP8 + SiTU), ``LayerPool`` (adaptive + elastic hot slots),
  and ``Prefill`` — with zero SGLang source modifications.
- SM120 attention and vision fixes (``attn.py``):
  - Clears ``out_dtype=torch.float32`` on ``Inkling2Attention.attn`` / ``attn_padded`` (FA4 SM120 requires matching
    BF16 output dtype; ``gate_sink`` on BF16 vs FP32 is bitwise identical on SM120) and disables SM100-only wave
    splits / presheared ``tile_m=128`` verify path by enabling compact verify (``rel_bias_compact_decode``).
  - Routes ``rel_bias_partials`` on SM120 to SGLang's Triton ``inkling2_decode_attention_sm90.decode_partials``
    instead of the SM100-only ``tcgen05`` kernel.
  - Enables ``Inkling2VisionAttention`` on SM120 via FA4 ``flash_attn_varlen_func`` (removing the SM90/100 cap gate).
"""

import functools
import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@functools.lru_cache
def sm120():
    from sglang.srt.runtime_context import get_platform   # no CUDA context needed
    return bool(get_platform().is_sm120)


@functools.lru_cache
def has_inkling2():
    """Whether the installed SGLang includes Inkling-2 (checked on disk without importing SGLang)."""
    try:
        spec = importlib.util.find_spec('sglang.srt')
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    return any((Path(loc) / 'models' / 'inkling2.py').is_file() for loc in spec.submodule_search_locations)


def install(registry, hook_type):
    if not has_inkling2():
        return
    from xtier.packs.inkling2 import attn, kv_fp4, moe, mtp

    moe.install(registry, hook_type)
    mtp.install(registry, hook_type)
    if sm120():
        attn.install(registry, hook_type)
        kv_fp4.install(registry, hook_type)
