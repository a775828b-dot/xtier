"""Planner: decides per MoE layer where its experts run.

Goal (see docs/PLAN.md): the CPU and the GPU finish a layer at the same time, under one global GPU budget for hot
experts. Step 2: every selected layer (XTIER_LAYERS) runs on XTier; the layers of XTIER_CACHE_FULL keep every expert
on the GPU, the others XTIER_CACHE_SLOTS hot experts (the rest on the CPU engine). With XTIER_DRAFT=1 the MoE layers
of a speculative draft run on XTier too, by default with every expert on the GPU (GPU-only W4A16, the draft must be
fast); XTIER_DRAFT_SLOTS gives them N hot slots or none instead (small GPUs). XTIER_CACHE_PLAN: per-layer slots
from a global budget (planner/budget.py, tools/plan_cache.py).
"""

from xtier import config


def layer_config(layer_idx: int):
    """XTier's decision for the FusedMoE layer being built, or None to leave it to SGLang."""
    from sglang.srt.runtime_context import get_flags

    from xtier.backend.method import LayerConfig

    if get_flags().moe.in_speculative_scope:   # a speculative draft's layers
        if not config.draft():
            return None
        return LayerConfig(layer_id=int(layer_idx), threads=config.threads(), cache_slots=config.draft_slots(),
                           draft=True)
    selected = config.layers()
    if selected is not None and layer_idx not in selected:
        return None
    planned = config.cache_plan().get(int(layer_idx))
    if planned is not None:   # as many slots as experts: the layer is fully resident (the pool's S >= E)
        slots = planned
    else:
        slots = -1 if layer_idx in config.cache_full() else config.cache_slots()
    return LayerConfig(layer_id=int(layer_idx), threads=config.threads(), cache_slots=slots)
