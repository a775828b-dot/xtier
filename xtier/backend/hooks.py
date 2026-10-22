"""Where XTier enters SGLang.

``FusedMoE.__init__`` asks ``create_kt_config_from_server_args`` for a KTransformers hybrid config and, when it gets
one, wraps the checkpoint's MoE method as ``KTEPWrapperMethod(gpu_method, config)`` before creating any weight. XTier
answers that question for the layers its planner takes (an XTier ``LayerConfig``) and replaces the wrapper class by
a subclass that turns an XTier config into ``XTierMoEMethod`` (real KTransformers configs keep the original wrapper).
After each model load a self-check counts the layers XTier runs and fails loudly if the hooks did not take.
"""

import logging

from xtier import config, planner

logger = logging.getLogger(__name__)
KT = 'sglang.srt.layers.moe.kt_ep_wrapper'


def install(registry, hook_type):
    from sglang.srt.layers.moe import kt_ep_wrapper

    from xtier.backend.method import LayerConfig, XTierMoEMethod

    original = kt_ep_wrapper.KTEPWrapperMethod

    class KTEPWrapperMethod(original):
        def __new__(cls, gpu_method, kt_config, *args, **kwargs):
            if isinstance(kt_config, LayerConfig):   # not an instance of cls: __init__ is skipped
                try:
                    return XTierMoEMethod(gpu_method, kt_config)
                except NotImplementedError as e:
                    if not kt_config.draft:
                        raise
                    # a draft layer XTier cannot run (e.g. BF16 experts) stays with SGLang
                    logger.info('xtier: draft MoE layer %d stays with SGLang: %s', kt_config.layer_id, e)
                    return gpu_method
            return super().__new__(cls)

    # FusedMoE's loader recognizes the wrapper by class name.
    KTEPWrapperMethod.__module__, KTEPWrapperMethod.__qualname__ = original.__module__, original.__qualname__

    def kt_config(original_fn, server_args, layer_idx, *args, **kwargs):
        cfg = planner.layer_config(layer_idx)
        return cfg if cfg is not None else original_fn(server_args, layer_idx, *args, **kwargs)

    def after_load(result, model_runner, *args, **kwargs):
        check(model_runner)
        return result

    registry.register(f'{KT}.create_kt_config_from_server_args', kt_config, hook_type.AROUND)
    registry.register(f'{KT}.KTEPWrapperMethod', KTEPWrapperMethod, hook_type.REPLACE)
    registry.register('sglang.srt.model_executor.model_runner.ModelRunner.load_model', after_load, hook_type.AFTER)


def check(model_runner):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    from xtier.backend.method import XTierMoEMethod

    moe = [m for m in model_runner.model.modules() if isinstance(m, FusedMoE)]
    ours = [m for m in moe if isinstance(m.quant_method, XTierMoEMethod)]
    draft = bool(getattr(model_runner, 'is_draft_worker', False))
    if draft and not config.draft():
        if ours:
            raise RuntimeError(f'xtier: {len(ours)} draft MoE layers on XTier without XTIER_DRAFT=1')
        return
    if moe and not ours and not draft:   # a draft's layers may stay with SGLang (formats XTier cannot run)
        raise RuntimeError(f'xtier: XTIER=1 but none of the {len(moe)} MoE layers runs on XTier (hooks not applied)')
    unplanned = [m.quant_method.cfg.layer_id for m in ours if m.quant_method.plan is None]
    if unplanned:
        raise RuntimeError(f'xtier: MoE layers {unplanned} have no CPU engine plan after loading')
    logger.info('xtier: %s model: %d of %d MoE layers on XTier (layers %s)', 'draft' if draft else 'target', len(ours),
                len(moe), [m.quant_method.cfg.layer_id for m in ours])
