"""Host fixes that do not belong to one model family.

- TileLang target: ``tilelang.backend.target.auto_detect_target`` tries its registered detectors in order and the HIP
  one comes first: on a CUDA host that also has a ROCm toolkit installed (``/opt/rocm/bin/hipcc``, no AMD GPU) every
  TileLang kernel is lowered for ROCm and fails (e.g. SGLang's QSA indexer: "N must be divisible by 16" from the MFMA
  layout). With a CUDA build of torch, XTier asks the CUDA detector first.
- Scheduler NUMA binding: SGLang binds each scheduler process to its GPU's NUMA node (SGLANG_AUTO_NUMA_BIND, on by
  default) right after loading the plugins. With XTier's placement active (XTIER_MAIN_CPUS) that intersects XTier's
  main CPUs with one node: on the RTX PRO 5000 host (112 workers) the scheduler, the CUDA callback thread and the
  Engram readers ended up on 2 CPUs. XTier places the process itself, so the automatic binding is skipped (an
  explicit --numa-node still applies).
"""

import logging

logger = logging.getLogger(__name__)
TILELANG_TARGET = 'tilelang.backend.target.auto_detect_target'
NUMA_NODE = 'sglang.srt.utils.numa_utils.get_numa_node_if_available'


def _auto_detect_target(original, *args, **kwargs):
    import torch
    if torch.version.hip is None and torch.cuda.is_available():
        from tilelang.backend import target
        spec = target._TARGET_DETECTORS.get('cuda')
        detected = spec.detect() if spec is not None else None
        if detected is not None:
            return detected
    return original(*args, **kwargs)


def _numa_node(original, server_args, gpu_id, *args, **kwargs):
    if getattr(server_args, 'numa_node', None) is None:
        logger.info("xtier: SGLang's automatic NUMA binding of the scheduler skipped (XTier places its threads)")
        return None
    return original(server_args, gpu_id, *args, **kwargs)


def install(registry, hook_type) -> list[str]:
    import importlib.util
    import os
    fixes = []
    if os.environ.get('XTIER_MAIN_CPUS'):
        registry.register(NUMA_NODE, _numa_node, hook_type.AROUND)
        fixes.append('no-auto-numa-bind')
    if importlib.util.find_spec('tilelang') is not None:
        registry.register(TILELANG_TARGET, _auto_detect_target, hook_type.AROUND)
        fixes.append('tilelang-cuda-target')
    return fixes
