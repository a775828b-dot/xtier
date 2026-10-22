"""Host fixes that do not belong to one model family.

- TileLang target: ``tilelang.backend.target.auto_detect_target`` tries its registered detectors in order and the HIP
  one comes first: on a CUDA host that also has a ROCm toolkit installed (``/opt/rocm/bin/hipcc``, no AMD GPU) every
  TileLang kernel is lowered for ROCm and fails (e.g. SGLang's QSA indexer: "N must be divisible by 16" from the MFMA
  layout). With a CUDA build of torch, XTier asks the CUDA detector first.
"""

import logging

logger = logging.getLogger(__name__)
TILELANG_TARGET = 'tilelang.backend.target.auto_detect_target'


def _auto_detect_target(original, *args, **kwargs):
    import torch
    if torch.version.hip is None and torch.cuda.is_available():
        from tilelang.backend import target
        spec = target._TARGET_DETECTORS.get('cuda')
        detected = spec.detect() if spec is not None else None
        if detected is not None:
            return detected
    return original(*args, **kwargs)


def install(registry, hook_type) -> list[str]:
    import importlib.util
    fixes = []
    if importlib.util.find_spec('tilelang') is not None:
        registry.register(TILELANG_TARGET, _auto_detect_target, hook_type.AROUND)
        fixes.append('tilelang-cuda-target')
    return fixes
