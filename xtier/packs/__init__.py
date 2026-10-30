"""Acceleration packs: per-model-family fixes and kernels, registered as hooks on SGLang (XTIER_PACKS=all|none|list).

A pack's hooks only act when its model's code runs, so every selected pack is installed in every scheduler process.
Packs in SERVING also hook the process that serves the API (how prompts are tokenized, how the server arguments are
resolved): ``install_serving``.
"""

import importlib
import os

PACKS = ('qwen4_exp', 'glm5_next', 'deepseek_v4', 'minimax_m3', 'speculative')


def selected() -> list[str]:
    spec = os.environ.get('XTIER_PACKS', 'all').strip()
    if spec in ('', 'all'):
        return list(PACKS)
    if spec == 'none':
        return []
    names = [s.strip() for s in spec.split(',') if s.strip()]
    unknown = [n for n in names if n not in PACKS]
    if unknown:
        raise ValueError(f'xtier: unknown packs {unknown} (available: {PACKS})')
    return names


def install(registry, hook_type) -> list[str]:
    names = selected()
    for name in names:
        importlib.import_module(f'xtier.packs.{name}').install(registry, hook_type)
    return names


# packs with hooks in the process that serves the API (prompt tokenization, argument resolution), installed there
# and nowhere else
SERVING = ('minimax_m3',)


def install_serving(registry, hook_type) -> list[str]:
    names = []
    for name in selected():
        if name in SERVING and importlib.import_module(f'xtier.packs.{name}').install_serving(registry, hook_type):
            names.append(name)
    return names
