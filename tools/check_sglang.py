"""The installed SGLang against this XTier, without a model: its main commit against the tested one
(xtier.SGLANG_COMMIT), every hook target XTier registers (all packs, all features on) resolving to an attribute, and
every name XTier imports from sglang importing. SGLang's HookRegistry logs a hook it cannot apply and goes on; this
lists them all before a server starts. Exit status 1 on a missing target or import. The packs that choose hooks by
GPU architecture ask the CUDA device.

  <SGLang venv>/bin/python tools/check_sglang.py
"""
import ast
import importlib
import os
import pathlib
import re
import sys
from importlib import metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES = dict(XTIER='1', XTIER_ENGRAM_ROWS='nvme', XTIER_ENGRAM_PREFETCH='1', XTIER_DRAFT='1', XTIER_FP8_WEIGHTS='1',
                XTIER_CACHE_ELASTIC='1', XTIER_FAST_LOAD='1')


def resolve(target):
    """The object a dotted target names (the longest importable module prefix, then attributes), or None."""
    parts = target.split('.')
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
        except Exception:
            continue
        try:
            for name in parts[i:]:
                obj = getattr(obj, name)
        except AttributeError:
            return None
        return obj
    return None


def hook_targets():
    from sglang.srt.plugins.hook_registry import HookType

    from xtier import packs, platform
    from xtier.backend import hooks
    from xtier.cache import elastic

    targets = {}

    class Recorder:
        @staticmethod
        def register(target, fn, hook_type=None, *args, **kwargs):
            targets.setdefault(target, f'{fn.__module__}.{fn.__qualname__}')

    hooks.install(Recorder, HookType)
    elastic.install(Recorder, HookType)
    packs.install(Recorder, HookType)
    packs.install_serving(Recorder, HookType)
    platform.install(Recorder, HookType)
    platform.install_serving(Recorder, HookType)
    return targets


def imported_names():
    """(module, name or None, file:line) of every import from sglang in xtier/."""
    for path in sorted((ROOT / 'xtier').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            where = f'{path.relative_to(ROOT)}:{node.lineno}' if hasattr(node, 'lineno') else ''
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or '').startswith('sglang'):
                for alias in node.names:
                    yield node.module, alias.name, where
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('sglang'):
                        yield alias.name, None, where


def main():
    for key, value in FEATURES.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(ROOT))
    from xtier import SGLANG_COMMIT

    version = metadata.version('sglang')
    m = re.search(r'\+g([0-9a-f]{7,})', version)
    same = bool(m) and m.group(1)[:7] == SGLANG_COMMIT[:7]
    print(f'SGLang {version}; XTier is tested with SGLang main {SGLANG_COMMIT}: '
          f'{"the same commit" if same else "a different or unknown commit"}')

    targets = hook_targets()
    missing = [(t, fn) for t, fn in sorted(targets.items()) if resolve(t) is None]
    names = list(imported_names())
    bad = []
    for module, name, where in names:
        if (resolve(f'{module}.{name}') if name else resolve(module)) is None:
            bad.append(f'{module}{"." + name if name else ""}  ({where})')
    for t, fn in missing:
        print(f'  hook target missing: {t}  (for {fn})')
    for b in bad:
        print(f'  import missing: {b}')
    print(f'{len(missing)} of {len(targets)} hook targets and {len(bad)} of {len(names)} imports missing')
    return 1 if missing or bad else 0


if __name__ == '__main__':
    sys.exit(main())
