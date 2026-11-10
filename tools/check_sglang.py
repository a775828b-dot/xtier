"""The installed SGLang against this XTier, without a model: its main commit against the tested one
(xtier.SGLANG_COMMIT), every hook target XTier registers (all packs, all features on) resolving to an attribute, and
every name XTier imports from sglang importing. SGLang's HookRegistry logs a hook it cannot apply and goes on; this
lists them all before a server starts. Exit status 1 on a missing target or import, and with --strict on another
commit too. The packs that choose hooks by GPU architecture ask the CUDA device (--emulate smXY: another one,
tools/arch_check.py).

Two more lists, for a SGLang that still has every target but changed what is behind them (2026-11-10 audit):
- the hooked functions whose code changed since the tested commit: tools/hook_sources.json keeps a fingerprint of each
  target's source (comments, docstrings and layout aside); a wrapper or a rewrite of a function that changed may no
  longer fit it. Read what changed (git log -p on the function in SGLang), then --write-sources records the commit's
  fingerprints (merged: targets of other GPU architectures stay).
- attributes of SGLang objects XTier reads with a default (getattr(obj, name, default)): renamed, they cost no error,
  only what the list says.
With --strict either list fails the check too.

  <SGLang venv>/bin/python tools/check_sglang.py [--strict] [--emulate smXY] [--write-sources]
"""
import ast
import hashlib
import importlib
import inspect
import json
import os
import pathlib
import re
import sys
import textwrap
from importlib import metadata

ROOT = pathlib.Path(__file__).resolve().parents[1]
FEATURES = dict(XTIER='1', XTIER_ENGRAM_ROWS='nvme', XTIER_ENGRAM_PREFETCH='1', XTIER_DRAFT='1', XTIER_FP8_WEIGHTS='1',
                XTIER_CACHE_ELASTIC='1', XTIER_FAST_LOAD='1')
SOURCES = ROOT / 'tools' / 'hook_sources.json'
_FB, _MOE = 'sglang.srt.model_executor.forward_batch_info.ForwardBatch', 'sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE'
ATTRIBUTES = [   # (class, attribute, what XTier loses without it)
    (_FB, 'seq_lens_sum', "the elastic cache's estimate of a prefill's memory: every extra slot released before each one"),
    (_FB, 'seq_lens_cpu', 'the same (the fallback)'),
    (_FB, 'input_ids', "the elastic cache's token count of a forward"),
    (_MOE, 'moe_tp_rank', 'with tensor parallelism every rank places its CPU threads as rank 0'),
    (_MOE, 'moe_tp_size', 'the same'),
    (_MOE, 'top_k', "the cold experts' RAM slot check assumes 10 routes a token"),
    ('sglang.srt.layers.sampler.Sampler', 'tp_sync_group', "the fix of GLM's crash with tools (512 MB, xtier/platform.py)"),
    ('sglang.srt.model_executor.model_runner.ModelRunner', 'is_draft_worker', "a draft model's forwards taken for the target's"),
    ('sglang.srt.sampling.sampling_batch_info.SamplingBatchInfo', 'top_ks', 'speculative truncation: greedy requests not told apart'),
    ('sglang.srt.plugins.hook_registry.HookRegistry', '_hooks', 'the stop on a hook that did not apply (xtier/plugin.py)'),
    ('sglang.srt.plugins.hook_registry.HookRegistry', '_patched', 'the same'),
]


def has_attribute(cls, name):
    """Whether instances of cls have the attribute: a class attribute, a field, an annotation, or an assignment to
    self.<name> in the class's (or a base class's) code."""
    for c in inspect.getmro(cls):
        if name in vars(c) or name in getattr(c, '__annotations__', {}):
            return True
        try:
            if re.search(rf'self\.{re.escape(name)}\s*(:[^=\n]*)?=(?!=)', inspect.getsource(c)):
                return True
        except (OSError, TypeError):
            pass
    return False


def fingerprint(obj):
    """A hash of the code behind a hook target, its comments, docstrings and layout aside; None without Python source."""
    if isinstance(obj, property):
        obj = obj.fget
    obj = getattr(obj, '__func__', obj)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError, TypeError, SyntaxError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()[:16]


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

    from xtier import plugin
    targets[f'sglang.srt.managers.scheduler.{plugin.SCHEDULER_ENTRY}'] = 'xtier.plugin._in_scheduler'
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
    args = sys.argv[1:]
    for key, value in FEATURES.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(ROOT))
    if '--emulate' in args:
        sys.path.insert(0, str(ROOT / 'tools'))
        import arch_check
        arch_check.emulate(args[args.index('--emulate') + 1].removeprefix('sm'))
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

    recorded = json.loads(SOURCES.read_text()) if SOURCES.exists() else {'sglang': '', 'targets': {}}
    now = {t: fingerprint(resolve(t)) for t in sorted(targets) if resolve(t) is not None}
    now = {t: f for t, f in now.items() if f is not None}
    changed = [t for t, f in now.items() if t in recorded['targets'] and recorded['targets'][t] != f]
    unknown = [t for t in now if t not in recorded['targets']]
    for t in changed:
        print(f'  hooked code changed since SGLang {recorded["sglang"][:10]}: {t}  (for {targets[t]})')
    for t in unknown:
        print(f'  hooked code without a recorded fingerprint: {t}  (for {targets[t]})')
    print(f'{len(changed)} of {len(now)} hooked functions changed, {len(unknown)} not recorded')

    gone = []
    for cls, name, loss in ATTRIBUTES:
        c = resolve(cls)
        if c is None or not has_attribute(c, name):
            gone.append(name)
            print(f'  attribute missing: {cls}.{name}  (XTier then loses: {loss})')
    print(f'{len(gone)} of {len(ATTRIBUTES)} attributes XTier reads with a default missing')

    if '--write-sources' in args:
        recorded['targets'].update(now)
        recorded['sglang'] = SGLANG_COMMIT
        SOURCES.write_text(json.dumps(recorded, indent=1, sort_keys=True) + '\n')
        print(f'{SOURCES.relative_to(ROOT)}: {len(now)} fingerprints recorded ({len(recorded["targets"])} in all)')
        return 0
    strict = '--strict' in args and (not same or changed or unknown or gone)
    return 1 if missing or bad or strict else 0


if __name__ == '__main__':
    sys.exit(main())
