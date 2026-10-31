"""The checks that stop a server from running silently without part of XTier (xtier/plugin.py, the backend), without a model: SGLang without the scheduler entry the
plugin goes by, a hook of XTier that does not apply (and one of another plugin, which is not XTier's to stop), a setting XTier does not know, a layer whose expert
tensors did not all arrive; and the job count of the FlashInfer prebuild (xtier/prebuild.py) under a memory limit.

  python tests/startup_checks_check.py
"""
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.srt.plugins import hook_registry as hr  # noqa: E402

from xtier import plugin, prebuild  # noqa: E402

bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def stops(fn):
    """The message of the SystemExit fn raises, else None."""
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


check('scheduler entry, this SGLang', stops(plugin._check_scheduler_entry) is None)
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name: SimpleNamespace(name=name, loader=SimpleNamespace(get_source=lambda n: 'def other():\n    pass\n'))
message = stops(plugin._check_scheduler_entry)
importlib.util.find_spec = find_spec
check('scheduler entry renamed: the server stops', bool(message) and plugin.SCHEDULER_ENTRY in message)

registry = hr.HookRegistry
plugin._require_hooks()
original = registry.apply_hooks


def hook(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def register(target, plugin_name):
    token = hr._current_plugin_source.set(hr.HookSource(plugin_name=plugin_name, dist_name=None))
    try:
        registry.register(target, hook, hr.HookType.AROUND)
    finally:
        hr._current_plugin_source.reset(token)


register('json.dumps', 'xtier')
check('a hook that applies', stops(original) is None)
register('sglang.no_such_module.no_such_function', 'someone_else')
check('another plugin\'s hook fails: not stopped', stops(original) is None)
register('sglang.no_such_module.other_function', 'xtier')
message = stops(original)
check('a hook of XTier fails: the server stops', bool(message) and 'other_function' in message, str(message))
for target in ('sglang.no_such_module.no_such_function', 'sglang.no_such_module.other_function'):
    registry._hooks.pop(target, None)

unknown = 'XTIER' + '_NO_SUCH_SETTING'   # not spelled out: tools/repo_check.py lists the names the sources mention
os.environ[unknown] = '1'
os.environ['XTIER_THREADS'] = '16'
err = io.StringIO()
with contextlib.redirect_stderr(err):
    plugin._check_settings()
check('an unknown setting is named, a known one not', unknown in err.getvalue() and 'XTIER_THREADS' not in err.getvalue(), err.getvalue().strip())

from xtier.backend.method import XTierMoEMethod  # noqa: E402

stub = SimpleNamespace(plan=None, cfg=SimpleNamespace(layer_id=7), _missing={'w13_weight': 3, 'w2_weight': 0})
try:
    XTierMoEMethod.process_weights_after_loading(stub, None)
    message = None
except RuntimeError as e:
    message = str(e)
check('expert tensors missing: the load fails', bool(message) and 'w13_weight' in message and 'layer 7' in message, str(message))
stub.plan = object()
XTierMoEMethod.process_weights_after_loading(stub, None)

free = prebuild._free_bytes()
jobs = prebuild._jobs(128)
check('prebuild jobs', 1 <= jobs <= 128 and free > 0, f'{free / 2**30:.0f} GiB free, {jobs} jobs')
prebuild._free_bytes = lambda: 21 * 2**30
check('prebuild jobs with little memory', prebuild._jobs(128) == 1)
print('STARTUP-CHECKS-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
