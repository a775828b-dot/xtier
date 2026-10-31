"""SGLang general-plugin entry point (entry point group ``sglang.srt.plugins``).

SGLang calls ``load()`` early in every process and then applies the hooks registered through
``sglang.srt.plugins.hook_registry.HookRegistry``. XTier registers its model hooks only in scheduler processes (the
ones that build models: importing the hook targets elsewhere would only cost startup time); ``install()`` does it
unconditionally, for tools that build a ModelRunner in their own process. The other processes, the one that serves
the API among them, get the few hooks on how prompts are tokenized and the host fixes of serving
(``install_serving()``). SGLang logs and swallows exceptions raised
here, so the backend checks after every model load that it took over (``xtier.backend.hooks.check``).
"""

import logging
import os
import sys
from pathlib import Path

from xtier import SGLANG_COMMIT, __version__, config, prebuild

logger = logging.getLogger(__name__)
_installed = []
_serving = []
SCHEDULER_ENTRY = 'run_scheduler_process'   # of sglang.srt.managers.scheduler


def _in_scheduler() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == SCHEDULER_ENTRY:
            return True
        frame = frame.f_back
    return False


def _check_scheduler_entry() -> None:
    """_in_scheduler goes by that function's name: under another name no process would install the hooks and the server
    would run without XTier. SystemExit, because SGLang swallows the exceptions of a plugin. From the source, the
    module is heavy to import."""
    import importlib.util
    import re

    spec = importlib.util.find_spec('sglang.srt.managers.scheduler')
    source = spec.loader.get_source(spec.name) if spec is not None else None
    if source is not None and not re.search(rf'^def {SCHEDULER_ENTRY}\(', source, re.M):
        raise SystemExit(f'xtier: XTIER=1, but this SGLang has no {SCHEDULER_ENTRY}: XTier would not be installed. '
                         'Install the SGLang commit README.md names, or unset XTIER.')


def _check_sglang() -> None:
    """A warning when the installed SGLang is not the main commit XTier is tested with."""
    import re
    from importlib import metadata

    try:
        version = metadata.version('sglang')
    except metadata.PackageNotFoundError:
        return
    m = re.search(r'\+g([0-9a-f]{7,})', version)
    if not m or m.group(1)[:7] != SGLANG_COMMIT[:7]:
        logger.warning('xtier %s is tested with SGLang main %s; this SGLang is %s, so some hooks may not apply '
                       '(tools/check_sglang.py lists them)', __version__, SGLANG_COMMIT, version)


def _require_hooks() -> None:
    """SGLang logs a hook it cannot apply and goes on: the server would run without that part of XTier, slower or with
    other results. Checked right after SGLang applied them, before anything is loaded."""
    from sglang.srt.plugins.hook_registry import HookRegistry

    apply_hooks = HookRegistry.apply_hooks

    def checked():
        apply_hooks()
        failed = sorted(target for target, hooks in HookRegistry._hooks.items() if target not in HookRegistry._patched
                        and any(getattr(h[-1], 'plugin_name', None) == 'xtier' for h in hooks))
        if failed:
            raise SystemExit(f'xtier: hooks not applied (SGLang logged why): {failed}; tools/check_sglang.py lists what '
                             'this SGLang lacks')

    if hasattr(HookRegistry, '_hooks') and hasattr(HookRegistry, '_patched'):
        HookRegistry.apply_hooks = checked


def _check_settings() -> None:
    """A setting XTier does not know (misspelled, or removed) would be ignored without a word. xtier/env_names.txt:
    tools/repo_check.py --write."""
    names = Path(__file__).with_name('env_names.txt')
    if names.exists():
        known = set(names.read_text().split())
        unknown = sorted(k for k in os.environ if k.startswith('XTIER_') and k not in known)
        if unknown:
            print(f'[xtier] settings XTier does not know are ignored: {unknown}', file=sys.stderr, flush=True)


def install() -> None:
    if _installed:
        return
    _check_sglang()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from xtier import packs, platform
    from xtier.backend import hooks

    hooks.install(HookRegistry, HookType)
    from xtier.cache import elastic
    if elastic.install(HookRegistry, HookType):
        logger.info('xtier: elastic expert cache on (XTIER_CACHE_ELASTIC=1)')
    names = packs.install(HookRegistry, HookType)
    fixes = platform.install(HookRegistry, HookType)
    _installed.append(True)
    logger.info('xtier %s: MoE backend hooks, packs %s and host fixes %s registered (pid %d)', __version__, names,
                fixes, os.getpid())


def install_serving() -> None:
    if _serving:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from xtier import packs, platform

    names = packs.install_serving(HookRegistry, HookType)
    fixes = platform.install_serving(HookRegistry, HookType)
    _serving.append(True)
    logger.info('xtier %s: serving hooks of packs %s and host fixes %s registered (pid %d)', __version__, names,
                fixes, os.getpid())


def load() -> None:
    if not config.enabled():
        logger.debug('xtier %s: XTIER is not 1, nothing registered', __version__)
        return
    if os.environ.get('XTIER_MAIN_CPUS'):
        # XTier places the threads itself. SGLang binds every scheduler to its GPU's NUMA node when the main process
        # starts it (numactl, SGLANG_NUMA_BIND_V2): XTier's main CPUs of that node only (2 on the RTX PRO 5000 host)
        # and a preferred memory policy that puts the scheduler's own allocations there, where the engine's
        # node-bound shards then find no pages (10-23: OOM kill, CONSTRAINT_MEMORY_POLICY nodemask=0). Off here, in
        # the main process, before any scheduler starts (an explicit --numa-node still applies).
        os.environ.setdefault('SGLANG_AUTO_NUMA_BIND', '0')
        from xtier.engine import placement
        if not placement.deferred():   # with TP the rank's placement waits for its first MoE layer
            placement.apply(config.threads())
    _require_hooks()
    if _in_scheduler():
        _check_settings()
        prebuild.run()   # nothing is loaded yet
        install()
        # SGLang configures logging only after loading the plugins: one line on stderr
        print(f'[xtier] {__version__}: hooks installed in scheduler pid {os.getpid()}, CPUs '
              f'{len(os.sched_getaffinity(0))}, SGLANG_AUTO_NUMA_BIND={os.environ.get("SGLANG_AUTO_NUMA_BIND", "")}',
              file=sys.stderr, flush=True)
    else:
        _check_scheduler_entry()
        install_serving()
