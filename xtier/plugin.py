"""SGLang general-plugin entry point (entry point group ``sglang.srt.plugins``).

SGLang calls ``load()`` early in every process and then applies the hooks registered through
``sglang.srt.plugins.hook_registry.HookRegistry``. XTier registers its model hooks only in scheduler processes (the
ones that build models: importing the hook targets elsewhere would only cost startup time); ``install()`` does it
unconditionally, for tools that build a ModelRunner in their own process. SGLang logs and swallows exceptions raised
here, so the backend checks after every model load that it took over (``xtier.backend.hooks.check``).
"""

import logging
import os
import sys

from xtier import __version__, config

logger = logging.getLogger(__name__)
_installed = []


def _in_scheduler() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == 'run_scheduler_process':
            return True
        frame = frame.f_back
    return False


def install() -> None:
    if _installed:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from xtier import packs, platform
    from xtier.backend import hooks

    hooks.install(HookRegistry, HookType)
    names = packs.install(HookRegistry, HookType)
    fixes = platform.install(HookRegistry, HookType)
    _installed.append(True)
    logger.info('xtier %s: MoE backend hooks, packs %s and host fixes %s registered (pid %d)', __version__, names,
                fixes, os.getpid())


def load() -> None:
    if not config.enabled():
        logger.debug('xtier %s: XTIER is not 1, nothing registered', __version__)
        return
    if os.environ.get('XTIER_MAIN_CPUS'):
        from xtier.engine import placement
        if not placement.deferred():   # with TP the rank's placement waits for its first MoE layer
            placement.apply(config.threads())
    if _in_scheduler():
        install()
