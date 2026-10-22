"""SGLang general-plugin entry point (entry point group ``sglang.srt.plugins``).

SGLang calls ``load()`` early in every process and then applies the hooks registered through
``sglang.srt.plugins.hook_registry.HookRegistry``. It logs and swallows exceptions raised here, so XTier must
check at model load that its MoE backend took over and fail loudly otherwise (see docs/PLAN.md).
Nothing is registered yet: the backend, the expert cache and the packs come with the steps of the plan.
"""

import logging

from xtier import __version__

logger = logging.getLogger(__name__)


def load() -> None:
    logger.info("xtier %s loaded; no hooks registered yet", __version__)
