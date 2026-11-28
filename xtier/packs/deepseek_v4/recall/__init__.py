"""Backward-compatible shim: universal Recall now lives in xtier.packs.speculative.recall."""
from xtier.packs.speculative.recall import enabled, install

__all__ = ["enabled", "install"]
