"""XTIER_TAIL_SKIP=K/N (keep/total, e.g. 5/6; off by default; changes the output): Strata's route tail skip
(STRATA_ROUTE_TAIL_SKIP=K: their default 7 of a top-10, for experts on NVMe, off by default with NVFP4 experts), here for
every expert the CPU computes, in RAM or on NVMe. N is the model's top-k (checked at the first call of every layer);
the draft model's layers (DSpark's: top-3) keep every route.
A decode-path CPU call (xtier_forward_decode: a decode step, a speculative verify window, a prompt of at most
XTIER_DECODE_MAX tokens; at most XTIER_TAIL_SKIP_TOKENS tokens, 8, one mailbox slot) neither reads nor computes an
expert that no token of the call has among its K largest routes (by weight within the token's top-N): its routes add
nothing, the other routes keep their weights (no renormalisation). The experts in the layer's GPU cache are computed on
the GPU, all of them. Larger calls (prefill) leave out nothing. csrc/engine/engine.cpp tail_skip.

XTIER_TAIL_SKIP_EVAL=1, to measure it: the same rule on each token of a larger call (a prompt) on its own, for the
experts outside the layer's GPU cache, the weights of the routes it would leave out set to zero, so that the
teacher-forced log probabilities of a text (bench/ppl_eval.py) show what the skip does to a decode of one token a step
(a verify window of several tokens leaves out fewer: every one of its tokens must leave the expert out)."""
import logging
import os
import re

import torch

logger = logging.getLogger(__name__)
_seen = [0, 0, 0]   # calls, routes, routes set to zero


def setting():
    """XTIER_TAIL_SKIP=K/N as (K, N); (0, 0) when off (unset, empty or 0). Digits/digits and nothing else: the engine
    reads it too, by the same rule, and refuses a malformed value at the plan's creation."""
    v = os.environ.get('XTIER_TAIL_SKIP', '')
    if v in ('', '0'):
        return 0, 0
    m = re.fullmatch(r'([0-9]{1,4})/([0-9]{1,4})', v)
    if not m or not 1 <= int(m[1]) < int(m[2]):
        raise ValueError(f'XTIER_TAIL_SKIP must be keep/total, e.g. 5/6 (got {v!r})')
    return int(m[1]), int(m[2])


def rank():
    """K of XTIER_TAIL_SKIP=K/N (the routes of rank K and lower are the tail), 0 when off."""
    return setting()[0]


def check(k):
    """A layer's top-k against XTIER_TAIL_SKIP=K/N: N must be k."""
    keep, total = setting()
    if total and total != k:
        raise ValueError(f'XTIER_TAIL_SKIP={keep}/{total}: this model routes every token to {k} experts '
                         f'(keep/total, total {k})')


def eval_rank():
    """K when XTIER_TAIL_SKIP_EVAL=1 applies the rule to prompts, else 0."""
    return rank() if os.environ.get('XTIER_TAIL_SKIP_EVAL', '0') == '1' else 0


def prefill_mask(pool, ids, weights, R):
    """weights with each token's routes at rank >= R (outside its R largest) to the experts a decode step computes on
    the CPU set to zero: the ones outside the layer's GPU cache (pool.table < 0); every expert without a cache, none in
    a fully cached layer."""
    if R <= 0 or R >= ids.shape[1] or (pool is not None and pool.full):
        return weights
    order = torch.argsort(weights, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(ids.shape[1], device=ids.device).expand_as(order).contiguous())
    drop = (ids >= 0) & (ranks >= R)
    if pool is not None:
        drop &= pool.table[ids.clamp(min=0).long()] < 0
    _seen[0] += 1
    _seen[1] += ids.numel()
    _seen[2] += int(drop.sum())
    if _seen[0] % 920 == 0:
        logger.info('xtier tail skip (eval, keep %d/%d): %.2f%% of the routes set to zero so far (%d of %d)', R,
                    ids.shape[1], 100 * _seen[2] / max(1, _seen[1]), _seen[2], _seen[1])
    return weights.masked_fill(drop, 0.0)
