"""XTIER_TAIL_SKIP=R (off by default; changes the output): Strata's route tail skip (STRATA_ROUTE_TAIL_SKIP: their default
7 of a top-10, off by default with NVFP4 experts). A decode-sized CPU call (at most XTIER_TAIL_SKIP_TOKENS tokens, 8: a
decode step, one request's speculative verify window) neither reads nor computes an expert that is not in RAM when every
token of the call routes it at rank R or lower (by weight within the token's top-k, 0 the largest): its routes add
nothing, the other routes keep their weights (no renormalisation). csrc/engine/engine.cpp tail_skip.

XTIER_TAIL_SKIP_EVAL=1, to measure it: the same rule on each token of a larger call (a prompt) on its own, the weights of
the routes it would leave out set to zero, so that the teacher-forced log probabilities of a text (bench/ppl_eval.py)
show what the skip does to a decode of one token a step (a verify window of several tokens leaves out fewer: every one of
its tokens must rank the expert that low)."""
import logging
import os

import torch

logger = logging.getLogger(__name__)
_seen = [0, 0, 0]   # calls, routes, routes set to zero


def rank():
    v = os.environ.get('XTIER_TAIL_SKIP', '').strip()
    return max(0, int(v)) if v else 0


def eval_rank():
    """R when XTIER_TAIL_SKIP_EVAL=1 applies the rule to prompts, else 0."""
    return rank() if os.environ.get('XTIER_TAIL_SKIP_EVAL', '0') == '1' else 0


def prefill_mask(plan, ids, weights, R):
    """weights with each token's routes to experts outside RAM (plan.slot_table() < 0) at rank >= R set to zero."""
    if plan is None or R <= 0 or R >= ids.shape[1]:
        return weights
    cold = torch.tensor([s < 0 for s in plan.slot_table()], dtype=torch.bool, device=ids.device)
    if not bool(cold.any()):
        return weights
    order = torch.argsort(weights, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(ids.shape[1], device=ids.device).expand_as(order).contiguous())
    drop = (ids >= 0) & cold[ids.clamp(min=0).long()] & (ranks >= R)
    _seen[0] += 1
    _seen[1] += ids.numel()
    _seen[2] += int(drop.sum())
    if _seen[0] % 920 == 0:
        logger.info('xtier tail skip (eval, rank >= %d): %.2f%% of the routes set to zero so far (%d of %d)', R,
                    100 * _seen[2] / max(1, _seen[1]), _seen[2], _seen[1])
    return weights.masked_fill(drop, 0.0)
