"""Automatic cache plan: slots per MoE layer under one GPU budget, from the cost model (cost.py).

Every layer picks a slot count s in {0, q, 2q, ..., E - q} or E (every expert cached). A dynamic program over the
budget (in units of q slots) minimizes

  decode(plan) + lam * prefill(plan)

where decode is the expected MoE time of one decode window (sum over the layers of cost.Model.layer_us) and prefill
the upload time of one prefill chunk (cost.Model.upload_us for every layer that is not fully cached). lam = 0 gives
the fastest decode the budget allows; ``best`` then raises lam as long as the decode time stays within ``tol`` of that
optimum, so the remaining freedom goes to fully cached layers (faster prefill) and not to slots that do not change
the decode. Exact for this separable model; about a second for 50 layers x 512 experts.
"""
import json
import logging
import os

import numpy as np

from xtier.planner.cost import Model, plan_us, usages as profile_usages

logger = logging.getLogger(__name__)


def options(E, q):
    return list(range(0, E, q)) + [E]


def solve(model, usages, budget, lam=0.0, q=None):
    """{layer: slots} minimizing decode + lam * prefill with at most `budget` slots over `usages` {layer: Usage}; slot
    counts in steps of q (default E / 64: 8 for 512 experts, 2 for 128)."""
    layers = sorted(usages)
    E = model.E
    q = q or max(1, E // 64)
    opts = options(E, q)
    units = np.array([(s + q - 1) // q for s in opts])
    B = int(budget) // q
    up = model.upload_us()
    INF = float('inf')
    dp = np.zeros(B + 1)
    choice = np.zeros((len(layers), B + 1), dtype=np.int16)
    for i, l in enumerate(layers):
        cost = np.array([model.layer_us(usages[l], s) + (0.0 if s >= E else lam * up) for s in opts])
        new = np.full(B + 1, INF)
        arg = np.zeros(B + 1, dtype=np.int16)
        for o, (u, c) in enumerate(zip(units, cost)):
            if u > B:
                continue
            cand = np.full(B + 1, INF)
            cand[u:] = dp[:B + 1 - u] + c
            better = cand < new
            new[better] = cand[better]
            arg[better] = o
        dp = new
        choice[i] = arg
    if not np.isfinite(dp[B]):
        raise ValueError(f'planner: a budget of {budget} slots does not fit the smallest plan')
    plan, b = {}, B
    for i in range(len(layers) - 1, -1, -1):
        o = int(choice[i, b])
        plan[layers[i]] = opts[o]
        b -= int(units[o])
    return plan


def prefill_us(model, plan):
    """Upload time of one prefill chunk: every layer that is not fully cached."""
    return sum(model.upload_us() for s in plan.values() if s < model.E)


LAMS = [0.0] + [x * 10 ** e for e in range(-5, 0) for x in (1, 1.5, 2, 3, 5, 7)]


def best(model, usages, budget, tol=0.003, q=None, bound=None, lams=LAMS):
    """The plan with the least prefill upload among those whose decode is within tol of the fastest one (or at most
    `bound` microseconds per window). Returns (plan, decode_us, prefill_us, lam)."""
    base = solve(model, usages, budget, 0.0, q)
    d0 = plan_us(model, usages, base)
    limit = bound if bound is not None else d0 * (1 + tol)
    chosen, lo, hi = (base, d0, prefill_us(model, base), 0.0), 0.0, None
    # decode grows and upload shrinks with lam (Lagrangian solutions): the answer is the largest feasible lam
    for lam in lams[1:]:
        p = solve(model, usages, budget, lam, q)
        d = plan_us(model, usages, p)
        if d > limit:
            hi = lam
            break
        lo, chosen = lam, (p, d, prefill_us(model, p), lam)
    for _ in range(8 if hi is not None else 0):
        lam = (lo * hi) ** 0.5 if lo > 0 else hi / 4
        p = solve(model, usages, budget, lam, q)
        d = plan_us(model, usages, p)
        if d > limit:
            hi = lam
        else:
            lo = lam
            if prefill_us(model, p) <= chosen[2]:
                chosen = (p, d, prefill_us(model, p), lam)
    return chosen


def even(layout, budget, E):
    """The hand layout with the budget's extra slots spread evenly over its partly cached layers."""
    part = sorted(l for l, s in layout.items() if 0 < s < E)
    out = dict(layout)
    extra = int(budget) - sum(layout.values())
    if part and extra > 0:
        for i, l in enumerate(part):
            out[l] = min(E - 1, out[l] + extra // len(part) + (1 if i < extra % len(part) else 0))
    return out


def guard(model, usages, plan, reference, min_gain=0.01):
    """The plan, or the reference when the plan uploads no less and its predicted decode gain is below min_gain:
    measured on GLM-5.3, a 0.8 % predicted gain of an uneven split did not show (the per-layer model is not that
    exact), so small reshuffles are not worth the risk. Returns (plan, kept_reference)."""
    if prefill_us(model, plan) < prefill_us(model, reference):
        return plan, False
    if plan_us(model, usages, plan) <= plan_us(model, usages, reference) * (1 - min_gain):
        return plan, False
    return reference, True


def from_env(full_layers, slots):
    """XTIER_CACHE_PLAN=auto: the plan of the profiled layers, computed at startup.

    XTIER_CACHE_PROFILE   routing profile with "counts" (tools/plan_trace.py makes one from a traced run)
    XTIER_PLANNER_PROBE   tools/plan_probe.py output for the model's layer shape on this host
    XTIER_CACHE_BUDGET    slots over the profiled layers (or XTIER_CACHE_BUDGET_GIB); default: those of the hand
                          layout XTIER_CACHE_FULL / XTIER_CACHE_SLOTS (`full_layers`, `slots`), i.e. the same memory,
                          plus XTIER_CACHE_BUDGET_ADD_GIB (memory measured free at the prefill peak, see docs/planner.md)
    XTIER_PLANNER_TOL     decode slack traded for fully cached layers (default 0.003)
    XTIER_PLANNER_KEEP=1  (default) the decode bound is the hand layout's predicted decode (x 1 + tol), and a plan that
                          uploads more than the hand layout is not taken; 0: bound by the fastest plan (x 1 + tol). The
                          model is measured right near the hand layouts but not far from them (Qwen3.8 with every layer
                          partly cached: predicted 1.4 % faster, measured 3.3 % slower), hence the default.
    XTIER_PLANNER_LAM     a fixed prefill weight instead (0: the fastest decode)
    XTIER_PLANNER_ADAPT   adaptive residency gain of cost.Model (default 1.0: the static profile ranking)
    XTIER_PLANNER_MIN_GAIN  below this predicted decode gain (default 0.01) a plan that does not upload less keeps the
                          hand layout (its extra budget spread evenly), see guard()
    """
    prof, probe = os.environ.get('XTIER_CACHE_PROFILE', ''), os.environ.get('XTIER_PLANNER_PROBE', '')
    if not prof or not probe:
        raise RuntimeError('xtier: XTIER_CACHE_PLAN=auto needs XTIER_CACHE_PROFILE and XTIER_PLANNER_PROBE')
    with open(prof) as f:
        d = json.load(f)
    with open(probe) as f:
        model = Model.from_probe(json.load(f), float(os.environ.get('XTIER_PLANNER_ADAPT', '1.0')))
    if 'counts' not in d or int(d['experts']) != model.E:
        raise RuntimeError(f'xtier: XTIER_CACHE_PLAN=auto: {prof} needs "counts" for {model.E} experts per layer')
    usages = profile_usages(d, model.tokens)
    E = model.E
    layout = {l: (E if l in full_layers else slots) for l in usages}
    budget = os.environ.get('XTIER_CACHE_BUDGET', '').strip()
    gib = os.environ.get('XTIER_CACHE_BUDGET_GIB', '').strip()
    budget = int(budget) if budget else int(float(gib) * 2**30 / model.slot_bytes()) if gib else sum(layout.values())
    budget += int(float(os.environ.get('XTIER_CACHE_BUDGET_ADD_GIB', '0') or 0) * 2**30 / model.slot_bytes())
    tol = float(os.environ.get('XTIER_PLANNER_TOL', '0.003'))
    lam = os.environ.get('XTIER_PLANNER_LAM', '').strip()
    ref = plan_us(model, usages, layout)
    if lam:
        plan = solve(model, usages, budget, float(lam))
        dec, pre, lam = plan_us(model, usages, plan), prefill_us(model, plan), float(lam)
    else:
        keep = os.environ.get('XTIER_PLANNER_KEEP', '1') == '1'
        plan, dec, pre, lam = best(model, usages, budget, tol, bound=ref * (1 + tol) if keep else None)
        if keep and pre > prefill_us(model, even(layout, budget, E)):
            plan = even(layout, budget, E)
    plan, kept = guard(model, usages, plan, even(layout, budget, E), float(os.environ.get('XTIER_PLANNER_MIN_GAIN', '0.01')))
    if kept:
        dec, pre = plan_us(model, usages, plan), prefill_us(model, plan)
    full = sorted(l for l, s in plan.items() if s >= E)
    logger.info('xtier planner: %d profiled layers, budget %d slots (%.1f GiB): %d full %s, the others %s slots%s; '
                'decode %.3f ms per window (hand layout %.3f), prefill upload %.0f ms per chunk (hand layout %.0f), '
                'lam %g', len(usages), budget, budget * model.slot_bytes() / 2**30, len(full), full,
                sorted({s for s in plan.values() if s < E}), ' (hand layout kept: gain below XTIER_PLANNER_MIN_GAIN)'
                if kept else '', dec / 1e3, ref / 1e3, pre / 1e3, prefill_us(model, layout) / 1e3, lam)
    return plan
