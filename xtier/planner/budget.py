"""Global plan of the expert cache: one GPU budget of slots split over the MoE layers.

Input: per-layer expert usage counts (the LKQstrata profile's "counts": decode windows that used each expert, over
"windows" windows). A slot added to layer l holds its next most used expert, which the CPU engine then no longer
reads in the windows that use it: the gain of a slot is that expert's window count. A layer that holds every expert
also drops its CPU part and the mailbox round trip of every call, worth ``full_bonus`` windows (the cost the
round trip has, in units of one expert read). Slots go to the largest gain per slot first, single slots and
"complete this layer" moves competing, until the budget is spent.

The model is the decode cost the cache removes (expert reads of the CPU engine per window); prefill and the GPU's
cost of the cached routes are not in it.
"""

import heapq


def plan(counts, budget, full_bonus=0.0):
    """counts: {layer: [window count per expert]}; budget: slots in all. -> {layer: slots} (len(counts[l]) = full)."""
    order = {l: sorted(c, reverse=True) for l, c in counts.items()}
    rest = {l: [0.0] * (len(c) + 1) for l, c in order.items()}   # rest[l][s]: gain of the experts after the first s
    for l, c in order.items():
        for s in range(len(c) - 1, -1, -1):
            rest[l][s] = rest[l][s + 1] + c[s]
    slots = {l: 0 for l in order}
    left = int(budget)
    while left > 0:
        best = None   # (gain per slot, layer, slots to add)
        for l, c in order.items():
            s, n = slots[l], len(c)
            if s >= n:
                continue
            single = (c[s] + (full_bonus if s + 1 == n else 0.0), l, 1)
            full = ((rest[l][s] + full_bonus) / (n - s), l, n - s) if n - s <= left else None
            for cand in (single, full):
                if cand is not None and (best is None or cand[0] > best[0]):
                    best = cand
        if best is None:
            break
        _, l, add = best
        add = min(add, left)
        slots[l] += add
        left -= add
    return slots


def cpu_reads(counts, slots, windows):
    """Expert reads the CPU engine still does per window (uncached experts weighted by how often they are used)."""
    total = 0.0
    for l, c in counts.items():
        total += sum(sorted(c, reverse=True)[slots.get(l, 0):])
    return total / max(windows, 1)


def full_layers(counts, slots):
    return sorted(l for l, c in counts.items() if slots.get(l, 0) >= len(c))
