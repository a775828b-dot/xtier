"""Planner cost model: the decode time of a model's XTier MoE layers under a cache plan.

Per decode window (one forward of the verify width m), a layer whose m tokens use n_g distinct cached experts and n_c
other ones costs (probe.py measures the coefficients on this host, for this layer shape):

  every expert cached (S = E):   gpu(n_g)                   GPU alone, no mailbox
  partly cached (0 < S < E):     max(gpu(n_g) + p0, cpu(n_c, r_c) + p1)   both sides at once, the slower one waits
  no cache (S = 0):              cpu(n_c, r_c)              mailbox only

with gpu(n) = g0 + g1 n (the GPU part reads each used expert once for all tokens) and cpu(n, r) = c0 + c1 n + c2 r
(r: routes, token-expert pairs, of the CPU part; on large experts a token's compute costs about a fifth of the read).
n_g, n_c and r_c come from the routing profile: an expert used in a fraction p of the windows adds p to the expected
distinct count and p (1 - p) to its variance, its routes per window to r; the slots of a layer hold its most used
experts. The expectation of the max uses Clark's formula for two normal variables, so a layer whose two sides are
balanced pays for the spread of the windows. The rest of a decode step (attention, dense layers, sampling) does not
depend on the plan and is not modelled.

Prefill: a layer that is not fully cached uploads all its experts once per prefill chunk (GPU prefill); a fully
cached one computes on its slots in place. ``upload_us`` is the copy time of one layer at the probe's bandwidth.
"""
import math


def _lsq(points):
    """Least squares y = a + b x over [(x, y), ...]."""
    n = len(points)
    sx = sum(p[0] for p in points); sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points); sxy = sum(p[0] * p[1] for p in points)
    d = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / d if d else 0.0
    return (sy - b * sx) / n, b


def _lsq2(points):
    """Least squares y = a + b x + c z over [(x, z, y), ...] (normal equations, 3 x 3)."""
    import numpy as np
    X = np.array([[1.0, p[0], p[1]] for p in points])
    y = np.array([p[2] for p in points])
    return [float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0]]


def fit(probe):
    """{'gpu': [g0, g1], 'cpu': [c0, c1, c2], 'part': [p0, p1], 'rms_*_us': ...} from the probe's points (version 1
    probes, without route counts: c2 = 0)."""
    g0, g1 = _lsq(probe['gpu_full'])
    cpu = probe['cpu_only']
    if cpu and len(cpu[0]) == 3:
        c0, c1, c2 = _lsq2(cpu)
        pts = probe['partial']   # [g, c, CPU routes, us]
    else:
        (c0, c1), c2 = _lsq(cpu), 0.0
        pts = [(g, c, 0.0, us) for g, c, us in probe['partial']]
    rms_cpu = math.sqrt(sum((c0 + c1 * p[0] + c2 * (p[1] if len(p) == 3 else 0) - p[-1]) ** 2 for p in cpu) / len(cpu))
    best = None
    for i in range(-40, 81):
        p0 = i * 0.5
        for j in range(-160, 81):
            p1 = j * 0.5
            err = sum((max(g0 + g1 * g + p0, c0 + c1 * c + c2 * r + p1) - us) ** 2 for g, c, r, us in pts)
            if best is None or err < best[0]:
                best = (err, p0, p1)
    err, p0, p1 = best if best else (0.0, 0.0, 0.0)
    return dict(gpu=[round(g0, 3), round(g1, 4)], cpu=[round(c0, 3), round(c1, 4), round(c2, 4)], part=[p0, p1],
                rms_cpu_us=round(rms_cpu, 2), rms_part_us=round(math.sqrt(err / max(len(pts), 1)), 2))


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def emax(ma, va, mb, vb):
    """E[max(A, B)] of independent normals A ~ (ma, va), B ~ (mb, vb) (Clark 1961)."""
    t = math.sqrt(va + vb)
    if t < 1e-9:
        return max(ma, mb)
    a = (ma - mb) / t
    return ma * _Phi(a) + mb * _Phi(-a) + t * _phi(a)


class Usage:
    """A layer's experts by usage: p[i] = fraction of the decode windows using the i-th most used expert, q[i] = its
    routes (token-expert pairs) per window. Without route counts in the profile every use of an expert is taken to
    carry the window's average routes per distinct expert (`tokens` = routes per window, m x top-k)."""

    def __init__(self, counts, windows, routes=None, tokens=None):
        order = sorted(range(len(counts)), key=lambda e: -counts[e])
        self.p = [counts[e] / windows for e in order]
        if routes is not None:
            self.q = [routes[e] / windows for e in order]
        else:
            per = tokens / sum(self.p) if tokens and sum(self.p) > 0 else 1.0
            self.q = [p * per for p in self.p]
        self.E = len(self.p)
        self.mean, self.var, self.rq = [0.0], [0.0], [0.0]   # prefix sums over the first s experts
        for p, q in zip(self.p, self.q):
            self.mean.append(self.mean[-1] + p)
            self.var.append(self.var[-1] + p * (1 - p))
            self.rq.append(self.rq[-1] + q)

    def split(self, s):
        """Per window with s slots (the top s cached): mean and variance of the distinct cached experts, of the
        distinct uncached ones, and the mean routes to uncached experts."""
        s = max(0, min(s, self.E))
        return (self.mean[s], self.var[s], self.mean[-1] - self.mean[s], self.var[-1] - self.var[s],
                self.rq[-1] - self.rq[s])

    def hit(self, s):
        return self.mean[min(s, self.E)] / max(self.mean[-1], 1e-9)


def usages(profile, tokens=None):
    """{layer: Usage} of a profile with "counts" (and optionally "routes", tools/plan_trace.py)."""
    W = int(profile['windows'])
    routes = profile.get('routes') or [None] * len(profile['counts'])
    return {int(l): Usage(c, W, r, tokens) for l, c, r in zip(profile['layer_ids'], profile['counts'], routes)}


class Model:
    """Per-call costs in microseconds (fit of probe.py) and the upload bandwidth of one layer shape.

    adapt: the adaptive residency (cache/pool.py swaps in what the current text uses) makes S slots of a partly cached
    layer hit like the profile's top adapt x S experts; measured on DeepSeek V4.1 (50 slots: hit 0.568 at run time,
    the seed profile's top 82 cover the same), so about 1.6 there. 1.0: the static profile ranking."""

    def __init__(self, fit, expert_bytes, experts, h2d_gbps=None, adapt=1.0, tokens=None):
        self.g0, self.g1 = fit['gpu']
        c = fit['cpu']
        self.c0, self.c1, self.c2 = c[0], c[1], (c[2] if len(c) > 2 else 0.0)
        self.p0, self.p1 = fit['part']
        self.expert_bytes, self.E = expert_bytes, experts
        self.h2d = h2d_gbps
        self.adapt = float(adapt)
        self.tokens = tokens   # routes per decode window of the probe (m x top-k)

    @classmethod
    def from_probe(cls, probe, adapt=1.0):
        m = probe['model']
        return cls(probe['fit'], m['expert_bytes'], m['E'], probe.get('h2d_gbps'), adapt, probe['m'] * m['k'])

    def effective(self, usage, s):
        """Slots of the profile ranking that hit like s adaptive slots (s for an uncached or fully cached layer)."""
        return s if s <= 0 or s >= usage.E else min(usage.E, int(round(self.adapt * s)))

    def _cpu(self, mc, rc):
        return self.c0 + self.c1 * mc + self.c2 * rc

    def layer_us(self, usage, s):
        """Expected decode time of one layer's MoE call with s slots (s >= E: every expert cached)."""
        mg, vg, mc, vc, rc = usage.split(self.effective(usage, s))
        if s >= usage.E:
            return self.g0 + self.g1 * mg
        if s <= 0:
            return self._cpu(mc, rc)
        per = rc / mc if mc > 1e-9 else 0.0   # routes per uncached expert: their spread follows the experts'
        return emax(self.g0 + self.g1 * mg + self.p0, (self.g1 ** 2) * vg,
                    self._cpu(mc, rc) + self.p1, ((self.c1 + self.c2 * per) ** 2) * vc)

    def sides_us(self, usage, s):
        """(GPU side, CPU side) expected microseconds of a partly cached layer, for reports."""
        mg, _, mc, _, rc = usage.split(self.effective(usage, s))
        return self.g0 + self.g1 * mg + self.p0, self._cpu(mc, rc) + self.p1

    def upload_us(self):
        """Copy time of one layer's experts at the probe's host-to-device bandwidth (a prefill chunk of a layer
        that is not fully cached)."""
        if not self.h2d:
            return 0.0
        return self.E * self.expert_bytes / (self.h2d * 1e9) * 1e6

    def slot_bytes(self):
        return self.expert_bytes


def plan_us(model, usages, slots):
    """Decode MoE microseconds per window of a plan {layer: slots} over the profiled layers {layer: Usage}."""
    return sum(model.layer_us(u, slots.get(l, 0)) for l, u in usages.items())
