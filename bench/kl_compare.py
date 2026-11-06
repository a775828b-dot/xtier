"""How far a configuration moves a model's predictions from a reference: two bench/ppl_eval.py --top N dumps of the same
segments (the reference first). Per scored position, the KL(reference || test) over the tokens both runs list among
their N most likely, the rest of each distribution as one more outcome: merging outcomes can only lower a KL, so this is
a lower bound of the full KL, close to it when the N tokens hold most of the mass. Reported: its mean and 99th
percentile, the positions whose most likely token changed, the mean NLL difference of the scored tokens.

  python bench/kl_compare.py bench/results/ppl/REF.top.json bench/results/ppl/TEST.top.json
"""
import json
import math
import sys


def kl(p_top, q_top):
    p, q = dict(p_top), dict(q_top)
    both = [t for t in p if t in q]
    pk, qk = [math.exp(p[t]) for t in both], [math.exp(q[t]) for t in both]
    rest_p, rest_q = max(1e-9, 1 - sum(pk)), max(1e-9, 1 - sum(qk))
    return sum(a * (math.log(a) - math.log(b)) for a, b in zip(pk, qk)) + rest_p * (math.log(rest_p) - math.log(rest_q))


def main():
    ref, test = (json.load(open(p)) for p in sys.argv[1:3])
    if [(s['corpus'], s['seg'], len(s['lp'])) for s in ref['segments']] != \
            [(s['corpus'], s['seg'], len(s['lp'])) for s in test['segments']]:
        sys.exit('kl_compare: the two dumps do not score the same segments')
    kls, flips, dnll, n = [], 0, 0.0, 0
    per = {}
    for a, b in zip(ref['segments'], test['segments']):
        seg = []
        for lp_a, lp_b, ta, tb in zip(a['lp'], b['lp'], a['top'], b['top']):
            seg.append(max(0.0, kl(ta, tb)))
            flips += max(ta, key=lambda x: x[1])[0] != max(tb, key=lambda x: x[1])[0]
            dnll += lp_a - lp_b
            n += 1
        kls += seg
        c = per.setdefault(a['corpus'], [0.0, 0])
        c[0] += sum(seg)
        c[1] += len(seg)
    kls.sort()
    print(json.dumps(dict(ref=ref['tag'], test=test['tag'], positions=n, kl_mean=round(sum(kls) / n, 5),
                          kl_p99=round(kls[int(0.99 * (n - 1))], 4), top1_changed=round(flips / n, 4),
                          nll_diff=round(dnll / n, 5),
                          kl_by_corpus={c: round(s / m, 5) for c, (s, m) in per.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
