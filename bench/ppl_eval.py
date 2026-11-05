"""Fixed-text perplexity of a running SGLang server (precision A/B between configurations).

  python bench/ppl_eval.py --port 18180 --tokenizer MODEL_DIR --tag NAME [--segments 6 --tokens 4096] [--corpus NAME=PATH]

Segments are cut deterministically from fixed corpora, so every configuration scores exactly the same token ids. The
default corpora are the ones of the production baselines (bench/baselines: Markdown docs, agent sessions, a Chinese
document; local files, not in the repository). Uses /generate with return_logprob and logprob_start_len=0 (prefill
logprobs of the given ids), flushing the prefix cache before every segment (each is a cold prefill). Reports mean
NLL / PPL overall and per corpus, and per-segment NLL so two runs can be compared pairwise. ``--prefix`` puts text
before every segment, unscored: the document-start token of a model that needs one (MiniMax M3's ``]~!b[``, which
its chat template writes; a segment without it scores as noise).
"""
import argparse
import json
import math
import os
import urllib.request

from transformers import AutoTokenizer

DEFAULT_CORPORA = {
    'docs': '/data/fast/workspaces/lkqmoe-20260928/runs/md-corpus.txt',
    'agent': '/data/fast/workspaces/lkqmoe-20260928/runs/dsh-corpus-all.txt',
    'zh': '/data/fast/workspaces/qwen-runtime/docs/qwen38-flash-next-opt-20260923.md',
}
P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18180)
P.add_argument('--tokenizer', required=True)
P.add_argument('--tag', required=True)
P.add_argument('--segments', type=int, default=6, help='segments per corpus')
P.add_argument('--tokens', type=int, default=4096)
P.add_argument('--corpus', action='append', help='NAME=PATH, repeatable (default: the baseline corpora)')
P.add_argument('--prefix', default='', help='text before every segment, not scored (e.g. a document-start token)')
P.add_argument('--out', default='bench/results/ppl')
P.add_argument('--top', type=int, default=0, help='also keep the N most likely tokens of every scored position in '
               'OUT/TAG.top.json, for the KL between two runs (bench/kl_compare.py)')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
prefix = tok.encode(A.prefix, add_special_tokens=False) if A.prefix else []
corpora = dict(c.split('=', 1) for c in A.corpus) if A.corpus else DEFAULT_CORPORA
segs = []
for name, path in corpora.items():
    text = open(path, encoding='utf-8', errors='ignore').read()
    ids = tok.encode(text[:4_000_000], add_special_tokens=False)
    n = min(A.segments, max(1, len(ids) // A.tokens))
    stride = max(A.tokens, (len(ids) - A.tokens) // max(1, n))
    for i in range(n):
        s = ids[i * stride: i * stride + A.tokens]
        if len(s) == A.tokens:
            segs.append((name, i, s))
rows, tops = [], []
for name, i, s in segs:
    urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{A.port}/flush_cache', b''), timeout=60)
    # the segment's first token stays unscored, as without a prefix
    body = dict(input_ids=prefix + s, sampling_params=dict(max_new_tokens=1, temperature=0),
                return_logprob=True, logprob_start_len=len(prefix) + 1 if prefix else 0)
    if A.top:
        body['top_logprobs_num'] = A.top
    req = urllib.request.Request(f'http://127.0.0.1:{A.port}/generate', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        meta = json.loads(r.read())['meta_info']
    scored = [j for j, x in enumerate(meta['input_token_logprobs']) if x[0] is not None]
    lps = [meta['input_token_logprobs'][j][0] for j in scored]
    nll = -sum(lps) / len(lps)
    rows.append(dict(corpus=name, seg=i, n=len(lps), nll=nll))
    if A.top:   # per scored position: [token id, log probability] of the most likely tokens
        top = meta['input_top_logprobs']
        tops.append(dict(corpus=name, seg=i, lp=[round(x, 6) for x in lps],
                         top=[[[t, round(lp, 6)] for lp, t, *_ in top[j]] for j in scored]))
    print(json.dumps(rows[-1]), flush=True)
summary = {}
for name in list(corpora) + ['all']:
    sel = [r for r in rows if name == 'all' or r['corpus'] == name]
    if sel:
        nll = sum(r['nll'] * r['n'] for r in sel) / sum(r['n'] for r in sel)
        summary[name] = dict(segments=len(sel), nll=round(nll, 5), ppl=round(math.exp(nll), 4))
print('SUMMARY', json.dumps(summary), flush=True)
os.makedirs(A.out, exist_ok=True)
json.dump(dict(tag=A.tag, rows=rows, summary=summary), open(os.path.join(A.out, f'{A.tag}.json'), 'w'), indent=1)
if A.top:
    json.dump(dict(tag=A.tag, top=A.top, segments=tops), open(os.path.join(A.out, f'{A.tag}.top.json'), 'w'))
