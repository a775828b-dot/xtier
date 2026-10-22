"""Prefill latency of a running SGLang server by prompt length: /generate with max_new_tokens=1 on token ids cut from a
fixed corpus, prefix cache flushed before each request (cold prefill), median of --repeat runs.

  python bench/ttft_probe.py --port 18180 --tokenizer DIR [--lengths 70,500,1000,2000,4096,8192] [--repeat 3]
"""
import argparse
import json
import statistics
import time
import urllib.request

from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18180)
P.add_argument('--tokenizer', required=True)
P.add_argument('--lengths', default='70,500,1000,2000,4096,8192')
P.add_argument('--repeat', type=int, default=3)
P.add_argument('--corpus', default='/data/fast/workspaces/lkqmoe-20260928/runs/md-corpus.txt')
P.add_argument('--out')
A = P.parse_args()
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
ids = tok.encode(open(A.corpus, encoding='utf-8', errors='ignore').read()[:400000], add_special_tokens=False)
base = f'http://127.0.0.1:{A.port}'
rows = []
for n in map(int, A.lengths.split(',')):
    times = []
    for r in range(A.repeat):
        urllib.request.urlopen(urllib.request.Request(base + '/flush_cache', b''), timeout=60)
        s = ids[r * 9973 % (len(ids) - n): r * 9973 % (len(ids) - n) + n]
        body = json.dumps(dict(input_ids=s, sampling_params=dict(max_new_tokens=1, temperature=0))).encode()
        t = time.time()
        urllib.request.urlopen(urllib.request.Request(base + '/generate', body, {'Content-Type': 'application/json'}),
                               timeout=1800).read()
        times.append(time.time() - t)
    med = statistics.median(times)
    rows.append(dict(tokens=n, seconds=round(med, 3), tok_s=round(n / med, 1), runs=[round(x, 3) for x in times]))
    print(json.dumps(rows[-1]), flush=True)
if A.out:
    json.dump(rows, open(A.out, 'w'), indent=1)
