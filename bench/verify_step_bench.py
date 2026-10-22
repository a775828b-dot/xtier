"""Cost of one speculative verify step of a running SGLang server (the production baselines' decode measure).

  python bench/verify_step_bench.py --port 18180 --model Qwen3.8-Flash-Next [--rounds 2] [--out FILE]

20 fixed prompts x temperature 0 and 0.7 x rounds, 768 output tokens, streamed one request at a time; verify steps
from sglang:spec_verify_calls_total in /metrics. ms_per_step = decode seconds / verify steps, accept = tokens / steps,
decode_tps = (tokens - 1) / decode seconds (first to last streamed token).
"""
import argparse
import ast
import json
import re
import time
import urllib.request
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18180)
P.add_argument('--model', required=True)
P.add_argument('--rounds', type=int, default=2)
P.add_argument('--tokens', type=int, default=768)
P.add_argument('--prompts', default='/data/fast/workspaces/ds41-1017/experiments/sts_traffic.py')
P.add_argument('--out')
A = P.parse_args()
URL = f'http://127.0.0.1:{A.port}'
src = Path(A.prompts).read_text(encoding='utf-8')
PROMPTS = next(ast.literal_eval(n.value) for n in ast.parse(src).body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'PROMPTS')


def verify_calls():
    t = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
    m = re.search(r'^sglang:spec_verify_calls_total\{[^}]*\} ([0-9.e+]+)$', t, re.M)
    if not m:
        raise SystemExit('no sglang:spec_verify_calls_total in /metrics')
    return float(m.group(1))


rows = []
for r in range(A.rounds):
    for temp in (0.0, 0.7):
        for i, p in enumerate(PROMPTS):
            body = json.dumps(dict(model=A.model, messages=[{'role': 'user', 'content': p}], max_tokens=A.tokens,
                                   temperature=temp, stream=True, stream_options={'include_usage': True})).encode()
            req = urllib.request.Request(URL + '/v1/chat/completions', body, {'Content-Type': 'application/json'})
            v0, first, usage = verify_calls(), None, None
            with urllib.request.urlopen(req, timeout=900) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith('data:') or line == 'data: [DONE]':
                        continue
                    ev = json.loads(line[5:])
                    ch = ev.get('choices') or []
                    d = (ch[0].get('delta') or {}) if ch else {}
                    if first is None and (d.get('content') or d.get('reasoning_content')):
                        first = time.time()
                    if ev.get('usage'):
                        usage = ev['usage']
            t1 = time.time()
            rows.append(dict(round=r, temp=temp, i=i, tokens=usage['completion_tokens'],
                             decode_s=(t1 - first) if first else 0.0, steps=verify_calls() - v0))
tok = sum(x['tokens'] - 1 for x in rows)
sec = sum(x['decode_s'] for x in rows)
steps = sum(x['steps'] for x in rows)
summary = dict(decode_tps=round(tok / sec, 2), ms_per_step=round(sec / steps * 1000, 3),
               accept=round(sum(x['tokens'] for x in rows) / steps, 3), requests=len(rows))
print('SUMMARY', json.dumps(summary), flush=True)
if A.out:
    json.dump(dict(rows=rows, summary=summary), open(A.out, 'w'), indent=1)
