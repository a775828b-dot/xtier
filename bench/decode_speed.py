"""Decode speed of a running SGLang server: streamed chat completions, decode tok/s from the first to the last token.

  python bench/decode_speed.py --port 18180 --model Qwen3.8-Flash-Next [--tokens 512] [--concurrency 1] [--rounds 2]

Prompts are fixed (code, prose, math, Chinese); temperature 0, thinking off. Prints per request and the aggregate
(sum of tokens / wall time of the round) per round.
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request

PROMPTS = [
    'Write a Python class implementing an LRU cache with get and put in O(1), with docstrings and a short usage example.',
    'Explain how a CPU branch predictor works, covering two-level adaptive predictors and why mispredictions are costly.',
    '用中文详细介绍一下长城的历史，包括修建的朝代、主要用途和今天的保护情况。',
    'Solve step by step: a train leaves at 9:15 at 84 km/h, another at 10:00 at 120 km/h on the same track. When does the second catch up?',
]
P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18180)
P.add_argument('--model', required=True)
P.add_argument('--tokens', type=int, default=512)
P.add_argument('--concurrency', type=int, default=1)
P.add_argument('--rounds', type=int, default=2)
P.add_argument('--out')
A = P.parse_args()
url = f'http://127.0.0.1:{A.port}/v1/chat/completions'


def run(prompt):
    body = dict(model=A.model, temperature=0, max_tokens=A.tokens, stream=True, ignore_eos=True,
                chat_template_kwargs={'enable_thinking': False}, messages=[{'role': 'user', 'content': prompt}],
                stream_options={'include_usage': True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0, first, last, n = time.time(), None, None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            d = json.loads(line[6:])
            if d.get('usage'):
                n = d['usage']['completion_tokens']
            delta = d['choices'][0].get('delta', {}) if d.get('choices') else {}
            if delta.get('content') or delta.get('reasoning_content'):
                now = time.time()
                first = first or now
                last = now
    return dict(tokens=n, ttft=first - t0, decode=(n - 1) / (last - first) if last and last > first else 0.0,
                first=first, last=last)


rounds = []
for r in range(A.rounds):
    prompts = [PROMPTS[(r * A.concurrency + i) % len(PROMPTS)] for i in range(A.concurrency)]
    with cf.ThreadPoolExecutor(A.concurrency) as ex:
        res = list(ex.map(run, prompts))
    wall = max(x['last'] for x in res) - min(x['first'] for x in res)
    agg = sum(x['tokens'] for x in res) / wall
    rounds.append(dict(aggregate=agg, requests=res))
    print(f'round {r}: aggregate {agg:.1f} tok/s; per request ' +
          ', '.join(f'{x["decode"]:.1f} tok/s (ttft {x["ttft"]:.2f} s, {x["tokens"]} tok)' for x in res), flush=True)
summary = dict(concurrency=A.concurrency, tokens=A.tokens,
               per_request=round(statistics.mean(x['decode'] for r in rounds for x in r['requests']), 1),
               aggregate=round(statistics.mean(r['aggregate'] for r in rounds), 1))
print('SUMMARY', json.dumps(summary))
if A.out:
    json.dump(dict(summary=summary, rounds=rounds), open(A.out, 'w'), indent=1)
