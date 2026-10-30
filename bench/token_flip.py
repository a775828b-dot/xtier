"""How much a copy test's token depends on the prompt's KV (a running server with logprobs):
  python bench/token_flip.py --copy bench/results/copy/TAG-c1.json [--port 30000] [--model M] [--cycles 12] [--segment 2]
         [--key 'SPLIT_TILE / PREFILL_TRUNC'] [--watch ATE,ATION]
Every cycle flushes the prefix cache (so the prompt's KV is computed again) and generates through the place --key marks in
the segment, at temperature 0, with the top 20 logprobs: what came next and the logprobs of the watched tokens (the last
one is the right one). Same weights, same text: differences between cycles are the prompt's KV.
"""
import argparse
import json
import statistics
import urllib.request

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=30000)
P.add_argument('--model', default='Qwen3.8-Flash-Next')
P.add_argument('--copy', required=True)
P.add_argument('--segment', type=int, default=2)
P.add_argument('--key', default='SPLIT_TILE / PREFILL_TRUNC')
P.add_argument('--watch', default='ATE,ATION')
P.add_argument('--cycles', type=int, default=12)
A = P.parse_args()
src = json.load(open(A.copy))[A.segment]['src']
user = '把 <src> 和 </src> 之间的内容逐字原样输出，不要增删改任何字符，不要加代码块标记，不要解释。\n<src>\n' + src + '\n</src>'
limit = src.index(A.key) // 4 + 40
watch = A.watch.split(',')
values = {w: [] for w in watch}
flips = 0
for cycle in range(A.cycles):
    urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{A.port}/flush_cache', method='POST'), timeout=60).read()
    body = dict(model=A.model, temperature=0, max_tokens=limit, stream=False, logprobs=True, top_logprobs=20,
                chat_template_kwargs={'enable_thinking': False}, messages=[{'role': 'user', 'content': user}])
    q = urllib.request.Request(f'http://127.0.0.1:{A.port}/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(q, timeout=900).read())
    items = d['choices'][0]['logprobs']['content']
    text = ''.join(x['token'] for x in items)
    place = text.find(A.key)
    if place < 0:
        print(f'cycle {cycle}: place not reached in {len(items)} tokens', flush=True)
        continue
    seen, k = 0, None
    for i, x in enumerate(items):
        seen += len(x['token'])
        if seen > place + len(A.key) - 1:
            k = i + 1
            break
    tops = {c['token']: c['logprob'] for c in items[k]['top_logprobs']}
    row = {w: tops.get(w) for w in watch}
    for w in watch:
        if row[w] is not None:
            values[w].append(row[w])
    flips += items[k]['token'] != watch[-1]
    cached = (d['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
    print(f'cycle {cycle}: cached {cached}, got {items[k]["token"]!r}; ' + ', '.join(f'{w} {row[w] if row[w] is None else round(row[w], 3)}' for w in watch), flush=True)
summary = {w: (round(min(v), 2), round(statistics.median(v), 2), round(max(v), 2), len(v)) for w, v in values.items() if v}
print(f'SUMMARY {A.cycles} cycles, other token than {watch[-1]!r} in {flips}; (min, median, max, seen in the top 20): {summary}', flush=True)
