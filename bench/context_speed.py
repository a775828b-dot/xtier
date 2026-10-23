"""Long-context speed of a running SGLang server (the README table): per context length, a cold prefill of a prompt
of about N tokens (agent-session text from a fixed corpus plus a review task; prefix cache flushed first) and an
answer of up to --output tokens, streamed through /v1/chat/completions with the server's default sampling (as
bench/agent_bench.py). Prefill tok/s = prompt tokens / time to the first streamed token; decode tok/s = (output
tokens - 1) / time from the first to the last streamed token (reasoning and answer tokens alike).

  python bench/context_speed.py --port 18190 --model DeepSeek-V4.1-Flash-1017 --tokenizer MODEL_DIR \\
      [--contexts 65536,520000] [--output 1024] [--out FILE]
"""
import argparse
import json
import time
import urllib.request

from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--model', required=True)
P.add_argument('--tokenizer', required=True)
P.add_argument('--contexts', default='65536,520000', help='prompt tokens per measurement (before the chat template)')
P.add_argument('--output', type=int, default=1024)
P.add_argument('--corpus', default='/data/fast/workspaces/lkqmoe-20260928/runs/dsh-corpus-all.txt')
P.add_argument('--out')
A = P.parse_args()
TASK = ('\n\n---\nThe text above is a log of earlier coding-agent sessions. Write a detailed review for a new team '
        'member: what the sessions worked on, the main problems and how they were solved, the decisions taken, and '
        'what is still open. Use sections and be thorough.')
URL = f'http://127.0.0.1:{A.port}'
tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)
ids = tok.encode(open(A.corpus, encoding='utf-8', errors='ignore').read(), add_special_tokens=False)


def post(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else b''
    req = urllib.request.Request(URL + path, data, {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


rows = []
for n in [int(x) for x in A.contexts.split(',')]:
    if n > len(ids):
        raise SystemExit(f'corpus has {len(ids)} tokens, {n} asked')
    text = tok.decode(ids[:n]) + TASK
    post('/flush_cache').read()
    body = dict(model=A.model, messages=[{'role': 'user', 'content': text}], max_tokens=A.output, stream=True,
                stream_options={'include_usage': True})
    t0 = time.time()
    first = last = None
    usage = {}
    with post('/v1/chat/completions', body, timeout=3600) as r:
        for raw in r:
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content'):
                    now = time.time()
                    first = first or now
                    last = now
    prompt, out = usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0)
    ttft = (first or time.time()) - t0
    decode = (out - 1) / (last - first) if first and last and last > first else 0.0
    row = dict(context=n, prompt_tokens=prompt, output_tokens=out, ttft_s=round(ttft, 2),
               prefill_tok_s=round(prompt / ttft, 1) if ttft > 0 else 0.0, decode_tok_s=round(decode, 1))
    rows.append(row)
    print(json.dumps(row), flush=True)
if A.out:
    with open(A.out, 'w') as f:
        json.dump(rows, f, indent=1)
