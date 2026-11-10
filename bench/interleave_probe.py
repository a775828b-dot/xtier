"""What a long cold prompt does to a request that is decoding beside it, on a running SGLang server (the measure behind
the console's "Prefill 中穿插 decode" option, SGLang's --prefill-decode-interval N):

  python bench/interleave_probe.py --port 18190 --model NAME [--tokens 200000] [--out FILE]

1. alone: a cold prompt of about --tokens tokens, 8 output tokens: its time to the first token.
2. mix: request A streams a long answer to a 16K-token prompt; 12 s into its decode request B sends another cold prompt
   of that size. Reported: B's time to the first token, A's stream chunks per second (one chunk a decode step) in the
   10 s before B, while B is read and after, A's longest wait between two chunks while B is read, and A's tokens per
   chunk (its completion tokens over its chunks), which turns the rates into tokens per second.
The prompts are this repository's own sources behind a unique first line, so nothing hits the prefix cache. Start the
server once per setting (the interval is a server argument) and compare the outputs."""
import argparse
import http.client
import json
import threading
import time
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18190)
P.add_argument('--model', required=True)
P.add_argument('--tokens', type=int, default=200000)
P.add_argument('--chars-per-token', type=float, default=3.75, help='of source code under the model\'s tokenizer')
P.add_argument('--out')
A = P.parse_args()
REPO = Path(__file__).resolve().parents[1]
SRC = ''.join(p.read_text(encoding='utf-8') for p in sorted((REPO / 'xtier').rglob('*.py')))


def text_of(tokens, tag):
    chars = int(tokens * A.chars_per_token)
    start = (len(tag) * 7919 + int(time.time())) % len(SRC)
    return f'[{tag} {time.time():.3f}]\n' + (SRC[start:] + SRC * (chars // len(SRC) + 1))[:chars]


def stream(content, max_tokens, stamps=None):
    """(seconds to the first streamed token, usage) of a chat completion."""
    body = {'model': A.model, 'stream': True, 'stream_options': {'include_usage': True}, 'max_tokens': max_tokens,
            'temperature': 0, 'messages': [{'role': 'user', 'content': content}]}
    conn = http.client.HTTPConnection('127.0.0.1', A.port, timeout=3600)
    t0 = time.time()
    conn.request('POST', '/v1/chat/completions', body=json.dumps(body, ensure_ascii=False).encode(),
                 headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    if resp.status != 200:
        raise SystemExit(f'HTTP {resp.status}: {resp.read()[:300]!r}')
    first, usage = None, {}
    for raw in resp:
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data:') or line[5:].strip() == '[DONE]':
            continue
        msg = json.loads(line[5:].strip())
        usage = msg.get('usage') or usage
        for ch in msg.get('choices') or []:
            d = ch.get('delta') or {}
            if d.get('content') or d.get('reasoning_content') or d.get('tool_calls'):
                now = time.time()
                first = first if first is not None else now - t0
                if stamps is not None:
                    stamps.append(now)
    conn.close()
    return first, usage


def rate(stamps, a, b):
    return round(sum(1 for s in stamps if a <= s < b) / max(b - a, 1e-9), 2)


QUESTION = '\n\nSay in one sentence what this code is for.'
first, usage = stream(text_of(A.tokens, 'alone') + QUESTION, 8)
result = {'alone': {'ttft_s': round(first, 3), 'prompt_tokens': usage.get('prompt_tokens')}}
print('alone', json.dumps(result['alone']), flush=True)

stamps, a_usage = [], {}
ta = threading.Thread(target=lambda: a_usage.update(stream(
    'Explain this code function by function, every function, in as much detail as you can:\n\n' + text_of(16000, 'mix-a'),
    14000, stamps)[1]))
ta.start()
while len(stamps) < 20 and ta.is_alive():
    time.sleep(0.2)
if len(stamps) < 20:
    raise SystemExit('request A did not decode')
time.sleep(12)
t0 = time.time()
first, usage = stream(text_of(A.tokens, 'mix-b') + QUESTION, 8)
t1 = t0 + first
a_done = not ta.is_alive()
time.sleep(10)
t2 = time.time()
ta.join()
edges = [t0] + [s for s in stamps if t0 <= s < t1] + [t1]
result['mix'] = {
    'b_ttft_s': round(first, 3), 'b_prompt_tokens': usage.get('prompt_tokens'),
    'a_chunks_per_s_before': rate(stamps, t0 - 10, t0), 'a_chunks_per_s_during': rate(stamps, t0, t1),
    'a_chunks_per_s_after': rate(stamps, t1 + 2, t2), 'a_longest_wait_during_s': round(max(b - a for a, b in zip(edges, edges[1:])), 3),
    'a_tokens_per_chunk': round((a_usage.get('completion_tokens') or 0) / max(len(stamps), 1), 2),
    'a_ended_before_b_first_token': a_done}
print('mix', json.dumps(result['mix']), flush=True)
if A.out:
    Path(A.out).write_text(json.dumps(result, indent=1))
