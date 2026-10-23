"""Long-context speed of a running SGLang server (the README table): per context length, a cold prefill of a prompt
of about N tokens (agent-session text from a fixed corpus plus a review task; prefix cache flushed first) and an
answer of up to --output tokens, streamed through /v1/chat/completions with the server's default sampling (as
bench/agent_bench.py). Prefill tok/s = prompt tokens / time to the first streamed token; decode tok/s = (output
tokens - 1) / time from the first to the last streamed token (reasoning and answer tokens alike). A first request
of --warmup tokens (16 output tokens, not measured) takes the one-time costs of a fresh server (kernel compilation
and autotuning of the prefill shapes), which otherwise land in the first measurement. The prompt is then sent
--repeats more times with its prefix cached (the same context, a new sampled answer each time): decode is the
median over all answers, since with speculative decoding one answer's speed moves by 10-30% with how well its text
is drafted. Each answer also reports SGLang's speculative statistics (return_spec_tokens_details): verify steps and
accept length, and the time per verify step, which does not depend on the text and compares configurations.

  python bench/context_speed.py --port 18190 --model DeepSeek-V4.1-Flash-1017 --tokenizer MODEL_DIR \\
      [--contexts 65536,520000] [--output 1024] [--warmup 32768] [--repeats 4] [--out FILE]
"""
import argparse
import json
import statistics
import time
import urllib.request

from transformers import AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--model', required=True)
P.add_argument('--tokenizer', required=True)
P.add_argument('--contexts', default='65536,520000', help='prompt tokens per measurement (before the chat template)')
P.add_argument('--output', type=int, default=1024)
P.add_argument('--warmup', type=int, default=32768, help='prompt tokens of the unmeasured first request (0: none)')
P.add_argument('--repeats', type=int, default=4, help='answers per context after the cold one, prefix cached')
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


def chat(n, max_tokens, start=0, flush=True):
    """A request of the corpus tokens [start, start + n) plus the task, the prefix cache flushed first unless not
    `flush`: (send time, stream)."""
    if start + n > len(ids):
        raise SystemExit(f'corpus has {len(ids)} tokens, {start + n} asked')
    body = dict(model=A.model, messages=[{'role': 'user', 'content': tok.decode(ids[start:start + n]) + TASK}],
                max_tokens=max_tokens, stream=True, stream_options={'include_usage': True},
                return_spec_tokens_details=True)
    if flush:
        post('/flush_cache').read()
    return time.time(), post('/v1/chat/completions', body, timeout=3600)


def answer(n, flush=True):
    """One streamed answer to the n-token prompt: dict of prompt / output tokens, time to the first token, decode
    tok/s, and with speculative decoding its verify steps, accept length and ms per verify step."""
    first = last = None
    usage, spec = {}, {}
    t0, r = chat(n, A.output, flush=flush)
    with r:
        for raw in r:
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            chunk = json.loads(line[5:])
            if chunk.get('usage'):
                usage = chunk['usage']
            if (chunk.get('sglext') or {}).get('spec_tokens_details'):
                spec = chunk['sglext']['spec_tokens_details']
            for choice in chunk.get('choices') or []:
                delta = choice.get('delta') or {}
                if delta.get('content') or delta.get('reasoning_content'):
                    now = time.time()
                    first = first or now
                    last = now
    out = usage.get('completion_tokens', 0)
    span = last - first if first and last and last > first else 0.0
    steps = int(spec.get('spec_verify_ct') or 0)
    return dict(prompt=usage.get('prompt_tokens', 0), out=out, ttft=(first or time.time()) - t0,
                decode=(out - 1) / span if span else 0.0, steps=steps, accept=float(spec.get('spec_accept_length') or 0),
                step_ms=1e3 * span / (steps - 1) if span and steps > 1 else 0.0)


if A.warmup:
    t0, r = chat(A.warmup, 16, start=len(ids) - A.warmup)   # the corpus end: no prefix shared with the prompts
    with r:
        r.read()
    print(f'warm-up: {A.warmup} tokens in {time.time() - t0:.1f} s', flush=True)
rows = []
for n in [int(x) for x in A.contexts.split(',')]:
    runs = [answer(n)] + [answer(n, flush=False) for _ in range(A.repeats)]
    cold = runs[0]
    row = dict(context=n, prompt_tokens=cold['prompt'], output_tokens=cold['out'], ttft_s=round(cold['ttft'], 2),
               prefill_tok_s=round(cold['prompt'] / cold['ttft'], 1) if cold['ttft'] > 0 else 0.0,
               decode_tok_s=round(statistics.median(r['decode'] for r in runs), 1),
               decode_runs=[round(r['decode'], 1) for r in runs], output_runs=[r['out'] for r in runs],
               repeat_ttft_s=[round(r['ttft'], 2) for r in runs[1:]])
    if all(r['steps'] > 1 for r in runs):
        row.update(step_ms=round(statistics.median(r['step_ms'] for r in runs), 2),
                   accept_len=round(statistics.mean(r['accept'] for r in runs), 2),
                   step_ms_runs=[round(r['step_ms'], 2) for r in runs], accept_runs=[round(r['accept'], 2) for r in runs])
    rows.append(row)
    print(json.dumps(row), flush=True)
if A.out:
    with open(A.out, 'w') as f:
        json.dump(rows, f, indent=1)
