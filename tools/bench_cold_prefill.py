#!/usr/bin/env python3
"""Cold-prefill probe: a fresh 100K-token prompt (random marker) so no radix/HiCache prefix can hit.

    python3 bench_cold_prefill.py [url] [words]

Prints prompt tokens, wall seconds and prompt-only tok/s. Each run uses a different first sentence, so the whole
token sequence differs and the server has to read it from scratch.
"""
import json
import sys
import time
import urllib.request
import uuid

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8001/v1/chat/completions'
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 500

sentence = 'The quick brown fox jumps over the lazy dog. '
para = (sentence * 20 + '\n') * REPEATS
text = f'Marker {uuid.uuid4().hex}. ' + para + '\nAnswer with one short sentence.'
body = {'model': 'DeepSeek-V4.1-Flash', 'messages': [{'role': 'user', 'content': text}],
        'max_tokens': 16, 'temperature': 0.0}
req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as resp:
    out = json.load(resp)
dt = time.time() - t0
p = out['usage']['prompt_tokens']
print(f'cold prompt_tokens={p} total={dt:.1f}s out={out["usage"]["completion_tokens"]}')
print(f'prompt-only approx = {p / dt:.0f} tok/s')
