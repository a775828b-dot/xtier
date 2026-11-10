"""What clients do to a server that it must survive, against a running server (2026-11-10 audit; about 5 minutes):

1. aborts: streaming requests whose client goes away in the middle of the prefill (after S seconds) or of the decode
   (after N chunks); after each, how long the GPU stays busy (the server noticed the abort, or worked the request off);
2. requests that must be refused (too long, bad parameters, broken JSON) or answered without harm;
3. three requests at once;
after every group a small request must still be answered. One JSON line per case, a summary at the end; exit status 1
when a case failed. The prompts are pieces of this repository's text behind an id of their own (no prefix-cache hit);
--corpus FILE takes another text.

  python bench/abuse_test.py --port 39504 --model DeepSeek-V4.1-Flash-1017 --out abuse.json [--quick]
"""
import argparse
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

P = argparse.ArgumentParser(description='A running server against aborted, malformed and concurrent requests.')
P.add_argument('--port', type=int, required=True)
P.add_argument('--model', required=True, help='the served model name')
P.add_argument('--out', required=True, help='JSON file of every case')
P.add_argument('--quick', action='store_true', help='fewer aborts')
P.add_argument('--corpus', help="a text file to cut prompts from (default: this repository's .md and .py files)")
A = P.parse_args()
port, model, out, quick = A.port, A.model, A.out, A.quick
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
ROOT = Path(__file__).resolve().parents[1]
CORPUS = (Path(A.corpus).read_text(encoding='utf-8', errors='ignore') if A.corpus else
          '\n\n'.join(p.read_text(encoding='utf-8', errors='ignore')
                      for p in sorted(ROOT.rglob('*.md')) + sorted((ROOT / 'xtier').rglob('*.py'))))
while len(CORPUS) < 3_000_000:   # the longest prompt: 2.8 M characters
    CORPUS += '\n\n' + CORPUS
rng = random.Random(20261110)
rows = []


def log(**kw):
    rows.append(kw)
    print(json.dumps(kw, ensure_ascii=False)[:420], flush=True)


def text(chars):
    """A fresh piece of the corpus behind its own id: no prefix-cache hit."""
    o = rng.randrange(0, len(CORPUS) - chars)
    return f'[{time.time_ns():x}] ' + CORPUS[o:o + chars]


def gpu_util():
    try:
        return int(subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5).stdout.split()[0])
    except Exception:   # noqa: BLE001
        return -1


def idle(timeout=900):
    """Seconds until the GPU has been idle for three samples (None: not within timeout)."""
    t0, calm = time.time(), 0
    while time.time() - t0 < timeout:
        calm = calm + 1 if 0 <= gpu_util() < 5 else 0
        if calm >= 3:
            return round(time.time() - t0 - 1.0, 1)
        time.sleep(0.5)
    return None


def post(body, timeout=900, raw=None):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    conn.request('POST', '/v1/chat/completions', raw if raw is not None else json.dumps(body),
                 {'Content-Type': 'application/json'})
    return conn


def ask(prompt='Reply with the single word: pong', max_tokens=24, **extra):
    """(status, seconds, start of the body) of a small request."""
    t0 = time.time()
    try:
        conn = post(dict(model=model, messages=[{'role': 'user', 'content': prompt}], max_tokens=max_tokens,
                         temperature=0, **extra), timeout=300)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, round(time.time() - t0, 2), data[:160].decode('utf-8', 'replace')
    except Exception as e:   # noqa: BLE001
        return -1, round(time.time() - t0, 2), repr(e)[:160]


def alive(tag):
    status, seconds, body = ask()
    log(case=f'alive after {tag}', status=status, seconds=seconds, ok=status == 200, body=body if status != 200 else '')
    return status == 200


def hang_up(conn, response=None):
    """The client goes away for good: the response's file object holds the socket open, so it is shut down first."""
    import socket
    try:
        conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    if response is not None:
        response.close()
    conn.close()


def abort(tag, chars, seconds=None, chunks=None, max_tokens=1024, prompt=None, limit=None):
    body = dict(model=model, stream=True, max_tokens=max_tokens, temperature=0.7,
                messages=[{'role': 'user', 'content': prompt or text(chars) + '\n\nSummarize the text above in detail.'}])
    t0, got, err, r = time.time(), 0, '', None
    try:
        conn = post(body)
        if seconds is not None:
            time.sleep(seconds)
        else:
            r = conn.getresponse()
            while got < chunks:
                line = r.readline()
                if not line:
                    break
                got += line.startswith(b'data: ')
        hang_up(conn, r)
    except Exception as e:   # noqa: BLE001
        err = repr(e)[:120]
    gone = round(time.time() - t0, 1)
    busy = idle()
    log(case=tag, prompt_chars=chars, client_left_after_s=gone, chunks=got, gpu_busy_after_abort_s=busy, error=err,
        ok=busy is not None and not err and (limit is None or busy <= limit))


def refused(tag, body=None, raw=None, expect=(400, 422), timeout=900):
    t0 = time.time()
    try:
        conn = post(body, timeout=timeout, raw=raw)
        r = conn.getresponse()
        data = r.read()
        conn.close()
        status, head = r.status, data[:200].decode('utf-8', 'replace')
    except Exception as e:   # noqa: BLE001
        status, head = -1, repr(e)[:200]
    log(case=tag, status=status, seconds=round(time.time() - t0, 2), ok=status in expect, body=head)


def user(content):
    return [{'role': 'user', 'content': content}]


t_start = time.time()
alive('start')
# 1. aborts in the prefill: a 120K-character prompt (about 30K tokens, two prefill chunks on DeepSeek) left after 1-6 s
for i in range(3 if quick else 8):
    abort(f'abort in prefill #{i}', 120_000, seconds=rng.choice([0.3, 1.0, 2.0, 4.0, 6.0]))
alive('aborts in prefill')
# a long one: about 150K tokens (seven chunks), left after 3 s: the server must not work it off
abort('abort in a long prefill', 600_000, seconds=3.0)
alive('abort in a long prefill')
# 2. aborts in the decode
for i in range(3 if quick else 8):
    abort(f'abort in decode #{i}', 8_000, chunks=rng.choice([1, 3, 10, 40]))
alive('aborts in decode')
# a long answer (up to 6000 tokens, most of a minute) left after 30 chunks: the server must stop decoding, not finish it
ESSAY = ('Write a very long and detailed story of at least 5000 words about a lighthouse keeper, chapter by chapter. '
         'Do not stop early and do not summarize.')
for i in range(2):
    abort(f'abort in a long decode #{i}', 0, chunks=30, max_tokens=6000, prompt=f'[{time.time_ns():x}] ' + ESSAY, limit=15)
alive('aborts in a long decode')
# 3. two aborted at once, while a third is answered
threads = [threading.Thread(target=abort, args=(f'abort, two at once #{i}', 60_000), kwargs=dict(seconds=2.0))
           for i in range(2)]
for t in threads:
    t.start()
alive('two aborts in flight')
for t in threads:
    t.join()
# 4. requests to refuse
refused('prompt longer than the context (about 700K tokens)', dict(model=model, max_tokens=16, messages=user(text(2_800_000))))
alive('too long a prompt')
refused('max_tokens 10^9', dict(model=model, max_tokens=10**9, messages=user('Say hi.')), expect=(200, 400))
refused('temperature -1', dict(model=model, max_tokens=8, temperature=-1, messages=user('Say hi.')))
refused('top_p 5', dict(model=model, max_tokens=8, top_p=5, messages=user('Say hi.')))
refused('no messages', dict(model=model, max_tokens=8, messages=[]))
refused('broken JSON', raw='{"model": "x", "messages": [')
refused('unknown role', dict(model=model, max_tokens=8, messages=[{'role': 'wizard', 'content': 'hi'}]), expect=(200, 400, 422))
refused('tool_choice names no tool', dict(model=model, max_tokens=8, messages=user('hi'),
                                           tool_choice={'type': 'function', 'function': {'name': 'nope'}}), expect=(200, 400, 422))
refused('tool with a broken schema', dict(model=model, max_tokens=64, messages=user('Call the tool.'), tools=[
    {'type': 'function', 'function': {'name': 't', 'parameters': {'type': 'object', 'properties': {'a': {'type': 'nonsense'}}}}}]),
    expect=(200, 400, 422))
refused('json_schema that cannot compile', dict(model=model, max_tokens=64, messages=user('Give JSON.'), response_format={
    'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {'type': 'object', 'properties': {'a': {'$ref': '#/nowhere'}}}}}),
    expect=(200, 400, 422))
refused('logprobs with top_logprobs 20', dict(model=model, max_tokens=8, logprobs=True, top_logprobs=20, messages=user('Say hi.')),
        expect=(200, 400))
refused('n = 3', dict(model=model, max_tokens=8, n=3, messages=user('Say hi.')), expect=(200, 400))
refused('stop and min_tokens above max_tokens', dict(model=model, max_tokens=8, min_tokens=64, stop=['a', 'b'],
                                                      messages=user('Say hi.')), expect=(200, 400))
alive('refused requests')
# 5. three at once (the configurations here run two)
results = []


def one(i):
    results.append((i,) + ask(text(20_000) + '\n\nOne sentence: what is this about?', max_tokens=48))


threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
log(case='three at once', statuses=[r[1] for r in sorted(results)], seconds=[r[2] for r in sorted(results)],
    ok=all(r[1] == 200 for r in results))
alive('three at once')
bad = [r for r in rows if not r.get('ok')]
log(case='SUMMARY', cases=len(rows), failed=len(bad), minutes=round((time.time() - t_start) / 60, 1),
    ok=not bad, failed_cases=[r['case'] for r in bad])
json.dump(rows, open(out, 'w'), ensure_ascii=False, indent=1)
print('ABUSE-TEST', 'PASS' if not bad else f'FAIL ({len(bad)})')
sys.exit(1 if bad else 0)
