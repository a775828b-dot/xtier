import json
import time
import urllib.request
import uuid
import threading
import sys

URL = 'http://127.0.0.1:8001/v1/chat/completions'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 300

sentence = 'The quick brown fox jumps over the lazy dog. '
para = (sentence * 20 + '\n') * CHARS
results = [None] * N
peak = [0]
maxq = [0]
stop = [False]


def run(i):
    text = f'Marker {i}-{uuid.uuid4().hex}. ' + para + '\nAnswer with one short sentence.'
    body = json.dumps({'model': 'DeepSeek-V4.1-Flash',
                       'messages': [{'role': 'user', 'content': text}],
                       'max_tokens': 32, 'temperature': 0.0}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            out = json.load(r)
        u = out['usage']
        results[i] = f'prompt={u["prompt_tokens"]} out={u["completion_tokens"]} wall={time.time()-t0:.1f}s'
    except Exception as e:
        results[i] = f'FAIL {str(e)[:120]}'


def sampler():
    while not stop[0]:
        try:
            m = urllib.request.urlopen('http://127.0.0.1:8001/metrics', timeout=3).read().decode()
            for l in m.splitlines():
                if l.startswith('sglang:num_running_reqs') and 'tp_rank="0"' in l:
                    peak[0] = max(peak[0], int(float(l.split()[-1])))
                if l.startswith('sglang:num_queue_reqs') and 'tp_rank="0"' in l:
                    maxq[0] = max(maxq[0], int(float(l.split()[-1])))
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=sampler, daemon=True).start()
t0 = time.time()
ths = [threading.Thread(target=run, args=(i,)) for i in range(N)]
for th in ths:
    th.start()
for th in ths:
    th.join()
stop[0] = True
ok = sum(1 for r in results if r and not r.startswith('FAIL'))
print(f'{N}x{len(para)//4//1000}K wall={time.time()-t0:.1f}s peak_running={peak[0]} max_queue={maxq[0]} ok={ok}/{N}')
for i, r in enumerate(results):
    print(f'  req{i}: {r}')
