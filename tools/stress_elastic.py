#!/usr/bin/env python3
"""压力复现 elastic shrink/grow：反复发长 prompt（制造 KV 抢占与 vmm_unmap），观察是否崩溃。

    python3 stress_elastic.py [SECONDS] [WORKERS] [PROMPT_LEN]

每轮请求用随机 marker（不命中前缀缓存），迫使 re-prefill；多个并发让 elastic 反复 release/grow。
"""
import json, sys, time, urllib.request, uuid, threading

URL = 'http://127.0.0.1:8001/v1/chat/completions'
SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
PLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
sentence = 'The quick brown fox jumps over the lazy dog. '
para = (sentence * 20 + '\n') * (PLEN // (len(sentence) * 20) + 1)

stop = [False]
ok = [0]; fail = [0]
errors = []
lock = threading.Lock()

def worker(wid):
    while not stop[0]:
        text = f'Marker {wid}-{uuid.uuid4().hex}. ' + para + '\nAnswer with one short sentence.'
        body = {'model': 'DeepSeek-V4.1-Flash', 'messages': [{'role': 'user', 'content': text}],
                'max_tokens': 16, 'temperature': 0.0}
        req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                json.load(r)
            with lock: ok[0] += 1
        except Exception as e:
            with lock:
                fail[0] += 1
                if len(errors) < 10: errors.append(str(e)[:150])

ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
for t in ths: t.start()
t0 = time.time()
while time.time() - t0 < SECONDS:
    time.sleep(10)
    alive = sum(1 for t in ths if t.is_alive())
    print(f'[{int(time.time()-t0):3d}s] ok={ok[0]} fail={fail[0]} workers_alive={alive}', flush=True)
stop[0] = True
for t in ths: t.join(timeout=30)
print(f'DONE ok={ok[0]} fail={fail[0]}')
for e in errors: print('  ERR:', e)
