#!/usr/bin/env python3
"""并发长请求压测：N 个各约 LEN token 的不同 prompt 同时发出，采样并发/TTFT/吞吐。

    python3 bench_parallel_long.py [N] [LEN] [OUT]

用随机 marker 让每个 prompt 不同（避免全部命中同一前缀），同时打印服务端能否并行。
"""
import json, sys, time, urllib.request, uuid, threading

URL = 'http://127.0.0.1:8001/v1/chat/completions'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 64

sentence = 'The quick brown fox jumps over the lazy dog. '

def make_prompt(seed):
    para = (sentence * 20 + '\n') * (LEN // (len(sentence) * 20) + 1)
    return f'Marker {seed}-{uuid.uuid4().hex}. ' + para + '\nAnswer with one short sentence.'

results = [None] * N
def run(i):
    text = make_prompt(i)
    body = {'model': 'DeepSeek-V4.1-Flash', 'messages': [{'role': 'user', 'content': text}],
            'max_tokens': OUT, 'temperature': 0.0}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            out = json.load(resp)
        dt = time.time() - t0
        u = out['usage']
        results[i] = dict(ok=True, prompt=u['prompt_tokens'], out=u['completion_tokens'], wall=dt,
                          tps=u['completion_tokens']/dt)
    except Exception as e:
        results[i] = dict(ok=False, err=str(e)[:120], wall=time.time()-t0)

# 采样服务端并发
peak = [0]
stop = [False]
def sampler():
    import urllib.request as U
    while not stop[0]:
        try:
            m = U.urlopen('http://127.0.0.1:8001/metrics', timeout=3).read().decode()
            for line in m.splitlines():
                if line.startswith('sglang:num_running_reqs') and 'tp_rank="0"' in line:
                    peak[0] = max(peak[0], int(float(line.split()[-1])))
        except Exception:
            pass
        time.sleep(0.5)
t = threading.Thread(target=sampler, daemon=True); t.start()

t0 = time.time()
ths = [threading.Thread(target=run, args=(i,)) for i in range(N)]
for th in ths: th.start()
for th in ths: th.join()
stop[0] = True
wall = time.time() - t0
print(f'N={N} LEN≈{LEN} wall={wall:.1f}s peak_running={peak[0]}')
for i, r in enumerate(results):
    if r['ok']:
        print(f'  req{i}: prompt={r["prompt"]} out={r["out"]} wall={r["wall"]:.1f}s tps={r["tps"]:.1f}')
    else:
        print(f'  req{i}: FAIL {r["err"]} wall={r["wall"]:.1f}s')
