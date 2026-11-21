#!/usr/bin/env python3
"""按真实 agent 场景压测：每个会话是长会话，多轮追加，前缀高度重合。

与 bench_latency.py 的区别：
  - 每个会话先建立一次长上下文（冷 prefill），然后做 TURNS 轮追加
  - 第 2 轮起每轮只 prefill 新追加的部分（~几百 token），模拟工具结果回填
  - 报告每轮的 TTFT / ITL，这才是 agent 用户真正体感到的

用法：
  python3 bench_agent_real.py --sessions 4 --ctx 60000 --turns 6 --out 64
"""
import argparse, json, statistics, threading, time, urllib.request, uuid, sys

URL = 'http://127.0.0.1:8001/v1/chat/completions'
METRICS = 'http://127.0.0.1:8001/metrics'
SENT = 'The quick brown fox jumps over the lazy dog. '


def block(chars):
    return (SENT * 20 + '\n') * max(1, chars // (len(SENT) * 20))


def stream(messages, out_tokens):
    body = json.dumps({'model': 'DeepSeek-V4.1-Flash', 'messages': messages,
                       'max_tokens': out_tokens, 'temperature': 0.0, 'stream': True}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    last = None
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: '):
                continue
            payload = line[6:]
            if payload == '[DONE]':
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = (d.get('choices') or [{}])[0]
            txt = (ch.get('delta') or {}).get('content')
            if txt:
                now = time.time()
                if first is None:
                    first = now
                last = now
                n += 1
    t1 = time.time()
    return dict(ttft=(first - t0) if first else None,
                total=t1 - t0, tokens=n,
                itl=((last - first) / max(1, n - 1)) if (first and n > 1) else None)


class Sampler:
    def __init__(self):
        self.peak_run = 0
        self.peak_q = 0
        self.stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            try:
                m = urllib.request.urlopen(METRICS, timeout=3).read().decode()
                for l in m.splitlines():
                    if l.startswith('sglang:num_running_reqs') and 'tp_rank="0"' in l:
                        self.peak_run = max(self.peak_run, int(float(l.split()[-1])))
                    if l.startswith('sglang:num_queue_reqs') and 'tp_rank="0"' in l:
                        self.peak_q = max(self.peak_q, int(float(l.split()[-1])))
            except Exception:
                pass
            time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, default=4)
    ap.add_argument('--ctx', type=int, default=60000, help='初始上下文字符量相关，约 token 数')
    ap.add_argument('--turns', type=int, default=6)
    ap.add_argument('--out', type=int, default=64)
    a = ap.parse_args()

    res = [dict(build=None, turns=[]) for _ in range(a.sessions)]
    errors = []
    barrier = threading.Barrier(a.sessions)

    def one(i):
        base = f'Session {i}-{uuid.uuid4().hex[:8]}.\n' + block(a.ctx)
        msgs = [{'role': 'user', 'content': base}]
        try:
            barrier.wait()
            # 冷建上下文
            r = stream(msgs, a.out)
            res[i]['build'] = r
            msgs.append({'role': 'assistant', 'content': 'ok ' * 5})
            # 追加轮
            for t in range(a.turns):
                add = f'Tool result {t}-{uuid.uuid4().hex[:6]}:\n' + block(1500)
                msgs.append({'role': 'user', 'content': add})
                r2 = stream(msgs, a.out)
                res[i]['turns'].append(r2)
                msgs.append({'role': 'assistant', 'content': 'ack ' * 5})
        except Exception as e:
            errors.append(f'sess{i}: {type(e).__name__}: {str(e)[:160]}')

    s = Sampler()
    t0 = time.time()
    ths = [threading.Thread(target=one, args=(i,)) for i in range(a.sessions)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    s.stop = True
    time.sleep(0.4)

    print(f'\n=== {a.sessions} sessions x ctx~{a.ctx} x {a.turns} turns, out={a.out} ===')
    print(f'wall={wall:.1f}s  peak_running={s.peak_run}  peak_queue={s.peak_q}')
    if errors:
        print('ERRORS:', errors)

    builds = [r['build'] for r in res if r['build']]
    if builds:
        bt = [b['ttft'] for b in builds if b['ttft']]
        print(f'\n[冷建上下文] TTFT: ' + ', '.join(f'{x:.1f}s' for x in sorted(bt)))
        print(f'             worst={max(bt):.1f}s  median={statistics.median(bt):.1f}s')

    all_ttft, all_itl = [], []
    print('\n[追加轮 TTFT / ITL]  行=会话, 列=轮次')
    for i, r in enumerate(res):
        if not r['turns']:
            continue
        tt = [t['ttft'] for t in r['turns']]
        it = [t['itl'] for t in r['turns'] if t['itl']]
        all_ttft += [x for x in tt if x]
        all_itl += it
        print(f'  s{i}: ' + '  '.join(f'{x:.2f}s' for x in tt))
        print(f'      ITL: ' + '  '.join(f'{x*1000:.0f}ms' for x in it))

    if all_ttft:
        all_ttft.sort()
        print(f'\n追加轮 TTFT  p50={statistics.median(all_ttft):.2f}s  '
              f'p95={all_ttft[int(len(all_ttft)*0.95)-1]:.2f}s  worst={max(all_ttft):.2f}s')
    if all_itl:
        all_itl.sort()
        print(f'追加轮 ITL   p50={statistics.median(all_itl)*1000:.0f}ms  '
              f'p95={all_itl[int(len(all_itl)*0.95)-1]*1000:.0f}ms  worst={max(all_itl)*1000:.0f}ms')


if __name__ == '__main__':
    main()
