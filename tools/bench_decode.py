#!/usr/bin/env python3
"""Decode 密集基准：专门用于 A/B 对比投机解码的 accept len 效果。

与 bench_agent_real.py 的区别：
  - 输出 token 数多（默认 1024），让 decode 成为主导，prefill 占比极小
  - 逐 token 记录到达间隔，能算出 ITL 的 p50/p95/max 和 >200ms 的占比
  - 报告每路 tok/s 和聚合 tok/s

用法：
  python3 bench_decode.py --sessions 4 --ctx 8000 --out 1024 --turns 2
"""
import argparse, json, random, statistics, threading, time, urllib.request, uuid

URL = 'http://127.0.0.1:8001/v1/chat/completions'
METRICS = 'http://127.0.0.1:8001/metrics'
SENT = 'The quick brown fox jumps over the lazy dog. '

# 多样化语料：纯重复会让模型很快 EOS，decode 就不成主导了
_SUBJ = ['the compiler', 'the kernel scheduler', 'the cache hierarchy', 'the interconnect fabric',
         'the memory controller', 'the tensor engine', 'the routing network', 'the speculative decoder',
         'the quantizer', 'the expert dispatcher', 'the attention kernel', 'the token sampler']
_VERB = ['analyzes', 'reconstructs', 'predicts', 'validates', 'schedules', 'dispatches',
         'reconciles', 'optimizes', 'calibrates', 'reorders', 'consolidates', 'overlaps']
_OBJ = ['the residual stream', 'the kv page table', 'the draft tree', 'the expert weights',
        'the activation tiles', 'the all-reduce ring', 'the sampling logits', 'the graph buckets',
        'the nvlink channels', 'the numa pages', 'the block tables', 'the rope caches']


def _corpus(n_lines):
    rnd = random.Random(20261120)
    out = []
    for i in range(n_lines):
        s = rnd.choice(_SUBJ); v = rnd.choice(_VERB); o = rnd.choice(_OBJ)
        out.append(f'Observation {i}: {s} {v} {o} while the {rnd.choice(_SUBJ)} tracks '
                   f'{rnd.choice(_OBJ)} under sustained {rnd.choice(["latency","throughput","concurrency","pressure"])}.')
    return '\n'.join(out)


def block(chars):
    """约 chars 字符的多样化上下文（所有会话共用同一份，保证前缀缓存命中）"""
    per = 130
    lines = max(20, int(chars / per))
    return _corpus(lines)


def stream(messages, out_tokens):
    body = json.dumps({'model': 'DeepSeek-V4.1-Flash', 'messages': messages,
                       'max_tokens': out_tokens, 'temperature': 0.0, 'stream': True}).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    stamps = []
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
                stamps.append(now)
    t1 = time.time()
    gaps = [(stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps))]
    gaps.sort()
    n = len(stamps)
    return dict(
        ttft=(first - t0) if first else None,
        total=t1 - t0, tokens=n,
        tps=(n / (t1 - first)) if first and t1 > first else 0.0,
        itl_mean=(sum(gaps) / len(gaps)) if gaps else None,
        itl_p50=(gaps[len(gaps) // 2]) if gaps else None,
        itl_p95=(gaps[int(len(gaps) * 0.95)]) if gaps else None,
        itl_max=(gaps[-1]) if gaps else None,
        stalls=sum(1 for g in gaps if g > 200),
        gaps=gaps,
    )


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
    ap.add_argument('--ctx', type=int, default=8000)
    ap.add_argument('--turns', type=int, default=2)
    ap.add_argument('--out', type=int, default=1024)
    a = ap.parse_args()

    res = [dict(build=None, turns=[]) for _ in range(a.sessions)]
    errors = []
    barrier = threading.Barrier(a.sessions)

    def one(i):
        base = (f'Session {i}-{uuid.uuid4().hex[:8]}.\n'
                + block(a.ctx)
                + '\n\n请基于上面的观察列表，输出一份编号的技术说明，'
                  '至少 900 条，每条一行，务必写满，不要提前结束。')
        msgs = [{'role': 'user', 'content': base}]
        try:
            barrier.wait()
            res[i]['build'] = stream(msgs, a.out)
            msgs.append({'role': 'assistant', 'content': '好的，下面开始编号说明。'})
            for t in range(a.turns):
                add = f'补充第 {t} 组的新观察，请接着往下写，仍然写满 900 条：\n' + block(1500)
                msgs.append({'role': 'user', 'content': add})
                res[i]['turns'].append(stream(msgs, a.out))
                msgs.append({'role': 'assistant', 'content': '继续。'})
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

    print(f'\n=== DECODE bench: {a.sessions} sessions x ctx~{a.ctx} x {a.turns} turns, out={a.out} ===')
    print(f'wall={wall:.1f}s  peak_running={s.peak_run}  peak_queue={s.peak_q}')
    if errors:
        print('ERRORS:', errors)

    all_gaps = []
    total_tok = 0
    decode_wall = 0.0
    print('\n[每路] 会话 | TTFT | 输出tok | tok/s | ITL p50 | p95 | max | >200ms')
    for i, r in enumerate(res):
        for label, rr in [('build', r['build'])] + [(f't{t}', x) for t, x in enumerate(r['turns'])]:
            if not rr or not rr['tokens']:
                continue
            all_gaps += rr['gaps']
            total_tok += rr['tokens']
            decode_wall = max(decode_wall, rr['total'])
            print(f'  s{i}.{label:5s} {rr["ttft"] if rr["ttft"] is None else round(rr["ttft"],2):>7} '
                  f'{rr["tokens"]:>8} {rr["tps"]:>7.1f} '
                  f'{rr["itl_p50"] if rr["itl_p50"] is None else round(rr["itl_p50"]):>8} '
                  f'{rr["itl_p95"] if rr["itl_p95"] is None else round(rr["itl_p95"]):>5} '
                  f'{rr["itl_max"] if rr["itl_max"] is None else round(rr["itl_max"]):>5} '
                  f'{rr["stalls"]:>6}')

    if all_gaps:
        all_gaps.sort()
        n = len(all_gaps)
        big = sum(1 for g in all_gaps if g > 200)
        vbig = sum(1 for g in all_gaps if g > 500)
        print(f'\n全样本 ITL  n={n}  p50={all_gaps[n//2]:.0f}ms  p95={all_gaps[int(n*0.95)]:.0f}ms  '
              f'max={all_gaps[-1]:.0f}ms  mean={sum(all_gaps)/n:.0f}ms')
        print(f'  >200ms: {big} ({big*100/n:.1f}%)   >500ms: {vbig} ({vbig*100/n:.1f}%)')
    print(f'\n聚合: 总输出 {total_tok} tok, 最长单路 wall={decode_wall:.1f}s, '
          f'聚合吞吐={total_tok/max(1e-9,decode_wall):.1f} tok/s')


if __name__ == '__main__':
    main()
