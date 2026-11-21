#!/usr/bin/env python3
"""复现用户的 Chatbox 场景：单请求 5000 字长文生成，逐 token 计时。

用法：
  python3 bench_longform.py --chars 5000 --concurrency 1
  python3 bench_longform.py --chars 5000 --concurrency 6   # 叠加并发
"""
import argparse, json, threading, time, urllib.request, uuid

URL = 'http://127.0.0.1:8001/v1/chat/completions'

PROMPT = ("请写一篇不少于 {n} 字的中文长文，主题是「多智能体系统中的推理加速技术」。"
          "要求：内容详实、连续输出、不要分点太碎、不要提前结束、必须写满 {n} 字。")


def stream_once(prompt, max_tokens):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens, 'temperature': 0.7, 'stream': True,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time(); first = None; stamps = []; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: '):
                continue
            p = line[6:]
            if p == '[DONE]':
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            c = (d.get('choices') or [{}])[0]
            t = (c.get('delta') or {}).get('content')
            if t:
                now = time.time()
                if first is None:
                    first = now
                stamps.append(now); text.append(t)
    t1 = time.time()
    gaps = sorted((stamps[i] - stamps[i - 1]) * 1000 for i in range(1, len(stamps)))
    n = len(stamps)
    txt = ''.join(text)
    return dict(
        ttft=first - t0 if first else None, total=t1 - t0, tokens=n, chars=len(txt),
        tps=n / (t1 - first) if first and t1 > first else 0,
        gap_mean=sum(gaps) / len(gaps) if gaps else None,
        gap_p50=gaps[len(gaps) // 2] if gaps else None,
        gap_p95=gaps[int(len(gaps) * .95)] if gaps else None,
        gap_max=gaps[-1] if gaps else None,
        s200=sum(1 for g in gaps if g > 200), s500=sum(1 for g in gaps if g > 500),
        s1000=sum(1 for g in gaps if g > 1000),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chars', type=int, default=5000)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=6000)
    a = ap.parse_args()
    prompt = PROMPT.format(n=a.chars)

    res = [None] * a.concurrency
    barrier = threading.Barrier(a.concurrency)

    def one(i):
        barrier.wait()
        try:
            res[i] = stream_once(prompt + f'\n（第 {i+1} 篇，编号 {uuid.uuid4().hex[:6]}）', a.max_tokens)
        except Exception as e:
            res[i] = dict(error=f'{type(e).__name__}: {str(e)[:160]}')

    t0 = time.time()
    ths = [threading.Thread(target=one, args=(i,)) for i in range(a.concurrency)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.time() - t0

    print(f'\n=== 长文 {a.chars} 字 x {a.concurrency} 并发 (wall={wall:.1f}s) ===')
    print(f'{"路":>4} {"TTFT":>8} {"tok":>6} {"字":>6} {"tok/s":>7} {"gap均值":>8} {"p50":>6} {"p95":>6} {"max":>7} {">200":>5} {">500":>5} {">1s":>5}')
    for i, r in enumerate(res):
        if r is None: continue
        if 'error' in r:
            print(f'  s{i}: ERROR {r["error"]}'); continue
        print(f'  s{i}: {r["ttft"]:>7.2f}s {r["tokens"]:>6} {r["chars"]:>6} {r["tps"]:>7.1f} '
              f'{r["gap_mean"]:>7.0f}ms {r["gap_p50"]:>5.0f}ms {r["gap_p95"]:>5.0f}ms {r["gap_max"]:>6.0f}ms '
              f'{r["s200"]:>5} {r["s500"]:>5} {r["s1000"]:>5}')
    tot = sum(r['tokens'] for r in res if r and 'error' not in r)
    print(f'\n聚合: {tot} tok / {wall:.1f}s = {tot/max(1e-9,wall):.1f} tok/s')


if __name__ == '__main__':
    main()
