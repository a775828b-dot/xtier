#!/usr/bin/env python3
"""多请求延迟分布基准：卡顿（stutter）不是吞吐问题，是 token 间隔的尾部问题。

每个会话：
  1. 先用一个长提示把上下文撑到 --ctx token（模拟 agent 的长会话）
  2. 然后流式生成 --out 个 token，记录每个 token 到达的时间

输出每个会话的 TTFT、token 间间隔（ITL）的 p50/p95/p99 和最大停顿，以及全局的停顿分布。

用法：
  python tools/bench_latency.py --sessions 6 --ctx 100000 --out 64
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
import uuid

URL = 'http://127.0.0.1:8001/v1/chat/completions'
SENT = 'The quick brown fox jumps over the lazy dog. '


def long_text(tokens):
    """造一个约 tokens 个 token 的提示（每句约 10 token）。"""
    need = tokens // 10
    return (SENT * 10 + '\n') * max(1, need // 10)


def stream_once(prompt, out_tokens, results, idx):
    body = json.dumps({
        'model': 'DeepSeek-V4.1-Flash',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': out_tokens, 'temperature': 0.0, 'stream': True,
    }).encode()
    req = urllib.request.Request(URL, body, {'Content-Type': 'application/json'})
    t0 = time.time()
    first = None
    times = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                delta = obj.get('choices') or []
                if delta and delta[0].get('delta', {}).get('content'):
                    now = time.time()
                    if first is None:
                        first = now
                    times.append(now)
    except Exception as e:
        results[idx] = {'error': str(e)[:120]}
        return
    itl = [times[i] - times[i - 1] for i in range(1, len(times))]
    results[idx] = {
        'ttft': (first - t0) if first else None,
        'tokens': len(times),
        'itl_p50': statistics.median(itl) if itl else None,
        'itl_p95': sorted(itl)[int(len(itl) * 0.95)] if itl else None,
        'itl_max': max(itl) if itl else None,
        'total': times[-1] - t0 if times else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', type=int, default=6)
    ap.add_argument('--ctx', type=int, default=100000, help='每个会话的提示 token 数')
    ap.add_argument('--out', type=int, default=64, help='生成的 token 数')
    args = ap.parse_args()

    base = long_text(args.ctx)
    results = [None] * args.sessions
    print(f'提示约 {len(base) // 4} 字符，会话 {args.sessions} 路，同时开始流式生成 {args.out} token', flush=True)
    t0 = time.time()
    ths = []
    for i in range(args.sessions):
        p = f'Marker {i}-{uuid.uuid4().hex}. ' + base + '\n用一句话回答。'
        ths.append(threading.Thread(target=stream_once, args=(p, args.out, results, i)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0

    print(f'\n墙钟 {wall:.1f}s')
    print(f'{"会话":>4} {"TTFT":>8} {"tok":>5} {"ITL p50":>9} {"ITL p95":>9} {"ITL max":>9} {"总时":>8}')
    maxes = []
    p95s = []
    ttfts = []
    for i, r in enumerate(results):
        if not r or 'error' in r:
            print(f'{i:>4}  FAIL {r}')
            continue
        print(f'{i:>4} {r["ttft"]:>7.1f}s {r["tokens"]:>5} {r["itl_p50"]*1000:>8.0f}ms '
              f'{r["itl_p95"]*1000:>8.0f}ms {r["itl_max"]*1000:>8.0f}ms {r["total"]:>7.1f}s')
        maxes.append(r['itl_max'])
        p95s.append(r['itl_p95'])
        ttfts.append(r['ttft'])
    if maxes:
        print(f'\n最坏停顿 {max(maxes)*1000:.0f} ms | ITL p95 最差 {max(p95s)*1000:.0f} ms | '
              f'TTFT 最差 {max(ttfts):.1f}s')


if __name__ == '__main__':
    main()
