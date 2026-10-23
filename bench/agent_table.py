"""agent_bench results as the production README's tables: per session (context start -> end, cold prefill TTFT / tok/s,
requests, output tokens, decode tok/s, DSpark accept, ms per step) and all, for each round given; then both rounds
together, and the long responses (>= 300 output tokens) only, the subset XTier can time on every kind of request.
    python3 agent_table.py LABEL R1.json R2.json [LABEL R1.json R2.json ...]"""
import json
import sys

ORDER = ['64k', '128k-write', '256k', '384k', '500k']


def timed(rows):
    return [r for r in rows if r.get('decode_s') and r.get('decode_tps')]


def agg(rows):
    rows = timed(rows)
    if not rows:
        return None
    out = sum(r['output_tokens'] for r in rows)
    dec = sum(r['decode_s'] for r in rows)
    steps = sum(r['verify_steps'] or 0 for r in rows)
    return {'n': len(rows), 'out': out, 'tps': out / dec, 'accept': out / steps if steps else None,
            'ms': dec * 1000 / steps if steps else None}


def fmt(a):
    if a is None:
        return '— | — | — | —'
    return f"{a['out']:,} | {a['tps']:.1f} | {a['accept']:.2f} | {a['ms']:.1f}"


args = sys.argv[1:]
while args:
    label, files, args = args[0], args[1:3], args[3:]
    rounds = [json.load(open(f)) for f in files]
    print(f'## {label}')
    for i, d in enumerate(rounds, 1):
        rows, s = d['rows'], d['summary']
        cold = {c['session']: c for c in s.get('cold_prefill', [])}
        print(f'\n第 {i} 轮（{len(rows)} 个请求，可计时 {len(timed(rows))} 个）\n')
        print('| 会话 | 上下文（起 → 止） | 冷 prefill：首字 s / tok/s | 请求数 | 可计时 | 输出 token | decode tok/s | 接受长度 | 每步 ms |')
        print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
        for name in ORDER:
            srows = [r for r in rows if r['session'] == name]
            if not srows:
                continue
            ctx = [r['prompt_tokens'] for r in srows]
            c = cold.get(name, {})
            first = srows[0]
            ttft = c.get('ttft_s', first['ttft_s'])
            # uncached prompt tokens / TTFT, as agent_bench and the production README (the session's system prompt hits)
            uncached = c.get('uncached_tokens') or first['prompt_tokens'] - (first.get('cached_tokens') or 0)
            a = agg(srows)
            print(f"| {name} | {min(ctx) // 1000}K → {max(ctx) // 1000}K | {ttft:.1f} / {uncached / ttft:,.0f} | "
                  f"{len(srows)} | {a['n'] if a else 0} | {fmt(a)} |")
        a = agg(rows)
        print(f"| **全部** | | | {len(rows)} | {a['n']} | {fmt(a)} |")
    both = [r for d in rounds for r in d['rows']]
    a, long = agg(both), agg([r for r in both if r['output_tokens'] >= 300])
    print(f"\n两轮合计：可计时 {a['n']} 个请求，decode {a['tps']:.1f} tok/s，接受长度 {a['accept']:.2f}，每步 {a['ms']:.1f} ms")
    print(f"长回复（≥ 300 token）：{long['n']} 个，decode {long['tps']:.1f} tok/s，接受长度 {long['accept']:.2f}，每步 {long['ms']:.1f} ms")
    for i, d in enumerate(rounds, 1):
        lr = agg([r for r in d['rows'] if r['output_tokens'] >= 300])
        print(f"  第 {i} 轮长回复：{lr['n']} 个，{lr['tps']:.1f} tok/s，接受长度 {lr['accept']:.2f}，每步 {lr['ms']:.2f} ms")
    cached = [r.get('cached_tokens') for r in both]
    print(f"cached_tokens 上报：{sum(c is not None for c in cached)} / {len(cached)} 个请求\n")
