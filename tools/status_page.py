#!/usr/bin/env python3
"""XTier 状态页：只读，没有任何启停接口。

    python3 tools/status_page.py [--host 127.0.0.1] [--port 8100]
                                 [--api http://127.0.0.1:8001] [--gpus 0,2] [--log PATH]

数据来源：
  - SGLang 的 /metrics（服务启动时已带 --enable-metrics）
  - /health、/get_server_info
  - nvidia-smi（显存、利用率）
  - 服务日志里的 "[xtier] XTier cache: ..." 行（命中率、每层 CPU 时间）

只实现 GET；页面每 2 秒拉一次 /api/status，不写任何东西。
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULTS = {
    'api': 'http://127.0.0.1:8001',
    'gpus': '0,2',
    'log': None,
}
STATE = dict(DEFAULTS)
CACHE_LINE = re.compile(r'\[xtier\] XTier cache: (.*)')


def http_get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')


def parse_metrics(text):
    """{name: value}，多 rank 时只取 tp_rank="0"（避免重复累加）。"""
    out, hist = {}, {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        head, _, raw = line.rpartition(' ')
        try:
            val = float(raw)
        except ValueError:
            continue
        name = head.split('{', 1)[0]
        labels = head[head.find('{'):] if '{' in head else ''
        if 'tp_rank="' in labels and 'tp_rank="0"' not in labels:
            continue
        if name in ('sglang:time_to_first_token_seconds_sum',
                    'sglang:time_to_first_token_seconds_count',
                    'sglang:inter_token_latency_seconds_sum',
                    'sglang:inter_token_latency_seconds_count'):
            hist[name] = hist.get(name, 0.0) + val
            continue
        out[name] = val
    out.update(hist)
    return out


def find_log(explicit):
    if explicit:
        return explicit
    cands = [p for p in glob.glob('/tmp/opencode/xtier-*.log') if 'memguard' not in p]
    return max(cands, key=os.path.getmtime) if cands else None


def last_cache_line(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 400_000))
            tail = f.read().decode('utf-8', 'replace').replace('\r', '\n')
    except OSError:
        return None
    hits = CACHE_LINE.findall(tail)
    if not hits:
        return None
    s = hits[-1]
    d = {}
    m = re.search(r'hit ([\d.]+)', s)
    if m:
        d['hit'] = float(m.group(1))
    m = re.search(r'([\d.]+) distinct experts', s)
    if m:
        d['experts'] = float(m.group(1))
    m = re.search(r'(\d+) us CPU per layer call', s)
    if m:
        d['cpu_us'] = float(m.group(1))
    m = re.search(r'(\d+) swaps in (\d+) rounds', s)
    if m:
        d['swaps'] = int(m.group(1))
        d['rounds'] = int(m.group(2))
    m = re.search(r'\+(\d+) of \+(\d+) slots', s)
    if m:
        d['slots_filled'] = int(m.group(1))
        d['slots_total'] = int(m.group(2))
    return d


def gpu_stats(ids):
    try:
        q = subprocess.run(
            ['nvidia-smi', '-i', ids,
             '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=4)
        out = []
        for line in q.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(',')]
            if len(p) >= 5:
                out.append({'index': p[0], 'used': float(p[1]), 'total': float(p[2]),
                            'util': float(p[3]), 'temp': float(p[4])})
        return out
    except Exception:
        return []


def build_status():
    api = STATE['api']
    st = {'ok': False, 'ts': time.time(), 'api': api, 'gpus': gpu_stats(STATE['gpus'])}
    t0 = time.time()
    try:
        m = parse_metrics(http_get(api + '/metrics'))
        st['latency_ms'] = round((time.time() - t0) * 1000, 1)
        st['ok'] = True
    except Exception as exc:
        st['error'] = f'{type(exc).__name__}: {exc}'
        st['xtier'] = last_cache_line(find_log(STATE['log']))
        return st

    max_tok = m.get('sglang:max_total_num_tokens') or 0
    used_tok = m.get('sglang:kv_used_tokens') or 0
    ttft_sum = m.get('sglang:time_to_first_token_seconds_sum', 0.0)
    ttft_cnt = m.get('sglang:time_to_first_token_seconds_count', 0.0)
    itl_sum = m.get('sglang:inter_token_latency_seconds_sum', 0.0)
    itl_cnt = m.get('sglang:inter_token_latency_seconds_count', 0.0)

    st.update({
        'model': 'DeepSeek-V4.1-Flash',
        'running': m.get('sglang:num_running_reqs', 0),
        'queue': m.get('sglang:num_queue_reqs', 0),
        'gen_tps': round(m.get('sglang:gen_throughput', 0.0), 1),
        'requests': int(m.get('sglang:num_requests_total', 0)),
        'prompt_tokens': int(m.get('sglang:prompt_tokens_total', 0)),
        'gen_tokens': int(m.get('sglang:generation_tokens_total', 0)),
        'kv_used': int(used_tok), 'kv_max': int(max_tok),
        'kv_pct': round(100.0 * used_tok / max_tok, 1) if max_tok else 0.0,
        'accept_len': round(m.get('sglang:spec_accept_length', 0.0), 2),
        'accept_rate': round(m.get('sglang:spec_accept_rate', 0.0), 2),
        'hit_rate': round(m.get('sglang:cache_hit_rate', 0.0), 3),
        'cached_tokens': int(m.get('sglang:cached_tokens_total', 0)),
        'ttft_ms': round(1000.0 * ttft_sum / ttft_cnt, 1) if ttft_cnt else None,
        'itl_ms': round(1000.0 * itl_sum / itl_cnt, 1) if itl_cnt else None,
        'kv_gb': round(m.get('sglang:kv_cache_memory_usage_gb', 0.0), 2),
        'hicache_used': int(m.get('sglang:hicache_host_used_tokens', 0)),
    })
    try:
        info = json.loads(http_get(api + '/get_server_info'))
        st['context'] = info.get('context_length')
        st['tp'] = info.get('tp_size')
        st['max_req'] = info.get('max_running_requests')
        st['pdi'] = info.get('prefill_decode_interval')
        st['chunk'] = info.get('chunked_prefill_size')
        st['policy'] = info.get('schedule_policy')
        st['dspark'] = info.get('speculative_dspark_block_size')
    except Exception:
        pass
    st['xtier'] = last_cache_line(find_log(STATE['log']))
    return st


PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>XTier 状态</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--ok:#3fb950;--bad:#f85149;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",sans-serif;padding:20px}
header{display:flex;align-items:baseline;gap:12px;margin-bottom:16px;flex-wrap:wrap}
h1{font-size:18px;margin:0;letter-spacing:.5px}
.badge{padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
.badge.on{color:var(--ok);border-color:var(--ok)}
.badge.off{color:var(--bad);border-color:var(--bad)}
.dim{color:var(--dim);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--dim);font-size:12px;margin-bottom:6px}
.card .v{font-size:24px;font-variant-numeric:tabular-nums;font-weight:600}
.card .v small{font-size:13px;font-weight:400;color:var(--dim)}
.bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:8px}
.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .4s}
table{width:100%;border-collapse:collapse;margin-top:12px;font-variant-numeric:tabular-nums}
td{padding:5px 0;border-bottom:1px solid #21262d}
td:first-child{color:var(--dim);width:45%}
td:last-child{text-align:right}
footer{margin-top:16px;color:var(--dim);font-size:12px}
</style></head><body>
<header><h1>XTier 状态</h1><span id=badge class="badge off">…</span>
<span class=dim id=meta></span></header>
<div class=grid>
  <div class=card><div class=k>并发 / 排队</div><div class=v id=conc>-</div></div>
  <div class=card><div class=k>生成吞吐</div><div class=v id=tps>-</div></div>
  <div class=card><div class=k>KV 占用</div><div class=v id=kv>-</div><div class=bar><i id=kvbar></i></div></div>
  <div class=card><div class=k>投机接受长度</div><div class=v id=acc>-</div></div>
  <div class=card><div class=k>前缀缓存命中</div><div class=v id=hit>-</div></div>
  <div class=card><div class=k>首 token 延迟</div><div class=v id=ttft>-</div></div>
</div>
<table>
 <tr><td>累计请求 / prompt / 生成 token</td><td id=counts>-</td></tr>
 <tr><td>每 token 间隔 (ITL)</td><td id=itl>-</td></tr>
 <tr><td>KV 池</td><td id=kvpool>-</td></tr>
 <tr><td>上下文 / TP / 并发上限</td><td id=cfg>-</td></tr>
 <tr><td>调度策略 / PDI / chunk / DSpark</td><td id=sched>-</td></tr>
 <tr><td>XTier 专家命中 / 每层 CPU</td><td id=xt>-</td></tr>
 <tr><td>XTier 槽位</td><td id=xtslot>-</td></tr>
 <tr><td>GPU</td><td id=gpu>-</td></tr>
</table>
<footer>只读页面：仅从 8001 的 /metrics 和 /health 取数，没有任何启停接口。每 2 秒刷新。</footer>
<script>
const $=id=>document.getElementById(id);
async function tick(){
  try{
    const r=await fetch('/api/status',{cache:'no-store'});const d=await r.json();
    const b=$('badge');
    if(d.ok){b.className='badge on';b.textContent='在线';}else{b.className='badge off';b.textContent='离线';}
    $('meta').textContent=d.ok?`${d.api} · 采集 ${d.latency_ms} ms · ${new Date(d.ts*1000).toLocaleTimeString()}`:(d.error||'');
    $('conc').innerHTML=`${d.running??'-'} <small>/ 排队 ${d.queue??'-'}</small>`;
    $('tps').innerHTML=`${d.gen_tps??'-'} <small>tok/s</small>`;
    $('kv').innerHTML=`${d.kv_pct??'-'}% <small>${d.kv_used??0} / ${d.kv_max??0}</small>`;
    $('kvbar').style.width=(d.kv_pct||0)+'%';
    $('acc').innerHTML=`${d.accept_len??'-'} <small>rate ${d.accept_rate??'-'}</small>`;
    $('hit').innerHTML=`${d.hit_rate!==undefined?(d.hit_rate*100).toFixed(1):'-'}% <small>${d.cached_tokens??0} tok</small>`;
    $('ttft').innerHTML=`${d.ttft_ms??'-'} <small>ms 平均</small>`;
    $('counts').textContent=`${d.requests??0} / ${d.prompt_tokens??0} / ${d.gen_tokens??0}`;
    $('itl').textContent=`${d.itl_ms??'-'} ms`;
    $('kvpool').textContent=`${d.kv_gb??0} GB；HiCache 主机缓存 ${d.hicache_used??0} tok`;
    $('cfg').textContent=`${d.context??'-'} · TP${d.tp??'-'} · max_running ${d.max_req??'-'}`;
    $('sched').textContent=`${d.policy??'-'} · PDI ${d.pdi??'-'} · chunk ${d.chunk??'-'} · DSPARK ${d.dspark??'-'}`;
    const x=d.xtier;
    $('xt').textContent=x?`命中 ${(x.hit*100).toFixed(1)}% · ${x.experts??'-'} 个专家 · ${x.cpu_us??'-'} µs/层`:'（还没有完成的请求统计）';
    $('xtslot').textContent=x&&x.slots_total?`已填 ${x.slots_filled} / ${x.slots_total}；累计换入 ${x.swaps??'-'} 次`:'-';
    $('gpu').textContent=(d.gpus||[]).map(g=>`GPU${g.index} ${g.used.toFixed(1)}/${g.total.toFixed(0)} GB · ${g.util.toFixed(0)}% · ${g.temp.toFixed(0)}°C`).join('   |   ')||'-';
  }catch(e){$('badge').className='badge off';$('badge').textContent='页面错误';$('meta').textContent=String(e);}
}
tick();setInterval(tick,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = 'xtier-status'

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            body = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if path == '/api/status':
            body = json.dumps(build_status(), ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        if path == '/healthz':
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')
            return
        self.send_response(404)
        self.send_header('Content-Length', '0')
        self.end_headers()


def main():
    ap = argparse.ArgumentParser(description='XTier read-only status page (GET only, no start/stop).')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--api', default=DEFAULTS['api'], help='SGLang base URL (default 127.0.0.1:8001)')
    ap.add_argument('--gpus', default=DEFAULTS['gpus'], help='nvidia-smi -i 的取值，默认 0,2')
    ap.add_argument('--log', default=None, help='服务日志路径（默认自动找最新的 /tmp/opencode/xtier-*.log）')
    args = ap.parse_args()
    STATE.update(api=args.api.rstrip('/'), gpus=args.gpus, log=args.log)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'XTier 状态页 http://{args.host}:{args.port}/  (只读；数据源 {STATE["api"]}；GPU {STATE["gpus"]})', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
