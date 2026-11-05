"""Read-only telemetry and reproducible HTTP benchmarks; never changes model launch settings."""
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[2]
KINDS = {'decode': '固定输出 decode', 'concurrency': '并发阶梯 1/2/4/8/16',
         'context': '上下文长度', 'cache': '前缀缓存冷/热', 'agent': 'Agent 多轮', 'copy': '照抄正确性'}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ADVICE = [
    '单请求速度和总吞吐分别比较；提高并发可能增加排队与单请求延迟。',
    'CPU 专家耗时高且 GPU 有空闲：先比较线程数，再比较实际热专家容量和命中率。一次只改一个参数。',
    'TTFT 随上下文增长：比较冷/热前缀与排队时间，勿将 TTFT 全部归因于 prefill。',
    '输出速度下降而接受长度也下降：同时比较验证步速，避免把内容差异当成算力变化。',
    '显存需要给 prefill、KV 和临时缓冲留余量；不要只按空闲显存增大缓存或并发。',
    '功耗不是优化目标；同时看 tok/s、TTFT、错误率与每输出 token 的采样能耗。',
    '关闭其他客户端请求后测试。服务级计数器可能包含其他流量，客户端测试不能自动隔离它们。',
]


def copy_summary(copies):
    """Case-level exact match after the copy benchmark's normalization; errors count as failures."""
    rows = []
    for index, item in enumerate(copies):
        error = item.get('error')
        exact = not error and not item['diffs']
        rows.append({**item, 'case': index + 1, 'exact': bool(exact),
                     'strict_exact': not error and item['src'] == item['out']})
    total = len(rows)
    correct = sum(row['exact'] for row in rows)
    strict = sum(row['strict_exact'] for row in rows)
    return dict(total=total, correct=correct, different=total-correct,
                accuracy=correct/total if total else None, strict_accuracy=strict/total if total else None,
                errors=sum(bool(row.get('error')) for row in rows), rows=rows)


def get(url):
    """Fetch local server JSON with a bounded timeout."""
    with OPENER.open(url, timeout=8) as r:
        return json.load(r)


def metrics(url):
    """Preserve stage labels and sum other metric series; absence stays absent."""
    with OPENER.open(url + '/metrics', timeout=8) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r'(sglang:[\w]+)(\{[^}]*\})?\s+([\d.eE+\-]+)$', line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name == 'sglang:scheduler_stage_seconds_total':
            name += labels or ''
        out[name] = out.get(name, 0) + float(value)
    return out


def telemetry(url=None, log=None):
    """Always sample hardware; a missing URL leaves inference metrics absent without polling a loading server."""
    result = {'time': time.time(), 'metrics': {}, 'gpus': [], 'errors': [], 'cpu_window': None,
              'gpu_kernel_ms': None, 'engram_wait_ms': None}
    if url is not None:
        try:
            result['metrics'] = metrics(url)
        except Exception as e:
            result['errors'].append('metrics: ' + str(e))
    try:
        fields = 'power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu'
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                                       text=True, timeout=5)
        keys = ('power_w', 'power_limit_w', 'util_pct', 'memory_mib', 'memory_total_mib', 'sm_mhz', 'mem_mhz', 'temp_c')
        result['gpus'] = [dict(zip(keys, [float(v.strip()) if re.fullmatch(r'[\d.]+', v.strip()) else None
                                       for v in line.split(',')])) for line in text.strip().splitlines()]
    except Exception as e:
        result['errors'].append('GPU: ' + str(e))
    if log:
        try:
            with Path(log).open('rb') as f:
                f.seek(max(0, Path(log).stat().st_size - 262144))
                text = f.read().decode(errors='replace')
            rows = re.findall(r'XTier cache: (\d+) windows, hit ([\d.]+), ([\d.]+) distinct experts and ([\d.]+) us CPU per layer call[^\n]*', text)
            if rows:
                n, hit, experts, us = rows[-1]
                result['cpu_window'] = dict(windows=int(n), hit=float(hit), experts=float(experts), cpu_us=float(us),
                                            note='最近日志窗口；可能早于当前测试，不是单次请求计时')
        except OSError as e:
            result['errors'].append('log: ' + str(e))
    return result


def percentile(values, q):
    """Nearest-rank percentile, including small samples."""
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def stream(url, model, prompt, tokens):
    """Measure first payload latency and approximate streamed decode speed with server token usage."""
    started = time.perf_counter()
    first = last = None
    usage, content, error = {}, '', None
    body = dict(model=model, messages=[{'role': 'user', 'content': prompt}], temperature=0, max_tokens=tokens,
                stream=True, stream_options={'include_usage': True}, ignore_eos=True,
                thinking={'type': 'disabled'}, chat_template_kwargs={'enable_thinking': False})
    try:
        req = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with OPENER.open(req, timeout=300) as response:
            for line in response:
                line = line.decode().strip()
                if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                    continue
                event = json.loads(line[5:])
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    d = choice.get('delta') or {}
                    if d.get('content') or d.get('reasoning_content'):
                        last = time.perf_counter()
                        first = first or last
                        content += d.get('content') or ''
        if not first or 'completion_tokens' not in usage:
            raise RuntimeError('流缺少内容或 usage，不能计算性能')
    except Exception as e:
        error = str(e)
    ended = time.perf_counter()
    n = usage.get('completion_tokens', 0)
    return dict(output_tokens=n, prompt_tokens=usage.get('prompt_tokens'),
                cached_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                ttft_s=first-started if first else None, latency_s=ended-started,
                decode_tps=(n-1)/(last-first) if first and last > first and n > 1 else None,
                error=error, content=content)


def round_result(rows, wall):
    """Include prefill/queue time in aggregate throughput and retain all failures."""
    good = [r for r in rows if not r['error']]
    return dict(requests=rows, wall_s=wall, errors=len(rows)-len(good),
                aggregate_tps=sum(r['output_tokens'] for r in good)/wall if wall else None,
                decode_tps_median=statistics.median(v) if (v := [r['decode_tps'] for r in good if r['decode_tps'] is not None]) else None,
                ttft_p50_s=percentile([r['ttft_s'] for r in good], .5),
                ttft_p95_s=percentile([r['ttft_s'] for r in good], .95),
                latency_p95_s=percentile([r['latency_s'] for r in good], .95))


def run(args):
    """Run fixed workloads; requested concurrency above the server cap is skipped unless explicitly queued."""
    info = get(args.url + '/server_info')
    cap = info.get('max_running_requests')
    context = int(info.get('context_length') or 0)
    configuration = {}
    state_file = Path.home()/'.local/state/xtier-console/xtier-state.json'
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        if args.url == f"http://127.0.0.1:{state.get('port')}":
            configuration = state.get('options', {})
    topology = {p.name: (p/'cpulist').read_text().strip()
                for p in Path('/sys/devices/system/node').glob('node[0-9]*')}
    report = dict(version=1, configuration=configuration, numa_cpus=topology, kind=args.kind, started=time.time(), settings=vars(args).copy(),
                  server={k: info.get(k) for k in ('context_length', 'max_running_requests', 'speculative_algorithm',
                          'speculative_num_draft_tokens', 'speculative_dspark_sps_table_path')},
                  rounds=[], samples=[], notes=ADVICE)
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            report['samples'].append(telemetry(args.url, args.log))
            stop.wait(2)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        if args.kind in ('agent', 'copy'):
            out = str(Path(args.out).with_suffix('')) + '-raw.json'
            if args.kind == 'agent':
                if context < 70000:
                    raise ValueError('64K Agent 测试需要至少约 70K 上下文余量')
                import sglang
                source = Path(sglang.__file__).resolve().parents[2]
                argv = [sys.executable, str(REPO/'bench/agent_bench.py'), 'panel', '--url', args.url,
                        '--model', args.model, '--tokenizer', info['tokenizer_path'], '--repo', str(REPO),
                        '--lsglang', str(source), '--sessions', '64k', '--steps', str(args.steps),
                        '--context-limit', str(context), '--max-tokens', str(args.tokens), '--seed', '42', '--out', out]
            else:
                # Existing copy benchmark reports textual differences, not just generation speed.
                import sglang
                source = Path(sglang.__file__).resolve().parent/'srt/managers/scheduler.py'
                argv = [sys.executable, str(REPO/'bench/copy_test.py'), '--port', args.url.rsplit(':', 1)[1],
                        '--model', args.model, '--source', str(source), '--chunks', '2', '--tag', Path(args.out).stem,
                        '--out', str(Path(args.out).parent/'copy')]
            print('COMMAND', __import__('shlex').join(argv), flush=True)
            completed = subprocess.run(argv, check=False)
            if args.kind == 'agent':
                if Path(out).is_file():
                    report['agent'] = json.loads(Path(out).read_text())
            else:
                copy_file = Path(args.out).parent/'copy'/(Path(args.out).stem+'-c1.json')
                if copy_file.is_file():
                    report['copy_results'] = copy_summary(json.loads(copy_file.read_text()))
            completed.check_returncode()
        else:
            levels = [1, 2, 4, 8, 16] if args.kind == 'concurrency' else [1]
            lengths = [1024, 8192, 32768, 65536] if args.kind == 'context' else [1024]
            tok = None
            if args.kind in ('context', 'cache'):
                from tokenizers import Tokenizer
                tok = Tokenizer.from_file(str(Path(info['tokenizer_path'])/'tokenizer.json'))
            for level in levels:
                if cap and level > cap and not args.queue:
                    report['rounds'].append(dict(concurrency=level, skipped=f'超过服务并发上限 {cap}；开启排队压力模式可测试客户端并发'))
                    continue
                for length in lengths:
                    if context and length + args.tokens + 128 >= context:
                        report['rounds'].append(dict(context=length, skipped='上下文余量不足'))
                        continue
                    nonce = uuid.uuid4().hex
                    text = 'Explain the following code and discuss its performance.\n' + ('def add(a, b): return a + b\n' * (length//8+1))
                    if tok:
                        text = tok.decode(tok.encode(text).ids[:length])
                    else:
                        text = 'Write a detailed tutorial on implementing a thread-safe LRU cache in Python.'
                    for repeat in range(args.rounds if args.kind != 'cache' else 2):
                        # Unique prefix per independent trial; cache test intentionally reuses it for warm phase.
                        prefix = nonce if args.kind == 'cache' else uuid.uuid4().hex
                        prompts = [f'Benchmark {prefix}-{i}\n{text}' for i in range(level)]
                        t0 = time.perf_counter()
                        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                            rows = list(pool.map(lambda p: stream(args.url, args.model, p, args.tokens), prompts))
                        for row in rows:
                            row.pop('content', None)
                        result = round_result(rows, time.perf_counter()-t0)
                        result.update(concurrency=level, context_target=length if tok else None, repeat=repeat,
                                      phase=('cold' if repeat == 0 else 'warm') if args.kind == 'cache' else 'independent',
                                      queued=bool(cap and level > cap))
                        report['rounds'].append(result)
                        print('ROUND', json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as e:
        report['error'] = str(e)
        raise
    finally:
        stop.set(); thread.join(timeout=20)
        report['finished'] = time.time()
        report['samples'] = list(report['samples'])
        report['observations'] = observations(report['samples'], report['rounds'])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print('RESULT', args.out, flush=True)
    if any(r.get('errors') for r in report['rounds']):
        raise RuntimeError('部分请求失败；已保留结果，不应作为成功基线')
    return report


def observations(samples, rounds):
    """Service-level counter deltas and trapezoidal sampled energy, never per-kernel attribution."""
    stages = {}
    if len(samples) > 1:
        first, last = samples[0]['metrics'], samples[-1]['metrics']
        stages = {k: v-first[k] for k, v in last.items()
                  if k.startswith('sglang:scheduler_stage_seconds_total') and k in first and v >= first[k]}
    energy = 0.0
    duration = 0.0
    for a, b in zip(samples, samples[1:]):
        pa = [g.get('power_w') for g in a.get('gpus', [])]
        pb = [g.get('power_w') for g in b.get('gpus', [])]
        dt = b['time']-a['time']
        if pa and pb and len(pa) == len(pb) and None not in pa+pb and 0 < dt < 30:
            energy += (sum(pa)+sum(pb))*.5*dt
            duration += dt
    return {'scheduler_stage_delta_s': stages, 'sampled_gpu_energy_j': energy if duration else None,
            'sampled_gpu_mean_w': energy/duration if duration else None, 'sampled_duration_s': duration,
            'note': 'GPU 能耗仅覆盖有效采样区间、所有可见 GPU，包含空闲及其他任务；调度计数器为服务全局值，非纯 GPU/Engram 计时'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=KINDS, default='decode')
    p.add_argument('--url', default='http://127.0.0.1:18190')
    p.add_argument('--model', default='DeepSeek-V4.1-Flash-1017')
    p.add_argument('--tokens', type=int, choices=(32, 128, 256, 512, 1024), default=512)
    p.add_argument('--rounds', type=int, choices=range(1, 6), default=3)
    p.add_argument('--steps', type=int, choices=range(1, 25), default=8)
    p.add_argument('--queue', action='store_true', help='允许客户端并发超过服务上限，结果标记为含排队')
    p.add_argument('--log')
    p.add_argument('--out', default='bench/results/performance.json')
    run(p.parse_args())


if __name__ == '__main__':
    main()
