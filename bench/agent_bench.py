# Copy of bench/agent_bench.py from github.com/a775828b-dot/lsglang_ds4.1 @ 641931f (Apache-2.0), the benchmark
# the production README measures with, plus two robustness fixes marked "XTier copy": a tool call whose arguments
# are not valid JSON goes into the history as {} (SGLang answers 400 to such a history; the model still sees the
# parse error in the tool result), and an HTTP error ends that session instead of the round. Timing is unchanged.
"""agent_bench: coding-agent benchmark for the DeepSeek V4.1 Flash 1017 server (OpenAI API, tools, streaming).

Shaped like a real coding-agent (dsh / Claude Code style) session measured through our gateway on 2026-10-19
(77 requests, 98.6% prefix-cache hits, mostly tool calls with short reasoning and some long file writes), and spread
over the whole 512K KV pool: five sessions start at about 64K, 128K, 256K, 384K and 480K tokens (the old five-point
grid). Each session gives the model a coding task on a sandbox copy of real code (this repository's adapter/ and
bench/, and the patched Lsglang sources), a history of earlier read_file calls that brings the prompt to its target
size, and OpenAI tools. The model then works as an agent for up to --steps requests: its tool calls run
against the sandbox and the results go back, so every later request re-reads a cached prefix.

Tools: read_file / list_dir / grep / edit_file / write_file / check_python (py_compile only). Nothing the model
writes is executed; every path stays inside the per-session sandbox (a temp directory, removed afterwards).

Per request: TTFT, output tokens, decode tok/s (first to last streamed chunk), DSpark verify steps (/metrics delta),
accept length and ms per verify step. The first request of each session is a cold prefill of the whole prompt
(prefill tok/s = uncached prompt tokens / TTFT; every run starts its system prompt with its own id). A session ends early when its next request could exceed --context-limit. Accept length follows the content and the sampling, so compare
configurations by "ms per step" first; decode tok/s is what a user sees on this kind of task.

Usage: agent_bench.py LABEL --tokenizer MODEL_DIR --lsglang PATCHED_LSGLANG [--repo THIS_REPO] [--url URL]
       [--sessions 64k,128k-write,256k,384k,500k] [--steps 24] [--temperature T] [--seed N] [--out FILE] [--dry-run]
Default sampling is the server's (the agent sent no temperature)."""
import argparse, json, os, re, shutil, statistics as st, sys, tempfile, time, urllib.request
from pathlib import Path

URL = 'http://127.0.0.1:39504'
MODEL = 'DeepSeek-V4.1-Flash-1017'
SRT = 'python/sglang/srt'
ARGS = None
RUN_ID = f'{time.time_ns():x}'   # starts the system prompt: no prefix-cache reuse across runs
# Sandbox contents: (root key, path under it, sandbox directory, file pattern). The task files (adapter/, bench/)
# and the large real sources that pad the history to the target context (sglang/srt/...).
TREES = [
    ('repo', 'adapter', 'adapter', '*.py'),
    ('repo', 'bench', 'bench', '*.py'),
    ('lsglang', f'{SRT}/layers', 'sglang/srt/layers', '*.py'),
    ('lsglang', f'{SRT}/managers', 'sglang/srt/managers', '*.py'),
    ('lsglang', f'{SRT}/models', 'sglang/srt/models', 'deepseek*.py'),
    ('lsglang', f'{SRT}/model_executor', 'sglang/srt/model_executor', '*.py'),
    ('lsglang', f'{SRT}/speculative', 'sglang/srt/speculative', '*.py'),
]

SYSTEM = (
    '你是一个在终端里工作的编码代理（类似 Claude Code / dsh）。工作目录是一个 Python 项目，包含 adapter（DeepSeek V4.1'
    ' 推理适配层：Engram 行存储、按层提前加载等）、bench（测速与工具脚本）和 sglang 的部分源码。\n'
    '规则：先用工具阅读相关代码再动手；修改用 edit_file（old_string 必须与文件内容逐字一致且唯一）或 write_file；'
    '改完用 check_python 做语法检查；回答用中文，简洁。上文已经读过的文件不要重复读取；读够了就动手修改，'
    '不要只阅读不修改。完成任务后给出简短总结，不再调用工具。')

TOOLS = [
    {'type': 'function', 'function': {'name': 'read_file', 'description': '读取文件内容（带行号）。',
        'parameters': {'type': 'object', 'properties': {
            'path': {'type': 'string', 'description': '相对工作目录的路径'},
            'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'list_dir', 'description': '列出目录。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
    {'type': 'function', 'function': {'name': 'grep', 'description': '在文件或目录中按正则搜索，返回 路径:行号:内容。',
        'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'path': {'type': 'string'}},
                       'required': ['pattern']}}},
    {'type': 'function', 'function': {'name': 'edit_file', 'description': '把文件中唯一出现的 old_string 替换为 new_string。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_string': {'type': 'string'},
                       'new_string': {'type': 'string'}}, 'required': ['path', 'old_string', 'new_string']}}},
    {'type': 'function', 'function': {'name': 'write_file', 'description': '写入（覆盖）文件。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                       'required': ['path', 'content']}}},
    {'type': 'function', 'function': {'name': 'check_python', 'description': '对 Python 文件做语法检查（py_compile，不执行）。',
        'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}},
]

# (target context tokens, task, files read in the pre-filled history: small task files first, then padding)
SESSIONS = {
    '64k': (64_000,
            '给 adapter/early_lk_moe.py 的逐层交接加统计：记录已交接的层数和每层交接耗时（秒），提供模块级函数 stats() '
            '返回汇总（层数、总耗时、最慢的层及其耗时），不要改变交接逻辑和调用顺序。再写一个不依赖 GPU 的单元测试 '
            'bench/test_early_lk_moe_stats.py，用假的 FusedMoE 类和 mock 覆盖统计逻辑。相关文件上文已读过，最多再读 3 个文件就开始修改。',
            ['adapter/early_lk_moe.py', 'adapter/sitecustomize.py']),
    '128k-write': (128_000,
                   '为 bench/agent_bench.py 里的 Sandbox 类写一份完整的 pytest 单元测试文件 bench/test_agent_bench_sandbox.py：'
                   '覆盖 read_file（含 start_line/end_line）、list_dir、grep、edit_file（唯一、零次、多次匹配）、write_file、'
                   'check_python（通过与语法错误）以及路径越界拒绝，至少 12 个用例，用 tmp_path 构造文件，不依赖 GPU 和网络。'
                   '相关代码上文已读过，直接用 write_file 写出完整文件，再用 check_python 检查。',
                   ['bench/agent_bench.py', 'adapter/early_lk_moe.py', 'sglang/srt/managers/schedule_batch.py']),
    '256k': (256_000,
             '阅读 adapter/engram_backend.py 里 Engram 提前查表（prefetch）相关代码，找出多请求并发或 CUDA graph 重放时'
             '可能出现的竞态或缓冲区复用问题。对你确认的问题用 edit_file 给出最小修复，并在修改处加注释说明原因。'
             '相关文件上文已读过，最多再读 3 个文件就开始修改。',
             ['adapter/engram_backend.py', 'sglang/srt/layers/logits_processor.py', 'sglang/srt/managers/schedule_batch.py',
              'sglang/srt/managers/scheduler.py']),
    '384k': (384_000,
             '重构 bench/agent_bench.py 的汇总部分：把 summarize 拆成按上下文分档和按会话两个函数，新增每个会话的'
             '冷启动首字与后续首字中位数，增加 --json-only 参数只输出 JSON；保持命令行和结果 JSON 现有字段兼容。'
             '相关文件上文已读过，最多再读 3 个文件就开始修改。',
             ['bench/agent_bench.py', 'sglang/srt/models/deepseek_v4.py', 'sglang/srt/managers/scheduler.py',
              'sglang/srt/managers/schedule_batch.py', 'sglang/srt/layers/logits_processor.py',
              'sglang/srt/model_executor/model_runner.py']),
    '500k': (480_000,
             '整理 adapter/sitecustomize.py 和 adapter/ds1017_quant.py：给所有函数补全类型注解和中文文档字符串，'
             '把 sitecustomize.py 里读取的环境变量开关整理成文件顶部的一张常量表（名称、默认值、含义），不要改变任何行为。'
             '两个文件上文已读过，直接用 edit_file 修改，完成后用 check_python 检查。',
             ['adapter/sitecustomize.py', 'adapter/ds1017_quant.py', 'sglang/srt/models/deepseek_v4.py',
              'sglang/srt/managers/scheduler.py']),
}


class Sandbox:
    def __init__(self, roots):
        self.root = Path(tempfile.mkdtemp(prefix='agent_bench_'))
        for key, sub, dst, pat in TREES:
            src = Path(roots[key]) / sub
            if not src.is_dir():
                shutil.rmtree(self.root, ignore_errors=True)
                raise SystemExit(f'agent_bench: missing source directory {src}')
            for f in src.rglob(pat):
                if '__pycache__' in f.parts or not f.is_file() or f.stat().st_size > 2_000_000:
                    continue
                t = self.root / dst / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)

    def path(self, p):
        q = (self.root / str(p).lstrip('/')).resolve()
        if q != self.root and self.root not in q.parents:
            raise ValueError('路径超出工作目录')
        return q

    def call(self, name, args):
        try:
            return getattr(self, 't_' + name)(**args)
        except TypeError as e:
            return f'参数错误：{e}'
        except Exception as e:  # tool errors go back to the model, as an agent harness does
            return f'错误：{type(e).__name__}: {e}'

    def t_read_file(self, path, start_line=None, end_line=None):
        lines = self.path(path).read_text(encoding='utf-8', errors='replace').splitlines()
        a = max(1, int(start_line or 1)); b = min(len(lines), int(end_line or len(lines)), a + 1999)
        return '\n'.join(f'{i:6d}\t{lines[i - 1]}' for i in range(a, b + 1)) + \
            ('' if b >= len(lines) else f'\n... (共 {len(lines)} 行，用 start_line/end_line 继续读)')

    def t_list_dir(self, path='.'):
        d = self.path(path)
        return '\n'.join(sorted((e.name + '/') if e.is_dir() else e.name for e in d.iterdir())[:300])

    def t_grep(self, pattern, path='.'):
        rx = re.compile(pattern); base = self.path(path); out = []
        files = [base] if base.is_file() else sorted(base.rglob('*.py'))
        for f in files:
            for i, line in enumerate(f.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
                if rx.search(line):
                    out.append(f'{f.relative_to(self.root)}:{i}:{line[:200]}')
                    if len(out) >= 200:
                        return '\n'.join(out) + '\n...（只显示前 200 条）'
        return '\n'.join(out) or '无匹配'

    def t_edit_file(self, path, old_string, new_string):
        f = self.path(path); s = f.read_text(encoding='utf-8')
        n = s.count(old_string)
        if n != 1:
            return f'错误：old_string 出现 {n} 次（必须恰好 1 次）'
        f.write_text(s.replace(old_string, new_string), encoding='utf-8')
        return f'已修改 {path}'

    def t_write_file(self, path, content):
        f = self.path(path); f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding='utf-8')
        return f'已写入 {path}（{len(content.splitlines())} 行）'

    def t_check_python(self, path):
        import py_compile
        try:
            py_compile.compile(str(self.path(path)), doraise=True, cfile=os.devnull)
            return '语法检查通过'
        except py_compile.PyCompileError as e:
            return f'语法错误：{e.msg}'

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


def history(sb, task, files, target, tok):
    """System + task + pre-filled read_file calls until the prompt reaches about `target` tokens."""
    msgs = [{'role': 'system', 'content': f'（测速运行 {RUN_ID}）' + SYSTEM}, {'role': 'user', 'content': task}]
    used = len(tok.encode(SYSTEM + task + json.dumps(TOOLS, ensure_ascii=False)).ids)
    pool = list(files) + sorted(str(p.relative_to(sb.root)) for p in (sb.root / 'sglang').rglob('*.py'))
    seen = set()
    for i, rel in enumerate(pool):
        if used >= target:
            break
        if rel in seen:
            continue
        seen.add(rel)
        text = sb.t_read_file(rel)
        n = len(tok.encode(text).ids)
        if used + n > target * 1.05 and i >= len(files):
            continue
        cid = f'call_pre_{i}'
        msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': rel})}}]})
        msgs.append({'role': 'tool', 'tool_call_id': cid, 'content': text})
        used += n + 20
    return msgs, used


def verify_calls():
    text = urllib.request.urlopen(ARGS.url + '/metrics', timeout=10).read().decode()
    m = re.search(r'^sglang:spec_verify_calls_total\{[^}]*\} ([0-9.e+]+)$', text, re.M)
    return float(m.group(1)) if m else 0.0


def request(msgs, args):
    body = dict(model=args.model, messages=msgs, tools=TOOLS, tool_choice='auto', max_tokens=args.max_tokens,
                stream=True, stream_options={'include_usage': True})
    if args.temperature is not None:
        body['temperature'] = args.temperature
    if args.seed is not None:
        body['seed'] = args.seed
    req = urllib.request.Request(args.url + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    v0 = verify_calls(); t0 = time.time(); first = last = None; usage = None
    content, reasoning, calls = [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            ev = json.loads(line[5:])
            if ev.get('usage'):
                usage = ev['usage']
            for ch in ev.get('choices') or []:
                d = ch.get('delta') or {}
                got = False
                if d.get('content'):
                    content.append(d['content']); got = True
                if d.get('reasoning_content'):
                    reasoning.append(d['reasoning_content']); got = True
                for tc in d.get('tool_calls') or []:
                    c = calls.setdefault(tc.get('index', 0), {'id': None, 'name': '', 'args': ''})
                    c['id'] = tc.get('id') or c['id']
                    fn = tc.get('function') or {}
                    c['name'] += fn.get('name') or ''
                    c['args'] += fn.get('arguments') or ''
                    got = True
                if got:
                    now = time.time(); first = first or now; last = now
    steps = verify_calls() - v0
    n = (usage or {}).get('completion_tokens', 0)
    dec_s = (last - first) if first and last and last > first else 0
    row = dict(prompt_tokens=(usage or {}).get('prompt_tokens'),
               cached_tokens=((usage or {}).get('prompt_tokens_details') or {}).get('cached_tokens'),
               output_tokens=n, ttft_s=round((first or time.time()) - t0, 3), decode_s=round(dec_s, 3),
               decode_tps=round((n - 1) / dec_s, 1) if dec_s and n > 1 else None, verify_steps=steps,
               accept=round(n / steps, 3) if steps else None,
               step_ms=round(1000 * dec_s / steps, 2) if steps and dec_s else None,
               tool_calls=[c['name'] for c in calls.values()], reasoning_chars=sum(map(len, reasoning)),
               text_chars=sum(map(len, content)))
    return row, ''.join(content), ''.join(reasoning), calls


def run_session(name, args, tok):
    target, task, files = SESSIONS[name]
    sb = Sandbox(dict(repo=args.repo, lsglang=args.lsglang))
    try:
        msgs, est = history(sb, task, files, target, tok)
        print(f'[{name}] pre-filled history: {len(msgs)} messages, ~{est} tokens', flush=True)
        if args.dry_run:
            return []
        rows = []
        for step in range(args.steps):
            try:
                row, content, reasoning, calls = request(msgs, args)
            except urllib.error.HTTPError as e:   # XTier copy: end the session, keep the round
                print(f'[{name}] step {step}: HTTP {e.code} {e.read()[:200]!r}, ending the session', flush=True)
                break
            row.update(session=name, step=step)
            rows.append(row)
            print(f'[{name}] step {step}: ctx {row["prompt_tokens"]} cached {row["cached_tokens"]} out {row["output_tokens"]} '
                  f'ttft {row["ttft_s"]}s decode {row["decode_tps"]} tok/s accept {row["accept"]} '
                  f'{row["step_ms"]} ms/step tools {row["tool_calls"]}', flush=True)
            if not calls:
                break
            # context guard: the next request = this prompt + this answer + the tool results + its own answer
            grown = row['prompt_tokens'] + row['output_tokens']
            tcs = []
            for i, c in sorted(calls.items()):
                cid = c['id'] or f'call_{step}_{i}'
                args_json = c['args']
                try:   # XTier copy: SGLang rejects a history whose tool-call arguments are not valid JSON (400)
                    json.loads(args_json or '{}')
                except json.JSONDecodeError:
                    args_json = '{}'
                tcs.append({'id': cid, 'type': 'function', 'function': {'name': c['name'], 'arguments': args_json},
                            'raw_args': c['args']})
            msgs.append({'role': 'assistant', 'content': content, 'tool_calls': tcs})
            for t in tcs:
                try:
                    a = json.loads(t.pop('raw_args') or '{}')
                    result = sb.call(t['function']['name'], a) if isinstance(a, dict) else '参数必须是 JSON 对象'
                except json.JSONDecodeError as e:
                    result = f'参数不是合法 JSON：{e}'
                msgs.append({'role': 'tool', 'tool_call_id': t['id'], 'content': str(result)[:60000]})
                grown += len(tok.encode(msgs[-1]['content']).ids) + 64
            if grown + args.max_tokens + 1024 > args.context_limit:
                print(f'[{name}] context guard: next request ~{grown} tokens, stopping the session', flush=True)
                break
        return rows
    finally:
        sb.close()


def summarize(rows):
    def agg(rs):
        rs = [r for r in rs if r['decode_s'] and r['verify_steps'] and r['output_tokens'] > 1]
        if not rs:
            return None
        toks = sum(r['output_tokens'] - 1 for r in rs); secs = sum(r['decode_s'] for r in rs)
        steps = sum(r['verify_steps'] for r in rs)
        return dict(requests=len(rs), output_tokens=toks + len(rs), decode_tps=round(toks / secs, 1),
                    median_request_tps=round(st.median(r['decode_tps'] for r in rs), 1),
                    accept=round(sum(r['output_tokens'] for r in rs) / steps, 3),
                    ms_per_step=round(1000 * secs / steps, 2))
    out = dict(all=agg(rows))
    for lo, hi in ((0, 98304), (98304, 196608), (196608, 327680), (327680, 458752), (458752, 1 << 30)):
        a = agg([r for r in rows if lo <= (r['prompt_tokens'] or 0) < hi])
        if a:
            out[f'ctx_{lo // 1024}k_{hi // 1024 if hi < (1 << 30) else "max"}'] = a
    out['sessions'] = {}
    for name in dict.fromkeys(r['session'] for r in rows):
        rs = [r for r in rows if r['session'] == name]
        a = agg(rs)
        if a:
            ctx = [r['prompt_tokens'] or 0 for r in rs]
            a.update(ctx_start=min(ctx), ctx_end=max(ctx), steps=len(rs))
            out['sessions'][name] = a
    later = [r['ttft_s'] for r in rows if r['step'] > 0]
    out['ttft_first_s'] = [r['ttft_s'] for r in rows if r['step'] == 0]
    out['ttft_later_median_s'] = round(st.median(later), 2) if later else None
    out['tool_call_requests'] = sum(1 for r in rows if r['tool_calls'])
    out['cold_prefill'] = [dict(session=r['session'], prompt_tokens=r['prompt_tokens'], cached_tokens=r['cached_tokens'] or 0,
                                ttft_s=r['ttft_s'],
                                prefill_tps=round((r['prompt_tokens'] - (r['cached_tokens'] or 0)) / r['ttft_s'], 1)
                                if r['ttft_s'] else None)
                           for r in rows if r['step'] == 0]
    outs = sorted(r['output_tokens'] for r in rows)
    out['output_tokens_p10_p50_p90'] = [outs[int(q * (len(outs) - 1))] for q in (0.1, 0.5, 0.9)] if outs else None
    out['long_output_share'] = round(sum(o for o in outs if o >= 500) / max(1, sum(outs)), 3)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('label')
    p.add_argument('--url', default=os.environ.get('AGENT_BENCH_URL', URL))
    p.add_argument('--model', default=MODEL, help='served model name')
    p.add_argument('--tokenizer', default=os.environ.get('MODEL'), help='model directory or its tokenizer.json')
    p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]), help='this repository (adapter/, bench/)')
    p.add_argument('--lsglang', default=os.environ.get('LSGLANG'), help='Lsglang source with the 1017 patch applied')
    p.add_argument('--out', default=None, help='result JSON (default: agent-bench-LABEL.json in the current directory)')
    p.add_argument('--sessions', default='64k,128k-write,256k,384k,500k')
    p.add_argument('--context-limit', type=int, default=524288, help='server context / KV pool (tokens)')
    p.add_argument('--steps', type=int, default=24)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--dry-run', action='store_true', help='build the sandboxes and histories only, no requests')
    args = p.parse_args()
    global ARGS
    ARGS = args
    if not args.tokenizer or not args.lsglang:
        p.error('--tokenizer and --lsglang are required (or set MODEL and LSGLANG)')
    from tokenizers import Tokenizer
    t = Path(args.tokenizer)
    tok = Tokenizer.from_file(str(t / 'tokenizer.json' if t.is_dir() else t))
    rows = []; t0 = time.time()
    for name in args.sessions.split(','):
        rows += run_session(name, args, tok)
    if args.dry_run:
        return
    summary = summarize(rows)
    print('SUMMARY', args.label, json.dumps(summary, ensure_ascii=False))
    out = Path(args.out or f'agent-bench-{args.label}.json')
    json.dump(dict(label=args.label, args=vars(args), wall_s=round(time.time() - t0, 1), summary=summary, rows=rows),
              open(out, 'w'), indent=1, ensure_ascii=False)
    print('saved', out)


if __name__ == '__main__':
    main()
