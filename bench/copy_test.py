"""Verbatim-copy test of a running SGLang server: the model must echo code chunks exactly (temperature 0, thinking off).
Any changed span points at the decode path (the check that caught the 10-21 variable-length verify bug).
--temperature T > 0 samples instead (T = 0.01: almost the greedy text through the sampling paths of the decode, e.g.
rejection sampling and its verify truncation, which greedy requests skip).

  python bench/copy_test.py --port 18180 --model Qwen3.8-Flash-Next --chunks 8 --concurrency 1 --tag NAME

Chunks: 90-line windows of a fixed source file (the production baselines' choice). --ignore-first-indent ignores the
indentation of the first line, which the Qwen reply always drops.
"""
import argparse
import concurrent.futures as cf
import difflib
import json
import os
import urllib.request
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, default=18180)
P.add_argument('--model', required=True)
P.add_argument('--chunks', type=int, default=8)
P.add_argument('--concurrency', type=int, default=1)
P.add_argument('--tag', required=True)
P.add_argument('--source', default='/data/fast/workspaces/ds41-1017/source/python/sglang/srt/managers/scheduler.py')
P.add_argument('--ignore-first-indent', action='store_true')
P.add_argument('--temperature', type=float, default=0.0)
P.add_argument('--out', default='bench/results/copy')
A = P.parse_args()
lines = Path(A.source).read_text(encoding='utf-8').splitlines()
chunks = ['\n'.join(lines[a:a + 90]) for a in (200, 900, 1600, 2300, 3000, 3700, 4400, 5100)]
chunks = [c for c in chunks if '｜' not in c][:A.chunks]
url = f'http://127.0.0.1:{A.port}/v1/chat/completions'


def run(c):
    body = dict(model=A.model, temperature=A.temperature, max_tokens=4000, stream=False,
                chat_template_kwargs={'enable_thinking': False},
                messages=[{'role': 'user', 'content': '把 <src> 和 </src> 之间的内容逐字原样输出，不要增删改任何字符，'
                           '不要加代码块标记，不要解释。\n<src>\n' + c + '\n</src>'}])
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return c, d['choices'][0]['message'].get('content') or ''


saved, bad = [], 0
with cf.ThreadPoolExecutor(A.concurrency) as ex:
    for c, out in ex.map(run, chunks):
        o = out.strip('\n')
        if o.startswith('```'):
            o = o.split('\n', 1)[1].rsplit('```', 1)[0].strip('\n')
        src = c
        if A.ignore_first_indent:
            src, o = src.lstrip(' '), o.lstrip(' ')
        diffs = [op for op in difflib.SequenceMatcher(None, src, o, autojunk=False).get_opcodes() if op[0] != 'equal']
        bad += bool(diffs)
        saved.append(dict(src=c, out=out, diffs=[list(d) for d in diffs]))
os.makedirs(A.out, exist_ok=True)
Path(A.out, f'{A.tag}-c{A.concurrency}.json').write_text(json.dumps(saved, ensure_ascii=False))
print(f'COPY {bad}/{len(chunks)} copies differ (concurrency {A.concurrency})')
