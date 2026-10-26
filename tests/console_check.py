"""Checks of the XTier console (xtier/console) without launching a model: the console entries of configs/ load, every
option value validates and builds a launch command, the measured defaults build the README configuration, the page
renders, and the standalone server answers (its POST guard included).

    python tests/console_check.py        (from the repository root; standard library only)
"""

import itertools
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.console import __main__ as server, core, ui  # noqa: E402

models = core.load_models()
assert list(models)[:3] == ['qwen38', 'glm53', 'ds41'], list(models)
console = core.Console(models, tempfile.mkdtemp(prefix='xtier-console-'), runner='setsid', guard_gib=0)

combos = 0
for mid, m in models.items():
    keys = [o['key'] for o in m['options']]
    assert len(keys) == len(set(keys)), mid
    b = m.get('bench') or {}   # the card's README speeds: how they were measured and the text shown
    assert b.get('kind') in ('agent', 'text') and isinstance(b.get('dec'), str) and isinstance(b.get('pre'), str), mid
    d = console.defaults(mid)
    assert console.validate(mid, {}) == d
    for o in m['options']:
        assert o['default'] in {c[0] for c in o['choices']}, (mid, o['key'])
        for cond in o.get('off_when', []):
            assert set(cond) <= set(keys), (mid, o['key'], cond)
    for n in m.get('notes', []):
        assert set(n['when']) <= set(keys), (mid, n)
    for values in itertools.product(*[[c[0] for c in o['choices']] for o in m['options']]):
        o = console.validate(mid, dict(zip(keys, values)))
        env, argv = console.build(mid, o)
        assert env['PORT'] == str(m['port']) and all(isinstance(v, str) for v in env.values()), (mid, o)
        assert Path(argv[0]).is_file() and os.access(argv[0], os.X_OK), argv[0]
        assert all(isinstance(a, str) for a in argv), (mid, o)
        console.describe(mid, o)
        combos += 1
    try:
        console.validate(mid, {keys[0]: 'nope'})
        raise AssertionError('accepted a bad value')
    except ValueError:
        pass

# the measured defaults = the README "+ 加速包" rows
env, argv = console.build('qwen38', console.defaults('qwen38'))
assert argv[0].endswith('run-qwen-mtp.sh') and argv[argv.index('--kv-cache-dtype') + 1] == 'nvfp4'
assert argv[argv.index('--chunked-prefill-size') + 1] == '24576' and ',27' in env['XTIER_CACHE_FULL']
assert 'XTIER_CACHE_ELASTIC' not in env   # off by default: Qwen's decode is GPU-bound
assert console.build('qwen38', {**console.defaults('qwen38'), 'elastic': 'on'})[0]['XTIER_CACHE_ELASTIC'] == '1'
assert all(o.get('switch') for m in console.models.values() for o in m['options'] if o['key'] == 'elastic')
env, argv = console.build('qwen38', {**console.defaults('qwen38'), 'packs': 'off'})
assert argv[argv.index('--kv-cache-dtype') + 1] == 'fp8_e4m3' and env['XTIER_QSA_FP4'] == '0'
env, argv = console.build('qwen38', {**console.defaults('qwen38'), 'kv': 'fp8', 'chunk': '32768'})
assert argv[argv.index('--chunked-prefill-size') + 1] == '24576' and ',27' not in env['XTIER_CACHE_FULL']
env, argv = console.build('glm53', console.defaults('glm53'))
assert '--speculative-use-rejection-sampling' in argv and env['XTIER_CACHE_SLOTS'] == '58'
assert env['XTIER_CACHE_ELASTIC'] == '1'
env, argv = console.build('glm53', {**console.defaults('glm53'), 'packs': 'off'})
assert '--speculative-use-rejection-sampling' not in argv and env['XTIER_FP8_WEIGHTS'] == '0'
assert 'XTIER_CACHE_ELASTIC' not in env
env, argv = console.build('ds41', console.defaults('ds41'))
assert env['DSPARK'] == '1' and env['BATCH'] == '22528' and env['XTIER_ENGRAM_CACHE_GIB'] == '4'
assert env['XTIER_CACHE_ELASTIC'] == '1' and env['XTIER_CACHE_ELASTIC_MAX_SLOTS'] == '68'
assert 'XTIER_CACHE_ELASTIC' not in console.build('ds41', {**console.defaults('ds41'), 'packs': 'off'})[0]

# the page
data = {**console.console_data(), 'status': console.status()}
json.dumps(data)
page = ui.page_html(data)
assert 'id=xtier-console' in page and 'xtierConsoleHooks' in page and 'build' not in json.dumps(data['models'])
assert 'xt-switch' in page and 'role=switch' in page

# the standalone server
httpd = ThreadingHTTPServer(('127.0.0.1', 0), server.make_handler(console, server.Jobs(console)))
threading.Thread(target=httpd.serve_forever, daemon=True).start()
base = f'http://127.0.0.1:{httpd.server_address[1]}'


def call(path, body=None, headers=None):
    req = urllib.request.Request(base + path, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


assert call('/')[0] == 200
status, body = call('/api/status')
assert status == 200 and json.loads(body)['state'] == 'offline'
assert call('/api/start', b'model=qwen38')[0] == 403   # no X-Requested-With: refused
status, body = call('/api/start', b'model=nope', {'X-Requested-With': 'fetch'})
assert status == 400 and not json.loads(body)['ok']
status, body = call('/api/job')
assert status == 200 and json.loads(body)['state'] == 'idle'
httpd.shutdown()

# the warm-up before "online": sampled chat requests of growing prompts, each to its end; a failure is reported
seen = []


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        seen.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        self.send_response(500 if len(seen) > 3 else 200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, *args):
        pass


fake = ThreadingHTTPServer(('127.0.0.1', 0), FakeOpenAI)
threading.Thread(target=fake.serve_forever, daemon=True).start()
deadline = time.monotonic() + 60
assert core.warm_up(fake.server_address[1], 'Some-Model', deadline) == ''
assert [p for p, _ in seen] == ['/v1/chat/completions'] * 3
sizes = [len(b['messages'][0]['content']) for _, b in seen]
assert sizes == sorted(sizes) and all(b['model'] == 'Some-Model' and b['max_tokens'] == 8 and b['top_k'] > 0
                                     and 0 < b['top_p'] < 1 and b['temperature'] > 0 for _, b in seen), seen
assert 'HTTP Error 500' in core.warm_up(fake.server_address[1], 'Some-Model', deadline)
fake.shutdown()
print(f'console ok: {len(models)} models, {combos} option combinations built, warm-up requests checked')
