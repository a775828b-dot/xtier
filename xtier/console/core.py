"""The console's logic: models, options, launching one server, following it.

Models come from configs/<name>/console.py, one per "model x hardware" launch configuration: a MODEL dict (the card,
the exposed options with their defaults = the measured configuration, README speeds) and build(options, port) ->
(environment additions, argv). One server at a time; it listens on the model's own port (SGLang's OpenAI-compatible
API) and outlives the console: with a user systemd it runs in a transient scope, otherwise in its own session.
"""

import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / 'configs'
# the MODEL fields the page gets (build functions and launch details stay on the server)
PUBLIC = ('name', 'code', 'port', 'served', 'spec', 'arch', 'hardware', 'bench', 'notes')


def common_options(spec, threads=('112', '96', '64'), contexts=('524288', '393216', '262144')):
    """The options every model has: the packs (off = the README "官方 main" configuration), speculative decoding, the
    context (= the KV pool) and the CPU engine's threads (the first context and thread count are the measured ones)."""
    return [
        {'key': 'packs', 'label': '加速包', 'default': 'on',
         'choices': [['on', '开 · 实测配置'], ['off', '关 · 官方 main 基线']],
         'hint': '关：只保留跑起来必需的修正，与 README「官方 main」那行同一配置'},
        {'key': 'spec', 'label': f'{spec} 投机解码', 'default': 'on', 'choices': [['on', '开'], ['off', '关']]},
        {'key': 'context', 'label': '上下文 / KV 池', 'default': contexts[0],
         'choices': [[c, f'{int(c) // 1024}K'] for c in contexts]},
        {'key': 'threads', 'label': 'CPU 专家线程', 'default': threads[0],
         'choices': [[threads[0], f'{threads[0]} · 实测']] + [[t, t] for t in threads[1:]]},
    ]


def load_models(names=None, configs=CONFIGS, ports=None):
    """{id: model} from configs/<name>/console.py (every one, or the directories in names) in MODEL['order'];
    ports overrides the listening port by model id."""
    dirs = [Path(configs) / n for n in names] if names else sorted(p.parent for p in Path(configs).glob('*/console.py'))
    models = {}
    for d in dirs:
        module_name = 'xtier_console_config_' + re.sub(r'\W', '_', d.name)
        spec = importlib.util.spec_from_file_location(module_name, d / 'console.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = dict(module.MODEL, config=d.name, build=module.build, tags=getattr(module, 'tags', None))
        if model['id'] in models:
            raise ValueError(f"model id {model['id']} in both {models[model['id']]['config']} and {d.name}")
        if ports and model['id'] in ports:
            model['port'] = ports[model['id']]
        models[model['id']] = model
    return dict(sorted(models.items(), key=lambda kv: (kv[1].get('order', 100), kv[0])))


# --- probes ---

def _get_ok(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def health(port, timeout=2.0):
    """SGLang's /health: runs a one-token generation first, which takes seconds right after start-up."""
    return _get_ok(f'http://127.0.0.1:{port}/health', timeout)


def models_ok(port, timeout=2.0):
    """/v1/models answers: a cheap liveness probe."""
    return _get_ok(f'http://127.0.0.1:{port}/v1/models', timeout)


# the warm-up prompts: this sentence (about 17 tokens) repeated, sized for the hybrid, split and GPU prefill regimes
_WARM_TEXT = 'The console sends a few requests through the server before it reports the model online. '
WARM_REPEATS = (20, 150, 400)


def warm_up(port, served, deadline):
    """Sampled chat requests of about 340, 2,500 and 6,800 prompt tokens, one after the other, each to its end. They run
    what /health's one-token generation does not reach, and so what a server builds on its first use: FlashInfer's
    sampling module, the kernels of each prefill regime, Triton kernels changed since the last start. With a cold JIT
    cache that is a minute or more (10-26: the first client request after a start waited 88 s). Until the monotonic
    deadline; returns '' or why a request failed (the server stays up either way)."""
    for tag, repeat in zip('ABC', WARM_REPEATS):
        body = {'model': served, 'max_tokens': 8, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 40,
                'messages': [{'role': 'user', 'content': f'Warm-up {tag}. ' + _WARM_TEXT * repeat + 'Reply with ok.'}]}
        req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=max(30.0, deadline - time.monotonic())) as resp:
                resp.read()
        except Exception as exc:   # noqa: BLE001 - a failed warm-up only leaves the first-use work to the first request
            return f'{type(exc).__name__}: {exc}'[:300]
    return ''


def _number(text):
    try:
        return float(text)
    except ValueError:
        return 0.0


def gpus():
    """The GPUs together: name, count, memory used / total (GiB), mean utilization (%), power (W); {} without nvidia-smi."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
        rows = [[x.strip() for x in line.split(',')] for line in out.splitlines() if line.count(',') >= 4]
        if not rows:
            return {}
        return {'name': rows[0][0].replace('NVIDIA ', ''), 'count': len(rows),
                'mem_used': round(sum(_number(r[1]) for r in rows) / 1024, 1),
                'mem_total': round(sum(_number(r[2]) for r in rows) / 1024, 1),
                'util': round(sum(_number(r[3]) for r in rows) / len(rows)),
                'power': round(sum(_number(r[4]) for r in rows))}
    except (OSError, subprocess.TimeoutExpired):
        return {}


def _user_systemd():
    """A user service manager to run servers in transient scopes (they then outlive the console and its session)."""
    if not shutil.which('systemd-run'):
        return False
    try:
        return subprocess.run(['systemctl', '--user', 'show-environment'], capture_output=True,
                              timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _systemctl(*args, timeout=30):
    return subprocess.run(['systemctl', '--user', *args], capture_output=True, text=True, timeout=timeout)


def scope_active(unit):
    try:
        return _systemctl('is-active', f'{unit}.scope', timeout=5).stdout.strip() in ('active', 'activating')
    except (OSError, subprocess.TimeoutExpired):
        return False


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


# --- the server log: loading phases and decode telemetry ---

_SHARDS = re.compile(r'loading shards:\s+(\d+)% Completed \| (\d+)/(\d+)')
_LAYER = re.compile(r'xtier: layer (\d+): \d+ experts')
# "#full token" on hybrid pools (DeepSeek, GLM, Qwen), "#token" on one pool (MiniMax)
_DECODE = re.compile(r'Decode batch, #running-req: (\d+), #(?:full )?token: (\d+)(?:.*?accept len: ([\d.]+))?.*?'
                     r'gen throughput \(token/s\): ([\d.]+)')
_PREFILL = re.compile(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)')
_VERSION = re.compile(r'\[xtier\] ([\w.+-]+): hooks installed')
# the log tail: the time of day only, the polling requests dropped, the batch lines in short
_STAMP = re.compile(r'^\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\]\s*')
_ACCESS = re.compile(r'INFO:\s+\S+ - "(\w+) (\S+) HTTP/[\d.]+" (\d+)')
_POLL = re.compile(r'"GET /(?:v1/models|health|metrics|get_model_info|get_server_info)\b')
_QUEUE = re.compile(r'#queue-req: (\d+)')
_PREFILL_TPS = re.compile(r'input throughput \(token/s\): ([\d.]+)')
_CACHE = re.compile(r'XTier cache: \d+ windows, hit ([\d.]+), ([\d.]+) distinct experts and (\d+) us CPU per layer call')
# the same lines with their time, to tell a running server from an idle one (SGLang logs a batch line only while busy)
_AT = r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] '
_DECODE_AT = re.compile(_AT + _DECODE.pattern, re.M)
_PREFILL_AT = re.compile(_AT + r'Prefill batch, #new-seq: (\d+).*?#running-req: (\d+)', re.M)


def _fresh(stamps, now, floor, cap=60.0):
    """Whether the newest of these log times is recent: within 2.5 of their usual gaps (at least floor seconds)."""
    times = [time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S')) for s in stamps[-6:]]
    gaps = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
    interval = gaps[len(gaps) // 2] if gaps else floor
    return now - times[-1] <= min(cap, max(floor, 2.5 * interval))


def _count(n):
    n = int(n)
    return str(n) if n < 1000 else f'{n / 1000:.1f}K' if n < 10000 else f'{n / 1000:.0f}K'


def _note(path, text):
    """One console line in the server's log."""
    try:
        with open(path, 'ab') as fh:
            fh.write(f"[xtier-console] {time.strftime('%F %T')} {text}\n".encode())
    except (OSError, TypeError):
        pass


def _read_tail(path, size=262144):
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            return fh.read().decode('utf-8', 'replace').replace('\r', '\n')
    except (OSError, TypeError):
        return ''


def _short(line):
    """A log line for the page: HH:MM:SS, SGLang's batch lines in short, an access line as method, path, status."""
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
    stamp = _STAMP.match(line)
    head, line = (stamp.group(1) + ' ', line[stamp.end():]) if stamp else ('', line)
    decode, prefill, access, cache = _DECODE.search(line), _PREFILL.search(line), _ACCESS.search(line), _CACHE.search(line)
    if decode:
        running, tokens, accept, tps = decode.groups()
        line = (f'decode {running} 请求 · {_count(tokens)} token' + (f' · 接受 {accept}' if accept else '')
                + f' · {float(tps):.1f} tok/s')
        queue = _QUEUE.search(decode.string)
        if queue and queue.group(1) != '0':
            line += f' · 排队 {queue.group(1)}'
    elif prefill:
        seqs, new, cached = prefill.groups()
        tps = _PREFILL_TPS.search(line)
        line = (f'prefill {seqs} 请求 · 新 {int(new):,} · 命中 {int(cached):,}'
                + (f' · {float(tps.group(1)):,.0f} tok/s' if tps else ''))
    elif access:
        line = ' '.join(access.groups())
    elif cache:
        hit, experts, us = cache.groups()
        line = f'热专家命中 {hit} · 每层 {experts} 个 CPU 专家 · CPU {us} µs / 层'
    return (head + line)[:240]


def log_tail(path, n=5):
    lines = [ln.strip() for ln in _read_tail(path, 16384).splitlines()
             if ln.strip() and 'Multi-thread loading' not in ln and not _POLL.search(ln)]
    return [_short(ln) for ln in lines[-n:]]


def _progress(text):
    """(phase, label, percent) of a loading server from its log: the latest phase reached wins."""
    if 'The server is fired up' in text:   # HTTP is up, the first /health generation still to pass
        return 'graphs', '预热', 97
    if 'Capturing batches' in text or 'CUDA graph begin' in text:
        return 'graphs', '捕获 CUDA 图', 94 if 'Capture draft' in text else 88
    if 'KV Cache is allocated' in text or 'xtier cache: ' in text:
        return 'pool', '建专家池 / KV 池', 80
    loads = text.count('Load weight begin')
    if loads >= 2:
        return 'draft', '加载草稿模型', 74
    if loads == 1:
        shards = _SHARDS.findall(text)
        return 'weights', '加载权重', 8 + (int(int(shards[-1][0]) * 0.62) if shards else 0)
    return 'boot', '启动进程', 3


class Console:
    """One XTier server at a time for the models given (load_models). state_dir keeps the state (xtier-state.json),
    the last options per model (xtier-launcher-settings.json) and, unless log_dir says otherwise, the server logs.
    runner: 'systemd' (a transient user scope named unit), 'setsid' (own session; stopped through its process group)
    or 'auto'. guard_gib: tools/mem_guard.sh beside the server stops it when the host's available memory drops below
    that many GiB (0 = no guard)."""

    def __init__(self, models, state_dir, *, log_dir=None, runner='auto', unit='xtier-console', guard_gib=20,
                 cwd=None):
        self.models = models
        self.state_dir = Path(state_dir)
        self.log_dir = Path(log_dir) if log_dir else self.state_dir / 'logs'
        self.state_file = self.state_dir / 'xtier-state.json'
        self.settings_file = self.state_dir / 'xtier-launcher-settings.json'
        self.runner = ('systemd' if _user_systemd() else 'setsid') if runner == 'auto' else runner
        self.unit, self.guard_gib, self.cwd = unit, guard_gib, Path(cwd or REPO)
        self._waiting = set()   # ports wait_ready() is still bringing up: status() says loading until it is done
        self._status = {'t': 0.0, 'body': None}
        self._lock = threading.Lock()

    # --- options ---

    def defaults(self, model):
        return {o['key']: o['default'] for o in self.models[model]['options']}

    def validate(self, model, form):
        """Option values from a form (dict of str or list of str), the default for missing ones; ValueError otherwise."""
        if model not in self.models:
            raise ValueError('未知的 XTier 模型')
        values = {}
        for option in self.models[model]['options']:
            raw = form.get(option['key'], option['default'])
            if isinstance(raw, list):
                raw = raw[0] if raw else option['default']
            raw = str(raw)
            if raw not in {c[0] for c in option['choices']}:
                raise ValueError(f"{option['label']}：不支持的取值 {raw}")
            values[option['key']] = raw
        return values

    def build(self, model, values):
        """(environment additions, argv) of the server for validated values."""
        m = self.models[model]
        return m['build'](values, m['port'])

    def describe(self, model, o):
        """One line for job labels and the status panel."""
        m = self.models[model]
        parts = [m['name'], '加速包' if o.get('packs', 'on') == 'on' else '官方 main']
        if m.get('spec'):
            parts.append(m['spec'] + ('' if o.get('spec', 'on') == 'on' else ' 关'))
        if o.get('context'):
            parts.append(f"{int(o['context']) // 1024}K")
        if m.get('tags'):
            parts += m['tags'](o)
        return 'XTier · ' + ' / '.join(parts)

    def console_data(self):
        """What the page renders: the models' card fields, their options and defaults, the saved choices."""
        return {'models': {k: {f: m[f] for f in PUBLIC if f in m} for k, m in self.models.items()},
                'options': {k: m['options'] for k, m in self.models.items()},
                'defaults': {k: self.defaults(k) for k in self.models},
                'saved': self.load_settings()}

    # --- saved options and state ---

    def load_settings(self):
        saved = _read_json(self.settings_file)
        out = {'model': saved.get('model') if saved.get('model') in self.models else next(iter(self.models), None)}
        for model in self.models:
            try:
                out[model] = self.validate(model, saved.get(model) or {})
            except ValueError:
                out[model] = self.defaults(model)
        return out

    def save_settings(self, model, values):
        saved = _read_json(self.settings_file)   # entries of models not loaded here stay
        saved.update({'model': model, model: values})
        _write_json(self.settings_file, saved)

    def read_state(self):
        return _read_json(self.state_file)

    def write_state(self, state):
        _write_json(self.state_file, state)

    # --- start / stop ---

    def active(self, state=None):
        """Is anything of the server left?"""
        state = self.read_state() if state is None else state
        if state.get('runner', self.runner) == 'systemd':
            return scope_active(self.unit)
        return bool(state.get('pgid')) and _group_alive(state['pgid'])

    def running(self):
        """The state of the running server, or None."""
        state = self.read_state()
        return state if state and self.active(state) else None

    def _wrap(self, unit, argv):
        if self.runner != 'systemd':
            return argv
        return ['systemd-run', '--user', '--scope', '--quiet', '--collect', f'--unit={unit}',
                '--property=KillMode=control-group', '--property=TimeoutStopSec=90', *argv]

    def start(self, model, values):
        """Launch the server of model (validated values), with the host memory guard beside it; returns the state."""
        self.stop()
        env_add, argv = self.build(model, values)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f"xtier-{model}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        # the console's environment minus anything that looks like a credential
        env = {k: v for k, v in os.environ.items() if not re.search(r'KEY|TOKEN|SECRET|PASSW', k, re.I)}
        env.update(PYTHONUNBUFFERED='1', **env_add)
        label = self.describe(model, values)
        with log.open('ab') as fh:
            fh.write(f"[xtier-console] {time.strftime('%F %T')} {label}\n".encode())
            fh.write(('[xtier-console] ' + ' '.join(f'{k}={v}' for k, v in env_add.items()) + ' ' + ' '.join(argv)
                      + '\n').encode())
            fh.flush()
            proc = subprocess.Popen(self._wrap(self.unit, argv), stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, env=env, cwd=str(self.cwd), start_new_session=True)
        if self.runner == 'systemd':   # the scope shows up once systemd-run has registered it: until then active() says no
            for _ in range(100):
                if scope_active(self.unit) or proc.poll() is not None:
                    break
                time.sleep(0.1)
        threading.Thread(target=proc.wait, name='xtier-reaper', daemon=True).start()
        m = self.models[model]
        state = {'model': model, 'options': values, 'label': label, 'runner': self.runner, 'pid': proc.pid,
                 'pgid': proc.pid, 'port': m['port'], 'served': m['served'], 'log': str(log),
                 'started_at': time.time(), 'ready_at': None, 'stopped_at': None, 'error': ''}
        if self.guard_gib:
            with log.with_suffix('.guard.log').open('ab') as gfh:
                guard = subprocess.Popen(
                    self._wrap(self.unit + '-guard', [str(REPO / 'tools/mem_guard.sh'), str(proc.pid),
                                                      str(self.guard_gib), '1']),
                    stdin=subprocess.DEVNULL, stdout=gfh, stderr=subprocess.STDOUT, start_new_session=True)
            threading.Thread(target=guard.wait, name='xtier-guard-reaper', daemon=True).start()
            state['guard_pgid'] = guard.pid
        self.write_state(state)
        return state

    def stop(self, reason=''):
        """Stop the server (if any) and its guard; True when nothing of the server is left."""
        state = self.read_state()
        was = bool(state) and self.active(state)
        if state.get('runner', self.runner) == 'systemd':
            for unit in (self.unit, self.unit + '-guard'):
                try:
                    _systemctl('stop', f'{unit}.scope', timeout=120)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        else:
            for key in ('pgid', 'guard_pgid'):
                if state.get(key):
                    _signal_group(state[key], signal.SIGTERM)
        for _ in range(90):
            if not state or not self.active(state):
                break
            time.sleep(1)
        else:
            if state.get('pgid'):
                _signal_group(state['pgid'], signal.SIGKILL)
                time.sleep(2)
        if was:
            state.update(stopped_at=time.time(), error=reason or state.get('error', ''))
            self.write_state(state)
        return not (state and self.active(state))

    def wait_ready(self, state, timeout):
        """Until /health passes and the warm-up requests ran (True, ''), the server exits or timeout seconds pass
        (False, why). The page shows the warm-up as the last loading phase ("预热")."""
        deadline = time.monotonic() + timeout
        self._waiting.add(state['port'])
        try:
            while time.monotonic() < deadline:
                if not self.active(state):
                    return False, '服务提前退出：' + ' | '.join(log_tail(state.get('log'), 6))
                if health(state['port']):
                    t0 = time.monotonic()
                    why = warm_up(state['port'], state.get('served'), deadline)
                    _note(state.get('log'), f'warm-up {"failed after" if why else "done in"} '
                                            f'{time.monotonic() - t0:.1f} s{": " + why if why else ""}')
                    state['ready_at'] = time.time()
                    self.write_state(state)
                    return True, ''
                time.sleep(3)
            return False, f'启动超时（{timeout // 60} 分钟）'
        finally:
            self._waiting.discard(state['port'])

    def wait_gpu_free(self, max_gib=2.0, timeout=60):
        """After a stop: until the GPUs hold at most max_gib (memory of other processes stays); True when they do."""
        deadline = time.monotonic() + timeout
        while True:
            g = gpus()
            if not g or g['mem_used'] <= max_gib:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(2)

    # --- status for the page ---

    def status(self):
        with self._lock:
            if self._status['body'] is not None and time.monotonic() - self._status['t'] < 1.5:
                return self._status['body']
        state = self.read_state()
        body = {'state': 'offline', 'model': state.get('model'), 'label': state.get('label', ''), 'gpu': gpus(),
                'port': state.get('port'), 'served': state.get('served')}
        if state and self.active(state):
            text = _read_tail(state.get('log'))
            version = _VERSION.findall(text)
            body.update(started_at=state.get('started_at'), options=state.get('options'),
                        elapsed=round(time.time() - state.get('started_at', time.time())),
                        version=version[-1] if version else '', layers=len(set(_LAYER.findall(text))))
            # online once wait_ready saw /health pass; /v1/models when nothing here waits (the console restarted)
            if state.get('ready_at') or (state.get('port') not in self._waiting and models_ok(state.get('port'))):
                decodes = _DECODE_AT.findall(text)[-40:]
                prefills = _PREFILL_AT.findall(text)
                body.update(state='online', progress=100, phase='online', phase_label='在线')
                now = time.time()
                if decodes:
                    stamp, running, tokens, accept, tps = decodes[-1]
                    spark = [float(d[4]) for d in decodes]
                    if _fresh([d[0] for d in decodes], now, 6.0):
                        body.update(running=int(running), tokens=int(tokens), accept=float(accept) if accept else None,
                                    tps=float(tps), spark=spark)
                    else:   # no decode line for a while: nothing decodes (a prefill may run, below)
                        body.update(running=0, tokens=int(tokens), accept=None, tps=0.0, spark=spark + [0.0])
                if prefills and _fresh([p[0] for p in prefills], now, 20.0) and (
                        not decodes or prefills[-1][0] >= decodes[-1][0]):
                    # a prefill after the newest decode line: its requests and those it joins run
                    body['running'] = int(prefills[-1][1]) + int(prefills[-1][2])
                    body.setdefault('tps', 0.0)
                last = _PREFILL.findall(text)
                if last:
                    body['last_prefill'] = int(last[-1][1])
                if state.get('ready_at'):
                    body['load_s'] = round(state['ready_at'] - state['started_at'])
            else:
                phase, label, pct = _progress(text)
                body.update(state='loading', progress=pct, phase=phase, phase_label=label)
            body['tail'] = log_tail(state.get('log'))
        elif state.get('started_at') and not state.get('stopped_at'):
            body.update(state='failed', error=state.get('error') or '进程已退出', tail=log_tail(state.get('log'), 6))
        with self._lock:
            self._status.update(t=time.monotonic(), body=body)
        return body
