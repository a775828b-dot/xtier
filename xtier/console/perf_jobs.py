"""One cancellable benchmark process and bounded, persistent reports for the console."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

from . import perf


def save(path, value):
    """Publish job metadata atomically for concurrent polling readers."""
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


class Performance:
    """Own benchmark lifecycle separately from the model process; never restart the model."""
    def __init__(self, console):
        self.console = console
        self.directory = console.state_dir/'benchmarks'
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob('*.config.json'):
            previous = json.loads(path.read_text())
            if previous.get('state') in ('running', 'cancelling'):
                previous.update(state='interrupted', note='管理服务重启；结果可能不完整')
                save(path, previous)
        self.gate = threading.Lock()
        self.lock = threading.Lock()
        self.proc = None
        self.job = {'state': 'idle'}
        self.live = {}
        self.live_at = 0
        self.live_source = None
        self.live_lock = threading.Lock()

    def running(self):
        return self.job['state'] in ('running', 'cancelling')

    def command(self, form, output):
        kind = form.get('kind', ['decode'])[0]
        if kind not in perf.KINDS:
            raise ValueError('未知测试类型')
        tokens = int(form.get('tokens', ['512'])[0])
        rounds = int(form.get('rounds', ['3'])[0])
        steps = int(form.get('steps', ['8'])[0])
        if tokens not in (32, 128, 256, 512, 1024) or not 1 <= rounds <= 5 or not 1 <= steps <= 24:
            raise ValueError('测试参数超出范围')
        state = self.console.read_state()
        if self.console.status()['state'] != 'online':
            raise ValueError('请先启动模型，等待服务在线')
        argv = [sys.executable, str(perf.REPO/'xtier/console/perf.py'), '--kind', kind,
                '--url', f"http://127.0.0.1:{state['port']}", '--model', state['served'],
                '--tokens', str(tokens), '--rounds', str(rounds), '--steps', str(steps),
                '--out', str(output)]
        if state.get('log'):
            argv += ['--log', state['log']]
        if form.get('queue', ['0'])[0] == '1':
            argv += ['--queue']
        return argv, state

    def start(self, form):
        with self.lock:
            if self.running():
                raise ValueError('已有测试正在运行')
            key = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            output = self.directory/(key+'.json')
            argv, state = self.command(form, output)
            log = self.directory/(key+'.log')
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            self.job = dict(id=key, state='running', command=shlex.join(argv), log=str(log), output=str(output),
                            configuration=dict(state.get('options') or {}), model=state['served'], started=time.time(),
                            test_options={k: v[0] for k, v in form.items() if k in ('kind', 'tokens', 'rounds', 'steps', 'queue')})
            # Snapshot the selected settings before testing; a later restart must not rewrite attribution.
            save(self.directory/(key+'.config.json'), self.job)
            try:
                with log.open('w') as f:
                    self.proc = subprocess.Popen(argv, cwd=self.console.cwd, env=env, stdout=f,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
            except Exception:
                self.job['state'] = 'failed'
                raise
            threading.Thread(target=self._wait, args=(self.proc, self.job), daemon=True).start()
            return dict(self.job)

    def _wait(self, proc, job):
        code = proc.wait()
        with self.lock:
            if job['state'] == 'cancelling':
                job.update(state='cancelled', returncode=code, finished=time.time())
            if job['state'] == 'running':
                job.update(state='done' if code == 0 else 'failed', returncode=code, finished=time.time())
            save(self.directory/(job['id']+'.config.json'), job)

    def cancel(self):
        with self.lock:
            if self.running() and self.proc:
                self.job.update(state='cancelling', finished=time.time())
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The reaper is about to observe normal exit.
                proc = self.proc
                def reap():
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass  # Group exited after timeout.
                threading.Thread(target=reap, daemon=True).start()
        return {'ok': True, 'note': '已停止测试客户端；已提交到服务的请求可能仍在收尾'}

    def snapshot(self, management_busy=False):
        """Keep hardware observations available while reporting test readiness independently."""
        state = self.console.read_state()
        service_state = self.console.status()['state']
        online = service_state == 'online' and bool(state.get('port'))
        ready = online and not management_busy
        blocked_reason = ('请等待启动或安装任务结束' if management_busy else
                          '请先启动模型，等待服务在线' if not online else '')
        source = (state.get('port'), state.get('started_at'), service_state)
        # Cache samples across clients, but invalidate immediately when the model changes state.
        with self.live_lock:
            if source != self.live_source or time.monotonic()-self.live_at > 3:
                url = f"http://127.0.0.1:{state['port']}" if online else None
                self.live = perf.telemetry(url, state.get('log') if online else None)
                self.live.update(service_state=service_state,
                                 service_id=f"{state.get('port')}:{state.get('started_at')}")
                self.live_source = source
                self.live_at = time.monotonic()
        with self.lock:
            job = dict(self.job)
        if job.get('log'):
            with Path(job['log']).open('rb') as f:
                f.seek(max(0, Path(job['log']).stat().st_size-12000))
                job['tail'] = f.read().decode(errors='replace').splitlines()[-30:]
        history = []
        for p in sorted(self.directory.glob('*.config.json'), reverse=True)[:20]:
            try:
                history.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue  # A concurrent atomicity window can be retried on the next poll.
        return dict(job=job, live=self.live, ready=ready, blocked_reason=blocked_reason,
                    history=history, advice=perf.ADVICE)

    def report(self, key):
        if not __import__('re').fullmatch(r'\d{8}-\d{6}-[a-f0-9]{6}', key):
            raise ValueError('无效结果编号')
        snapshot = json.loads((self.directory/(key+'.config.json')).read_text())
        path = self.directory/(key+'.json')
        if path.is_file():
            report = json.loads(path.read_text())
        else:
            report = {'pending': True, 'rounds': [], 'note': '结果尚未生成' if snapshot['state'] in ('running', 'cancelling')
                      else '测试未产生完整结果；配置快照已保留。'}
        # Older reports only retained the copy counts; recover details from this run's own artifact.
        if report.get('copy_results') and 'rows' not in report['copy_results']:
            raw = self.directory / 'copy' / (key + '-c1.json')
            if raw.is_file():
                report['copy_results'] = perf.copy_summary(json.loads(raw.read_text()))
        report['launch_snapshot'] = snapshot
        return report
