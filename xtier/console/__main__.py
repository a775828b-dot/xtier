"""python -m xtier.console: the XTier console as a small web server.

    python -m xtier.console [--host 127.0.0.1] [--port 8100] [--config NAME ...] [--state-dir DIR]
                            [--runner auto|systemd|setsid] [--unit xtier-console] [--mem-guard-gib 20]

Pick a model (configs/*/console.py), adjust its options (defaults = the measured configuration), launch / stop; the
page follows the loading phases and the decode telemetry. The model server listens on its own port. The console
listens on 127.0.0.1 by default; from another machine use an SSH tunnel (ssh -L 8100:127.0.0.1:8100 host).
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import core, ui

API = {'start': 'api/start', 'stop': 'api/stop', 'status': 'api/status', 'job': 'api/job'}


class Jobs:
    """The launch in the background (stop the running server, start, wait until ready), one at a time."""

    def __init__(self, console):
        self.console = console
        self.lock = threading.Lock()
        self.job = {'id': 0, 'state': 'idle'}

    def launch(self, model, values):
        with self.lock:
            if self.job['state'] == 'running':
                return None, '已有启动任务在进行，等它结束或先停止'
            self.job = {'id': self.job['id'] + 1, 'state': 'running', 'label': self.console.describe(model, values),
                        'message': '', 'error': '', 'cancelled': False}
            job = self.job
        threading.Thread(target=self._run, args=(job, model, values), name='xtier-launch', daemon=True).start()
        return job, ''

    def _run(self, job, model, values):
        c, m = self.console, self.console.models[model]
        try:
            c.stop()
            if not c.wait_gpu_free():
                raise RuntimeError(f"GPU 上还有 {core.gpus().get('mem_used')} GiB 被其他进程占用，先停掉再启动")
            if job['cancelled']:
                raise RuntimeError('已手动停止')
            state = c.start(model, values)
            c.save_settings(model, values)
            ok, detail = c.wait_ready(state, m['ready_timeout'])
            if job['cancelled']:
                raise RuntimeError('已手动停止')
            if not ok:
                c.stop(detail)
                raise RuntimeError(detail)
            job.update(state='done', message=f"{job['label']} 已就绪：http://127.0.0.1:{m['port']}/v1，模型名 {m['served']}")
        except Exception as exc:   # the page shows it
            job.update(state='failed', error=str(exc))

    def stop(self):
        with self.lock:
            if self.job['state'] == 'running':
                self.job['cancelled'] = True
        return self.console.stop('手动停止')


def make_handler(console, jobs):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'xtier-console'

        def log_message(self, fmt, *args):   # the page polls every 2 s
            pass

        def _send(self, code, body, ctype='application/json; charset=utf-8'):
            data = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, value, code=200):
            self._send(code, json.dumps(value, ensure_ascii=False))

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                data = {**console.console_data(), 'status': console.status(), 'api': API}
                return self._send(200, ui.page_html(data), 'text/html; charset=utf-8')
            if path == '/api/status':
                return self._json(console.status())
            if path == '/api/job':
                return self._json({k: v for k, v in jobs.job.items() if k != 'cancelled'})
            self._json({'error': 'not found'}, 404)

        def do_POST(self):
            # a custom header no cross-site form can send: a page elsewhere cannot start or stop servers here
            if self.headers.get('X-Requested-With') != 'fetch':
                return self._json({'ok': False, 'error': 'forbidden'}, 403)
            length = min(int(self.headers.get('Content-Length') or 0), 65536)
            form = parse_qs(self.rfile.read(length).decode('utf-8', 'replace'))
            path = urlparse(self.path).path
            if path == '/api/start':
                model = (form.get('model') or [''])[0]
                try:
                    values = console.validate(model, form)
                except ValueError as exc:
                    return self._json({'ok': False, 'error': str(exc)}, 400)
                job, err = jobs.launch(model, values)
                if job is None:
                    return self._json({'ok': False, 'error': err}, 409)
                return self._json({'ok': True, 'job': job['id'], 'label': job['label']})
            if path == '/api/stop':
                label = console.read_state().get('label') or 'XTier'
                ok = jobs.stop()
                return self._json({'ok': ok, 'message': f'已停止 {label}' if ok else '', 'error': '' if ok else '停止超时'})
            self._json({'ok': False, 'error': 'not found'}, 404)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m xtier.console', description='XTier console: launch and follow '
                                 'one XTier model server at a time (models from configs/*/console.py).')
    ap.add_argument('--host', default='127.0.0.1', help='listen address (default 127.0.0.1)')
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--config', action='append', metavar='NAME',
                    help='configs/NAME to offer (repeatable; default: every configs/*/console.py)')
    ap.add_argument('--state-dir', default=str(Path.home() / '.local/state/xtier-console'),
                    help='state, last options per model and server logs')
    ap.add_argument('--runner', choices=('auto', 'systemd', 'setsid'), default='auto',
                    help='systemd: a transient user scope; setsid: an own session (auto: systemd when available)')
    ap.add_argument('--unit', default='xtier-console', help='name of the systemd scope')
    ap.add_argument('--mem-guard-gib', type=int, default=20,
                    help='stop the server when the host has less available memory (GiB; 0 = no guard)')
    args = ap.parse_args(argv)
    models = core.load_models(args.config)
    if not models:
        sys.exit(f'no console entries (configs/*/console.py) under {core.CONFIGS}')
    console = core.Console(models, args.state_dir, runner=args.runner, unit=args.unit, guard_gib=args.mem_guard_gib)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(console, Jobs(console)))
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        print(f'warning: listening on {args.host} - anyone who can reach it can start and stop model servers')
    print(f"XTier console on http://{args.host}:{args.port}/ ({', '.join(models)}; runner {console.runner}; "
          f'state {args.state_dir})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
