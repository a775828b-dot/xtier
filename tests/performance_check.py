"""HTTP benchmark contract checks without GPU or model inference."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.console import perf


class Handler(BaseHTTPRequestHandler):
    requests = 0
    fail = False
    def log_message(self, *args):
        pass
    def do_GET(self):
        data = {'context_length': 131072, 'max_running_requests': 2}
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(data).encode())
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        Handler.requests += 1
        self.send_response(200); self.end_headers()
        events = [{'choices': [{'delta': {'content': 'A'}}]}, {'choices': [{'delta': {'content': 'B'}}]},
                  {'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': 8}}]
        if Handler.fail:
            events = [{'error': {'message': 'test failure'}}]
        for event in events:
            self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode()); self.wfile.flush(); time.sleep(.005)
        self.wfile.write(b'data: [DONE]\n\n')


class CopyAccuracyTests(unittest.TestCase):
    def test_sps_table_path_does_not_imply_planner_activity(self):
        info = {'speculative_dspark_sps_table_path': 'table.json', 'internal_states': [
            {'dspark_info_record': {'mode': 'static', 'gamma': 5, 'records': [
                {'forward_ct': 10, 'budget': None}]}}]}
        result = perf.dspark_evidence(info, [9])
        self.assertEqual(result['ranks'][0]['planned_steps'], 0)

    def test_sps_evidence_excludes_warmup_and_counts_actual_decisions(self):
        info = {'internal_states': [{'dspark_info_record': {'mode': 'compact', 'gamma': 5,
            'simulate_acc_len': None, 'records': [
                {'forward_ct': 10, 'budget': 1, 'predicted_step_ms': 20, 'num_running_reqs': 1},
                {'forward_ct': 11, 'budget': 3, 'predicted_step_ms': 30, 'num_running_reqs': 1},
                {'forward_ct': 12, 'budget': 5, 'predicted_step_ms': 40, 'num_running_reqs': 1}]}}]}
        result = perf.dspark_evidence(info, [10])['ranks'][0]
        self.assertEqual(result['observed_steps'], 2)
        self.assertEqual(result['planned_steps'], 2)
        self.assertEqual(result['reduced_steps'], 1)
        self.assertEqual(result['budget_counts'], {'3': 1, '5': 1})

    def test_exact_match_counts_cases_not_matching_characters(self):
        result = perf.copy_summary([
            {'src': 'a', 'out': 'a', 'diffs': []},
            {'src': 'b', 'out': '\nb\n', 'diffs': []},
            {'src': 'cd', 'out': 'c', 'diffs': [['delete', 1, 2, 1, 1]]},
            {'src': 'e', 'out': '', 'diffs': [], 'error': 'HTTP 500'},
        ])
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['strict_accuracy'], .25)
        self.assertEqual(result['errors'], 1)
        self.assertFalse(result['rows'][-1]['exact'])

    def test_empty_samples_are_unscored(self):
        self.assertIsNone(perf.copy_summary([])['accuracy'])


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = 'http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()
    def test_stream_usage_and_content(self):
        r = perf.stream(self.url, 'test', 'prompt', 128)
        self.assertIsNone(r['error']); self.assertEqual(r['output_tokens'], 8)
        self.assertEqual(r['content'], 'AB'); self.assertGreater(r['decode_tps'], 0)
        self.assertGreater(r['latency_s'], r['ttft_s'])
    def test_failure_not_success(self):
        Handler.fail = True
        try:
            r = perf.stream(self.url, 'test', 'prompt', 128)
            self.assertIn('test failure', r['error'])
            result = perf.round_result([r], 2)
            self.assertEqual(result['errors'], 1); self.assertEqual(result['aggregate_tps'], 0)
        finally:
            Handler.fail = False
    def test_concurrency_cap_skips_and_queue_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(perf, 'telemetry', return_value={'time':0,'metrics':{}}):
            args = argparse.Namespace(kind='concurrency', url=self.url, model='test', tokens=128,
                                      rounds=1, steps=1, queue=False, log=None, out=tmp+'/out.json')
            r = perf.run(args)
            self.assertEqual([x['concurrency'] for x in r['rounds'] if x.get('skipped')], [4,8,16])
            args.queue = True
            r = perf.run(args)
            self.assertEqual([x['concurrency'] for x in r['rounds'] if x.get('queued')], [4,8,16])
            self.assertFalse(any(x.get('errors') for x in r['rounds']))
    def test_throughput_includes_full_wall(self):
        row = dict(error=None, output_tokens=100, decode_tps=100, ttft_s=9, latency_s=10)
        self.assertEqual(perf.round_result([row], 10)['aggregate_tps'], 10)
    def test_sampled_energy_and_counter_reset(self):
        samples=[{'time':0,'metrics':{'sglang:scheduler_stage_seconds_total{stage="a"}':5},'gpus':[{'power_w':100}]},
                 {'time':2,'metrics':{'sglang:scheduler_stage_seconds_total{stage="a"}':1},'gpus':[{'power_w':200}]}]
        r=perf.observations(samples, [])
        self.assertEqual(r['sampled_gpu_energy_j'],300); self.assertEqual(r['scheduler_stage_delta_s'],{})


class TelemetryAvailabilityTests(unittest.TestCase):
    def test_offline_hardware_does_not_poll_inference_or_invent_counters(self):
        with patch.object(perf, 'metrics') as fetch_metrics, patch.object(
                perf.subprocess, 'check_output', return_value='78, 600, 0, 35000, 86000, 2400, 12000, 50'):
            sample = perf.telemetry()
        fetch_metrics.assert_not_called()
        self.assertEqual(sample['gpus'][0]['power_w'], 78)
        self.assertEqual(sample['metrics'], {})
        self.assertIsNone(sample['cpu_window'])
        self.assertEqual(sample['errors'], [])
        self.assertGreater(sample['time'], 0)

    def test_failed_metrics_keep_hardware_observations(self):
        with patch.object(perf, 'metrics', side_effect=OSError('not ready')), patch.object(
                perf.subprocess, 'check_output', return_value='80, 600, 3, 36000, 86000, 2400, 12000, 51'):
            sample = perf.telemetry('http://127.0.0.1:1')
        self.assertEqual(sample['gpus'][0]['memory_mib'], 36000)
        self.assertEqual(sample['metrics'], {})
        self.assertEqual(sample['errors'], ['metrics: not ready'])

    def test_loading_keeps_hardware_and_readiness_ignores_gpu_sampling_failure(self):
        from types import SimpleNamespace
        from xtier.console.perf_jobs import Performance
        with tempfile.TemporaryDirectory() as tmp:
            state = {'port': 18190, 'started_at': 1, 'log': '/model.log'}
            status = {'state': 'loading'}
            console = SimpleNamespace(state_dir=Path(tmp), cwd=Path(tmp),
                                      read_state=lambda: state, status=lambda: status)
            runner = Performance(console)
            with patch.object(perf, 'telemetry', return_value={
                    'time': 1, 'metrics': {}, 'gpus': [{'power_w': 78}], 'errors': []}) as sample:
                loading = runner.snapshot()
            sample.assert_called_once_with(None, None)
            self.assertFalse(loading['ready'])
            self.assertEqual(loading['live']['gpus'][0]['power_w'], 78)
            self.assertEqual(loading['live']['service_state'], 'loading')
            status['state'] = 'online'
            with patch.object(perf, 'telemetry', return_value={
                    'time': 2, 'metrics': {}, 'gpus': [], 'errors': ['GPU: unavailable']}) as sample:
                online = runner.snapshot()
                busy = runner.snapshot(management_busy=True)
            sample.assert_called_once_with('http://127.0.0.1:18190', '/model.log')
            self.assertTrue(online['ready'])
            self.assertFalse(busy['ready'])
            self.assertEqual(busy['blocked_reason'], '请等待启动或安装任务结束')


class LifecycleTests(unittest.TestCase):
    def test_cancel_reaps_client_without_stopping_model(self):
        from types import SimpleNamespace
        from xtier.console.perf_jobs import Performance
        with tempfile.TemporaryDirectory() as tmp:
            console = SimpleNamespace(state_dir=Path(tmp), cwd=Path(tmp))
            runner = Performance(console)
            state = {'served': 'test', 'options': {'threads': '112'}}
            with patch.object(runner, 'command', return_value=([sys.executable, '-c', 'import time; time.sleep(60)'], state)):
                runner.start({"kind": ["decode"], "tokens": ["128"]})
            state["options"]["threads"] = "64"
            pending = runner.report(runner.job["id"])
            self.assertTrue(pending["pending"])
            self.assertEqual(pending["launch_snapshot"]["configuration"]["threads"], "112")
            self.assertEqual(pending["launch_snapshot"]["test_options"], {"kind": "decode", "tokens": "128"})
            proc = runner.proc
            runner.cancel()
            proc.wait(timeout=5)
            for _ in range(50):
                if runner.job['state'] == 'cancelled':
                    break
                time.sleep(.02)
            self.assertEqual(runner.job['state'], 'cancelled')
            self.assertFalse(runner.running())
            with self.assertRaises(ValueError):
                runner.report('../../etc/passwd')



if __name__ == '__main__':
    unittest.main()
