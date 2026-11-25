"""Tiered, diff-aware fast release verification for XTier.

As the supported model roster grows (Qwen3.8, GLM-5.3, DeepSeek V4.1, Hunyuan-4-Flash, ...),
running the full multi-hour release checklist on every change does not scale. This runner
implements a 3-tier verification pyramid:

  Tier 0 (~10 s, CPU only):
    repo_check, planner_check, console_check --fast, check_sglang --strict (+ --emulate sm86
    when core/platform/SGLang hooks changed).
  Tier 1 (~1-3 min, single-layer GPU unit tests):
    Auto-selects affected unit and single-layer checks from `git diff --name-only <base>`.
  Tier 2 (~5-7 min / model, single boot per model):
    Boots a minimal representative model set (1 model for single-pack edits, 2-3 anchor models
    covering NVFP4 + MXFP4 + SWA/DSA + EAGLE/DSpark + Engram when core changes, or --models),
    and runs a compact 4-gate live probe per boot:
      (1) Boot & fatal-log scan (Traceback / OutOfMemory / FATAL / Scheduler hit / does not know)
      (2) Fast verbatim copy (4 chunks, conc=2)
      (3) Fast PPL (2 segments x 4096 tokens)
      (4) Warm decode throughput (bs=1 + bs=2/4) + 64K needle retrieval + 0 OOM retries.

Tier 3 (full 520K/950K, 15-min dsh, abuse_test, 2^31 wrap, SM86 sim, clean checkout rehearsal)
remains in docs/release-checklist.md for major releases and SGLang upgrades.

Examples:
  python tools/fast_verify.py --tier 1 --base origin/main
  python tools/fast_verify.py --tier 2 --base origin/main --models hy4flash,glm53
  python tools/fast_verify.py --tier 2 --base v0.1.8 --dry-run
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FATAL_LOG_RE = re.compile(r'Traceback|OutOfMemoryError|CUDA out of memory|FATAL|Scheduler hit|does not know')

# Tier 3 full release checklist covers the latest 3 flagship models; older/other models validate
# through the Tier 2 fast channel (`--models fast-track`).
TIER3_MODELS = ('inkling2', 'hy4flash', 'ds41')
FAST_TRACK_MODELS = ('qwen38', 'glm53')

# Model metadata for Tier 2 compact live probes (mapped by xtier/console model id).
MODEL_META = {
    'inkling2': {
        'tokenizer': '/data/fast/cold1/models/Inkling-2-Small',
        'copy_args': ['--reasoning-effort', 'low', '--temperature', '0.01'],
        'chat_kwargs': {'reasoning_effort': 'low'},
        'conc': 2,
    },
    'hy4flash': {
        'tokenizer': '/data/fast/nvme1/models/huggingface/atakans/Hy4-flash-NVFP4',
        'copy_args': ['--reasoning-effort', 'no_think'],
        'chat_kwargs': {'reasoning_effort': 'no_think'},
        'conc': 4,
    },
    'glm53': {
        'tokenizer': '/data/fast/nvme1/models/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4',
        'copy_args': [],
        'chat_kwargs': {'enable_thinking': False},
        'conc': 2,
    },
    'ds41': {
        'tokenizer': '/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',
        'copy_args': [],
        'chat_kwargs': {'enable_thinking': False},
        'conc': 4,
    },
    'qwen38': {
        'tokenizer': '/data/fast/nvme1/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4',
        'copy_args': ['--ignore-first-indent'],
        'chat_kwargs': {'enable_thinking': False},
        'conc': 2,
    },
}

# Preset checkpoint directories (mirrors tests/layers.py) so missing checkpoints skip cleanly.
PRESET_PATHS = {
    'glm53': '/data/fast/nvme1/models/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4',
    'qwen38': '/data/fast/nvme1/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4',
    'ds41_1017': '/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017',
    'hy4flash': '/data/fast/nvme1/models/huggingface/atakans/Hy4-flash-NVFP4',
}


def cuda_env(base_env=None):
    env = dict(os.environ if base_env is None else base_env)
    if 'CUDA_HOME' not in env:
        venv_root = Path(sys.prefix)
        for cand in sorted(venv_root.glob('lib/python*/site-packages/nvidia/cu*'), reverse=True):
            if (cand / 'bin' / 'nvcc').exists():
                env['CUDA_HOME'] = str(cand)
                env['PATH'] = f"{cand / 'bin'}:{env.get('PATH', '')}"
                break
    return env


def changed_files(base_ref):
    files = set()
    for cmd in (
        ['git', 'diff', '--name-only', f'{base_ref}...HEAD'],
        ['git', 'diff', '--name-only'],
        ['git', 'diff', '--name-only', '--cached'],
        ['git', 'ls-files', '--others', '--exclude-standard'],
    ):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            files.update(line.strip() for line in r.stdout.splitlines() if line.strip())
    return sorted(f for f in files if not f.startswith('bench/results/'))


def detect_scopes(files):
    scopes = set()
    for f in files:
        if f.startswith(('csrc/', 'xtier/backend/', 'xtier/cache/', 'xtier/tables/')) or f in (
            'xtier/arch.py', 'xtier/__init__.py', 'xtier/hooks.py', 'xtier/plugin.py',
        ):
            scopes.add('core')
        if f.startswith('xtier/cache/cold') or f.startswith('tools/cold_'):
            scopes.add('cold')
        if f.startswith(('xtier/console/', 'configs/')):
            scopes.add('console')
        if f.startswith('xtier/platform/'):
            scopes.add('platform')
        if f.startswith(('xtier/packs/inkling2/', 'configs/inkling2-')):
            scopes.add('inkling2')
        if f.startswith(('xtier/packs/hy4_flash/', 'configs/hy4-flash-')):
            scopes.add('hy4')
        if f.startswith(('xtier/packs/glm5_next/', 'configs/glm53-')):
            scopes.add('glm53')
        if f.startswith(('xtier/packs/deepseek_v4/', 'configs/ds41-')):
            scopes.add('ds41')
        if f.startswith(('xtier/packs/qwen3_next/', 'configs/qwen38-')):
            scopes.add('qwen38')
        if f in ('SGLANG_COMMIT', 'tools/hook_sources.json', 'constraints.txt') or f.startswith('xtier/packs/__init__'):
            scopes.add('hooks')
    if not scopes:
        scopes.add('core')
    return sorted(scopes)


def select_tier1_tests(scopes):
    py = sys.executable
    ordered = []

    def add(name, cmd, preset=None):
        if any(n == name for n, _, _ in ordered):
            return
        ordered.append((name, cmd, preset))

    sc = set(scopes)
    if 'core' in sc:
        add('fp8_weights_check', [py, 'tests/fp8_weights_check.py'])
        add('elastic_check', [py, 'tests/elastic_check.py'])
        add('tables_check', [py, 'tests/tables_check.py'])
        add('prefill_cpu_share_check', [py, 'tests/prefill_cpu_share_check.py'])
        for preset in ('glm53', 'ds41_1017', 'hy4flash'):
            add(f'cache_check:{preset}', [py, 'tests/cache_check.py', '--preset', preset], preset)
            add(f'mailbox_check:{preset}', [py, 'tests/mailbox_check.py', '--preset', preset], preset)
    if 'inkling2' in sc:
        add('inkling2_check', [py, 'tests/inkling2_check.py'])
        add('fp8_weights_check', [py, 'tests/fp8_weights_check.py'])
        add('fp4_decode_check', [py, 'tests/fp4_decode_check.py'])
        add('fp8_decode_check', [py, 'tests/fp8_decode_check.py'])
    if 'hy4' in sc:
        add('hy4_flash_check', [py, 'tests/hy4_flash_check.py'])
        add('fp8_weights_check', [py, 'tests/fp8_weights_check.py'])
        add('cache_check:hy4flash', [py, 'tests/cache_check.py', '--preset', 'hy4flash'], 'hy4flash')
        add('mailbox_check:hy4flash', [py, 'tests/mailbox_check.py', '--preset', 'hy4flash'], 'hy4flash')
    if 'glm53' in sc:
        add('fp8_weights_check', [py, 'tests/fp8_weights_check.py'])
        add('glm_indexer_torch_check', [py, 'tests/glm_indexer_torch_check.py'])
        add('spec_truncate_check', [py, 'tests/spec_truncate_check.py'])
        add('cache_check:glm53', [py, 'tests/cache_check.py', '--preset', 'glm53'], 'glm53')
    if 'ds41' in sc:
        add('ds_sparse_mla_check', [py, 'tests/ds_sparse_mla_check.py'])
        add('ds_mxfp8_rows_check', [py, 'tests/ds_mxfp8_rows_check.py'])
        add('ds_wo_a_check', [py, 'tests/ds_wo_a_check.py'])
        add('ds_w8a16_check', [py, 'tests/ds_w8a16_check.py'])
        add('draft_lmhead_fp8_check', [py, 'tests/draft_lmhead_fp8_check.py'])
        add('cache_check:ds41_1017', [py, 'tests/cache_check.py', '--preset', 'ds41_1017'], 'ds41_1017')
    if 'qwen38' in sc:
        add('qsa_decode_check', [py, 'tests/qsa_decode_check.py'])
        add('qsa_fp4_check', [py, 'tests/qsa_fp4_check.py'])
        add('qsa_fp8_check', [py, 'tests/qsa_fp8_check.py'])
        add('qsa_prefill_check', [py, 'tests/qsa_prefill_check.py'])
        add('hc_mix_check', [py, 'tests/hc_mix_check.py'])
        add('ple_blocks_check', [py, 'tests/ple_blocks_check.py'])
        add('cache_check:qwen38', [py, 'tests/cache_check.py', '--preset', 'qwen38'], 'qwen38')
    if 'platform' in sc:
        add('fp4_decode_check', [py, 'tests/fp4_decode_check.py'])
        add('fp8_decode_check', [py, 'tests/fp8_decode_check.py'])
        add('fp8_encode_check', [py, 'tests/fp8_encode_check.py'])
        add('sm89_block_fp8_check', [py, 'tests/sm89_block_fp8_check.py'])
        add('cpu_image_check', [py, 'tests/cpu_image_check.py'])
    return ordered


def select_tier2_models(scopes, explicit_models):
    if explicit_models:
        out = []
        for raw in explicit_models.split(','):
            tok = raw.strip()
            if not tok:
                continue
            if tok == 'tier3':
                out.extend(TIER3_MODELS)
            elif tok == 'fast-track':
                out.extend(FAST_TRACK_MODELS)
            elif tok == 'all':
                out.extend(MODEL_META.keys())
            else:
                out.append(tok)
        return list(dict.fromkeys(out))
    sc = set(scopes)
    models = []
    if 'inkling2' in sc:
        models.append('inkling2')
    if 'hy4' in sc:
        models.append('hy4flash')
    if 'glm53' in sc:
        models.append('glm53')
    if 'ds41' in sc:
        models.append('ds41')
    if 'qwen38' in sc:
        models.append('qwen38')
    if 'core' in sc:
        # Minimal anchor pair covering NVFP4+EAGLE+FP8 (glm53) and MXFP4+DSpark+Engram (ds41),
        # plus any newly touched model already appended above.
        for anchor in ('glm53', 'ds41'):
            if anchor not in models:
                models.append(anchor)
    if not models:
        models = ['glm53']
    return models


def run_step(name, cmd, env=None, dry_run=False):
    if dry_run:
        print(f'  [DRY-RUN] {name}: {" ".join(cmd)}', flush=True)
        return {'name': name, 'status': 'dry-run', 'sec': 0.0}
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    lines = [ln for ln in (r.stdout or '').splitlines() if ln.strip()]
    tail = lines[-1] if lines else ''
    failed = r.returncode != 0 or any('FAIL' in ln for ln in lines)
    status = 'FAIL' if failed else ('SKIP' if tail.startswith('SKIP:') else 'PASS')
    print(f'  [{status:4s}] {name:28s} ({dt:5.1f}s) {tail[:90]}', flush=True)
    if failed:
        if r.stdout:
            print('--- STDOUT ---\n' + r.stdout[-1200:], flush=True)
        if r.stderr:
            print('--- STDERR ---\n' + r.stderr[-1200:], flush=True)
    return {'name': name, 'status': status, 'sec': round(dt, 2), 'tail': tail}


def req_json(port, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b'{}'
    r = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data, {'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def flush_cache(port):
    r = urllib.request.Request(f'http://127.0.0.1:{port}/flush_cache', method='POST')
    with urllib.request.urlopen(r, timeout=60) as resp:
        resp.read()


def run_live_model(mid, port_override, dry_run=False):
    from xtier.console import core

    if mid not in MODEL_META:
        raise ValueError(f'Unknown Tier 2 model id: {mid}')
    meta = MODEL_META[mid]
    if not Path(meta['tokenizer']).is_dir():
        print(f'  [SKIP] live:{mid} (checkpoint missing: {meta["tokenizer"]})', flush=True)
        return {'model': mid, 'status': 'SKIP', 'reason': 'checkpoint missing'}

    all_models = core.load_models()
    if mid not in all_models:
        raise ValueError(f'Model {mid} not found in configs/*/console.py')
    m = dict(all_models[mid])
    port = port_override or m['port']
    m['port'] = port
    console = core.Console({mid: m}, tempfile.mkdtemp(prefix=f'xtier-fv-{mid}-'), runner='setsid', guard_gib=0)
    defaults = console.defaults(mid)
    env_add, argv = console.build(mid, defaults)
    if dry_run:
        print(f'  [DRY-RUN] live:{mid} port={port} argv={" ".join(argv)} env_keys={sorted(env_add)}', flush=True)
        return {'model': mid, 'status': 'dry-run'}

    env = cuda_env()
    env['XTIER_FAST_LOAD'] = '1'
    env.update(env_add)
    tmpdir = Path(tempfile.mkdtemp(prefix=f'xtier-fv-run-{mid}-'))
    log_path = tmpdir / f'{mid}.log'
    res = {'model': mid, 'port': port, 'status': 'FAIL'}
    t_all = time.perf_counter()
    print(f'\n=== Tier 2 Live Probe: {mid} (port={port}, served={m["served"]}) ===', flush=True)
    log_f = log_path.open('wb')
    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        # 1. Wait ready
        t0 = time.perf_counter()
        ready = False
        while time.perf_counter() - t0 < 900:
            if proc.poll() is not None:
                raise RuntimeError(f'server exited early with rc={proc.returncode} (see {log_path})')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(2)
        if not ready:
            raise TimeoutError(f'{mid} did not become ready within 900 s')
        boot_s = time.perf_counter() - t0
        res['boot_s'] = round(boot_s, 1)
        print(f'  [1/5] READY in {boot_s:.1f}s', flush=True)

        # 2. Fast Copy (4 chunks, conc=2)
        flush_cache(port)
        copy_out = tmpdir / 'copy'
        copy_cmd = [
            sys.executable, 'bench/copy_test.py',
            '--port', str(port), '--model', m['served'],
            '--chunks', '4', '--concurrency', '2',
            '--tag', f'fv-{mid}', '--out', str(copy_out),
            *meta['copy_args'],
        ]
        # Fallback source file if host's default scheduler.py path isn't present
        default_src = Path('/data/fast/workspaces/ds41-1017/source/python/sglang/srt/managers/scheduler.py')
        if not default_src.exists():
            copy_cmd += ['--source', str(ROOT / 'xtier/backend/prefill.py')]
        rc_c = subprocess.run(copy_cmd, cwd=ROOT, capture_output=True, text=True)
        copy_line = (rc_c.stdout.strip().splitlines() or [''])[-1]
        res['copy'] = copy_line
        print(f'  [2/5] Copy (4x conc=2): rc={rc_c.returncode} | {copy_line}', flush=True)

        # 3. Fast PPL (2 segments x 4096)
        flush_cache(port)
        ppl_out = tmpdir / 'ppl'
        ppl_cmd = [
            sys.executable, 'bench/ppl_eval.py',
            '--port', str(port), '--tokenizer', meta['tokenizer'],
            '--tag', f'fv-{mid}', '--segments', '2', '--tokens', '4096', '--out', str(ppl_out),
        ]
        rc_p = subprocess.run(ppl_cmd, cwd=ROOT, capture_output=True, text=True)
        summary_lines = [ln for ln in rc_p.stdout.splitlines() if 'SUMMARY' in ln]
        ppl_summary = summary_lines[-1] if summary_lines else ''
        res['ppl'] = ppl_summary
        print(f'  [3/5] PPL (2x4096): rc={rc_p.returncode} | {ppl_summary}', flush=True)
        if rc_p.returncode != 0:
            raise RuntimeError(f'ppl_eval failed: {rc_p.stderr[-400:]}')

        # 4. Warm decode throughput (bs=1 + bs=conc)
        prompts = [
            'Explain Multi-Head Latent Attention (MLA) and its KV compression math concisely.',
            'Summarize Linux NUMA memory locality and hugepages best practices.',
            'Compare FP4 and FP8 quantization trade-offs for MoE LLM inference.',
            'Write a lock-free SPSC ring buffer in C++20 with acquire/release ordering.',
        ]

        def gen_one(p, max_tok=256):
            st = time.perf_counter()
            out = req_json(port, '/v1/chat/completions', {
                'model': m['served'], 'temperature': 0.7, 'max_tokens': max_tok,
                'chat_template_kwargs': meta['chat_kwargs'],
                'messages': [{'role': 'user', 'content': p}],
            })
            return out['usage']['completion_tokens'], time.perf_counter() - st

        # 1 short warmup + 1 measured bs=1 + 1 measured bs=conc
        gen_one(prompts[0], 192)
        c1, d1 = gen_one(prompts[1], 256)
        bs1_tps = c1 / d1
        conc = meta['conc']
        st = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            rs = list(ex.map(lambda p: gen_one(p, 256), prompts[:conc]))
        wall = time.perf_counter() - st
        bsn_tps = sum(x[0] for x in rs) / wall
        res['bs1_tps'] = round(bs1_tps, 1)
        res[f'bs{conc}_tps'] = round(bsn_tps, 1)
        print(f'  [4/5] Decode: bs=1 {bs1_tps:.1f} tok/s | bs={conc} {bsn_tps:.1f} tok/s agg', flush=True)

        # 5. 64K Needle retrieval + prefill throughput
        flush_cache(port)
        unit = 'Distributed inference telemetry log entry for verification and memory pooling validation. '
        needle = '\n[CATALOG RECORD] The archive catalog ID for Observation-2026 is 8472915.\n'
        prompt_64k = (unit * 2400 + needle + unit * 2400
                      + '\nQuestion: According to the catalog record above, what is the 7-digit archive catalog ID for Observation-2026? Reply with ONLY the 7 digits.')
        st = time.perf_counter()
        nres = req_json(port, '/v1/chat/completions', {
            'model': m['served'], 'temperature': 0.0, 'max_tokens': 192,
            'chat_template_kwargs': meta['chat_kwargs'],
            'messages': [{'role': 'user', 'content': prompt_64k}],
        })
        ndt = time.perf_counter() - st
        nu = nres['usage']
        nans = nres['choices'][0]['message'].get('content') or ''
        npass = '8472915' in nans
        ntps = nu['prompt_tokens'] / ndt
        res['needle_64k'] = {'prompt_tokens': nu['prompt_tokens'], 'tps': round(ntps, 0), 'pass': npass, 'ans': nans[:60]}
        print(f'  [5/5] 64K Needle: {nu["prompt_tokens"]} tok in {ndt:.1f}s ({ntps:.0f} tok/s) pass={npass} ans={nans[:40]!r}', flush=True)

        # Log audit
        log_f.flush()
        log_txt = log_path.read_text(errors='replace')
        fatals = [ln for ln in log_txt.splitlines() if FATAL_LOG_RE.search(ln)]
        retries = log_txt.count('memory mapping failed')
        res['fatal_lines'] = len(fatals)
        res['allocator_retries'] = retries
        res['total_s'] = round(time.perf_counter() - t_all, 1)
        ok = npass and len(fatals) == 0 and retries == 0 and rc_c.returncode == 0
        res['status'] = 'PASS' if ok else 'FAIL'
        print(f'  >>> live:{mid} {res["status"]} in {res["total_s"]}s (fatals={len(fatals)}, alloc_retries={retries})', flush=True)
        if fatals:
            print('  FATAL LOG SAMPLE:', fatals[:5], flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log_f.close()
        time.sleep(2)
    return res


def main():
    ap = argparse.ArgumentParser(description='Tiered, diff-aware fast release verification for XTier')
    ap.add_argument('--tier', type=int, choices=(0, 1, 2), default=2, help='Highest tier to run (0=static, 1=+unit, 2=+live)')
    ap.add_argument('--base', default='origin/main', help='Git ref to diff against (default: origin/main)')
    ap.add_argument('--models', default='', help='Comma-separated Tier 2 models or aliases: tier3, fast-track, all (e.g. hy4flash,glm53)')
    ap.add_argument('--port', type=int, default=0, help='Optional fixed port for Tier 2 servers')
    ap.add_argument('--out', default='', help='Optional JSON summary output path')
    ap.add_argument('--dry-run', action='store_true', help='Print verification plan without executing')
    args = ap.parse_args()

    files = changed_files(args.base)
    scopes = detect_scopes(files)
    t0 = time.perf_counter()
    print(f'=== XTier Fast Verify (tier={args.tier}, base={args.base}) ===', flush=True)
    print(f'Changed files ({len(files)}): {files[:12]}{" ..." if len(files) > 12 else ""}', flush=True)
    print(f'Detected scopes: {scopes}', flush=True)

    py = sys.executable
    env = cuda_env()
    report = {'tier': args.tier, 'base': args.base, 'scopes': scopes, 'tier0': [], 'tier1': [], 'tier2': []}
    any_fail = False

    # --- Tier 0: Static & Contract Gate (<15 s) ---
    print('\n--- Tier 0: Static & Contract Gate ---', flush=True)
    hook_env = dict(env, XTIER='1', XTIER_HY4_HYBRID_SWA='1', XTIER_HY4_KV_FP4='1',
                    XTIER_INKLING2_KV_FP4='1', XTIER_DSPARK_TRUNCATE='0.20',
                    XTIER_DRAFT_LMHEAD_FP8='1', XTIER_INKLING2_MARKOV_PAD2='1',
                    XTIER_ENGRAM_ROWS='nvme', XTIER_FP8_WEIGHTS='1')
    tier0_steps = [
        ('repo_check', [py, 'tools/repo_check.py'], env),
        ('planner_check', [py, 'tests/planner_check.py'], env),
        ('console_check --fast', [py, 'tests/console_check.py', '--fast'], env),
        ('check_sglang --strict', [py, 'tools/check_sglang.py', '--strict'], hook_env),
    ]
    if any(s in scopes for s in ('core', 'platform', 'hooks')):
        tier0_steps.append(('check_sglang sm86', [py, 'tools/check_sglang.py', '--strict', '--emulate', 'sm86'], hook_env))
    for name, cmd, e in tier0_steps:
        r = run_step(name, cmd, env=e, dry_run=args.dry_run)
        report['tier0'].append(r)
        if r['status'] == 'FAIL':
            any_fail = True

    # --- Tier 1: Diff-Aware Unit & Single-Layer GPU Gate (~1-3 min) ---
    if args.tier >= 1 and not any_fail:
        print('\n--- Tier 1: Diff-Aware Unit & Single-Layer GPU Gate ---', flush=True)
        for name, cmd, preset in select_tier1_tests(scopes):
            if preset and not Path(PRESET_PATHS[preset]).is_dir():
                print(f'  [SKIP] {name:28s} (checkpoint missing)', flush=True)
                report['tier1'].append({'name': name, 'status': 'SKIP', 'sec': 0.0})
                continue
            r = run_step(name, cmd, env=env, dry_run=args.dry_run)
            report['tier1'].append(r)
            if r['status'] == 'FAIL':
                any_fail = True

    # --- Tier 2: Representative Live E2E Gate (~5-7 min / model) ---
    if args.tier >= 2 and not any_fail:
        models = select_tier2_models(scopes, args.models)
        print(f'\n--- Tier 2: Representative Live E2E Gate (models={models}) ---', flush=True)
        for mid in models:
            r = run_live_model(mid, args.port or None, dry_run=args.dry_run)
            report['tier2'].append(r)
            if r['status'] == 'FAIL':
                any_fail = True

    report['total_s'] = round(time.perf_counter() - t0, 1)
    report['status'] = 'FAIL' if any_fail else 'PASS'
    print(f'\n=== FAST VERIFY {report["status"]} in {report["total_s"]}s ===', flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'Report written to {args.out}', flush=True)
    sys.exit(1 if any_fail else 0)


if __name__ == '__main__':
    main()
