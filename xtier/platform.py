"""Host fixes that do not belong to one model family.

- TileLang target: ``tilelang.backend.target.auto_detect_target`` tries its registered detectors in order and the HIP
  one comes first: on a CUDA host that also has a ROCm toolkit installed (``/opt/rocm/bin/hipcc``, no AMD GPU) every
  TileLang kernel is lowered for ROCm and fails (e.g. SGLang's QSA indexer: "N must be divisible by 16" from the MFMA
  layout). With a CUDA build of torch, XTier asks the CUDA detector first.
- Checkpoint page cache: ``--weight-loader-drop-cache-after-load`` evicts each checkpoint file with POSIX_FADV_DONTNEED
  once its tensors are loaded, which skips every page still mapped: safetensors' lazy tensors are views of one private
  mapping of the file, alive as long as any of them (the loader's last yielded tensor, a copy queued on a thread pool).
  So nothing of the file is evicted, and a 300 GB checkpoint fills the page cache while XTier's CPU shards fill the
  same RAM; with MGLRU the kernel then swaps anonymous memory too, although most of the page cache is cold (DeepSeek
  V4.1 1017 on the 377 GB host: swap from the moment free memory ran out). XTier first pages the file's mappings of
  this process out (MADV_PAGEOUT: their clean file pages leave the page cache), then lets SGLang drop the rest. When
  the file held XTier experts (``bulk_copy``: the loading thread copied them, i.e. most of the file), a background
  thread meanwhile reads the next two checkpoint files of the directory into the page cache (files up to 16 GiB: not
  DeepSeek's 100 GB Engram files; POSIX_FADV_WILLNEED would read only one read-ahead window), so the loading thread
  copies from the page cache instead of faulting the files in page by page. A pass that takes a few tensors per file
  (a speculative draft loading from the target's checkpoint) reads no file ahead.
- Token-id sync at TP 1: SGLang's sampler all-reduces the sampled ids over the TP group whenever a request has a
  grammar (tool calls with a constrained parser, e.g. GLM-5.3's glm47) and skips that only by flag, not by group size.
  At TP 1 the all-reduce over one rank is the identity, but its first call makes NCCL set up a communicator and
  cudaMalloc its buffers (512 MB); after a large prefill the device is full and the server dies (GLM-5.3 on the RTX
  PRO 5000, 10-24: "Failed to CUDA calloc 536870912 bytes" on the first tool request after a 64K prefill). XTier skips
  the call when the group has one rank.
- MXFP8 GEMMs at prefill sizes on SM120: SGLang tunes FlashInfer's GEMM tactics on a dummy decode forward, so the
  autotune cache holds the MXFP8 GEMM for 1-16 rows only; every prefill call misses and FlashInfer falls back to
  tactic 0, the 128 x 32 tile of its SM120 CUTLASS configs (128 x 32, 128 x 64, 128 x 128, each without and with A/B
  swap). From 256 rows the 128 x 128 tile (tactic 4) is 1.1-5.7x faster on the RTX PRO 5000 (MiniMax-M3.1's
  8192 -> 6144 projection at 16384 rows: 20.5 -> 4.0 ms) with the same output bit for bit (every tile blocks K by
  128); a miss from 256 rows takes it. Tuned shapes, tuning runs and pinned calls (SGLang's skip_ops) are left alone.
- GC freeze after start (the process that serves the API, ``install_serving``): with ``--skip-server-warmup`` SGLang's
  warm-up thread posts /freeze_gc at once, before uvicorn has bound the port, so the request is refused (a warning
  with a traceback in every start) and no process freezes its startup objects. XTier waits until the port accepts
  connections (``XTIER_FREEZE_GC_WAIT_S``, 120 s at most), then lets SGLang post it.
- FP8 activation quantization below SM89 (RTX 30 series): SGLang's Triton ``act_quant`` (the DSA indexer's queries)
  stores through float8e4nv, which Triton has only from SM89 on, so it does not compile; its TileLang version (the
  ROCm path; CUDA's cuda_fp8.h converts in software there) gives the same bytes and scales: bitwise on the RTX PRO
  5000 against the Triton kernel, both scale formats, and again built for sm_86 (tools/arch_check.py --emulate sm86).
"""

import ctypes
import functools
import logging
import os
import threading

logger = logging.getLogger(__name__)
TILELANG_TARGET = 'tilelang.backend.target.auto_detect_target'
DROP_FILE_CACHE = 'sglang.srt.model_loader.weight_utils._drop_file_cache_after_load'
TOKEN_SYNC = 'sglang.srt.layers.sampler.Sampler._sync_token_ids_across_tp'
MXFP8_TUNER = 'flashinfer.autotuner.autotuner.AutoTuner.choose_one'
FREEZE_GC = 'sglang.srt.entrypoints.http_server._freeze_gc_after_server_warmup'
ACT_QUANT = 'sglang.kernels.ops.quantization.act_quant_triton.act_quant'
MXFP8_LARGE_TILE, MXFP8_LARGE_ROWS = 4, 256   # SM120 CUTLASS MXFP8: tactic 4 = 128 x 128 x 128, no A/B swap
_MADV_PAGEOUT = 21
READ_AHEAD_FILES, READ_AHEAD_MAX = 2, 16 << 30


def _auto_detect_target(original, *args, **kwargs):
    import torch
    if torch.version.hip is None and torch.cuda.is_available():
        from tilelang.backend import target
        spec = target._TARGET_DETECTORS.get('cuda')
        detected = spec.detect() if spec is not None else None
        if detected is not None:
            return detected
    return original(*args, **kwargs)


def _mappings(path):
    """[start, end) of this process's mappings of the file `path`."""
    real, ranges = os.path.realpath(path), []
    with open('/proc/self/maps') as f:
        for line in f:
            fields = line.rstrip('\n').split(None, 5)
            if len(fields) == 6 and fields[5] == real:
                start, end = (int(x, 16) for x in fields[0].split('-'))
                ranges.append((start, end))
    return ranges


@functools.lru_cache
def _checkpoint_files(directory):
    return tuple(sorted(n for n in os.listdir(directory) if n.endswith('.safetensors')))


_queue, _queued, _ahead, _reader = [], set(), set(), []   # _ahead: read ahead, not loaded yet (at most 2 files)
_ready = threading.Condition()
_bulk = [False]   # the file being loaded held XTier experts


def bulk_copy():
    """The loading thread copies XTier experts out of the current checkpoint file (backend.hooks)."""
    _bulk[0] = True


def _read_files():
    view = memoryview(bytearray(16 << 20))
    while True:
        with _ready:
            while not _queue:
                _ready.wait()
            path = _queue.pop(0)
        try:
            with open(path, 'rb', buffering=0) as f:
                while f.readinto(view):
                    pass
        except OSError as e:
            logger.debug('xtier: read-ahead of %s failed: %s', path, e)


def _read_ahead_next(path):
    """Queue the checkpoint files after `path` (name order of its directory) for reading into the page cache."""
    directory, name = os.path.split(path)
    try:
        names = _checkpoint_files(directory)
    except OSError:
        return
    if name not in names:
        return
    i = names.index(name)
    with _ready:
        _ahead.discard(path)
        for n in names[i + 1:i + 1 + READ_AHEAD_FILES]:
            p = os.path.join(directory, n)
            try:
                small = os.path.getsize(p) <= READ_AHEAD_MAX
            except OSError:
                continue
            if small and p not in _queued and len(_ahead) < READ_AHEAD_FILES:
                _queued.add(p)
                _ahead.add(p)
                _queue.append(p)
        if not _reader:
            _reader.append(threading.Thread(target=_read_files, name='xtier-read-ahead', daemon=True))
            _reader[0].start()
        _ready.notify()


def _drop_file_cache(original, path, *args, **kwargs):
    if _bulk[0]:
        _bulk[0] = False
        _read_ahead_next(path)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        for start, end in _mappings(path):
            libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), _MADV_PAGEOUT)
    except OSError as e:
        logger.debug('xtier: page-out of %s failed: %s', path, e)
    return original(path, *args, **kwargs)


def _sync_token_ids(original, self, *args, **kwargs):
    import torch.distributed as dist
    group = getattr(self, 'tp_sync_group', None)
    if group is not None and dist.get_world_size(group) == 1:
        return None   # an all-reduce over one rank: nothing to agree on, and no NCCL communicator to create
    return original(self, *args, **kwargs)


@functools.lru_cache
def _sm120_mxfp8_tiles(device_index, runner_type, tactics):
    """Whether this is FlashInfer's SM120 MXFP8 CUTLASS runner with its six configs (three tiles x A/B swap)."""
    import torch
    return (runner_type == 'CutlassMxfp8GemmRunner' and tactics == 6
            and torch.cuda.get_device_capability(device_index)[0] == 12)


def _choose_one(original, self, custom_op, runners, tuning_config, inputs, *args, **kwargs):
    runner, tactic = original(self, custom_op, runners, tuning_config, inputs, *args, **kwargs)
    if (tactic == -1 and custom_op == 'mxfp8_gemm' and not self.is_tuning_mode
            and custom_op not in getattr(self, '_effective_skip_ops', ())
            and inputs and inputs[0].dim() == 2 and inputs[0].shape[0] >= MXFP8_LARGE_ROWS and inputs[0].is_cuda
            and _sm120_mxfp8_tiles(inputs[0].device.index, type(runner).__name__,
                                   len(runner.get_valid_tactics(inputs, None)))):
        return runner, MXFP8_LARGE_TILE
    return runner, tactic


def _freeze_gc_when_listening(original, server_args, *args, **kwargs):
    """In the warm-up thread: wait until the HTTP server accepts connections, then SGLang's POST /freeze_gc."""
    import socket
    import time
    from urllib.parse import urlsplit
    try:
        url = urlsplit(server_args.url())
        address = (url.hostname, url.port or (443 if url.scheme == 'https' else 80)) if url.hostname else None
    except Exception:   # an address SGLang builds differently: post as it would
        address = None
    if address is not None:
        deadline = time.monotonic() + float(os.environ.get('XTIER_FREEZE_GC_WAIT_S', '120'))
        while time.monotonic() < deadline:
            try:
                socket.create_connection(address, timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
    return original(server_args, *args, **kwargs)


def _act_quant(original, *args, **kwargs):
    from xtier.arch import HW_FP8
    if HW_FP8:
        return original(*args, **kwargs)
    from sglang.kernels.ops.quantization.act_quant_tilelang import act_quant
    return act_quant(*args, **kwargs)


def install_serving(registry, hook_type) -> list[str]:
    """Fixes of the process that serves the API."""
    registry.register(FREEZE_GC, _freeze_gc_when_listening, hook_type.AROUND)
    return ['freeze-gc-after-listen']


def install(registry, hook_type) -> list[str]:
    import importlib.util
    fixes = []
    if importlib.util.find_spec('tilelang') is not None:
        registry.register(TILELANG_TARGET, _auto_detect_target, hook_type.AROUND)
        fixes.append('tilelang-cuda-target')
    if os.path.exists('/proc/self/maps'):
        registry.register(DROP_FILE_CACHE, _drop_file_cache, hook_type.AROUND)
        fixes.append('checkpoint-page-out')
    registry.register(TOKEN_SYNC, _sync_token_ids, hook_type.AROUND)
    fixes.append('tp1-token-sync')
    if importlib.util.find_spec('tilelang') is not None:   # acts below SM89 only
        registry.register(ACT_QUANT, _act_quant, hook_type.AROUND)
        fixes.append('fp8-act-quant-below-sm89')
    if importlib.util.find_spec('flashinfer') is not None and os.environ.get('XTIER_MXFP8_PREFILL_TILE', '1') != '0':
        registry.register(MXFP8_TUNER, _choose_one, hook_type.AROUND)
        fixes.append('sm120-mxfp8-prefill-tile')
    return fixes
