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
- JIT build cache behind a symlinked cache directory: SGLang's content-addressed JIT cache keeps a build's own
  generated sources out of its dependency list by ``path.is_relative_to(build_dir)``, with ``path`` resolved but
  ``build_dir`` (the staging directory under ``~/.cache/sglang/jit``) not. When ``~/.cache`` is a symlink (moved to a
  data disk on a rented machine), the generated wrapper is recorded by its absolute staging path
  (``.staging-<uuid>/cuda.cu``), no published build ever matches again, and every start compiles every JIT kernel
  anew: on an RTX 3090 the Marlin W8A16 GEMM takes about 5 minutes, once per TP rank (the second rank finds no usable
  build either). XTier resolves ``build_dir`` first.
- CPU multimodal image/video decoding & bounded worker pool (``install_serving``, ``XTIER_CPU_IMAGE=1`` default,
  ``XTIER_MM_CPU_WORKERS``): SGLang's multimodal processors default to ``gpu_image_decode=True`` (nvJPEG on CUDA),
  GPU fast image preprocessing (``retry_on_cpu_after_gpu_oom=False``), and ``SGLANG_CPU_WORKERS=os.cpu_count()``
  (128 forked workers on a dual-EPYC host). On a tight-VRAM XTier host this opens an extra CUDA context (~0.5-1 GiB)
  in the tokenizer process, risks OOM on multi-image requests, and oversubscribes the non-MoE CPUs. XTier keeps
  image/video decoding and preprocessing on the CPU and bounds multimodal CPU workers inside the reserved non-MoE
  CPU headroom (4-8 workers on large hosts).
- Block-FP8 W8A8 Triton SWAP_AB / SPLIT_K on SM89 (Ada: L4 / L20 / L40S / RTX 4090): SGLang gates
  ``_w8a8_block_fp8_matmul_hopper`` (plain Triton, no TMA/wgmma) on ``get_platform().is_sm90``, so SM89 tuned
  configs carrying ``SWAP_AB`` or ``SPLIT_K > 1`` fall through to ``_w8a8_block_fp8_matmul`` and raise KeyError.
  Until upstream widens the gate to exact capability (8, 9), XTier dispatches those SM89 configs (non-deterministic
  mode) to ``_w8a8_block_fp8_matmul_hopper`` + ``_reduce_block_fp8_split_k`` without patching SGLang source.
- SWA radix insert full span (``SWAComponent.prepare_for_caching_req``): when a conversation partially matches an
  older SWA-less Full-KV prefix, SGLang returned ``effective_cache_len`` and truncated the radix insert at the matched
  prefix, creating a fixed point so later turns never extended the tree (Lsglang #37). XTier keeps the
  ``swa_branching_seqlen`` boundary metadata and returns ``None`` so the insert spans the full request.
- Startup eager draft kernel & QSA draft-graph metadata prewarm (``EagleDraftWorker._capture_cuda_graphs``,
  ``XTIER_DRAFT_PREWARM=1`` default): ``prepare_for_draft``, ``build_chain_tree``, ``prepare_draft_extend_*``,
  ``prepare_verify_commit_outputs``, and QSA's multi-step draft-graph metadata run outside CUDA graphs and specialize
  on batch size / mamba track interval; first served hit pays ~500 ms ``cuModuleLoadData`` and can wedge TP > 1 while
  peer ranks wait in collectives (Lsglang 1.7.7). Prewarming at init with ``accept_lens = ones`` and
  ``accept_index = zeros`` loads every specialization without touching ``eagle.py`` or triggering the ``accepted - 1``
  unmapped-page read (``NVRM Xid 31``).
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
PARALLEL_TOKENIZE = 'sglang.srt.managers.tokenizer_manager.TokenizerManager._tokenize_texts'
INIT_TOKENIZER = 'sglang.srt.managers.tokenizer_manager.TokenizerManager.init_tokenizer_and_processor'
MM_LOAD_IMAGE = 'sglang.srt.multimodal.media_io.load_image'
MM_LOAD_VIDEO = 'sglang.srt.multimodal.media_io.load_video'
MM_PROC_INIT = 'sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor.__init__'
MM_PROC_DEVICE = 'sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor._fast_image_processor_device'
MM_PROC_COMPETES = (
    'sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor'
    '._preprocessing_competes_with_the_scheduler'
)
MM_PROC_WORKERS = (
    'sglang.srt.multimodal.processors.base_processor.BaseMultimodalProcessor'
    '._resolve_auto_mm_processor_worker_num'
)
ACT_QUANT = 'sglang.kernels.ops.quantization.act_quant_triton.act_quant'
JIT_DEPS = 'sglang.kernels.jit.utils.compile.cache._to_entries'
BLOCK_FP8_TRITON = 'sglang.kernels.ops.gemm.fp8_kernel.w8a8_block_fp8_matmul_triton'
SWA_PREPARE_CACHE = 'sglang.srt.mem_cache.unified_cache.components.swa.SWAComponent.prepare_for_caching_req'
EAGLE_CAPTURE_GRAPHS = 'sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._capture_cuda_graphs'
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


def _jit_deps_resolved(original, *args, **kwargs):
    if 'build_dir' in kwargs:
        kwargs['build_dir'] = kwargs['build_dir'].resolve()
    return original(*args, **kwargs)


@functools.lru_cache(maxsize=8)
def _is_sm89(device_index: int) -> bool:
    import torch
    return torch.cuda.get_device_capability(device_index) == (8, 9)


_SM89_HOPPER_HAS_I64_OFFSETS = None


def _w8a8_block_fp8_sm89(original, A, B, As, Bs, block_size, output_dtype=None):
    import torch
    if output_dtype is None:
        output_dtype = torch.float16
    if (not A.is_cuda or A.ndim < 2 or A.shape[-1] == 0 or not _is_sm89(A.device.index)
            or B.ndim != 2 or len(block_size) != 2):
        return original(A, B, As, Bs, block_size, output_dtype)
    import triton
    from sglang.kernels.ops.gemm import fp8_kernel as fk
    N, K = B.shape
    configs = fk.get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])
    if not configs:
        return original(A, B, As, Bs, block_size, output_dtype)
    M = A.numel() // A.shape[-1]
    config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    if not (config.get('SWAP_AB', False) or config.get('SPLIT_K', 1) > 1):
        return original(A, B, As, Bs, block_size, output_dtype)
    ctx = fk.get_context()
    if ctx.is_config_namespace_published('exec') and fk.get_exec().deterministic.enable_deterministic_inference:
        return original(A, B, As, Bs, block_size, output_dtype)
    M, N, K, C = fk.prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)
    block_n, block_k = block_size
    split_k = config.get('SPLIT_K', 1)
    assert split_k & (split_k - 1) == 0
    partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32) if split_k > 1 else C
    needs_masking = bool(K % config['BLOCK_SIZE_K'] != 0)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), split_k)

    global _SM89_HOPPER_HAS_I64_OFFSETS
    if _SM89_HOPPER_HAS_I64_OFFSETS is None:
        import inspect
        _SM89_HOPPER_HAS_I64_OFFSETS = (
            'USE_INT64_OFFSETS' in inspect.signature(fk._w8a8_block_fp8_matmul_hopper.fn).parameters
        )
    kernel_kwargs = dict(needs_masking=needs_masking)
    if _SM89_HOPPER_HAS_I64_OFFSETS:
        kernel_kwargs['USE_INT64_OFFSETS'] = max(A.numel(), partials.numel()) >= 2**31
    fk._w8a8_block_fp8_matmul_hopper[grid](
        A, B, partials, As, Bs, M, N, K, block_n, block_k,
        A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0),
        **config, **kernel_kwargs,
    )
    if split_k > 1:
        fk._reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](partials, C, M * N, split_k, 256)
    return C


_BPE_SPLIT_PATTERN = None


def _try_parallel_bpe_encode(tok, text: str, add_special_tokens: bool = True, target_chunk: int = 16384):
    """Split a long single prompt on exact BPE pre-tokenization boundaries (right after ``\n`` and right before an
    ASCII alphanumeric ``[A-Za-z0-9]``, outside any newline-bearing added token) so HuggingFace's Rust ``tokenizers``
    engine encodes the slices across Rayon worker threads instead of on one CPU core (520K chars: 1.4 s -> 0.14 s,
    bitwise identical token IDs across DeepSeek V4.1, GLM-5.3, and Qwen3.8)."""
    global _BPE_SPLIT_PATTERN
    import re
    if _BPE_SPLIT_PATTERN is None:
        _BPE_SPLIT_PATTERN = re.compile(r'\n(?=[A-Za-z0-9])')

    info = getattr(tok, '_xtier_parallel_bpe_info', None)
    if info is None:
        ref = tok(['A\nB'])['input_ids'][0]
        no_sp = tok(['A\n', 'B'], add_special_tokens=False)['input_ids']
        all_sp = tok(['A\n', 'B'])['input_ids']
        if no_sp[0] + no_sp[1] == ref:
            mode = 'none'
        elif len(ref) == len(no_sp[0]) + len(no_sp[1]) + 1 and all_sp[0] + no_sp[1] == ref:
            mode = ('bos', ref[0])
        elif len(ref) == len(no_sp[0]) + len(no_sp[1]) + 1 and no_sp[0] + [all_sp[1][-1]] == ref:
            mode = ('eos', ref[-1])
        elif len(ref) == len(no_sp[0]) + len(no_sp[1]) + 2 and [ref[0]] + no_sp[0] + no_sp[1] + [ref[-1]] == ref:
            mode = ('both', ref[0], ref[-1])
        else:
            mode = 'unsupported'
        added = getattr(tok, 'added_tokens_encoder', None) or {}
        nl_specials = tuple(s for s in added if s and '\n' in s)
        info = (mode, nl_specials)
        try:
            setattr(tok, '_xtier_parallel_bpe_info', info)
        except Exception:
            pass

    mode, nl_specials = info
    if mode == 'unsupported':
        return None

    n = len(text)
    forbidden = []
    for sp in nl_specials:
        start = 0
        ln = len(sp)
        while True:
            idx = text.find(sp, start)
            if idx < 0:
                break
            forbidden.append((idx, idx + ln))
            start = idx + ln
    if forbidden:
        forbidden.sort()

    def in_forbidden(pos: int) -> bool:
        for a, b in forbidden:
            if pos <= a:
                return False
            if a < pos < b:
                return True
        return False

    cuts = [0]
    next_pos = target_chunk
    while next_pos < n - 4096:
        m = _BPE_SPLIT_PATTERN.search(text, next_pos)
        if m is None:
            break
        cut = m.end()
        if not in_forbidden(cut):
            cuts.append(cut)
            next_pos = cut + target_chunk
        else:
            next_pos = cut + 64
    if len(cuts) <= 1:
        return None
    cuts.append(n)
    parts = [text[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    part_ids = tok(parts, add_special_tokens=False)['input_ids']
    flat = []
    if add_special_tokens and isinstance(mode, tuple) and mode[0] in ('bos', 'both'):
        flat.append(mode[1])
    for pids in part_ids:
        flat.extend(pids)
    if add_special_tokens and isinstance(mode, tuple) and mode[0] == 'eos':
        flat.append(mode[1])
    elif add_special_tokens and isinstance(mode, tuple) and mode[0] == 'both':
        flat.append(mode[2])
    return flat


async def _parallel_tokenize_texts(original, self, texts, is_cross_encoder=False, *args, **kwargs):
    if (
        not is_cross_encoder
        and not args
        and not kwargs
        and isinstance(texts, str)
        and len(texts) >= 32768
        and getattr(self, 'async_dynamic_batch_tokenizer', None) is None
        and not getattr(getattr(self, 'model_config', None), 'is_embedding_gemma', False)
    ):
        tok = getattr(self, 'tokenizer', None)
        if tok is not None and getattr(tok, 'is_fast', False):
            try:
                ids = _try_parallel_bpe_encode(tok, texts)
                if ids is not None:
                    return ids, None
            except Exception as e:
                logger.debug('xtier: parallel BPE tokenization fell back: %s', e)
    return await original(self, texts, is_cross_encoder, *args, **kwargs)


def _wrap_tokenizer_encode(original, self, *args, **kwargs):
    ret = original(self, *args, **kwargs)
    tok = getattr(self, 'tokenizer', None)
    if (
        tok is not None
        and getattr(tok, 'is_fast', False)
        and not getattr(tok, '_xtier_encode_wrapped', False)
        and not getattr(getattr(self, 'model_config', None), 'is_embedding_gemma', False)
    ):
        orig_encode = tok.encode

        def _fast_encode(text, *eargs, **ekwargs):
            if (
                isinstance(text, str)
                and len(text) >= 32768
                and not eargs
                and (not ekwargs or tuple(ekwargs.keys()) == ('add_special_tokens',))
            ):
                try:
                    ids = _try_parallel_bpe_encode(tok, text, add_special_tokens=ekwargs.get('add_special_tokens', True))
                    if ids is not None:
                        return ids
                except Exception as e:
                    logger.debug('xtier: parallel BPE encode fell back: %s', e)
            return orig_encode(text, *eargs, **ekwargs)

        try:
            setattr(tok, 'encode', _fast_encode)
            setattr(tok, '_xtier_encode_wrapped', True)
        except Exception:
            pass
    return ret


def _mm_cpu_budget() -> int:
    """Safe CPU worker count for multimodal decoding/preprocessing inside XTier's reserved non-MoE cores.

    Honors an explicit ``XTIER_MM_CPU_WORKERS``; otherwise derives from the process CPU affinity (narrowed by
    ``xtier.engine.placement``) or ``total_cpus - XTIER_THREADS``, using at most half the reserved headroom and
    capped at 8 so bursts of high-res images/video never starve the HTTP loop or contend with MoE workers.
    """
    raw = os.environ.get('XTIER_MM_CPU_WORKERS', '').strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning('xtier: invalid XTIER_MM_CPU_WORKERS=%r; using auto budget', raw)
    total = os.cpu_count() or 4
    try:
        avail = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        avail = total
    if avail >= total:
        try:
            from xtier import config
            moe_threads = config.threads()
        except Exception:
            moe_threads = 0
        if 0 < moe_threads < total:
            avail = total - moe_threads
    if avail < 2:
        return 1
    return max(2, min(8, avail // 2))


def _mm_load_image_cpu(original, image_file, *args, **kwargs):
    if args:
        args = (False, *args[1:])
    else:
        kwargs['gpu_image_decode'] = False
    return original(image_file, *args, **kwargs)


def _mm_load_video_cpu(original, video_file, *args, **kwargs):
    if args:
        args = (False, *args[1:])
    else:
        kwargs['use_gpu'] = False
    return original(video_file, *args, **kwargs)


def _mm_fast_device_cpu(original, self, processor, *args, **kwargs):
    return 'cpu'


def _mm_not_competing(original, self, *args, **kwargs):
    return False


def _mm_processor_init(original, self, *args, **kwargs):
    cpu_image = os.environ.get('XTIER_CPU_IMAGE', '1') != '0'
    if cpu_image:
        cls = type(self)
        if getattr(cls, 'gpu_image_decode', False) is not False:
            cls.gpu_image_decode = False
        self.gpu_image_decode = False
        if getattr(cls, 'video_preprocessing_device', None) != 'cpu':
            cls.video_preprocessing_device = 'cpu'
        self.video_preprocessing_device = 'cpu'
    budget = _mm_cpu_budget()
    os.environ.setdefault('SGLANG_CPU_WORKERS', str(budget))
    if budget < 4:
        os.environ.setdefault('SGLANG_IO_WORKERS', str(budget))
    return original(self, *args, **kwargs)


def _mm_auto_workers(original, self, *args, **kwargs):
    budget = _mm_cpu_budget()
    if os.environ.get('XTIER_MM_CPU_WORKERS', '').strip():
        return budget
    n = max(1, int(original(self, *args, **kwargs)))
    if os.environ.get('XTIER_CPU_IMAGE', '1') != '0':
        target = min(4, budget)
        if n < target:
            n = target
    return min(n, budget)



def _swa_prepare_for_caching_req(original, self, req, insert_params, token_ids_len, is_finished):
    """Keep SWA branch boundary metadata without truncating radix insert keys (Lsglang #37)."""
    ret = original(self, req, insert_params, token_ids_len, is_finished)
    if ret is not None and getattr(insert_params, 'swa_branching_seqlen', None) is not None:
        return None
    return ret


def _draft_prewarm_batch_sizes(max_running_requests) -> tuple[int, ...]:
    """Batch sizes whose Triton specializations diverge (1, <=16 exhaustive for derived bs*width, then powers of 2)."""
    cap = int(max_running_requests or 0)
    if cap <= 0:
        return (1,)
    if cap <= 16:
        return tuple(range(1, cap + 1))
    return tuple(sorted(set(range(1, 17)) | {p for p in (32, 64, 128, 256) if p <= cap}))


def _draft_track_intervals(page_size, server_args=None) -> tuple[int, ...]:
    """Live ``mamba_track_interval`` constexpr values for ``prepare_verify_commit_outputs``."""
    if not page_size:
        return (0,)
    intervals = {0}
    try:
        from sglang.srt.runtime_context import mamba_track_grid
        grid_val = int(mamba_track_grid(int(page_size)))
        if grid_val > 0:
            intervals.add(grid_val)
    except Exception:
        try:
            import math
            from sglang.srt.runtime_context import get_server_args, mamba_cache_chunk_size
            grid = math.lcm(int(mamba_cache_chunk_size()), int(page_size))
            args = server_args if server_args is not None else get_server_args()
            track = int(getattr(args, 'mamba_track_interval', 0) or 0)
            val = int(math.lcm(grid, track)) if track > 0 else int(grid)
            if val > 0:
                intervals.add(val)
        except Exception:
            pass
    return tuple(sorted(intervals))


def _prewarm_eager_draft_kernels(
    req_to_token,
    *,
    topk: int,
    num_steps: int,
    num_draft_tokens,
    batch_sizes=(1,),
    track_intervals=(0,),
) -> int:
    """Device-load eager draft-step Triton kernels before serving, using accept_lens=1 to avoid OOB column -1 reads."""
    import torch
    if topk != 1 or req_to_token is None or not req_to_token.is_cuda:
        return 0
    from sglang.kernels.ops.speculative.cache_locs import assign_draft_cache_locs_contiguous_func
    from sglang.kernels.ops.speculative.eagle import (
        build_chain_tree,
        prepare_draft_extend_layout,
        prepare_draft_extend_lengths,
        prepare_verify_commit_outputs,
    )

    device = req_to_token.device
    steps = int(num_steps)
    width = int(num_draft_tokens) if num_draft_tokens else steps + 1
    if steps <= 0 or width <= 0:
        return 0
    window_variants = sorted({steps, width})
    launches = 0

    for bs in sorted({max(1, int(b)) for b in batch_sizes}):
        if req_to_token.shape[1] <= bs * max(window_variants) + 1:
            continue
        req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=device)
        seq_lens = torch.zeros(bs, dtype=torch.int64, device=device)

        for window_cols in window_variants:
            out_cache_loc = torch.empty((bs * topk * window_cols,), dtype=torch.int64, device=device)
            positions = torch.empty((bs * topk,), dtype=torch.int64, device=device)
            for mrope in (None, torch.zeros((3, bs), dtype=torch.int64, device=device)):
                assign_draft_cache_locs_contiguous_func(
                    req_pool_indices,
                    req_to_token,
                    seq_lens,
                    out_cache_loc,
                    topk,
                    window_cols,
                    positions=positions,
                    mrope=mrope,
                )
                launches += 1

        for bonus_dtype in (torch.int32, torch.int64):
            for draft_dtype in (torch.int64, torch.int32):
                for with_mrope in (True, False):
                    build_chain_tree(
                        torch.zeros((bs,), dtype=bonus_dtype, device=device),
                        torch.zeros((bs, steps), dtype=draft_dtype, device=device),
                        seq_lens,
                        None,
                        req_pool_indices=req_pool_indices,
                        req_to_token=req_to_token,
                        with_mrope=with_mrope,
                    )
                    launches += 1

        prepare_draft_extend_lengths(seq_lens, width, 0)
        launches += 1
        for with_mrope in (True, False):
            prepare_draft_extend_layout(seq_lens, width, with_mrope)
            launches += 1

        # accept_lens=ones (never zeros) keeps (accepted - 1) == 0 in-bounds under VMM page-aligned allocations,
        # while compiling the identical Triton cubin as served steps (accept_lens is a tensor pointer, not constexpr).
        predict = torch.zeros((bs * width,), dtype=torch.int32, device=device)
        accept_index = torch.zeros((bs, width), dtype=torch.int32, device=device)
        accept_lens = torch.ones((bs,), dtype=torch.int32, device=device)
        for interval in track_intervals:
            prepare_verify_commit_outputs(
                predict,
                accept_index,
                accept_lens,
                seq_lens,
                num_draft_tokens=width,
                mamba_track_interval=int(interval),
            )
            launches += 1

    return launches


def _prewarm_qsa_draft_graph_metadata(draft_attn_backend) -> int:
    """Device-load QSA multi-step draft-graph metadata kernels on cloned buffers so live graph state stays intact."""
    import torch
    if draft_attn_backend is None or not hasattr(draft_attn_backend, '_draft_graph_metadata_args'):
        return 0
    attn_backends = getattr(draft_attn_backend, 'attn_backends', None)
    if not attn_backends:
        return 0
    first = attn_backends[0]
    req_to_token = getattr(first, 'req_to_token', None)
    graph_meta = getattr(first, '_cuda_graph_metadata', None)
    if req_to_token is None or not req_to_token.is_cuda or not graph_meta:
        return 0

    from sglang.srt.layers.attention.qsa.graph_metadata import (
        launch_draft_graph_metadata,
        prepare_draft_graph_metadata,
    )
    from sglang.srt.model_executor.forward_mode import ForwardMode

    device = req_to_token.device
    buckets = sorted(bs for mode, bs in graph_meta if mode == ForwardMode.DECODE)
    launches = 0
    for bs in buckets:
        try:
            metadata = tuple(b._cuda_graph_metadata[(ForwardMode.DECODE, bs)] for b in attn_backends)
            args = prepare_draft_graph_metadata(
                metadata,
                first.req_to_token,
                metadata[0].indexer_metadata.token_to_kv_pool,
            )
        except (KeyError, AttributeError):
            continue
        buffers, r2t, row_stride, max_pages, ratio, full_page = args
        cloned = tuple(tuple(buf.clone() for buf in group) for group in buffers)
        launch_draft_graph_metadata(
            (cloned, r2t, row_stride, max_pages, ratio, full_page),
            torch.zeros(bs, dtype=torch.int64, device=device),
            torch.zeros(bs, dtype=torch.int64, device=device),
            bs,
            0,
        )
        launches += 1
    return launches


def _eagle_capture_cuda_graphs(original, self, *args, **kwargs):
    import torch
    if getattr(self, 'topk', 0) == 1 and torch.cuda.is_available():
        try:
            draft_runner = getattr(self, 'draft_runner', None)
            target_worker = getattr(self, 'target_worker', None)
            target_runner = getattr(target_worker, 'model_runner', None) if target_worker is not None else None
            server_args = getattr(draft_runner, 'server_args', None)
            alloc = getattr(target_runner, 'token_to_kv_pool_allocator', None)
            page_size = getattr(alloc, 'page_size', None)
            req_to_token = getattr(getattr(draft_runner, 'req_to_token_pool', None), 'req_to_token', None)
            max_reqs = getattr(server_args, 'max_running_requests', None)
            track_intervals = _draft_track_intervals(page_size, server_args=server_args)
            launches = _prewarm_eager_draft_kernels(
                req_to_token,
                topk=int(self.topk),
                num_steps=int(getattr(self, 'speculative_num_steps', 0)),
                num_draft_tokens=getattr(self, 'speculative_num_draft_tokens', None),
                batch_sizes=_draft_prewarm_batch_sizes(max_reqs),
                track_intervals=track_intervals,
            )
            if launches:
                logger.info(
                    'xtier: prewarmed %d eager draft kernel launch(es) (steps=%s, draft_tokens=%s, '
                    'max_reqs=%s, page_size=%s, track_intervals=%s)',
                    launches,
                    getattr(self, 'speculative_num_steps', None),
                    getattr(self, 'speculative_num_draft_tokens', None),
                    max_reqs,
                    page_size,
                    track_intervals,
                )
        except Exception as e:
            logger.warning('xtier: eager draft kernel prewarm skipped: %s', e)

    ret = original(self, *args, **kwargs)

    if torch.cuda.is_available():
        try:
            qsa_launches = _prewarm_qsa_draft_graph_metadata(getattr(self, 'draft_attn_backend', None))
            if qsa_launches:
                logger.info('xtier: prewarmed %d QSA draft-graph metadata launch(es)', qsa_launches)
        except Exception as e:
            logger.warning('xtier: QSA draft-graph metadata prewarm skipped: %s', e)
    return ret


def install_serving(registry, hook_type) -> list[str]:
    """Fixes of the process that serves the API."""
    import importlib.util
    fixes = ['freeze-gc-after-listen']
    registry.register(FREEZE_GC, _freeze_gc_when_listening, hook_type.AROUND)
    if os.environ.get('XTIER_PARALLEL_TOKENIZE', '1') != '0':
        registry.register(PARALLEL_TOKENIZE, _parallel_tokenize_texts, hook_type.AROUND)
        registry.register(INIT_TOKENIZER, _wrap_tokenizer_encode, hook_type.AROUND)
        fixes.append('parallel-bpe-tokenize')
    has_mm = importlib.util.find_spec('sglang.srt.multimodal.processors.base_processor') is not None
    if has_mm:
        if os.environ.get('XTIER_CPU_IMAGE', '1') != '0':
            registry.register(MM_LOAD_IMAGE, _mm_load_image_cpu, hook_type.AROUND)
            registry.register(MM_LOAD_VIDEO, _mm_load_video_cpu, hook_type.AROUND)
            registry.register(MM_PROC_DEVICE, _mm_fast_device_cpu, hook_type.AROUND)
            registry.register(MM_PROC_COMPETES, _mm_not_competing, hook_type.AROUND)
            fixes.append('cpu-image-decode')
        registry.register(MM_PROC_INIT, _mm_processor_init, hook_type.AROUND)
        registry.register(MM_PROC_WORKERS, _mm_auto_workers, hook_type.AROUND)
        fixes.append('mm-cpu-worker-budget')
    return fixes



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
    if importlib.util.find_spec('sglang.kernels.jit.utils.compile') is not None:
        registry.register(JIT_DEPS, _jit_deps_resolved, hook_type.AROUND)
        fixes.append('jit-cache-symlinked-dir')
    if importlib.util.find_spec('sglang.kernels.ops.gemm.fp8_kernel') is not None:
        registry.register(BLOCK_FP8_TRITON, _w8a8_block_fp8_sm89, hook_type.AROUND)
        fixes.append('sm89-block-fp8-split-k')
    if importlib.util.find_spec('sglang.srt.mem_cache.unified_cache.components.swa') is not None:
        registry.register(SWA_PREPARE_CACHE, _swa_prepare_for_caching_req, hook_type.AROUND)
        fixes.append('swa-radix-full-insert')
    if (importlib.util.find_spec('sglang.srt.speculative.eagle_worker_v2') is not None
            and os.environ.get('XTIER_DRAFT_PREWARM', '1') != '0'):
        registry.register(EAGLE_CAPTURE_GRAPHS, _eagle_capture_cuda_graphs, hook_type.AROUND)
        fixes.append('eagle-draft-prewarm')
    return fixes
