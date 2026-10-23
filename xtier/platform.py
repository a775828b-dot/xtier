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
    return fixes
