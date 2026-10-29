"""The DSA indexer's key pool on GPUs whose Triton has no float8e4nv (RTX 30 series).

SGLang writes the pool (an FP8 key and an FP32 scale per token) from four Triton kernels of kpool_fp8_index.py; each
stores its quantized keys through the pool's float8_e4m3fn view, which Triton cannot type below SM89. Here those four
kernels are rebuilt from SGLang's own source with that one store changed to xtier.arch.e4m3_bits into the pool's bytes
(the bits of the hardware conversion, tests/fp8_encode_check.py), and a stand-in passes every float8_e4m3fn argument
as its uint8 view. The launchers stay SGLang's: on the first call of one (a hook on each, registered below SM89 only)
the module's kernel names are pointed at the stand-ins.
"""
import importlib
import inspect
import linecache

import torch

MODULE = 'sglang.srt.layers.attention.dsa.kpool_fp8_index'
KERNELS = ('_kpool_softmax_rotate_write_cache_kernel', '_kpool_decode_update_and_maybe_write_cache_kernel',
           '_kpool_assemble_softmax_rotate_write_cache_kernel', '_kpool_write_tail_and_maybe_compress_kernel')
LAUNCHERS = ('kpool_softmax_rotate_write_cache', 'kpool_decode_update_and_maybe_write_cache',
             'kpool_assemble_softmax_rotate_write_cache', 'kpool_write_tail_and_maybe_compress')
STORE = 'tl.store(buf_fp8_ptr + out_k_offsets, quantized,'
_done = []


def _bytes(a):
    return a.view(torch.uint8) if isinstance(a, torch.Tensor) and a.dtype == torch.float8_e4m3fn else a


class _Bytes:
    """kernel[grid](...) of a rebuilt kernel, every float8_e4m3fn tensor argument passed as its uint8 view."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def run(*args, **kwargs):
            return launch(*[_bytes(a) for a in args], **{k: _bytes(v) for k, v in kwargs.items()})
        return run


def _rebuild():
    from xtier.arch import e4m3_bits
    module = importlib.import_module(MODULE)
    namespace = dict(vars(module), _xtier_e4m3_bits=e4m3_bits)
    for name in KERNELS:
        src = inspect.getsource(getattr(module, name).fn)
        if src.count(STORE) != 1:
            raise RuntimeError(f'xtier: {MODULE}.{name}: expected one FP8 store of the quantized keys')
        src = src.replace(STORE, 'tl.store(buf_fp8_ptr + out_k_offsets, _xtier_e4m3_bits(quantized),')
        filename = f'<xtier kpool_fp8: {name}>'
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
        exec(compile(src, filename, 'exec'), namespace)
        setattr(module, name, _Bytes(namespace[name]))


def launcher(original, *args, **kwargs):
    """AROUND hook of the pool's launchers."""
    if not _done:
        _rebuild()
        _done.append(True)
    return original(*args, **kwargs)
