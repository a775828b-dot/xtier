"""DeepSeek V4.1's FP4 fake quantization of compressed KV below SM89 (RTX 30 series): the E4M3 rounding of its block
scales in software.

SGLang's fused RoPE tail and FP4 round trip of the ratio-1/2 latents (fp4_rope_fake_quant.rope_tail_fake_quant_fp4_row)
rounds each 16-value block's scale to FP8 E4M3 with float8e4nv conversions, which Triton has not below SM89. Here that
row function is rebuilt from SGLang's source with the round trip as xtier.arch.e4m3_bits then e4m3_f32 (bit for bit
the hardware conversions: tests/fp8_encode_check.py, tests/fp8_decode_check.py); on the first call of the launcher (a
hook registered below SM89 only) the module's row function is pointed at it, so its kernel compiles with it. The rest
stays SGLang's.
"""
import importlib
import inspect
import linecache

MODULE = 'sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant'
ROW = 'rope_tail_fake_quant_fp4_row'
LAUNCHER = f'{MODULE}.rope_tail_fake_quant_fp4'
ROUND = 'scale = scale.to(tl.float8e4nv).to(tl.float32)'
_done = []


def _rebuild():
    from xtier.arch import e4m3_bits, e4m3_f32
    module = importlib.import_module(MODULE)
    src = inspect.getsource(getattr(module, ROW).fn)
    if src.count(ROUND) != 1:
        raise RuntimeError(f'xtier: {MODULE}.{ROW}: expected one E4M3 rounding of the block scales')
    src = src.replace(ROUND, 'scale = _xtier_e4m3_f32(_xtier_e4m3_bits(scale))')
    filename = f'<xtier fake_quant_fp8: {ROW}>'
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    namespace = dict(vars(module), _xtier_e4m3_bits=e4m3_bits, _xtier_e4m3_f32=e4m3_f32)
    exec(compile(src, filename, 'exec'), namespace)
    setattr(module, ROW, namespace[ROW])


def launcher(original, *args, **kwargs):
    """AROUND hook of rope_tail_fake_quant_fp4."""
    if not _done:
        _rebuild()
        _done.append(True)
    return original(*args, **kwargs)
