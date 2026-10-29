"""DeepSeek V4.1's mHC pre-mix below SM90 (RTX 30 / 40 series): SGLang's TileLang split-K GEMM at hc_hidden_size
20480.

Without DeepGEMM (SM90 and newer) SGLang's TileLang ``mhc_pre`` runs the mix GEMM of up to 2048 tokens on a split-K
TileLang kernel whose hidden block comes from a table of DeepSeek V4's two widths (16384 = 4096 x hc 4: 256; 28672 =
7168 x 4: 128). DeepSeek V4.1 Flash's 5120 x 4 = 20480 is missing and raises NotImplementedError, already in the
model's load-time prewarm. The kernel takes 20480 with V4 Pro's block of 128 (32 splits of 640). Here ``mhc_pre`` is
rebuilt from SGLang's source with 20480 in that branch; the kernels and the rest of the function stay SGLang's (FP32
operands, as DeepGEMM's TF32 GEMM on newer GPUs).
"""
import functools
import inspect
import linecache
import textwrap

MHC_PRE = 'sglang.kernels.ops.layernorm.mhc.mhc_pre'
BRANCH = 'elif hc_hidden_size == 28672:'


@functools.lru_cache
def _rebuilt(original):
    src = textwrap.dedent(inspect.getsource(original))
    if src.count(BRANCH) != 1:
        if '20480' in src:
            return original
        raise RuntimeError(f'xtier: {MHC_PRE}: expected one hidden block for hc_hidden_size 28672')
    src = src.replace(BRANCH, 'elif hc_hidden_size in (20480, 28672):')
    filename = '<xtier mhc_splitk: mhc_pre>'
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    namespace = {}
    exec(compile(src, filename, 'exec'), original.__globals__, namespace)
    return namespace[original.__name__]


def mhc_pre(original, *args, **kwargs):
    """AROUND hook of SGLang's mhc_pre (registered below SM90)."""
    return _rebuilt(original)(*args, **kwargs)
