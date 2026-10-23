"""PLE layer in token blocks (packs/qwen4_exp/ple_blocks.py) against SGLang's whole-chunk ``Qwen4ExpPLELayer.forward``
at Qwen3.8's shape (hidden 2560 x 4 hyper-connections, PLE dim 2560, kernel 4, dilation 3) with random weights: the
output, the conv state after the chunk and the radix-tree track snapshot, for chunk lengths around the block size;
and the peak memory of both. The projections run per block (a GEMM of other M), so the output may differ in the last
bits; the convolution part is the same columns.

  python tests/ple_blocks_check.py
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.srt.models import qwen4_exp  # noqa: E402

from xtier.packs.qwen4_exp import ple_blocks  # noqa: E402

HIDDEN, HC, PLE_DIM, K, DIL = 2560, 4, 2560, 4, 3
C, S = HIDDEN * HC, (K - 1) * DIL
torch.manual_seed(20261023)
Layer = qwen4_exp.Qwen4ExpPLELayer


def norm():
    return qwen4_exp.Qwen4ExpPLEGroupedNorm(C, eps=1e-6, group_size=HIDDEN).cuda().bfloat16()


wk = (torch.randn(C, PLE_DIM, device='cuda') * 0.02).bfloat16()
wv = (torch.randn(HIDDEN, PLE_DIM, device='cuda') * 0.02).bfloat16()
layer = SimpleNamespace(layer_id=0, hc_count=HC, hidden_size=HIDDEN, conv_channels=C, short_conv_state_len=S,
                        short_conv_dilation=DIL, _prefetch_state=None,
                        key_proj=lambda e: (e @ wk.t(), None), value_proj=lambda e: (e @ wv.t(), None),
                        norm_key=norm(), norm_query=norm(), norm_conv=norm(),
                        conv1d=torch.nn.Conv1d(C, C, K, groups=C, dilation=DIL, bias=False).cuda().bfloat16())
with torch.no_grad():
    layer.conv1d.weight.normal_(0, 0.3)
    for n in (layer.norm_key, layer.norm_query, layer.norm_conv):
        for p in n.parameters():
            p.uniform_(0.5, 1.5)
layer._apply_ple_norm = types.MethodType(Layer._apply_ple_norm, layer)
layer._short_conv = types.MethodType(Layer._short_conv, layer)
INITIAL_STATE = torch.randn(8, C, S, device='cuda').bfloat16()


class Pool:
    def __init__(self):
        self.cache = INITIAL_STATE.clone()   # the same incoming state for both implementations

    def short_conv_layer_cache(self, layer_id):
        return self.cache


def run(fn, hidden, embeddings, track_offset):
    pool = Pool()
    qwen4_exp.get_req_to_token_pool = lambda: pool
    qwen4_exp._ple_track_targets = (lambda fb, b: None) if track_offset is None else (
        lambda fb, b: (torch.tensor([5], device='cuda'), torch.tensor([track_offset], device='cuda')))
    T = hidden.shape[0]
    mode = SimpleNamespace(is_target_verify=lambda: False, is_decode=lambda: False)
    batch = SimpleNamespace(processed_tokens=T, physical_tokens=T, use_decode_fast_path=False, mode=mode,
                            lengths=torch.tensor([T], device='cuda'), row_width=T,
                            req_indices=torch.zeros(T, dtype=torch.long, device='cuda'),
                            token_offsets=torch.arange(T, device='cuda'),
                            valid_tokens=torch.ones(T, dtype=torch.bool, device='cuda'),
                            state_indices=torch.tensor([2], device='cuda'))
    layer.ple_embedding = type('E', (), {'enable_ple_fusion': False, '__call__': lambda self, b, fb: embeddings})()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn(layer, hidden, None, batch)
    torch.cuda.synchronize()
    return out, pool.cache.clone(), (torch.cuda.max_memory_allocated() - base) / 2**20


ok = True
for T, track in ((4097, None), (8192, 3), (20000, 12345), (32768, 32768), (32768, None)):
    hidden = torch.randn(T, C, device='cuda').bfloat16()
    embeddings = torch.randn(T, PLE_DIM, device='cuda').bfloat16()
    ref, ref_state, ref_mb = run(Layer.forward, hidden, embeddings, track)
    got, got_state, got_mb = run(lambda *a: ple_blocks.ple_forward(Layer.forward, *a), hidden, embeddings, track)
    rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
    same, state_rel = torch.equal(ref, got), ((got_state.float() - ref_state.float()).norm()
                                               / ref_state.float().norm()).item()
    ok &= rel < 1e-3 and state_rel < 1e-3
    print(f'{T:6d} tokens, track {track}: bitwise {same}, relative L2 {rel:.1e} (state {state_rel:.1e}); '
          f'peak {ref_mb:.0f} -> {got_mb:.0f} MiB')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
