"""GLM-5.3's DSA indexer below SM90 (RTX 30 / 40 series): its DeepGEMM calls in torch, FP32.

SGLang's k-pool indexer (dsa_indexer_kpool.py, kpool_plan.py) calls DeepGEMM directly: fp8_mqa_logits in prefill,
fp8_paged_mqa_logits in decode, their schedule (get_paged_mqa_logits_metadata) and get_num_sms;
SGLANG_FP8_PAGED_MQA_LOGITS_TORCH only switches the other indexer. DeepGEMM imports below SM90 but asserts
"Unsupported architecture" in every kernel. Under tools/arch_check.py --emulate the calls reached the real SM120 and
passed; the RTX 4090 found them. Below SM90 these four functions are replaced, everywhere: the logits as SGLang's
torch fallbacks (dsa/utils.py, dsv4/indexer.py) compute them, with the per-head scores in FP32 where those round them
to BF16 (FP8 products are exact in FP32, accumulated in FP32 as in DeepGEMM), prefill rows chunked to bound the
[rows, keys] scores; no schedule (these logits take none); the SM count from torch.
"""
import torch

SCORE_BUDGET = 256 << 20   # bytes of FP32 scores per prefill chunk
_arange = {}


def _positions(n, device):
    key = (n, device)
    if key not in _arange:
        _arange[key] = torch.arange(n, device=device)
    return _arange[key]


def fp8_mqa_logits(q_fp8, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0):
    """Prefill: logits [q, keys] = sum_h w[q, h] relu(q_h . k) * k_scale, outside [ks, ke) 0 (-inf if clean)."""
    k_fp8, k_scale = kv
    num_q, num_heads, head_dim = q_fp8.shape
    num_kv = k_fp8.shape[0]
    k_t = k_fp8.reshape(num_kv, head_dim).float().t()
    storage = torch.zeros(num_q, (num_kv + 3) // 4 * 4, dtype=torch.float32, device=q_fp8.device)
    logits = storage[:, :num_kv]   # the padded row stride SGLang's top-k expects
    w = weights.float()
    rows = max(1, SCORE_BUDGET // max(1, num_kv * 4))
    for r0 in range(0, num_q, rows):
        q = q_fp8[r0:r0 + rows].float()
        for h in range(num_heads):
            logits[r0:r0 + rows].addcmul_(torch.mm(q[:, h], k_t).relu_(), w[r0:r0 + rows, h:h + 1])
    logits *= k_scale.reshape(1, num_kv).float()
    positions = _positions(num_kv, q_fp8.device).unsqueeze(0)
    valid = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))
    return logits.masked_fill_(~valid, float('-inf') if clean_logits else 0.0)


def fp8_paged_mqa_logits(q_fp8, kvcache_fp8, weights, context_lens, block_table, schedule_metadata, max_seq_len,
                         clean_logits=False, indices=None):
    """Decode (one query per row, graph-capturable): logits [rows, max_seq_len] over each row's pages."""
    if indices is not None:
        raise NotImplementedError('xtier: torch paged MQA logits take no index tensor')
    if context_lens.dim() == 2:
        if context_lens.shape[1] != 1:
            raise NotImplementedError('xtier: torch paged MQA logits take one query per row (next_n 1)')
        context_lens = context_lens[:, 0]
    batch, _, num_heads, head_dim = q_fp8.shape
    block = kvcache_fp8.shape[1]
    pages = block_table.shape[1]
    flat = kvcache_fp8.view(torch.uint8).reshape(-1, block * (head_dim + 4))
    rows = flat[block_table.clamp(min=0)]   # [batch, pages, block * (head_dim + 4)]
    values = rows[..., :block * head_dim].contiguous().view(torch.float8_e4m3fn).float()
    values = values.reshape(batch, pages * block, head_dim)
    scales = rows[..., block * head_dim:].contiguous().view(torch.float32).reshape(batch, pages * block)
    scores = torch.bmm(values, q_fp8[:, 0].float().transpose(1, 2))   # [batch, keys, heads], FP32
    logits = (scores.relu_() * weights.float().unsqueeze(1)).sum(dim=2) * scales
    positions = _positions(pages * block, logits.device).unsqueeze(0)
    fill = float('-inf') if clean_logits else 0.0
    logits = logits.masked_fill(positions >= context_lens.unsqueeze(1), fill)
    if pages * block < max_seq_len:
        logits = torch.nn.functional.pad(logits, (0, max_seq_len - pages * block), value=fill)
    return logits[:, :max_seq_len]


def get_paged_mqa_logits_metadata(*args, **kwargs):
    return None


def get_num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _around(fn):
    def hook(original, *args, **kwargs):
        return fn(*args, **kwargs)
    return hook


# AROUND hooks (registered below SM90) by target; get_num_sms is a torch op, not a Python function
HOOKS = {f'deep_gemm.{fn.__name__}': _around(fn)
         for fn in (fp8_mqa_logits, fp8_paged_mqa_logits, get_paged_mqa_logits_metadata, get_num_sms)}
