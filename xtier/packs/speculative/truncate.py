"""Verify truncated at the draft's confidence (XTIER_SPEC_TRUNCATE_Q, default 0.2; from the GLM-5.3 pipeline).

After the draft, per sampled request of a chain draft (top-k 1): keep the leading drafts with q(x) >= the threshold, at
least one (greedy requests, top_k 1, keep all). The q row of the first dropped draft becomes zero: SGLang's chain
sampler rejects a proposal its draft gives no mass and resamples from the residual (p - q)+, which is p for an all-zero
q row, so the verify is exactly that of the shorter chain and stays lossless (its block verification too). The dropped
verify tokens skip their routed experts through XTier's decode-path token flags (xtier.backend.method.active_flags):
their outputs are unused, as they feed neither the accept test nor, attention being causal, the kept tokens.
"""

import triton
import triton.language as tl

BLOCK_V = 4096
_flags = []   # the core's token flags once enable() ran


@triton.jit
def _truncate(PROBS, TOKENS, TOPK, FLAGS, TH, stride_b, stride_s, V, S: tl.constexpr, HAS_TOPK: tl.constexpr,
              BLOCK: tl.constexpr, BLOCK_V: tl.constexpr):
    """Request b: q [S, V] of its drafts TOKENS[b, 1:]; keep = leading drafts with q >= TH (>= 1, all for top_k 1);
    flags of its S + 1 verify tokens = position <= keep; q row keep := 0 when a draft is dropped."""
    b = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    drafts = j < S
    tok = tl.load(TOKENS + b * (S + 1) + 1 + j, mask=drafts, other=0).to(tl.int64)
    q = tl.load(PROBS + b * stride_b + j * stride_s + tok, mask=drafts, other=1.0)
    keep = tl.maximum(tl.min(tl.where(drafts & (q < TH), j, S)), 1)
    if HAS_TOPK:
        keep = tl.where(tl.load(TOPK + b) == 1, S, keep)
    tl.store(FLAGS + b * (S + 1) + j, (j <= keep).to(tl.int32), mask=j < S + 1)
    if keep < S:
        row = PROBS + b * stride_b + keep * stride_s
        for v in range(0, V, BLOCK_V):
            o = v + tl.arange(0, BLOCK_V)
            tl.store(row + o, tl.zeros([BLOCK_V], dtype=tl.float32), mask=o < V)


def enable(device):
    """Before the CUDA graphs are captured: the target's verify graphs must read the token flags."""
    from xtier.backend.method import active_flags
    if not _flags:
        _flags.append(active_flags(device))


def apply(batch, topk: int, threshold: float) -> bool:
    """On the verify's stream before its forward: truncate the verify input of ``batch`` in place, flag the dropped
    tokens. False when there is nothing to truncate (greedy batch, no or one draft step, tree drafts)."""
    if not _flags or topk != 1 or batch.forward_mode.is_idle():
        return False
    verify_input = batch.spec_info
    probs = getattr(verify_input, 'draft_probs', None)
    if probs is None or probs.dim() != 3:
        return False
    flags = _flags[0]
    bs, steps, vocab = probs.shape
    if steps < 2 or verify_input.draft_token.numel() != bs * (steps + 1) or bs * (steps + 1) > flags.numel():
        return False
    if probs.stride(-1) != 1:
        probs = verify_input.draft_probs = probs.contiguous()
    top_ks = getattr(batch.sampling_info, 'top_ks', None)
    has_topk = top_ks is not None and top_ks.device == probs.device
    _truncate[(bs,)](probs, verify_input.draft_token, top_ks if has_topk else flags, flags, threshold,
                     probs.stride(0), probs.stride(1), vocab, S=steps, HAS_TOPK=has_topk,
                     BLOCK=triton.next_power_of_2(steps + 1), BLOCK_V=BLOCK_V)
    return True


def reset():
    """After the verify forward was enqueued: every token computes again."""
    _flags[0].fill_(1)
