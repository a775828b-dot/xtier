"""Inkling-2-Small DSpark MTP (5-layer draft) SM120 acceleration hooks:
- Markov head step_bias pad-to-2 on SM120 (XTIER_INKLING2_MARKOV_PAD2, default 1):
  Inkling2MTPHead.step_bias runs gamma=7 times per decode step ([bs, 256] x [201024, 256] + bias).
  At bs=1, SM120 cuBLAS picks a slow legacy GEMV (870 us / 7 calls); padding to M=2 switches to
  the Tensor-Core path (312 us / 7 calls, -0.56 ms/step, 0 extra VRAM, exact BF16).
- DSpark draft LM head FP8 copy (XTIER_DRAFT_LMHEAD_FP8, default 1 on SM89+):
  project_through_lm_head reads [201024, 4608] BF16 (1.73 GiB -> 1.53 ms/step); row-scaled FP8
  reads 0.86 GiB -> 0.85-0.92 ms/step (-0.61..-0.69 ms/step, 100% lossless because target verify
  uses the unmodified BF16 lm_head with exact rejection sampling).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
MTP_MODEL = 'sglang.srt.models.inkling2_mtp.Inkling2ForConditionalGenerationMTP'
MTP_HEAD = 'sglang.srt.models.inkling2_mtp.Inkling2MTPHead'
DRAFT_PROJECT = 'sglang.srt.models.dspark.project_through_lm_head'
DSPARK_WORKER = 'sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2'


def _quantize_lm_head(lm_head):
    weight = lm_head.weight.data
    n, k = weight.shape
    qweight = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n, 1), dtype=torch.float32, device=weight.device)
    for start in range(0, n, 8192):
        rows = weight[start:start + 8192].float()
        sc = rows.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        qweight[start:start + 8192] = (rows / sc).to(torch.float8_e4m3fn)
        scale[start:start + 8192] = sc
    lm_head._xtier_draft_fp8 = (qweight, scale.t().contiguous())
    logger.info('xtier: Inkling2 DSpark draft LM head FP8 copy %s (%.2f GiB)',
                tuple(qweight.shape), qweight.numel() / 2**30)


def _attach_shared(result, self, *args, **kwargs):
    head = getattr(self, 'lm_head', None)
    if (head is not None and getattr(head, 'weight', None) is not None and head.weight.dtype == torch.bfloat16
            and head.weight.dim() == 2 and getattr(head, '_xtier_draft_fp8', None) is None):
        _quantize_lm_head(head)
    return result


def _draft_project(original, hidden, lm_head):
    packed = getattr(lm_head, '_xtier_draft_fp8', None)
    if packed is None:
        return original(hidden, lm_head)
    qweight, scale = packed
    x = hidden.reshape(-1, hidden.shape[-1]).float()
    sx = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    out = torch._scaled_mm((x / sx).to(torch.float8_e4m3fn), qweight.t(), scale_a=sx, scale_b=scale,
                           out_dtype=torch.bfloat16)
    return out.view(*hidden.shape[:-1], qweight.shape[0])


def _markov_step_bias(original, self, prev_tokens, *, hidden_states, state):
    if (prev_tokens.numel() == 1 and prev_tokens.is_cuda
            and self._fold_weight is not None and self._fold_bias is not None):
        rnn_state, f_rnn = state
        rnn_state, latent = self._rnn_step(rnn_state, prev_tokens, hidden_states, f_rnn)
        if latent.dim() == 2 and latent.shape[0] == 1:
            bias = torch.addmm(
                self._fold_bias,
                latent.to(self._fold_weight.dtype).expand(2, -1).contiguous(),
                self._fold_weight.T,
            )[:1]
            if self.vocab_shard is None:
                bias = self._gather_logits(bias)
            return bias, (rnn_state, f_rnn)
        bias = torch.addmm(self._fold_bias, latent.to(self._fold_weight.dtype), self._fold_weight.T)
        if self.vocab_shard is None:
            bias = self._gather_logits(bias)
        return bias, (rnn_state, f_rnn)
    return original(self, prev_tokens, hidden_states=hidden_states, state=state)




def _build_acceptance_estimator(original, self, *args, **kwargs):
    est = original(self, *args, **kwargs)
    if est is not None:
        return est
    if (
        getattr(self.draft_model, 'confidence_head', None) is None
        and not getattr(self, '_greedy_draft', False)
        and float(os.environ.get('XTIER_DSPARK_TRUNCATE', '0')) > 0
    ):
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.speculative.online_acceptance_estimator import (
            OnlineAcceptanceEstimator,
        )

        logger.info(
            'xtier: Inkling2 static-verify OnlineAcceptanceEstimator enabled (gamma=%d, threshold=%s)',
            self.gamma, os.environ.get('XTIER_DSPARK_TRUNCATE'),
        )
        return OnlineAcceptanceEstimator(
            num_positions=self.gamma,
            device=self.device,
            log_rank=get_parallel().tp_rank == 0,
        )
    return None


def _observe_acceptance(original, self, *, features, accept, layout):
    if layout is None:
        from xtier.packs.speculative import dspark_truncate

        caps = dspark_truncate._state.get('caps')
        if caps is not None:
            layout = dspark_truncate._Cutoff(caps[:features.shape[0]])
    return original(self, features=features, accept=accept, layout=layout)


def _mix_norm(original, self, targets, row, norm):
    # SGLang Triton _mix_rmsnorm indexes targets with int32 row * stride(0);
    # for rows * taps * hidden >= 2^31 (rows >= 58255 at taps=8, hidden=4608),
    # signed int32 overflow triggers CUDA_ERROR_ILLEGAL_ADDRESS. Chunk rows safely.
    stride0 = targets.stride(0) if targets.ndim >= 2 else 0
    max_rows = (1 << 30) // stride0 if stride0 > 0 else targets.shape[0]
    if targets.shape[0] <= max_rows or max_rows <= 0:
        return original(self, targets, row, norm)
    out = targets.new_empty((targets.shape[0], targets.shape[-1]))
    for start in range(0, targets.shape[0], max_rows):
        end = min(start + max_rows, targets.shape[0])
        out[start:end] = original(self, targets[start:end], row, norm)
    return out


def install(registry, hook_type):
    registry.register(f'{MTP_MODEL}._mix_norm', _mix_norm, hook_type.AROUND)
    if os.environ.get('XTIER_DRAFT_LMHEAD_FP8', '1') != '0' and torch.cuda.get_device_capability() >= (8, 9):
        registry.register(f'{MTP_MODEL}.attach_shared_modules', _attach_shared, hook_type.AFTER)
        registry.register(DRAFT_PROJECT, _draft_project, hook_type.AROUND)
    if os.environ.get('XTIER_INKLING2_MARKOV_PAD2', '1') != '0' and torch.cuda.get_device_capability()[0] == 12:
        registry.register(f'{MTP_HEAD}.step_bias', _markov_step_bias, hook_type.AROUND)
    if float(os.environ.get('XTIER_DSPARK_TRUNCATE', '0')) > 0:
        registry.register(f'{DSPARK_WORKER}._build_acceptance_estimator', _build_acceptance_estimator, hook_type.AROUND)
        registry.register(f'{DSPARK_WORKER}._observe_acceptance', _observe_acceptance, hook_type.AROUND)
