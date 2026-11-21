"""Acceleration pack for SGLang's EAGLE speculative decoding (any model; from the GLM-5.3 pipeline's lsglang patch).

- Draft temperature = request temperature x XTIER_SPEC_DRAFT_TEMP_SCALE (default 0.8) for sampled requests: lossless
  for any scale (the accept test uses the q the draft sampled from); 0.8 gave the 1017-era GLM pipeline the highest
  expected acceptance sum(min(p, q)). Greedy requests (top_k 1) keep proposing the draft argmax (SGLang's code).
- Verify truncated at the draft's confidence (truncate.py, XTIER_SPEC_TRUNCATE_Q, default 0.2, 0 = off): the drafts
  after the first one with q(x) < 0.2 are not verified (lossless, as a shorter chain) and their verify tokens skip the
  routed experts (XTier's decode-path token flags), on a hybrid MoE mostly CPU work.

(Hot-token draft vocab rejection sampling was upstreamed into SGLang 4ce41378f9, which also introduces
``use_fused_draft_sample``; XTier no longer overrides ``speculative_use_rejection_sampling`` or scatters q itself.)

All act only with rejection sampling on; XTIER_SPEC_HOT_RS=0 turns the pack off.
"""

import logging
import os

logger = logging.getLogger(__name__)
WORKER = 'sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker'
SPEC_WORKER = 'sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2'
PROPOSAL = 'sglang.srt.speculative.spec_utils.sample_draft_proposal'
RENORM = 'sglang.srt.speculative.spec_utils.renorm_draft_probs'
TEMP_SCALE = float(os.environ.get('XTIER_SPEC_DRAFT_TEMP_SCALE', '0.8'))
TRUNCATE_Q = float(os.environ.get('XTIER_SPEC_TRUNCATE_Q', '0.2'))


def _alloc_memory_pool(original, self, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec
    result = original(self, *args, **kwargs)
    if get_spec().speculative_use_rejection_sampling and TRUNCATE_Q > 0 and self.topk == 1:
        from xtier.packs.speculative import truncate
        truncate.enable(self.device)   # the target's CUDA graphs are captured after the memory pools
        logger.info('xtier: verify truncated at draft q < %s', TRUNCATE_Q)
    return result


def _verify(original, self, batch, *args, **kwargs):
    from xtier.packs.speculative import truncate
    truncated = truncate.apply(batch, self.topk, TRUNCATE_Q)
    try:
        return original(self, batch, *args, **kwargs)
    finally:
        if truncated:
            truncate.reset()


def _proposal(original, next_token_logits, temperatures, *args, **kwargs):
    if TEMP_SCALE != 1.0 and temperatures is not None:
        temperatures = temperatures * TEMP_SCALE   # greedy rows (top_k 1) propose the argmax in SGLang's code
    return original(next_token_logits, temperatures, *args, **kwargs)


class _ScaledTemperatures:
    """A batch's sampling info with the draft temperatures."""

    def __init__(self, sampling_info):
        self._info = sampling_info
        self.temperatures = sampling_info.temperatures * TEMP_SCALE

    def __getattr__(self, name):
        return getattr(self._info, name)


def _renorm(original, next_token_logits, sampling_info, use_rejection_sampling, *args, **kwargs):
    """The draft extend after a prefill proposes from renorm_draft_probs, not sample_draft_proposal."""
    if TEMP_SCALE != 1.0 and use_rejection_sampling and sampling_info is not None:
        sampling_info = _ScaledTemperatures(sampling_info)
    return original(next_token_logits, sampling_info, use_rejection_sampling, *args, **kwargs)


def install(registry, hook_type):
    from xtier.packs.speculative import dspark_truncate

    dspark_truncate.install(registry, hook_type)
    if os.environ.get('XTIER_SPEC_HOT_RS', '1') == '0':
        return
    registry.register(f'{WORKER}.alloc_memory_pool', _alloc_memory_pool, hook_type.AROUND)
    registry.register(PROPOSAL, _proposal, hook_type.AROUND)
    registry.register(RENORM, _renorm, hook_type.AROUND)
    if TRUNCATE_Q > 0:
        registry.register(f'{SPEC_WORKER}.verify', _verify, hook_type.AROUND)
