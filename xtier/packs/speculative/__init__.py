"""Acceleration pack for SGLang's EAGLE speculative decoding (any model; from the GLM-5.3 pipeline's lsglang patch).

- Rejection sampling with a hot-token draft vocab: SGLang refuses ``--speculative-use-rejection-sampling`` together
  with ``--speculative-token-map`` ("FIXME: support reduced (hot) draft vocab by scattering draft probs into the target
  vocab"). Here the draft's proposal distribution q stays over the hot vocab inside the draft (its CUDA graph buffers
  get the hot width, as SGLang's draft-extend graph runner already does) and is scattered into the target vocab just
  before the verify input is built: q is zero outside the map, where the draft can never propose, so the accept test
  coin * q(x) < p(x) and the residual distribution (p - q)+ stay exact.
- Draft temperature = request temperature x XTIER_SPEC_DRAFT_TEMP_SCALE (default 0.8) for sampled requests: lossless
  for any scale (the accept test uses the q the draft sampled from); 0.8 gave the 1017-era GLM pipeline the highest
  expected acceptance sum(min(p, q)). Greedy requests (top_k 1) keep proposing the draft argmax (SGLang's code).
- Verify truncated at the draft's confidence (truncate.py, XTIER_SPEC_TRUNCATE_Q, default 0.2, 0 = off): the drafts
  after the first one with q(x) < 0.2 are not verified (lossless, as a shorter chain) and their verify tokens skip the
  routed experts (XTier's decode-path token flags), on a hybrid MoE mostly CPU work.

All act only with rejection sampling on; XTIER_SPEC_HOT_RS=0 turns the pack off.
"""

import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)
WORKER = 'sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker'
SPEC_WORKER = 'sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2'
DRAFT_GRAPHS = 'sglang.srt.speculative.eagle_draft_cuda_graph_runner.EAGLEDraftCudaGraphRunner'
DRAFT_INPUT = 'sglang.srt.speculative.eagle_info.EagleDraftInput'
VERIFY_INPUT = 'sglang.srt.speculative.eagle_worker_common.build_eagle_verify_input'
PROPOSAL = 'sglang.srt.speculative.spec_utils.sample_draft_proposal'
RENORM = 'sglang.srt.speculative.spec_utils.renorm_draft_probs'
TEMP_SCALE = float(os.environ.get('XTIER_SPEC_DRAFT_TEMP_SCALE', '0.8'))
TRUNCATE_Q = float(os.environ.get('XTIER_SPEC_TRUNCATE_Q', '0.2'))
_hot = {}   # 'ids': hot token ids (long, on the device), 'target_vocab': the target's vocab size


def _alloc_memory_pool(original, self, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec
    if not get_spec().speculative_use_rejection_sampling:
        return original(self, *args, **kwargs)
    # SGLang's vocab check is the last statement and the only reader of the flag in there: hide it for this call
    with get_spec().override(speculative_use_rejection_sampling=False):
        result = original(self, *args, **kwargs)
    hot = getattr(self, 'hot_token_id', None)
    target_vocab = self.target_worker.model_config.vocab_size
    if hot is not None and hot.shape[0] != target_vocab:
        _hot.update(ids=hot.to(torch.long), target_vocab=target_vocab)
        logger.info('xtier: rejection sampling with a hot draft vocab of %d of %d tokens (q scattered before verify)',
                    hot.shape[0], target_vocab)
    if TRUNCATE_Q > 0 and self.topk == 1:
        from xtier.packs.speculative import truncate
        truncate.enable(self.device)   # the target's CUDA graphs are captured after the memory pools
        logger.info('xtier: verify truncated at draft q < %s', TRUNCATE_Q)
    return result


@contextlib.contextmanager
def _hot_vocab(model_config):
    """The draft graphs' q buffer has the draft's width: the hot vocab while the runner builds and captures."""
    if 'ids' not in _hot:
        yield
        return
    full = model_config.vocab_size
    model_config.vocab_size = int(_hot['ids'].shape[0])
    try:
        yield
    finally:
        model_config.vocab_size = full


def _draft_graphs_init(original, self, *args, **kwargs):
    worker = kwargs.get('eagle_worker', args[0] if args else None)
    runner = getattr(worker, 'draft_runner', None) if worker is not None else None
    if runner is None or 'ids' not in _hot:
        return original(self, *args, **kwargs)
    with _hot_vocab(runner.model_config):
        return original(self, *args, **kwargs)


def _idle_input(original, cls, *args, **kwargs):
    if 'ids' in _hot and kwargs.get('vocab_size') == _hot['target_vocab']:
        kwargs['vocab_size'] = int(_hot['ids'].shape[0])
    return original(cls, *args, **kwargs)


def _scatter(draft_probs):
    ids = _hot['ids']
    if ids.device != draft_probs.device:
        ids = _hot['ids'] = ids.to(draft_probs.device)
    full = draft_probs.new_zeros(*draft_probs.shape[:-1], _hot['target_vocab'])
    full.index_copy_(full.dim() - 1, ids, draft_probs)
    return full


def _verify_input(original, *args, **kwargs):
    if 'ids' in _hot:
        if 'draft_probs' in kwargs:
            probs = kwargs['draft_probs']
            if probs is not None and probs.shape[-1] == _hot['ids'].shape[0]:
                kwargs['draft_probs'] = _scatter(probs)
        elif len(args) > 5 and args[5] is not None and args[5].shape[-1] == _hot['ids'].shape[0]:
            args = (*args[:5], _scatter(args[5]), *args[6:])
    return original(*args, **kwargs)


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
    registry.register(f'{DRAFT_GRAPHS}.__init__', _draft_graphs_init, hook_type.AROUND)
    registry.register(f'{DRAFT_INPUT}.create_idle_input', _idle_input, hook_type.AROUND)
    registry.register(VERIFY_INPUT, _verify_input, hook_type.AROUND)
    registry.register(PROPOSAL, _proposal, hook_type.AROUND)
    registry.register(RENORM, _renorm, hook_type.AROUND)
    if TRUNCATE_Q > 0:
        registry.register(f'{SPEC_WORKER}.verify', _verify, hook_type.AROUND)
