"""DSpark's static verify truncated at the draft's confidence (XTIER_DSPARK_TRUNCATE = survival threshold; 0 = off).

SGLang's DSpark draft carries a confidence head (per drafted token, the probability that the target accepts it given
its accepted prefix), which SGLang builds only for its ragged verify modes; here it is built for the static verify as
well. A request keeps its leading drafts while their survival (the product of those probabilities) stays >= the
threshold, at least one. The kept count k caps the request's verify at k + 1 rows the way SGLang's cap-accept mode
does, in every accept the static verify can run: the static epilogue's in-graph greedy accept reads its verify lengths,
the sampling accept region and the eager accept take the caps as their cutoff. A dropped draft is thus rejected
whatever the target computes for it, the step's tokens are those of a k-token block (lossless), and the dropped rows
skip their routed experts through XTier's decode-path token flags (attention being causal, they feed neither the kept
rows nor the accept): on a hybrid MoE mostly CPU work. XTIER_DSPARK_CONFIDENCE=1 builds the head without truncating
(SGLang's STS data collection, SGLANG_DSPARK_STS_COLLECT_PATH, records it).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)
THRESHOLD = float(os.environ.get('XTIER_DSPARK_TRUNCATE', '0'))
COLLECT = os.environ.get('XTIER_DSPARK_CONFIDENCE', '0') == '1'
DSPARK = 'sglang.srt.speculative.dspark_components'
HEAD = 'sglang.srt.models.dspark.build_confidence_head'
WORKER = f'{DSPARK}.dspark_worker_v2.DSparkWorkerV2'
PLANNER = f'{DSPARK}.dspark_planner.DSparkVerifyPlanner'
EPILOGUE = f'{DSPARK}.dspark_verify.DsparkVerifyEpilogue'
EXECUTOR = f'{DSPARK}.dspark_verify.TargetVerifyExecutor'
ACCEPT = f'{DSPARK}.dspark_verify.accept_draft_tokens'
MAX_BS = 64
_state = {}   # caps: int32 [MAX_BS] verify rows per request (stride when not truncated), flags, stride; keep: this step


class _Cutoff:
    """What accept_draft_tokens reads of a ragged layout: the verify lengths."""

    def __init__(self, verify_lens):
        self.verify_lens = verify_lens


def _confidence_head(original, config, *args, **kwargs):
    head = original(config, *args, **kwargs)
    if head is not None:
        return head
    from sglang.srt.models.dspark import DSparkConfidenceHead   # as SGLang builds it outside static verify

    markov_rank = int(getattr(config, 'markov_rank', 0))
    with_markov = bool(getattr(config, 'confidence_head_with_markov', markov_rank > 0))
    logger.info('xtier: DSpark confidence head in static verify (%s)',
                f'verify truncated at survival < {THRESHOLD}' if THRESHOLD > 0 else 'not truncating')
    return DSparkConfidenceHead(hidden_size=int(config.hidden_size), markov_rank=markov_rank, with_markov=with_markov)


def _alloc_memory_pool(result, self, *args, **kwargs):
    """Before the target's CUDA graphs are captured: the token flags they read, the caps the accept graphs read."""
    from xtier.backend.method import active_flags

    stride = int(self.verify_num_draft_tokens)
    _state.update(flags=active_flags(self.device), stride=stride,
                  caps=torch.full((MAX_BS,), stride, dtype=torch.int32, device=self.device))
    return result


def _schedule_layout(original, self, *args, **kwargs):
    """The step's caps, from the confidence the worker hands its planner (static verify: no layout)."""
    layout = original(self, *args, **kwargs)
    _state.pop('keep', None)
    if 'caps' not in _state:
        return layout
    if _state.pop('dirty', False):   # the accepts after the previous verify have read its caps
        _state['caps'].fill_(_state['stride'])
    confidence = kwargs.get('confidence')
    if layout is not None or confidence is None or self.mode_value != 'static':
        return layout
    bs, gamma = confidence.shape
    if bs > MAX_BS or gamma + 1 != _state['stride'] or bs * (gamma + 1) > _state['flags'].numel():
        return layout
    survival = torch.cumprod(confidence.float(), dim=1)
    keep = (survival >= THRESHOLD).sum(dim=1).clamp_(min=1)
    _state['caps'][:bs].copy_(keep + 1)
    _state.update(keep=keep, dirty=True)
    return layout


def _begin_static_step(result, self, bs, *args, **kwargs):
    if 'keep' in _state:
        self.verify_lens_buf[:bs].copy_(_state['caps'][:bs])
        self._static_step_state = None   # refilled at the next untruncated step
    return result


def _accept(original, *args, **kwargs):
    caps = _state.get('caps')
    if caps is not None and kwargs.get('cutoff_layout') is None:
        kwargs['cutoff_layout'] = _Cutoff(caps[:kwargs['candidates'].shape[0]])
    return original(*args, **kwargs)


def _run_non_compact(original, self, *args, **kwargs):
    keep = _state.get('keep')
    if keep is None:
        return original(self, *args, **kwargs)
    stride, flags = _state['stride'], _state['flags']
    rows = keep.numel() * stride
    flags[:rows].copy_((torch.arange(stride, device=keep.device) <= keep[:, None]).view(-1))
    try:
        return original(self, *args, **kwargs)
    finally:
        flags[:rows].fill_(1)   # after the verify on its stream; the caps stay for the accepts that follow it
        _state.pop('keep', None)


def install(registry, hook_type):
    if THRESHOLD <= 0 and not COLLECT:
        return
    registry.register(HEAD, _confidence_head, hook_type.AROUND)
    if THRESHOLD > 0:
        registry.register(f'{WORKER}.alloc_memory_pool', _alloc_memory_pool, hook_type.AFTER)
        registry.register(f'{PLANNER}.schedule_layout', _schedule_layout, hook_type.AROUND)
        registry.register(f'{EPILOGUE}.begin_static_step', _begin_static_step, hook_type.AFTER)
        registry.register(ACCEPT, _accept, hook_type.AROUND)
        registry.register(f'{EXECUTOR}.run_non_compact', _run_non_compact, hook_type.AROUND)
