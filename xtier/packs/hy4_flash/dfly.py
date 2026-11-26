"""Zero-patch native 5-layer DFly2 speculative decoding for Hunyuan-4-Flash.

Implements the bundled ``model.dfly_layers.*`` (~0.85B, 5 dense MLA layers) block-parallel
speculative drafter as a zero-patch XTier plugin using SGLang's ``SpeculativeAlgorithm.register``
and ``HookRegistry``. Mirrors upstream SGLang PR #51938 and automatically yields to upstream's
builtin ``SpeculativeAlgorithm.DFLY`` once merged.

Key properties:
- Zero extra KV cache VRAM: each draft layer directly reads the target DSA layer's KV pages
  (``dfly_target_kv_layers = [25, 29, 33, 37, 41]``) and captured target IndexPool top-2048 picks,
  writing its own 6-token draft block into the pending ``out_cache_loc`` verify slots that the
  target forward immediately overwrites during verification.
- Single fused CUDA graph per decode step: ``HYV4FlashDFly.draft_into_verify`` runs at the entry of
  ``HYV4FlashForConditionalGeneration.forward`` during ``TARGET_VERIFY``.
- Full compatibility with XTier's Hy4-Flash pack: ModelOpt NVFP4 checkpoints (BF16 DFly weights,
  optionally compressed to FP8 by ``fp8_weights.py``), 1M Hybrid SWA/NVFP4 KV pool
  (``gather_unpack_nvfp4_to_fp8`` pre-sized for ``max_rows * draft_index_topk``), FP8 IndexPool
  logits, and NVMe Engram RowStore (verify history staged and committed post-accept).
"""

from __future__ import annotations

from copy import deepcopy
import dataclasses
import logging
import os
from typing import Iterable, Optional, Tuple

import msgspec
import torch
from torch import nn
import triton
import triton.language as tl

from xtier.arch import e4m3_f32

logger = logging.getLogger(__name__)

CHECKPOINT_PREFIX = 'model.dfly_layers.'
DEFAULT_DFLY_BLOCK_SIZE = 6

# Hook targets
REJECT_HYV4 = 'sglang.srt.arg_groups.model_overrides.deepseek_v2._reject_unsupported_hyv4_flash'
HY4_CONFIG_INIT = 'sglang.srt.configs.hy_v4_flash.HYV4FlashConfig.__init__'
CREATE_DUMMY_VERIFY = 'sglang.srt.speculative.spec_info.create_dummy_verify_input'
SPEC_PREPARE_DECODE = 'sglang.srt.speculative.spec_utils.spec_prepare_for_decode'
GRAPH_GET_SPEC_INFO = 'sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.get_spec_info'
SCHED_ADD_REQ = 'sglang.srt.managers.scheduler.Scheduler._add_request_to_queue'
ROW_LAYOUT_INIT = 'sglang.srt.models.hunyuan_v4_flash._RowLayout.__init__'
INDEXPOOL_UPDATE = 'sglang.srt.models.hunyuan_v4_flash.FlashIndexPool._update_cache'
INDEXPOOL_SELECT = 'sglang.srt.models.hunyuan_v4_flash.FlashIndexPool._select'
INDEXPOOL_FWD = 'sglang.srt.models.hunyuan_v4_flash.FlashIndexPool.forward'
ENGRAM_HASHER_FWD = 'sglang.srt.models.hunyuan_v4_flash.FlashEngramHasher.forward'
HY4_MODEL_INIT = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashForConditionalGeneration.__init__'
HY4_LOAD_WEIGHTS = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashForConditionalGeneration.load_weights'
HY4_FWD = 'sglang.srt.models.hunyuan_v4_flash.HYV4FlashForConditionalGeneration.forward'

_in_target_verify_select = [False]


def _is_dfly_algo(algo) -> bool:
    if algo is None:
        return False
    fn = getattr(algo, 'is_dfly', None)
    if callable(fn):
        return bool(fn())
    return getattr(algo, 'name', None) == 'DFLY'


def _has_builtin_dfly() -> bool:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    return hasattr(SpeculativeAlgorithm, 'DFLY')


class HYV4FlashDFlyConfig(msgspec.Struct, frozen=True, kw_only=True):
    num_layers: int
    mask_token_id: int
    target_kv_layers: tuple[int, ...]
    target_index_layers: tuple[int, ...]
    target_index_topk: int
    draft_index_topk: int
    block_size: int


def parse_dfly_config(text_config, block_size: int) -> HYV4FlashDFlyConfig:
    from sglang.srt.configs.hy_v4_flash import SPARSE_ATTENTION

    def get(name, default=None):
        return getattr(text_config, f'dfly_{name}', default)

    num_layers = get('num_hidden_layers')
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError('DFly requires a positive dfly_num_hidden_layers')
    if get('mtp_fusion', 'slot0') != 'slot0':
        raise ValueError('DFly requires slot0 fusion')
    if get('enable_hidden_correction', True) is not True:
        raise ValueError('DFly requires the hidden correction')
    if get('enable_ihc', False) or get('mlp_type', 'dense') != 'dense':
        raise ValueError('DFly requires dense FFNs without iHC')
    if get('attn_causal', True) is not True:
        raise ValueError('DFly requires causal attention')
    if get('context_local_window') not in (None, 0):
        raise ValueError('DFly context_local_window is not supported')
    mask_token_id = get('mask_token_id')
    if type(mask_token_id) is not int or not 0 <= mask_token_id < text_config.vocab_size:
        raise ValueError('DFly dfly_mask_token_id is outside the vocabulary')
    topk = get('target_index_topk', text_config.index_topk)
    if topk != text_config.index_topk:
        raise ValueError('DFly target_index_topk disagrees with the target')
    kv_layers = tuple(get('target_kv_layers') or ())
    index_layers = get('target_index_layer')
    if type(index_layers) is int:
        index_layers = [index_layers] * num_layers
    index_layers = tuple(index_layers or ())
    n = text_config.num_hidden_layers
    for name, layers in (('kv', kv_layers), ('index', index_layers)):
        if len(layers) != num_layers or any(type(i) is not int or not 1 <= i < n for i in layers):
            raise ValueError(f'DFly needs one target {name} layer per draft layer')
        if any(text_config.layer_types[i] != SPARSE_ATTENTION for i in layers):
            raise ValueError('DFly target layers must be sparse attention layers')
    if len(set(kv_layers)) != num_layers or len(set(index_layers)) != num_layers:
        raise ValueError('DFly requires distinct target KV and index layers')
    if not 2 <= block_size <= 17:
        raise ValueError('DFly requires a draft block of 2..17 tokens')
    return HYV4FlashDFlyConfig(
        num_layers=num_layers,
        mask_token_id=mask_token_id,
        target_kv_layers=kv_layers,
        target_index_layers=index_layers,
        target_index_topk=topk,
        draft_index_topk=-(-(topk + block_size) // 128) * 128,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Draft top-k packing & per-request state (mirrors dfly_utils.py)
# ---------------------------------------------------------------------------


@triton.jit
def _build_draft_topk_kernel(
    state_topk,
    req_pool_indices,
    verify_slots,
    out_rows,
    out_counts,
    state_layer_stride,
    state_req_stride,
    out_layer_stride,
    out_row_stride,
    WIDTH: tl.constexpr,
    TARGET_TOPK: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    req_row = tl.program_id(0)
    layer = tl.program_id(1)
    req = tl.load(req_pool_indices + req_row).to(tl.int64)
    cols = tl.arange(0, TARGET_TOPK)
    history = tl.load(state_topk + layer * state_layer_stride + req * state_req_stride + cols)
    valid = history >= 0
    packed = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    num_valid = tl.sum(valid.to(tl.int32), axis=0)
    block = tl.arange(0, BLOCK_WIDTH)
    slots = tl.load(verify_slots + req_row * WIDTH + block, mask=block < WIDTH, other=-1).to(tl.int32)
    for j in tl.static_range(WIDTH):
        row = req_row * WIDTH + j
        base = out_rows + layer * out_layer_stride + row * out_row_stride
        tl.store(base + packed, history, mask=valid)
        tl.store(base + num_valid + block, slots, mask=block <= j)
        if layer == 0:
            tl.store(out_counts + row, num_valid + j + 1)


def build_draft_topk(
    *,
    state_topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    verify_slots: torch.Tensor,
    out_rows: torch.Tensor,
    out_counts: torch.Tensor,
) -> None:
    bs, width = verify_slots.shape
    layers, _, target_topk = state_topk.shape
    out_rows[:, : bs * width].fill_(-1)
    _build_draft_topk_kernel[(bs, layers)](
        state_topk,
        req_pool_indices,
        verify_slots,
        out_rows,
        out_counts,
        state_topk.stride(0),
        state_topk.stride(1),
        out_rows.stride(0),
        out_rows.stride(1),
        WIDTH=width,
        TARGET_TOPK=target_topk,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )


class DFlyRequestState:
    def __init__(
        self,
        *,
        pool_size: int,
        hidden_size: int,
        num_layers: int,
        target_topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden = torch.zeros((pool_size, hidden_size), dtype=dtype, device=device)
        self.topk = torch.full((num_layers, pool_size, target_topk), -1, dtype=torch.int32, device=device)

    def write(
        self,
        *,
        req_pool_indices: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        self.hidden[req_pool_indices] = hidden
        self.topk[:, req_pool_indices] = topk


# ---------------------------------------------------------------------------
# Draft model (mirrors models/hunyuan_v4_flash_dfly.py)
# ---------------------------------------------------------------------------


class DFlyExternalIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rows: Optional[torch.Tensor] = None

    def forward(self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
        return self.rows[: q_lora.shape[0]]


class HYV4FlashDFlyLayer(nn.Module):
    def __init__(self, config, index: int, target_layer_id: int, quant_config):
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.models.hunyuan_v4_flash import HYV4FlashAttention

        super().__init__()
        self.prefix = f'{CHECKPOINT_PREFIX}{index}'
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.indexer = DFlyExternalIndexer()
        # The target layer id makes RadixAttention address that layer's KV pages.
        self.self_attn = HYV4FlashAttention(
            config,
            target_layer_id,
            quant_config,
            f'{self.prefix}.self_attn',
            alt_stream=None,
        )
        # Swap out the default FlashIndexPool for the external draft top-k feeder.
        self.self_attn.indexer = self.indexer
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = None

    def materialize_mlp(self, width: int, device, dtype) -> None:
        from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_world_size
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

        if self.mlp is not None:
            if self.mlp.gate_up_proj.output_sizes[0] != width:
                raise ValueError('Inconsistent DFly FFN widths in the checkpoint')
            return
        if width % get_tensor_model_parallel_world_size():
            raise ValueError('DFly FFN width must be divisible by TP')
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                self.mlp = DeepseekV2MLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=width,
                    hidden_act=self.hidden_act,
                    quant_config=self.quant_config,
                    reduce_results=True,
                    prefix=f'{self.prefix}.mlp',
                )
        finally:
            torch.set_default_dtype(previous)

    def forward(self, positions, hidden, residual, forward_batch, zero_allocator):
        from sglang.srt.layers.layer_boundary import get_attn_tp_context
        from sglang.srt.layers.layer_boundary.adapters.attention import AttentionInputs

        if residual is not None:
            hidden = hidden + residual
        residual = hidden
        x = self.input_layernorm(hidden)
        context = get_attn_tp_context()
        context.set_attn_inputs(AttentionInputs(x, forward_batch, self.self_attn.prepare_qkv_latent))
        try:
            x = self.self_attn(positions, x, forward_batch, zero_allocator)
        finally:
            context.clear_attn_inputs()
        hidden = x + residual
        residual = hidden
        return self.mlp(self.post_attention_layernorm(hidden)), residual


class HYV4FlashDFlyFusion(nn.Module):
    def __init__(self, config):
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.linear import ReplicatedLinear

        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d,
            d,
            bias=False,
            quant_config=None,
            prefix=f'{CHECKPOINT_PREFIX}mtp_fusion.eh_proj',
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(self, anchors, target_hidden):
        fused, _ = self.eh_proj(torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), -1))
        return anchors + self.gate * fused


class HYV4FlashDFlyHiddenCorrection(nn.Module):
    def __init__(self, config):
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear

        super().__init__()
        d = config.hidden_size
        width = getattr(config, 'dfly_hidden_correction_intermediate_size', None) or d
        prefix = f'{CHECKPOINT_PREFIX}hidden_correction'
        self.hidden_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [width, width],
            bias=False,
            quant_config=None,
            prefix=f'{prefix}.gate_up_proj',
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            width,
            d,
            bias=False,
            quant_config=None,
            reduce_results=True,
            prefix=f'{prefix}.down_proj',
        )

    def forward(self, hidden, previous_embed):
        combined = torch.cat((self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1)
        gate_up, _ = self.gate_up_proj(combined)
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


def _split_kv_b_proj(attn) -> None:
    w = attn.kv_b_proj.weight
    if w.dtype != torch.bfloat16:
        raise ValueError('The DFly kv_b_proj must be bf16')
    w_kc, w_vc = w.unflatten(0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)).split(
        [attn.qk_nope_head_dim, attn.v_head_dim], dim=1
    )
    attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    attn.w_vc = w_vc.contiguous().transpose(1, 2)


@triton.jit
def _fp8_lmhead_argmax_kernel(
    X, Q, S, OUT_VAL, OUT_IDX,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """Fused group-128 FP8 W8A16 LM-head projection + per-block argmax (draft proposals only)."""
    pn = tl.program_id(0)
    rows_n = pn * BN + tl.arange(0, BN)
    nmask = rows_n < N
    rows_m = tl.arange(0, BM)
    mmask = rows_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    d128 = tl.arange(0, 128)
    ng = K // 128
    for pg in range(0, ng):
        x = tl.load(X + rows_m[:, None] * K + (pg * 128 + d128)[None, :], mask=mmask[:, None], other=0.0)
        w = e4m3_f32(tl.load(
            Q + rows_n[:, None] * K + (pg * 128 + d128)[None, :], mask=nmask[:, None], other=0
        ))
        sc = tl.load(S + rows_n * ng + pg, mask=nmask, other=0.0)
        wb = (w * sc[:, None]).to(tl.bfloat16)
        acc = tl.dot(x, tl.trans(wb), acc)
    acc = tl.where(nmask[None, :], acc, float('-inf'))
    best_v, best_local = tl.max(acc, axis=1, return_indices=True)
    best_i = pn * BN + best_local
    tl.store(OUT_VAL + rows_m * tl.num_programs(0) + pn, best_v, mask=mmask)
    tl.store(OUT_IDX + rows_m * tl.num_programs(0) + pn, best_i, mask=mmask)


def _fp8_lmhead_argmax(hidden: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor):
    rows, k = hidden.shape
    n = qweight.shape[0]
    bn = 64
    bm = 16 if rows <= 16 else 32
    nblk = triton.cdiv(n, bn)
    x = hidden if hidden.is_contiguous() else hidden.contiguous()
    vals = torch.empty((rows, nblk), dtype=torch.float32, device=x.device)
    idxs = torch.empty((rows, nblk), dtype=torch.int32, device=x.device)
    _fp8_lmhead_argmax_kernel[(nblk,)](
        x, qweight.view(torch.uint8), scales, vals, idxs, rows, n, k,
        BM=bm, BN=bn, num_warps=4, num_stages=2,
    )
    local_max, blk = vals.max(dim=-1)
    local_arg = idxs.gather(1, blk[:, None]).squeeze(1).to(torch.int64)
    return local_max, local_arg



@triton.jit
def _dfly_conf_dot_kernel(
    X, W, B, OUT,
    STEP, N_STEPS, H: tl.constexpr, BH: tl.constexpr,
):
    row = tl.program_id(0)
    acc = tl.zeros((BH,), dtype=tl.float32)
    for k0 in range(0, H, BH):
        offs = k0 + tl.arange(0, BH)
        mask = offs < H
        x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * w
    logit = tl.sum(acc, axis=0) + tl.load(B).to(tl.float32)
    prob = 1.0 / (1.0 + tl.exp(-logit))
    tl.store(OUT + row * N_STEPS + STEP, prob)


@triton.jit
def _dfly_conf_caps_kernel(
    PROBS, CAPS, FLAGS, THRESH,
    N_STEPS: tl.constexpr, WIDTH: tl.constexpr, HAS_FLAGS: tl.constexpr,
):
    row = tl.program_id(0)
    surv = 1.0
    keep = 0
    for step in tl.static_range(N_STEPS):
        prob = tl.load(PROBS + row * N_STEPS + step)
        surv = surv * prob
        keep += tl.where(surv >= THRESH, 1, 0)
    keep = tl.maximum(keep, 1)
    cap = keep + 1
    tl.store(CAPS + row, cap)
    if HAS_FLAGS:
        for col in tl.static_range(WIDTH):
            tl.store(FLAGS + row * WIDTH + col, tl.where(col < cap, 1, 0))


class HYV4FlashDFly(nn.Module):
    def __init__(self, config, dfly_config: HYV4FlashDFlyConfig, quant_config=None):
        from sglang.srt.layers.layernorm import RMSNorm

        super().__init__()
        self.dfly_config = dfly_config
        # ModelOpt NVFP4 checkpoints keep DFly layers in BF16 (exclude_modules lists unfused names,
        # not fused_qkv_a_proj_with_mqa / mlp.gate_up_proj), so construct DFly unquantized there.
        if quant_config is not None and quant_config.get_name() == 'modelopt_fp4':
            quant_config = None
        self.layers = nn.ModuleList(
            HYV4FlashDFlyLayer(config, i, target_layer, quant_config)
            for i, target_layer in enumerate(dfly_config.target_kv_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_fusion = HYV4FlashDFlyFusion(config)
        self.hidden_correction = HYV4FlashDFlyHiddenCorrection(config)
        self.confidence_head = nn.Linear(config.hidden_size, 1, bias=True)
        self.embed_tokens = None
        self.lm_head = None
        self.state: Optional[DFlyRequestState] = None
        self.draft_rows: Optional[torch.Tensor] = None
        self.draft_counts: Optional[torch.Tensor] = None
        self.candidates_out: Optional[torch.Tensor] = None
        self.active_flags: Optional[torch.Tensor] = None
        self.verify_caps: Optional[torch.Tensor] = None
        self._conf_probs: Optional[torch.Tensor] = None
        self._truncate_threshold = float(os.environ.get('XTIER_DSPARK_TRUNCATE', '0'))
        self._fp8_lmhead = None
        self._fp8_lmhead_cnt: Optional[torch.Tensor] = None

    def bind_target(self, embed_tokens, lm_head) -> None:
        self.__dict__['embed_tokens'] = embed_tokens
        self.__dict__['lm_head'] = lm_head

    def bind_buffers(
        self,
        *,
        state: DFlyRequestState,
        draft_rows: torch.Tensor,
        draft_counts: torch.Tensor,
        candidates_out: torch.Tensor,
        active_flags: Optional[torch.Tensor] = None,
        verify_caps: Optional[torch.Tensor] = None,
    ) -> None:
        self.state = state
        self.draft_rows = draft_rows
        self.draft_counts = draft_counts
        self.candidates_out = candidates_out
        self.active_flags = active_flags
        self.verify_caps = verify_caps
        if verify_caps is not None:
            self._conf_probs = torch.empty(
                (verify_caps.shape[0], self.dfly_config.block_size - 1),
                dtype=torch.float32,
                device=verify_caps.device,
            )
        else:
            self._conf_probs = None
        for i, layer in enumerate(self.layers):
            layer.indexer.rows = draft_rows[i]

    def draft_into_verify(self, input_ids: torch.Tensor, positions: torch.Tensor, forward_batch) -> None:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        num_rows = input_ids.shape[0]
        width = self.dfly_config.block_size
        bs = num_rows // width
        req_pool_indices = forward_batch.req_pool_indices[:bs]
        build_draft_topk(
            state_topk=self.state.topk,
            req_pool_indices=req_pool_indices,
            verify_slots=forward_batch.out_cache_loc[:num_rows].view(bs, width),
            out_rows=self.draft_rows,
            out_counts=self.draft_counts,
        )
        backend = get_attn_backend()
        metadata = backend.forward_metadata
        backend.forward_metadata = dataclasses.replace(
            metadata, dsa_cache_seqlens_int32=self.draft_counts[:num_rows]
        )
        try:
            block_ids = input_ids.view(bs, width)
            drafts = self.propose(
                block_ids=block_ids,
                positions=positions,
                target_hidden=self.state.hidden[req_pool_indices],
                forward_batch=forward_batch,
            )
        finally:
            backend.forward_metadata = metadata
        block_ids[:, 1:] = drafts
        self.candidates_out[:num_rows].copy_(input_ids)
        if self._truncate_threshold > 0 and self.verify_caps is not None and self._conf_probs is not None:
            flags_arg = self.active_flags if self.active_flags is not None else self.verify_caps
            _dfly_conf_caps_kernel[(bs,)](
                self._conf_probs,
                self.verify_caps,
                flags_arg,
                float(self._truncate_threshold),
                N_STEPS=width - 1,
                WIDTH=width,
                HAS_FLAGS=self.active_flags is not None,
                num_warps=1,
            )

    @torch.no_grad()
    def propose(
        self,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        from sglang.srt.utils import BumpAllocator

        bs, width = block_ids.shape
        embeds = self.embed_tokens(block_ids.reshape(-1)).view(bs, width, -1)
        embeds[:, 0] = self.mtp_fusion(embeds[:, 0], target_hidden)
        hidden = embeds.reshape(bs * width, -1)
        residual = None
        zero_allocator = BumpAllocator(
            buffer_size=2 * len(self.layers),
            dtype=torch.float32,
            device=hidden.device,
        )
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual, forward_batch, zero_allocator)
        hidden = self.norm(hidden + residual).view(bs, width, -1)

        n_steps = width - 1
        want_conf = self._truncate_threshold > 0 and self._conf_probs is not None
        previous = block_ids[:, 0]
        drafts = []
        for step in range(n_steps):
            corrected = self.hidden_correction(hidden[:, step + 1], self.embed_tokens(previous))
            if want_conf:
                cx = corrected if corrected.is_contiguous() else corrected.contiguous()
                _dfly_conf_dot_kernel[(bs,)](
                    cx,
                    self.confidence_head.weight,
                    self.confidence_head.bias,
                    self._conf_probs,
                    step,
                    n_steps,
                    H=cx.shape[1],
                    BH=1024,
                    num_warps=4,
                )
            previous = self._greedy(corrected)
            drafts.append(previous)
        return torch.stack(drafts, dim=1)

    def prepare_fp8_draft(self) -> None:
        from xtier.arch import HW_FP8
        from xtier.packs.glm5_next import fp8_weights

        if (HW_FP8 and os.environ.get('XTIER_DRAFT_LMHEAD_FP8', '1') != '0'
                and self._fp8_lmhead is None and self.lm_head is not None):
            shard = self.lm_head.shard_indices
            w = self.lm_head.weight[: shard.num_org_elements]
            if w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2 and w.shape[1] % fp8_weights.GROUP == 0:
                q, s = fp8_weights.quantize(w.data)
                self._fp8_lmhead = (q, s)
                self._fp8_lmhead_cnt = torch.zeros(
                    triton.cdiv(q.shape[0], 32), dtype=torch.int32, device=q.device
                )
                logger.info(
                    'xtier: Hy4-Flash DFly2 draft LM head FP8 (group-128): %.2f GiB -> %.2f GiB',
                    w.numel() * 2 / (2**30), (q.numel() + s.numel() * 4) / (2**30),
                )
        if os.environ.get('XTIER_FP8_WEIGHTS', '0') == '1':
            gu = self.hidden_correction.gate_up_proj
            w = getattr(gu, 'weight', None)
            if (isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16
                    and w.shape[1] % fp8_weights.GROUP == 0 and getattr(gu, '_xtier_fp8_scale', None) is None):
                q, s = fp8_weights.quantize(w.data)
                gu.weight = nn.Parameter(q, requires_grad=False)
                gu._xtier_fp8_scale = s
                gu._xtier_w8_cnt = torch.zeros(triton.cdiv(q.shape[0], 32), dtype=torch.int32, device=q.device)
                gu._xtier_w8a8 = False

    def _greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        from sglang.srt.runtime_context import get_parallel

        head = self.lm_head
        shard = head.shard_indices
        if self._fp8_lmhead is not None and hidden.shape[0] <= 32:
            local_max, local_arg = _fp8_lmhead_argmax(hidden, *self._fp8_lmhead)
        else:
            weight = head.weight[: shard.num_org_elements]
            logits = torch.mm(hidden, weight.T, out_dtype=torch.float32)
            local_max, local_arg = logits.max(dim=-1)
        local_ids = local_arg + shard.org_vocab_start_index
        tp_group = get_parallel().tp_group
        if tp_group.world_size == 1:
            return local_ids
        rows = hidden.shape[0]
        maxes = local_max.new_empty((tp_group.world_size * rows,))
        ids = local_ids.new_empty((tp_group.world_size * rows,))
        tp_group.all_gather_into_tensor(maxes, local_max.contiguous())
        tp_group.all_gather_into_tensor(ids, local_ids.contiguous())
        best = maxes.view(-1, rows).argmax(dim=0)
        return ids.view(-1, rows).gather(0, best[None]).squeeze(0)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        params = dict(self.named_parameters())
        pending_mlp: list[Tuple[str, torch.Tensor]] = []
        qkv_a: dict[str, dict[str, torch.Tensor]] = {}
        loaded: set[str] = set()
        device = self.norm.weight.device
        dtype = self.norm.weight.dtype

        def load(name, value, shard_id=None):
            if name not in params:
                raise ValueError(f'Unexpected DFly weight {name!r}')
            param = params[name]
            loader = getattr(param, 'weight_loader', default_weight_loader)
            if shard_id is None:
                loader(param, value)
            else:
                loader(param, value, shard_id)
            loaded.add(name)

        for name, value in weights:
            if name.startswith('confidence_head.proj.'):
                name = name.replace('confidence_head.proj.', 'confidence_head.', 1)
            if name.endswith('.weight_scale'):
                name += '_inv'
            if name.startswith('final_norm.'):
                name = name.replace('final_norm.', 'norm.', 1)
            elif name.split('.', 1)[0].isdigit():
                name = f'layers.{name}'
            if '.mlp.' in name and name.startswith('layers.'):
                layer = self.layers[int(name.split('.')[1])]
                if name.endswith(('.gate_proj.weight', '.up_proj.weight')):
                    layer.materialize_mlp(value.shape[0], device, dtype)
                elif name.endswith('.down_proj.weight'):
                    layer.materialize_mlp(value.shape[1], device, dtype)
                if layer.mlp is None:
                    pending_mlp.append((name, value))
                    continue
                params = dict(self.named_parameters())
                for pending_name, pending_value in pending_mlp:
                    self._load_mapped(load, pending_name, pending_value)
                pending_mlp.clear()
            if '.q_a_proj.' in name or '.kv_a_proj_with_mqa.' in name:
                kind = 'q' if '.q_a_proj.' in name else 'kv'
                fused = name.replace('.q_a_proj.', '.fused_qkv_a_proj_with_mqa.')
                fused = fused.replace('.kv_a_proj_with_mqa.', '.fused_qkv_a_proj_with_mqa.')
                parts = qkv_a.setdefault(fused, {})
                parts[kind] = value
                if len(parts) == 2:
                    load(fused, torch.cat([parts['q'], parts['kv']], dim=0))
                    del qkv_a[fused]
                continue
            self._load_mapped(load, name, value)
        if pending_mlp or qkv_a:
            raise ValueError('Incomplete DFly checkpoint: missing paired weights')
        missing = {
            name for name in dict(self.named_parameters()) if not name.endswith(('.k_scale', '.v_scale'))
        } - loaded
        if missing:
            raise ValueError(f'Missing DFly weights: {sorted(missing)}')
        for layer in self.layers:
            _split_kv_b_proj(layer.self_attn)
        logger.info(
            'xtier: loaded Hy4-Flash DFly2 drafter (%d layers, target_kv=%s, block_size=%d)',
            self.dfly_config.num_layers,
            list(self.dfly_config.target_kv_layers),
            self.dfly_config.block_size,
        )

    @staticmethod
    def _load_mapped(load, name, value):
        for shard, shard_id in (('.gate_proj.', 0), ('.up_proj.', 1)):
            if shard in name:
                load(name.replace(shard, '.gate_up_proj.'), value, shard_id)
                return
        load(name, value)


# ---------------------------------------------------------------------------
# Speculative V2 worker (mirrors speculative/dfly_worker_v2.py)
# ---------------------------------------------------------------------------


def _commit_accept(candidates, accept_len, bonus_tokens):
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    out_tokens[:, :-1].copy_(candidates[:, 1:])
    out_tokens[:, -1].fill_(0)
    out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus_tokens[:, None])
    return out_tokens, accept_len.to(torch.int32) + 1


def _engram_commit_verify(hasher, req_pool_indices: torch.Tensor, rows: torch.Tensor) -> None:
    recent = hasher.verify_recent[rows]
    hasher.history[req_pool_indices] = recent.flip(-1).to(hasher.history.dtype)


def _sampling_accept_capped(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info,
    caps: torch.Tensor,
    max_top_k: Optional[int],
    uniform_top_k_value: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lossless chain sampling accept capped at per-request verify lengths ``caps`` (2..block_size)."""
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.dflash_utils import (
        _get_or_create_chain_verify_buffers,
        build_speculative_verify_target_probs,
    )
    from sglang.srt.model_executor.runner_utils.pool import borrow_graph_pool
    from sglang.kernels.ops.speculative.verify_sampling import tree_speculative_sampling_target_only

    bs, draft_token_num = candidates.shape
    device = next_token_logits.device
    threshold_single = float(get_spec().speculative_accept_threshold_single)
    threshold_acc = max(float(get_spec().speculative_accept_threshold_acc), 1e-9)
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=draft_token_num, device=device)
    row_ids = torch.arange(bs, dtype=torch.long, device=device)
    next_tok = retrieve_next_token.clone()
    next_tok[row_ids, (caps - 1).to(torch.long)] = -1
    with borrow_graph_pool(user='DFLY verify probabilities'):
        coins = torch.rand((bs, draft_token_num), dtype=torch.float32, device=device)
        coins_final = torch.rand((bs,), dtype=torch.float32, device=device)
        target_probs = build_speculative_verify_target_probs(
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            draft_token_num=draft_token_num,
            bs=bs,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
            use_sparse_topk=True,
        )
        draft_probs = torch.zeros_like(target_probs)
        candidates_i64 = candidates if candidates.dtype == torch.int64 else candidates.to(torch.int64)
        tree_speculative_sampling_target_only(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates_i64,
            retrive_index=retrieve_index,
            retrive_next_token=next_tok,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=threshold_single,
            threshold_acc=threshold_acc,
            deterministic=True,
        )
    correct_len = accept_token_num
    accept_pos = accept_index[row_ids, correct_len.to(torch.long)].to(torch.long)
    bonus = predicts[accept_pos].to(torch.int64)
    return correct_len, bonus


def _build_dfly_worker_class():
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
    from sglang.kernels.ops.speculative.dflash import compute_dflash_correct_drafts_and_bonus
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.layers.dp_attention import get_dp_tp_group
    from sglang.srt.layers.logprob_processor import compute_spec_logprobs
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
    from sglang.srt.runtime_context import get_exec, get_schedule, get_spec
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
    from sglang.srt.speculative.dflash_utils import (
        apply_dflash_verify_logits_adjustments,
        commit_block_accept_tokens_to_penalties,
        compute_dflash_sampling_correct_drafts_and_bonus,
        is_dflash_sampling_verify_available,
    )
    from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
    from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

    class DFlyWorkerV2(BaseSpecWorker):
        def __init__(
            self,
            server_args: ServerArgs,
            gpu_id: int,
            nccl_port: int,
            target_worker: TpModelWorker,
        ):
            super().__init__()
            self.server_args = server_args
            self._target_worker = target_worker
            self._draft_worker = None
            self.model_runner = target_worker.model_runner
            self.device = self.model_runner.device
            self.dfly = getattr(self.model_runner.model, 'dfly', None)
            if self.dfly is None:
                raise ValueError('DFLY requires a target model with a bundled DFly draft')
            self.dfly_config = self.dfly.dfly_config
            self.block_size = int(get_spec().speculative_num_draft_tokens)
            if self.block_size != self.dfly_config.block_size:
                raise ValueError('DFly draft and verify block sizes disagree')
            self._tp_sync = SpecTpSync(get_dp_tp_group())
            self._block_offsets = torch.arange(self.block_size, dtype=torch.int64, device=self.device)
            self._max_rows = self._max_verify_rows()
            self._topk_capture = self._bind_topk_capture()
            self.state: Optional[DFlyRequestState] = None

        def _max_verify_rows(self) -> int:
            graph = get_exec().graph.cuda_graph_config
            graph_bs = graph.decode.max_bs if graph is not None else None
            max_bs = max(graph_bs or 0, get_schedule().max_running_requests or 0, 1)
            return max_bs * self.block_size

        def _bind_topk_capture(self) -> torch.Tensor:
            config = self.dfly_config
            capture = torch.full(
                (config.num_layers, self._max_rows, config.target_index_topk),
                -1,
                dtype=torch.int32,
                device=self.device,
            )
            model = self.model_runner.model.model
            for i, layer_id in enumerate(config.target_index_layers):
                indexer = model.layers[layer_id].self_attn.indexer
                indexer.dfly_topk_capture = capture[i]
            hasher = model.engram_hasher
            if hasher is not None:
                hasher.verify_recent = torch.zeros(
                    (self._max_rows, hasher.max_ngram_size - 1),
                    dtype=torch.int64,
                    device=self.device,
                )
            return capture

        def alloc_memory_pool(
            self,
            memory_pool_config=None,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
        ):
            super().alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
            config = self.dfly_config
            self.state = DFlyRequestState(
                pool_size=req_to_token_pool.req_to_token.shape[0],
                hidden_size=self.model_runner.model_config.hidden_size,
                num_layers=config.num_layers,
                target_topk=config.target_index_topk,
                dtype=self.model_runner.dtype,
                device=self.device,
            )
            flags = None
            caps = None
            if self.dfly._truncate_threshold > 0:
                from xtier.backend.method import active_flags

                max_bs = max(1, self._max_rows // self.block_size)
                flags = active_flags(self.device)
                caps = torch.full((max_bs,), self.block_size, dtype=torch.int32, device=self.device)
                logger.info(
                    'xtier: Hy4-Flash DFly2 confidence-head verify truncation enabled (survival >= %.3f)',
                    self.dfly._truncate_threshold,
                )
            self.dfly.bind_buffers(
                state=self.state,
                draft_rows=torch.full(
                    (config.num_layers, self._max_rows, config.draft_index_topk),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                draft_counts=torch.zeros(self._max_rows, dtype=torch.int32, device=self.device),
                candidates_out=torch.zeros(self._max_rows, dtype=torch.int64, device=self.device),
                active_flags=flags,
                verify_caps=caps,
            )
            from xtier.packs.hy4_flash import kv_fp4

            if kv_fp4.kv_fp4_enabled():
                kv_fp4.ensure_gather_buffers(self._max_rows * config.draft_index_topk, self.device)
            self.dfly.prepare_fp8_draft()

        def init_attention_backends(self):
            if not isinstance(self.model_runner.attn_backend, DeepseekSparseAttnBackend):
                raise ValueError('DFLY requires the DSA attention backend')

        def forward_batch_generation(
            self,
            batch: ScheduleBatch,
            on_publish=None,
            grammar_barrier=None,
            pp_proxy_tensors=None,
        ) -> GenerationBatchResult:
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                return self._forward_prefill(batch, on_publish)
            if batch.forward_mode.is_idle():
                return self._forward_idle(on_publish)
            return self._forward_decode(batch, on_publish)

        def _forward_prefill(self, batch: ScheduleBatch, on_publish):
            result = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.LAST
            )
            result.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            if batch.forward_mode.is_idle():
                result.next_draft_input = DFlashDraftInputV2.create_idle_input(device=self.device)
                return result
            logits_output = result.logits_output
            bs = batch.batch_size()
            self.state.write(
                req_pool_indices=batch.req_pool_indices,
                hidden=logits_output.hidden_states[:bs],
                topk=self._topk_capture[:, :bs],
            )
            logits_output.hidden_states = None
            result.next_draft_input = make_draft_input_v2(
                bonus_tokens=result.next_token_ids, new_seq_lens=batch.seq_lens
            )
            return result

        def _forward_idle(self, on_publish):
            next_draft_input = make_draft_input_v2(
                bonus_tokens=torch.empty((0,), dtype=torch.int64, device=self.device),
                new_seq_lens=torch.empty((0,), dtype=torch.int64, device=self.device),
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
                accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
                next_draft_input=next_draft_input,
                speculative_num_draft_tokens=self.block_size,
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        def _forward_decode(self, batch: ScheduleBatch, on_publish):
            draft_input = batch.spec_info
            if not isinstance(draft_input, DFlashDraftInputV2):
                raise RuntimeError('DFLY expects DFlashDraftInputV2 on the running batch')
            batch.seq_lens.record_stream(torch.cuda.current_stream())
            bs = len(batch.seq_lens)
            width = self.block_size
            prefix_lens = batch.seq_lens
            verify_forward_batch = self._prepare_verify(batch, draft_input)
            try:
                target_out = self.target_worker.forward_batch_generation(
                    batch=None, forward_batch=verify_forward_batch, is_verify=True
                )
            finally:
                if self.dfly.active_flags is not None:
                    self.dfly.active_flags[: bs * width].fill_(1)
            logits_output = target_out.logits_output
            candidates = self.dfly.candidates_out[: bs * width].view(bs, width)
            num_correct_drafts, bonus = self._accept(
                batch, draft_input, candidates, logits_output.next_token_logits
            )
            out_tokens, commit_lens = _commit_accept(candidates, num_correct_drafts, bonus)
            if batch.sampling_info is not None:
                commit_block_accept_tokens_to_penalties(
                    batch.sampling_info, out_tokens, commit_lens
                )
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            if on_publish is not None:
                on_publish(new_seq_lens)
            if batch.return_logprob:
                compute_spec_logprobs(
                    batch, logits_output, out_tokens.reshape(-1), chain_stride=width
                )
            source_rows = torch.arange(bs, device=self.device) * width + num_correct_drafts
            self._commit_state(batch, logits_output.hidden_states, source_rows)
            logits_output.hidden_states = None
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=out_tokens.reshape(-1),
                accept_lens=commit_lens,
                can_run_cuda_graph=target_out.can_run_cuda_graph,
                next_draft_input=make_draft_input_v2(bonus_tokens=bonus, new_seq_lens=new_seq_lens),
                speculative_num_draft_tokens=width,
                new_seq_lens=new_seq_lens,
                routed_experts_output=target_out.routed_experts_output,
                indexer_topk_output=target_out.indexer_topk_output,
                kv_weight_version_record=target_out.kv_weight_version_record,
            )

        def _prepare_verify(self, batch: ScheduleBatch, draft_input: DFlashDraftInputV2) -> ForwardBatch:
            bs = len(batch.seq_lens)
            width = self.block_size
            block_ids = torch.full(
                (bs, width),
                self.dfly_config.mask_token_id,
                dtype=torch.int64,
                device=self.device,
            )
            block_ids[:, 0] = draft_input.bonus_tokens
            positions = batch.seq_lens.to(torch.int64)[:, None] + self._block_offsets
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + width,
                batch_size=bs,
                draft_token_num=width,
                device=self.device,
            )
            verify_input = DFlashVerifyInput(
                draft_token=block_ids.view(-1),
                positions=positions.view(-1),
                draft_token_num=width,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            seq_lens_cpu, seq_lens_sum = batch.seq_lens_cpu, batch.seq_lens_sum
            if seq_lens_cpu is not None:
                batch.seq_lens_cpu = seq_lens_cpu + width
                batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            elif draft_input.nxt_kv_lens_cpu is not None:
                batch.seq_lens_cpu = draft_input.nxt_kv_lens_cpu
                batch.seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
            verify_forward_batch, _ = verify_input.prepare_for_verify(
                batch, self.target_worker, defer_metadata_init=True
            )
            batch.seq_lens_cpu, batch.seq_lens_sum = seq_lens_cpu, seq_lens_sum
            return verify_forward_batch

        def _accept(self, batch, draft_input, candidates, next_token_logits):
            sampling_info = batch.sampling_info
            if sampling_info is not None:
                apply_dflash_verify_logits_adjustments(
                    next_token_logits=next_token_logits,
                    sampling_info=sampling_info,
                    draft_token_num=self.block_size,
                )
            bs = candidates.shape[0]
            caps = self.dfly.verify_caps[:bs] if self.dfly.verify_caps is not None else None
            if (
                sampling_info is not None
                and not sampling_info.is_all_greedy
                and is_dflash_sampling_verify_available()
            ):
                if caps is not None:
                    num_correct_drafts, bonus = _sampling_accept_capped(
                        candidates=candidates,
                        next_token_logits=next_token_logits,
                        sampling_info=sampling_info,
                        caps=caps,
                        max_top_k=draft_input.max_top_k,
                        uniform_top_k_value=draft_input.uniform_top_k_value,
                    )
                else:
                    num_correct_drafts, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                        candidates=candidates,
                        next_token_logits=next_token_logits,
                        sampling_info=sampling_info,
                        max_top_k=draft_input.max_top_k,
                        uniform_top_k_value=draft_input.uniform_top_k_value,
                    )
                self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE, num_correct_drafts)
                self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_SAMPLE, bonus)
                return num_correct_drafts, bonus
            target_predict = torch.argmax(next_token_logits, dim=-1).view(candidates.shape)
            self._tp_sync.sync(SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, target_predict)
            num_correct_drafts, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates, target_predict=target_predict
            )
            if caps is not None:
                max_drafts = (caps - 1).to(num_correct_drafts.dtype)
                num_correct_drafts = torch.minimum(num_correct_drafts, max_drafts)
                bonus = target_predict.gather(1, num_correct_drafts.to(torch.int64)[:, None]).squeeze(1)
            return num_correct_drafts, bonus

        def _commit_state(self, batch, hidden_states, source_rows) -> None:
            self.state.write(
                req_pool_indices=batch.req_pool_indices,
                hidden=hidden_states[source_rows],
                topk=self._topk_capture[:, source_rows],
            )
            hasher = self.model_runner.model.model.engram_hasher
            if hasher is not None:
                _engram_commit_verify(hasher, batch.req_pool_indices, source_rows)

    return DFlyWorkerV2


_DFLY_WORKER_CLS = [None]


def _get_dfly_worker_cls(_server_args):
    if _DFLY_WORKER_CLS[0] is None:
        _DFLY_WORKER_CLS[0] = _build_dfly_worker_class()
    return _DFLY_WORKER_CLS[0]


# ---------------------------------------------------------------------------
# SpeculativeAlgorithm.register("DFLY") via CustomSpecAlgo
# ---------------------------------------------------------------------------


def _dfly_block_size_from_checkpoint(cfg) -> Optional[int]:
    from sglang.srt.hf_transformers import get_config

    hf_config = get_config(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        revision=cfg.revision,
    )
    block_size = getattr(hf_config, 'dfly_block_size', None)
    if block_size is not None:
        return int(block_size)
    spec = getattr(hf_config, 'speculative_config', None)
    if isinstance(spec, dict) and spec.get('num_speculative_tokens') is not None:
        return int(spec['num_speculative_tokens']) + 1
    return None


def _register_dfly() -> bool:
    """Register DFLY via SGLang's CustomSpecAlgo plugin API if not built in."""
    if _has_builtin_dfly():
        return False
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo, get_spec as get_registered_spec

    if get_registered_spec('DFLY') is not None:
        return True

    class DFlySpecAlgo(CustomSpecAlgo):
        def is_dfly(self) -> bool:
            return True

        def carries_draft_hidden_states(self) -> bool:
            return False

        def need_topk(self) -> bool:
            return False

        def create_future_map(
            self,
            device,
            req_to_token_pool,
            needs_cpu_seq_lens: bool = True,
            needs_confidence_relay: bool = False,
        ):
            from sglang.srt.managers.overlap_utils import FutureMap

            return FutureMap(
                device,
                self,
                req_to_token_pool,
                needs_cpu_seq_lens,
                needs_confidence_relay,
            )

        def handle_server_args(self, cfg) -> None:
            from sglang.srt.runtime_context import attn_dp_enabled_of

            if not cfg.device.startswith('cuda'):
                raise ValueError('DFLY speculative decoding requires CUDA.')
            if attn_dp_enabled_of(cfg) or cfg.pp_size != 1 or cfg.attn_cp_size != 1:
                raise ValueError('DFLY does not support DP attention, PP or CP.')
            if cfg.disaggregation_mode != 'null':
                raise ValueError('DFLY does not support PD disaggregation.')
            if cfg.enable_lora:
                raise ValueError('DFLY does not support LoRA.')
            if cfg.speculative_draft_model_path not in (None, cfg.model_path):
                raise ValueError('DFLY drafts live in the target checkpoint.')
            if cfg.speculative_draft_attention_backend is not None:
                raise ValueError('DFLY drafts run on the target attention backend.')
            for name in ('speculative_num_steps', 'speculative_eagle_topk'):
                if getattr(cfg, name) not in (None, 1):
                    raise ValueError(f'DFLY requires {name} == 1.')
                setattr(cfg, name, 1)
            block_size = cfg.speculative_num_draft_tokens
            if block_size is None:
                block_size = _dfly_block_size_from_checkpoint(cfg)
                if block_size is None:
                    block_size = DEFAULT_DFLY_BLOCK_SIZE
                    logger.warning(
                        'The checkpoint declares no DFly block size; defaulting to %d.',
                        block_size,
                    )
                cfg.speculative_num_draft_tokens = block_size
            if not 2 <= int(block_size) <= 17:
                raise ValueError('DFLY requires a block size in [2, 17].')
            if cfg.enable_mixed_chunk:
                cfg.enable_mixed_chunk = False
                logger.warning('Mixed chunked prefill is disabled for DFLY.')
            if cfg.max_running_requests is None:
                cfg.max_running_requests = 48
                logger.warning(
                    'Max running requests is reset to 48 for speculative decoding. '
                    'You can override this by explicitly setting --max-running-requests.'
                )

    SpeculativeAlgorithm.register('DFLY', supports_overlap=True, spec_class=DFlySpecAlgo)(
        _get_dfly_worker_cls
    )
    return True


# ---------------------------------------------------------------------------
# Zero-patch SGLang hooks
# ---------------------------------------------------------------------------


class _AllowDflyCfg:
    def __init__(self, cfg, *, allow_dfly: bool = False, allow_hicache: bool = False):
        self._cfg = cfg
        self._allow_dfly = allow_dfly
        self._allow_hicache = allow_hicache

    @property
    def speculative_algorithm(self):
        return None if self._allow_dfly else self._cfg.speculative_algorithm

    @property
    def enable_hierarchical_cache(self):
        return False if self._allow_hicache else self._cfg.enable_hierarchical_cache

    def __getattr__(self, name):
        return getattr(self._cfg, name)


def _reject_hyv4_hook(original, cfg):
    import os
    algo = getattr(cfg, 'speculative_algorithm', None)
    allow_dfly = isinstance(algo, str) and algo.upper() == 'DFLY'
    allow_hicache = bool(getattr(cfg, 'enable_hierarchical_cache', False)) and os.environ.get('XTIER_KV_OFFLOAD') == '1'
    if allow_dfly or allow_hicache:
        return original(_AllowDflyCfg(cfg, allow_dfly=allow_dfly, allow_hicache=allow_hicache))
    return original(cfg)


def _hy4_config_init(result, self, *args, **kwargs):
    for key, value in kwargs.items():
        if key.startswith('dfly_'):
            setattr(self.text_config, key, deepcopy(value))
    return result


def _create_dummy_verify_input(
    original, spec_algorithm, custom_mask, num_tokens_per_req, is_draft_worker
):
    if _is_dfly_algo(spec_algorithm):
        from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
        from sglang.srt.runtime_context import get_spec
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        return DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=get_spec().speculative_num_draft_tokens,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.NULL if is_draft_worker else CaptureHiddenMode.FULL
            ),
        )
    return original(spec_algorithm, custom_mask, num_tokens_per_req, is_draft_worker)


def _spec_prepare_for_decode(original, batch):
    if _is_dfly_algo(batch.spec_algorithm):
        batch.spec_info.prepare_for_decode(batch)
        return None
    return original(batch)


def _graph_get_spec_info(original, self, num_tokens, ragged_verify_layout=None):
    if _is_dfly_algo(self.model_runner.spec_algorithm):
        from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy

        _, build_custom_mask = resolve_dflash_verify_mask_policy(self.model_runner.attn_backend)
        return DFlashVerifyInput(
            draft_token=None,
            positions=None,
            draft_token_num=self.captured_req_width,
            custom_mask=(
                None
                if (self.model_runner.is_draft_worker or not build_custom_mask)
                else self.buffers.custom_mask
            ),
            capture_hidden_mode=(
                CaptureHiddenMode.NULL
                if self.model_runner.is_draft_worker
                else CaptureHiddenMode.FULL
            ),
            ragged_verify_layout=ragged_verify_layout,
        )
    return original(self, num_tokens, ragged_verify_layout=ragged_verify_layout)


def _scheduler_add_request(original, self, req):
    if _is_dfly_algo(self.spec_algorithm) and not req.finished():
        from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
        from sglang.srt.speculative.dflash_utils import validate_dflash_request

        error_msg = validate_dflash_request(
            req,
            self.enable_overlap,
            bound_runtime=isinstance(self.draft_worker, BaseSpecWorker)
            and self.draft_worker.has_bound_runtime,
        )
        if error_msg is None:
            params = req.sampling_params
            if any(
                x is not None
                for x in (params.json_schema, params.regex, params.ebnf, params.structural_tag)
            ):
                error_msg = 'DFLY speculative decoding does not support constrained decoding.'
        if error_msg is not None:
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
    return original(self, req)


def _row_layout_init(original, self, forward_batch, metadata, num_rows):
    if forward_batch.forward_mode.is_target_verify():
        device = forward_batch.seq_lens.device
        lengths = metadata.get_seqlens_expanded()[:num_rows].to(torch.int64)
        self.lengths = lengths
        self.positions = (lengths - 1).contiguous()
        self.seq_lens_cpu = forward_batch.seq_lens_cpu
        self.row_to_batch = torch.arange(num_rows, device=device)
        self.metadata = metadata
        self.num_rows = num_rows
        return None
    return original(self, forward_batch, metadata, num_rows)


def _verify_pool_groups(pool, forward_batch, layout, rows: int) -> None:
    ratio = pool.ratio
    bs = forward_batch.batch_size
    width = rows // bs
    device = layout.positions.device
    starts = forward_batch.seq_lens[:bs].to(torch.int64)
    groups = starts[:, None] // ratio + torch.arange(width // ratio + 1, device=device)
    live = (groups + 1) * ratio <= starts[:, None] + width
    members = groups[..., None] * ratio + torch.arange(ratio, device=device)
    first_rows = torch.arange(bs, device=device) * width
    live &= (forward_batch.out_cache_loc[first_rows] != 0)[:, None]
    member_slots = layout.slots(members, first_rows[:, None]).view(-1, ratio)
    # Slot 0 is the pool's padding slot: no request reads it.
    write_slots = torch.where(live.view(-1), member_slots[:, 0], 0)
    first_positions = members[..., 0].reshape(-1)
    if pool.rope_store is not None:
        first_positions = pool.rope_store.buffer[member_slots[:, 0]]
    pool._pool_groups(member_slots, first_positions, write_slots)


def _indexpool_update_cache(original, self, keys, forward_batch, layout):
    if forward_batch.forward_mode.is_target_verify():
        from sglang.srt.models.hunyuan_v4_flash import layout_clamp

        raw_buf = self._raw_key_buffer(keys.device)
        out_loc = forward_batch.out_cache_loc[: keys.shape[0]].to(torch.int64)
        raw_buf[layout_clamp(out_loc, raw_buf.shape[0])] = keys
        _verify_pool_groups(self, forward_batch, layout, keys.shape[0])
        return None
    return original(self, keys, forward_batch, layout)


def _indexpool_select(original, self, layout, q_fp8, weights, decode):
    if _in_target_verify_select[0]:
        decode = True
    return original(self, layout, q_fp8, weights, decode=decode)


def _capture_dfly_topk(pool, indices, forward_batch) -> None:
    from sglang.srt.model_executor.forward_context import get_attn_backend

    mode = forward_batch.forward_mode
    capture = pool.dfly_topk_capture
    if mode.is_target_verify():
        capture[: indices.shape[0]].copy_(indices)
    elif mode.is_extend():
        lens = forward_batch.extend_seq_lens.to(torch.int64)
        last = torch.cumsum(lens, 0) - 1
        seeds = get_attn_backend().prepare_mtp_seed_indices(indices[last], mode)
        capture[: last.shape[0]].copy_(seeds)


def _indexpool_forward(original, self, x, q_lora, positions, forward_batch, layer_id, return_indices=True):
    prev = _in_target_verify_select[0]
    _in_target_verify_select[0] = forward_batch.forward_mode.is_target_verify()
    try:
        indices = original(self, x, q_lora, positions, forward_batch, layer_id, return_indices=return_indices)
    finally:
        _in_target_verify_select[0] = prev
    if getattr(self, 'dfly_topk_capture', None) is not None and not forward_batch.forward_mode.is_idle():
        _capture_dfly_topk(self, indices, forward_batch)
    return indices


def _engram_hasher_forward(original, self, input_ids, forward_batch, mm_pad_ids=None):
    if not forward_batch.forward_mode.is_target_verify():
        return original(self, input_ids, forward_batch, mm_pad_ids=mm_pad_ids)
    n = self.max_ngram_size
    device = input_ids.device
    history = self._ensure_history(device)
    num_tokens = input_ids.shape[0]
    req = forward_batch.req_pool_indices.to(torch.int64)
    ids = input_ids.to(torch.int64)
    token_index = torch.arange(num_tokens, device=device)
    width = num_tokens // req.shape[0]
    row = token_index // width
    offset = token_index % width
    past = history[req]
    shifts = torch.arange(n, device=device)
    in_batch = shifts[None] <= offset[:, None]
    from_batch = ids[(token_index[:, None] - shifts).clamp_min(0)]
    hist_col = (n - 1 - (shifts[None] - offset[:, None])).clamp(0, n - 2)
    from_hist = past.to(torch.int64)[row].gather(1, hist_col)
    tokens = torch.where(in_batch, from_batch, from_hist)
    positions = forward_batch.positions[:num_tokens].to(torch.int64)
    verify_recent = getattr(self, 'verify_recent', None)
    if verify_recent is not None:
        verify_recent[:num_tokens].copy_(tokens[:, : n - 1])
    if self.visual_token_ids:
        tokens = self._restore_visual_ids(tokens, mm_pad_ids)
    blocked = positions[:, None] < shifts
    compressed = torch.where(blocked, self.pad_id, self.token_map[tokens])
    products = compressed[:, None, :] * self.multipliers[None]
    rolling, hashes = products[..., 0], []
    for i in range(1, n):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling[..., None] % self.primes[None, :, i - 1])
    hash_ids = torch.cat(hashes, -1) + self.offsets[None]
    no_hash = input_ids == self.pad_token_id
    if self.visual_token_ids:
        for token_id in self.visual_token_ids:
            no_hash |= input_ids == token_id
    return hash_ids.masked_fill(no_hash[:, None, None], -1)


def _hy4_model_init(result, self, config, quant_config=None, prefix=''):
    from sglang.srt.runtime_context import get_spec

    algo = get_spec().speculative_algorithm
    if getattr(self, 'dfly', None) is None and isinstance(algo, str) and algo.upper() == 'DFLY':
        text_config = config.get_text_config()
        self.dfly = HYV4FlashDFly(
            text_config,
            parse_dfly_config(text_config, int(get_spec().speculative_num_draft_tokens)),
            quant_config,
        )
        self.dfly.bind_target(self.model.embed_tokens, self.lm_head)
    return result


def _hy4_load_weights(original, self, weights):
    dfly = getattr(self, 'dfly', None)
    if dfly is None:
        return original(self, weights)
    dfly_weights = []

    def filtered():
        for name, loaded_weight in weights:
            if name.startswith(CHECKPOINT_PREFIX):
                dfly_weights.append((name.removeprefix(CHECKPOINT_PREFIX), loaded_weight))
                continue
            yield name, loaded_weight

    res = original(self, filtered())
    dfly.load_weights(dfly_weights)
    dfly_weights.clear()
    return res


def _hy4_forward(original, self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
    dfly = getattr(self, 'dfly', None)
    reset_flags = False
    if dfly is not None and forward_batch.forward_mode.is_target_verify():
        pos = (forward_batch.mrope_positions[0] if (self.is_mrope_enabled and forward_batch.mrope_positions is not None) else positions)
        dfly.draft_into_verify(input_ids, pos, forward_batch)
        reset_flags = dfly.active_flags is not None
    try:
        return original(self, input_ids, positions, forward_batch, input_embeds=input_embeds, **kwargs)
    finally:
        if reset_flags:
            dfly.active_flags[: input_ids.shape[0]].fill_(1)


def install_serving(registry, hook_type):
    if _has_builtin_dfly():
        return
    _register_dfly()
    registry.register(REJECT_HYV4, _reject_hyv4_hook, hook_type.AROUND)
    registry.register(HY4_CONFIG_INIT, _hy4_config_init, hook_type.AFTER)


def install(registry, hook_type):
    if _has_builtin_dfly():
        return
    _register_dfly()
    registry.register(REJECT_HYV4, _reject_hyv4_hook, hook_type.AROUND)
    registry.register(HY4_CONFIG_INIT, _hy4_config_init, hook_type.AFTER)
    registry.register(CREATE_DUMMY_VERIFY, _create_dummy_verify_input, hook_type.AROUND)
    registry.register(SPEC_PREPARE_DECODE, _spec_prepare_for_decode, hook_type.AROUND)
    registry.register(GRAPH_GET_SPEC_INFO, _graph_get_spec_info, hook_type.AROUND)
    registry.register(SCHED_ADD_REQ, _scheduler_add_request, hook_type.AROUND)
    registry.register(ROW_LAYOUT_INIT, _row_layout_init, hook_type.AROUND)
    registry.register(INDEXPOOL_UPDATE, _indexpool_update_cache, hook_type.AROUND)
    registry.register(INDEXPOOL_SELECT, _indexpool_select, hook_type.AROUND)
    registry.register(INDEXPOOL_FWD, _indexpool_forward, hook_type.AROUND)
    registry.register(ENGRAM_HASHER_FWD, _engram_hasher_forward, hook_type.AROUND)
    registry.register(HY4_MODEL_INIT, _hy4_model_init, hook_type.AFTER)
    registry.register(HY4_LOAD_WEIGHTS, _hy4_load_weights, hook_type.AROUND)
    registry.register(HY4_FWD, _hy4_forward, hook_type.AROUND)
