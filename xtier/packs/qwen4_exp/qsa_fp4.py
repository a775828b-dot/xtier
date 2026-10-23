"""QSA sparse attention on an NVFP4 KV cache (``--kv-cache-dtype nvfp4``, as the production pipeline).

SGLang main stores the cache (NVFP4KVCacheMethod: packed E2M1 pairs, an E4M3 scale per 16 values, a global FP32
scale; x = e2m1 * block scale * global scale) but its QSA backend reads the pool as plain K/V: the decode / verify
gather and the chunked-prefill packing hand the packed tensor to Triton, and the paged decode takes the head size
from it. Here, for a layer on an NVFP4 pool:
- decode / verify (``_forward_paged_attention``): SGLang's code on BF16-shaped stand-ins of the pool buffers (they
  size the scratch and the paged views and are never read); its gather of the selected rows
  (``qwen_sparse_kv_extraction_compact_triton``) reads the packed storage and dequantizes in registers;
- prefill with a cached prefix (``forward_extend``): ``pack_qsa_prefill_kv`` the same way.
The global scale is the one the pool wrote with: the hybrid pool of these models passes 1.0. It passes it as a Python
float, which NVFP4KVQuantizeUtil.quantize turns into a host-to-device copy that CUDA graph capture rejects (the target
verify graph fails to capture); a device tensor of 1.0, made when the NVFP4 pool is created, replaces it.
"""

import contextlib

import torch
import triton
import triton.language as tl

BLOCK_TOPK = 16
BLOCK_T = 32
_active = []      # the raw buffers of the layer whose attention runs: (k, v, k scales, v scales, k global, v global)
_stand_ins = {}   # (heads, dim, device) -> BF16 [1, heads, dim] for expanded stand-ins
_ones = {}        # device -> FP32 [1] = 1.0, made with the NVFP4 pool (outside graph capture)


def _one(device):
    one = _ones.get(device)
    if one is None:
        one = _ones[device] = torch.ones(1, dtype=torch.float32, device=device)
    return one


def raw_buffers(pool, layer):
    """The packed K/V, block scales and global scales of ``layer`` if ``pool`` is an NVFP4 KV pool, else None."""
    method = pool.get_kv_cache_quant_method() if hasattr(pool, 'get_kv_cache_quant_method') else None
    if getattr(method, 'name', None) != 'nvfp4':
        return None
    k, v, k_sf, v_sf = pool.get_raw_kv_buffer(layer.layer_id)
    if hasattr(pool, 'full_kv_pool'):   # HybridLinearKVPool.set_kv_buffer writes with k_scale = v_scale = 1.0
        gk = gv = _one(k.device)
    else:
        gk = method.k_scales_gpu[layer.layer_id:layer.layer_id + 1]
        gv = method.v_scales_gpu[layer.layer_id:layer.layer_id + 1]
    return k.view(torch.uint8), v.view(torch.uint8), k_sf.view(torch.uint8), v_sf.view(torch.uint8), gk, gv


@contextlib.contextmanager
def _layer(raw):
    _active.append(raw)
    try:
        yield
    finally:
        _active.pop()


def _stand_in(raw):
    k = raw[0]
    m, heads, half = k.shape
    key = (heads, half * 2, k.device)
    t = _stand_ins.get(key)
    if t is None:
        t = _stand_ins[key] = torch.empty((1, heads, half * 2), dtype=torch.bfloat16, device=k.device)
    return t.expand(m, heads, half * 2)


@triton.jit
def _e2m1(nib):
    """E2M1 nibble (sign in bit 3) -> fp32."""
    m = nib & 7
    v = tl.where(m < 4, m.to(tl.float32) * 0.5, tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))))
    return v * tl.where((nib & 8) != 0, -1.0, 1.0)   # a product: -0.0 stays -0.0


@triton.jit
def _dequant_rows(fp4, sf, g, rows, mask, head, HEADS: tl.constexpr, DIM: tl.constexpr):
    """[R, DIM] fp32 of rows `rows` (int64 [R]), head `head` of a packed [m, HEADS, DIM / 2] buffer."""
    d2 = tl.arange(0, DIM // 2)
    s = tl.arange(0, DIM // 16)
    packed = tl.load(fp4 + rows[:, None] * (HEADS * DIM // 2) + head * (DIM // 2) + d2[None, :], mask=mask[:, None],
                     other=0).to(tl.int32)
    vals = tl.interleave(_e2m1(packed & 0xF), _e2m1(packed >> 4))
    scales = tl.load(sf + rows[:, None] * (HEADS * DIM // 16) + head * (DIM // 16) + s[None, :], mask=mask[:, None],
                     other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    for _ in tl.static_range(4):
        scales = tl.interleave(scales, scales)
    return vals * scales * tl.load(g)


@triton.jit
def _gather_fp4(k, v, k_sf, v_sf, gk, gv, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v,
                topk: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr, req_stride: tl.constexpr,
                idx_stride: tl.constexpr, pad_cols, BLOCK_TOPK: tl.constexpr, ZERO_FILL: tl.constexpr):
    """SGLang's _compact_kv (qsa/sparse_attn.py) on packed NVFP4 rows: same validity rules and layouts."""
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, DIM)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(req_to_token + req * req_stride + tl.where(valid, positions, 0), mask=valid, other=0).to(tl.int64)
    dst = (pack_start + cols).to(tl.int64)[:, None] * HEADS * DIM + head * DIM + dims[None, :]
    if ZERO_FILL:   # page-aligned rows: zeros up to the stride, never stale bytes (see SGLang's kernel)
        store = (cols < pad_cols)[:, None] & (dims[None, :] < DIM)
    else:
        store = valid[:, None] & (dims[None, :] < DIM)
    out_dtype = out_k.dtype.element_ty
    tl.store(out_k + dst, _dequant_rows(k, k_sf, gk, slots, valid, head, HEADS, DIM).to(out_dtype), mask=store)
    tl.store(out_v + dst, _dequant_rows(v, v_sf, gv, slots, valid, head, HEADS, DIM).to(out_dtype), mask=store)


@triton.jit
def _pack_fp4(k, v, k_sf, v_sf, gk, gv, packed_k, packed_v, req_to_token, req_indices, cu_k, SR0, SR1,
              HEADS: tl.constexpr, DIM: tl.constexpr, BLOCK_T: tl.constexpr):
    """Request pid 1, tokens of block pid 0, head pid 2: rows req_to_token[req, t] -> packed [cu_k[b] + t]."""
    batch, head = tl.program_id(1), tl.program_id(2)
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    start = tl.load(cu_k + batch).to(tl.int64)
    valid = t < tl.load(cu_k + batch + 1).to(tl.int64) - start
    req = tl.load(req_indices + batch).to(tl.int64)
    slots = tl.load(req_to_token + req * SR0 + t * SR1, mask=valid, other=0).to(tl.int64)
    dims = tl.arange(0, DIM)
    dst = (start + t)[:, None] * HEADS * DIM + head * DIM + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < DIM)
    tl.store(packed_k + dst, _dequant_rows(k, k_sf, gk, slots, valid, head, HEADS, DIM).to(packed_k.dtype.element_ty),
             mask=mask)
    tl.store(packed_v + dst, _dequant_rows(v, v_sf, gv, slots, valid, head, HEADS, DIM).to(packed_v.dtype.element_ty),
             mask=mask)


def _check(raw):
    k = raw[0]
    dim = k.shape[2] * 2
    if not (k.is_contiguous() and raw[2].is_contiguous() and dim % 32 == 0 and dim & (dim - 1) == 0):
        raise ValueError(f'xtier qsa_fp4: packed [m, heads, dim / 2] rows with a power-of-2 head size expected, got '
                         f'{tuple(k.shape)}')
    return k.shape[1], dim


def gather(raw, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk, zero_fill_cols=0):
    heads, dim = _check(raw)
    num_cols = zero_fill_cols if zero_fill_cols > 0 else topk
    _gather_fp4[(batch, heads, triton.cdiv(num_cols, BLOCK_TOPK))](
        *raw, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, topk, heads, dim,
        req_to_token.stride(0), indices.stride(0), num_cols, BLOCK_TOPK=BLOCK_TOPK, ZERO_FILL=zero_fill_cols > 0,
        num_warps=4)


def pack(raw, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype):
    heads, dim = _check(raw)
    device = raw[0].device
    packed_k = torch.empty((total_k, heads, dim), dtype=output_dtype, device=device)
    packed_v = torch.empty((total_k, heads, dim), dtype=output_dtype, device=device)
    batch = req_indices.numel()
    _pack_fp4[(triton.cdiv(max_k, BLOCK_T), batch, heads)](
        *raw, packed_k, packed_v, req_to_token, req_indices, cu_k, *req_to_token.stride(), heads, dim,
        BLOCK_T=BLOCK_T, num_warps=4)
    return packed_k, packed_v


# --- hooks (registered by the pack's install) ---

def create_buffers(result, self, *args, **kwargs):
    """After NVFP4KVCacheMethod.create_buffers (pool creation, no capture yet): the 1.0 global scale on the device."""
    _one(result['k_buffer'][0].device)


def quantize(original, tensor, global_scale, *args, **kwargs):
    if isinstance(global_scale, (int, float)) and global_scale == 1.0 and tensor.device in _ones:
        global_scale = _ones[tensor.device]
    return original(tensor, global_scale, *args, **kwargs)


def forward_extend(original, self, q, k, v, layer, forward_batch, *args, **kwargs):
    raw = raw_buffers(self.token_to_kv_pool, layer)
    if raw is None:
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
    with _layer(raw):
        return original(self, q, k, v, layer, forward_batch, *args, **kwargs)


def pack_qsa_prefill_kv(original, *, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None):
    if not _active:
        return original(k=k, v=v, req_to_token=req_to_token, req_indices=req_indices, cu_k=cu_k, total_k=total_k,
                        max_k=max_k, output_dtype=output_dtype)
    return pack(_active[-1], req_to_token, req_indices, cu_k, total_k, max_k, output_dtype or torch.bfloat16)


def extraction(original, k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
               zero_fill_cols=0):
    if not _active:
        return original(k, v, req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk,
                        zero_fill_cols=zero_fill_cols)
    gather(_active[-1], req_to_token, req_indices, indices, seq_lens, cu_k, out_k, out_v, batch, topk, zero_fill_cols)


def paged_attention(original, self, q, layer, forward_batch, topk_indices):
    """SGLang's _forward_paged_attention for CUDA, the pool buffers replaced by stand-ins of their BF16 shape."""
    raw = raw_buffers(self.token_to_kv_pool, layer)
    if raw is None or not q.is_cuda:
        return original(self, q, layer, forward_batch, topk_indices)
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend
    from sglang.srt.layers.attention.qsa.sparse_attn import (qwen_sparse_fa2_cu_seqlens_triton,
                                                             qwen_sparse_kv_extraction_compact_triton)
    k_buffer = v_buffer = _stand_in(raw)
    metadata = self._resolve_metadata(forward_batch)
    topk_indices = self._expand_block_indices(topk_indices, metadata).to(torch.int32).contiguous()
    with _layer(raw):
        trtllm_decode = backend._resolve_trtllm_sparse_decode()
        if trtllm_decode is not None:
            return self._forward_trtllm_sparse(q, k_buffer, v_buffer, layer, forward_batch, metadata, topk_indices,
                                               trtllm_decode)
        batch, topk = topk_indices.shape
        if metadata.is_cuda_graph:
            valid_counts, cu_seqlens_k = metadata.fa2_valid_counts, metadata.fa2_cu_seqlens_k
            cu_seqlens_q = metadata.fa2_cu_seqlens_q
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
            cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
        qwen_sparse_fa2_cu_seqlens_triton(metadata.sequence_lengths, topk_indices, valid_counts, cu_seqlens_k, batch,
                                          topk)
        capacity = self._cuda_graph_max_tokens * topk if metadata.is_cuda_graph else batch * topk
        packed_k, packed_v = self._get_fa2_scratch(capacity, k_buffer.shape[1], k_buffer.shape[2], q.dtype,
                                                   k_buffer.device)
        rows = metadata.row_req_pool_indices
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer, v_buffer, self.req_to_token_pool.req_to_token,
            rows if rows is not None else forward_batch.req_pool_indices, topk_indices, metadata.sequence_lengths,
            cu_seqlens_k, packed_k, packed_v, batch, topk)
        output = backend._resolve_flash_attn_varlen_func()(
            q=q, k=packed_k, v=packed_v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=1,
            max_seqlen_k=topk, softmax_scale=layer.scaling, causal=True)
        return output.reshape(q.shape[0], -1)
