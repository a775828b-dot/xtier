"""PLE layer of a long prefill chunk in token blocks (around ``Qwen4ExpPLELayer.forward``).

SGLang runs the PLE layer of a prefill chunk over the whole chunk at once, and most of its tensors are [tokens x
hc_hidden] BF16 (32768 x 10240: 640 MiB each): the key projection and its norm, the query norm, the gated value and its
norm, the padded copy, the state concatenation and the output of the short convolution, the sum, a zeros tensor and
the masked result, many alive at the same time (about 5 GB). That is where 32K prefill chunks run out of GPU memory
(production runs 32K chunks). For a single-request chunk of more than XTIER_PLE_BLOCK tokens (default 4096; 0
turns this off) everything after the embedding lookup runs per block of that many tokens: the same per-token
projections, norms and gate, and the depthwise causal convolution with the state_len input columns before the block
([incoming state | chunk tokens] as SGLang lays them out), so every output column sees the same inputs and weights.
The conv state after the chunk and the radix-tree track snapshot are the same columns of that layout. Other batches
(several requests, decode, verify) keep SGLang's code.
"""

import math
import os

import torch
import torch.nn.functional as F

BLOCK = int(os.environ.get('XTIER_PLE_BLOCK', '4096'))


def _columns(state, x, offsets, length):
    """Columns [offset, offset + length) of [state | x^T] for each offset: [len(offsets), channels, length]."""
    cols = offsets.to(torch.long).unsqueeze(1) + torch.arange(length, device=x.device, dtype=torch.long)
    in_x = cols >= length
    from_x = x[(cols - length).clamp(min=0)].permute(0, 2, 1)                       # [n, C, length]
    from_state = state[0][:, cols.clamp(max=max(length - 1, 0))].permute(1, 0, 2)   # [n, C, length]
    return torch.where(in_x.unsqueeze(1), from_x, from_state)


def _blocked(self, batch, tokens):
    return (BLOCK > 0 and tokens > BLOCK and not batch.use_decode_fast_path and not batch.mode.is_target_verify()
            and batch.lengths.shape[0] == 1 and batch.row_width == tokens)


def ple_forward(original, self, hidden_states, forward_batch, batch):
    tokens = batch.processed_tokens
    if not _blocked(self, batch, tokens):
        return original(self, hidden_states, forward_batch, batch)
    from sglang.srt.models import qwen4_exp

    hidden_states = hidden_states[:tokens]
    if self._prefetch_state is not None:
        embeddings = self._consume_prefetched_embeddings(forward_batch)
    else:
        embeddings = self.ple_embedding(batch, forward_batch)
    hc_count, hidden_size = self.hc_count, self.hidden_size
    if hidden_states.shape[-1] != hc_count * hidden_size:
        raise RuntimeError('PLE hidden size does not match its hyper-connection layout: '
                           f'expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}')
    conv_state = qwen4_exp.get_req_to_token_pool().short_conv_layer_cache(self.layer_id)
    length = self.short_conv_state_len
    state = conv_state.index_select(0, batch.state_indices).to(dtype=hidden_states.dtype)   # [1, C, length]
    weight = self.conv1d.weight.to(dtype=hidden_states.dtype)
    out = torch.empty_like(hidden_states)
    normed = torch.empty_like(hidden_states)   # the convolution input of every token (track snapshot)
    before = state
    for t0 in range(0, tokens, BLOCK):
        t1 = min(t0 + BLOCK, tokens)
        n = t1 - t0
        key, _ = self.key_proj(embeddings[t0:t1])
        value, _ = self.value_proj(embeddings[t0:t1])
        key_normed = self._apply_ple_norm(self.norm_key, key.reshape(n, hc_count, hidden_size))
        query_normed = self._apply_ple_norm(self.norm_query, hidden_states[t0:t1].reshape(n, hc_count, hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate)
        gated_value = gate * value.unsqueeze(-2)
        normed[t0:t1] = self._apply_ple_norm(self.norm_conv, gated_value).flatten(-2)
        block = torch.cat([before, normed[t0:t1].t().unsqueeze(0)], dim=-1)   # [1, C, length + n]
        y = F.conv1d(block, weight, bias=None, dilation=self.short_conv_dilation, groups=self.conv_channels)
        out[t0:t1] = gated_value.flatten(-2) + F.silu(y[0].t())
        before = block[:, :, block.shape[-1] - length:] if length else block[:, :, :0]
    conv_state[batch.state_indices] = before.to(dtype=conv_state.dtype)   # the last state_len columns
    track = qwen4_exp._ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, track_offsets = track
        conv_state[track_indices] = _columns(state, normed, track_offsets, length).to(dtype=conv_state.dtype)
    out.masked_fill_(~batch.valid_tokens[:tokens].reshape(tokens, 1), 0)
    return qwen4_exp._pad_token_rows(out, batch.physical_tokens)
