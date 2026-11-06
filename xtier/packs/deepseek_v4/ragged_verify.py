"""Packed DSpark request mapping for Engram and low-ratio attention."""

import torch


def request_rows(starts, num_tokens):
    """Map packed tokens to requests on device, including graph padding suffixes."""
    return torch.searchsorted(
        starts, torch.arange(num_tokens, device=starts.device, dtype=starts.dtype), right=True,
    ) - 1


def token_req_indices(original, forward_batch, *, num_tokens=None):
    layout = forward_batch.ragged_verify_layout
    if layout is None or not forward_batch.forward_mode.is_target_verify():
        return original(forward_batch, num_tokens=num_tokens)
    if num_tokens is None:
        num_tokens = forward_batch.positions.shape[0]
    rows = request_rows(layout.extend_start_loc, num_tokens)
    return forward_batch.req_pool_indices.to(torch.int64)[rows]


def forced_budget(result, self, frac):
    """A calibration budget must bypass the uninitialized-table verify-all shortcut."""
    from sglang.srt.speculative.dspark_components.dspark_sps import is_uninitialized_sps_table

    if self._budget_planner is not None:
        self._is_verify_all = (self.is_compact_mode and frac is None
                               and is_uninitialized_sps_table(self._budget_planner.sps_table))
    return result


def forward(original, self, input_ids, forward_batch):
    """Hash each packed request against its own committed history and token prefix."""
    layout = forward_batch.ragged_verify_layout
    if (layout is None or not forward_batch.forward_mode.is_target_verify()
            or input_ids.shape[0] == 0):
        return original(self, input_ids, forward_batch)

    from sglang.srt.layers import engram

    assert self.history is not None, 'EngramHasher.init_history was not called'
    starts = layout.extend_start_loc
    slots = forward_batch.req_pool_indices
    if starts.shape[0] != slots.shape[0] or starts.shape[0] == 0:
        raise ValueError('Engram ragged verification requires one start per request slot')
    # Device-only mapping also works during CUDA graph capture. Padded suffixes
    # belong to the last row; they never commit history or affect earlier tokens.
    row = request_rows(starts, input_ids.shape[0])
    if engram._cuda_kernels(input_ids):
        hashes, _ = engram.engram_hash_ids(
            input_ids, forward_batch.positions, mode=engram.MODE_EXTEND,
            history=self.history, token_map=self.token_map,
            multipliers=self.multipliers, primes=self.primes, offsets=self.offsets,
            pad_id=self.pad_id, num_real=input_ids.shape[0], req_slots=slots,
            row=row, starts=starts, image_token_id=self.image_token_id,
            mm_pad_shift=engram.MM_PAD_SHIFT_VALUE,
        )
    else:
        hashes, _ = self._torch_hash_ids(
            input_ids, forward_batch.positions, engram.MODE_EXTEND,
            self.history[slots], input_ids.shape[0], 1, row, starts,
        )
    return hashes
