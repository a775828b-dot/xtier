"""Packed Engram verification matches independent requests without changing history."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from xtier.packs.deepseek_v4.ragged_verify import forward, forced_budget, token_req_indices


class RaggedHashTests(unittest.TestCase):
    def test_attention_mapping_uses_request_starts(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=torch.tensor([3, 1]),
            ragged_verify_layout=SimpleNamespace(extend_start_loc=torch.tensor([0, 2])),
        )
        actual = token_req_indices(None, batch, num_tokens=7)
        torch.testing.assert_close(actual, torch.tensor([3, 3, 1, 1, 1, 1, 1]))

    def test_calibration_disables_then_restores_verify_all(self):
        from sglang.srt.speculative.dspark_components.dspark_sps import build_uninitialized_sps_table
        planner = SimpleNamespace(
            is_compact_mode=True, _is_verify_all=True,
            _budget_planner=SimpleNamespace(sps_table=build_uninitialized_sps_table(max_batch_tokens=12)),
        )
        forced_budget(None, planner, 0.2)
        self.assertFalse(planner._is_verify_all)
        forced_budget(None, planner, None)
        self.assertTrue(planner._is_verify_all)

    def check_device(self, device):
        hasher = EngramHasher.__new__(EngramHasher)
        torch.nn.Module.__init__(hasher)
        hasher.max_ngram_size = 3
        hasher.pad_id = 0
        hasher.image_token_id = 31
        hasher._cpu_hash_params = None
        for name, values in (
            ('token_map', list(range(64))),
            ('multipliers', [[3, 5, 7], [11, 13, 17]]),
            ('primes', [[[101], [103]], [[107], [109]]]),
            ('offsets', [[0, 101], [0, 107]]),
        ):
            hasher.register_buffer(name, torch.tensor(values, device=device))
        hasher.init_history(4, device)
        hasher.history[:] = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [0, 0]], device=device)
        for lengths in ([1, 5], [3, 2], [6, 1], [1, 1]):
            for padding in (0, 3):
                with self.subTest(device=device, lengths=lengths, padding=padding):
                    sequences = [torch.tensor([10, 31, 12, 13, 14, 15][:n], device=device)
                                 for n in lengths]
                    positions = [torch.arange(n, device=device) + offset
                                 for n, offset in zip(lengths, (0, 20))]
                    slots = torch.tensor([3, 1], device=device)
                    before = hasher.history.clone()
                    expected = []
                    for i, sequence in enumerate(sequences):
                        batch = SimpleNamespace(
                            forward_mode=ForwardMode.TARGET_VERIFY, req_pool_indices=slots[i:i+1],
                            positions=positions[i], spec_info=SimpleNamespace(draft_token_num=len(sequence)),
                        )
                        expected.append(EngramHasher.forward(hasher, sequence, batch))
                    ids = torch.cat([*sequences, torch.zeros(padding, dtype=torch.long, device=device)])
                    batch = SimpleNamespace(
                        forward_mode=ForwardMode.TARGET_VERIFY, req_pool_indices=slots,
                        positions=torch.cat([*positions, torch.zeros(padding, dtype=torch.long, device=device)]),
                        ragged_verify_layout=SimpleNamespace(
                            extend_start_loc=torch.tensor([0, lengths[0]], dtype=torch.int32, device=device)),
                    )
                    actual = forward(EngramHasher.forward, hasher, ids, batch)
                    torch.testing.assert_close(actual[:sum(lengths)], torch.cat(expected), rtol=0, atol=0)
                    torch.testing.assert_close(hasher.history, before, rtol=0, atol=0)

    def test_cpu_request_boundaries_and_padding(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_request_boundaries_and_padding(self):
        self.check_device('cuda')


if __name__ == '__main__':
    unittest.main()
