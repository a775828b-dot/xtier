"""Unit & GPU/CUDA-graph verification for Universal Recall + short-suffix + zero-sync GPU propose_after.

Covers:
  1. CPU _OwnSuffixIndex (4..15-gram immediate prompt/output indexing) + RecallEngine.propose_after()
     non-mutating chaining + build_plan() multi-hop + XTIER_RECALL_SUFFIX_MIN=0 disable + ds41 shims.
  2. GPU GpuSuffixPool + _suffix_propose_after_kernel:
       - self-clobber avoidance (latest tail does not clobber earlier match in HT_POS),
       - incremental decode append,
       - lag=0, lag=1 (overlap 1-token lag), lag=3 (multi-token overlap pure-GPU 4-gram & no stale backward match),
       - confidence gating (min_conf), skip_mask, hash/token verification,
       - _onehot_masked_kernel (draft_probs 0/1 & corrected_logits -inf/0 on fp32 & bf16, untouched rows bit-exact).
  3. DSpark adapter (_on_dspark_propose): anchor Recall hit + GPU propose_after partial-tail hit + miss,
     verifying draft_tokens, epilogue_buf, verify_ids, confidence, and partial-tail corrected_logits one-hot.
  4. EAGLE adapter (on_eagle_verify) + truncate.py interplay: fixed & adaptive steps, draft_probs=None & !=None,
     recalled/chained tokens kept (q=1.0) while unmatched low-q neural tails still truncate.
  5. DFly2 adapter inside a real captured torch.cuda.CUDAGraph:
       - anchor Recall hit + GPU propose_after partial-tail hit + miss,
       - dynamic greedy (min_match=4) -> non-greedy (min_match=5) switch via static GPU _recall_cfg without recapture.
  6. Idempotent install(), XTIER_RECALL=0 opt-out, and after_step() resource release.

Usage:
    python tests/recall_universal_check.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xtier.packs.speculative import recall, truncate  # noqa: E402
from xtier.packs.speculative.recall import core as recall_core  # noqa: E402
from xtier.packs.speculative.recall import gpu_suffix, runtime  # noqa: E402


class _FakeReq:
    def __init__(self, rid, origin, out=None, finished=False, retracted=False, req_pool_idx=0):
        self.rid = rid
        self.origin_input_ids = list(origin)
        self.output_ids = list(out or [])
        self._finished = finished
        self.is_retracted = retracted
        self.req_pool_idx = req_pool_idx
        self.cached_tokens = 0
        self.finished_reason = None

    def finished(self):
        return self._finished


class _FakeMode:
    def __init__(self, idle=False):
        self._idle = idle

    def is_idle(self):
        return self._idle


def _reset_runtime():
    recall_core._ENGINE = None
    runtime._state["engine"] = None
    runtime._state["engine_failed"] = False
    runtime._state["group"] = types.SimpleNamespace(world_size=1, rank_in_group=0)
    runtime._state["recall_hot"] = frozenset()
    runtime._state["paused"] = False
    runtime._state["gpu_pool"] = None
    runtime._state["gpu_pool_failed"] = False
    runtime._fifo.clear()
    runtime._counts.clear()


def test_cpu_suffix_and_propose_after():
    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL_BACKEND"] = "memory"
    os.environ["XTIER_RECALL_SUFFIX_MIN"] = "4"
    os.environ["XTIER_RECALL_CHAIN"] = "1"

    ident = {"tokenizer_digest": "test-digest", "vocab_size": 4096, "model": "test-model"}
    # 1a. Default suffix_min is 8; explicit XTIER_RECALL_SUFFIX_MIN=4 still works for short 4-gram tests.
    eng_def = recall_core.RecallEngine(
        recall_core.RecallRecord("", ident, backend="memory"),
        gamma=5,
        environ={},
    )
    assert eng_def.suffix_min == 8, f"Expected default suffix_min=8, got {eng_def.suffix_min}"

    eng = recall_core.RecallEngine(
        recall_core.RecallRecord("", ident, backend="memory"),
        gamma=5,
        environ={"XTIER_RECALL_SUFFIX_MIN": "4"},
    )
    assert eng.suffix_min == 4

    # 1a-ii. Short 4-gram match inside prompt fires immediately on step 1 (before record flush).
    prompt = [100, 101, 102, 103, 501, 502, 503, 504, 505, 506, 900, 901, 902, 100, 101, 102, 103]
    eng.begin_stream("r1", prompt)
    prop = eng.propose("r1", n=5)
    assert prop is not None, "Expected 4-gram self-prompt match"
    assert prop.match_len == 4, f"Expected match_len=4, got {prop.match_len}"
    assert prop.tokens[:5].tolist() == [501, 502, 503, 504, 505], prop.tokens.tolist()

    # 1a-iii. Default suffix_min=8 rejects ambiguous 4-gram and accepts unique 8-gram.
    eng_def.begin_stream("r1_4", prompt)
    assert eng_def.propose("r1_4", n=5) is None
    prompt8 = [100, 101, 102, 103, 104, 105, 106, 107, 501, 502, 503, 504, 505, 900, 100, 101, 102, 103, 104, 105, 106, 107]
    eng_def.begin_stream("r1_8", prompt8)
    prop8 = eng_def.propose("r1_8", n=5)
    assert prop8 is not None and prop8.match_len == 8 and prop8.tokens[:5].tolist() == [501, 502, 503, 504, 505]

    # 1b. Non-mutating propose_after() extends after neural draft tokens without mutating stream state.
    prompt2 = [200, 201, 202, 203, 601, 602, 603, 604, 605, 700, 701, 702, 200, 201]
    eng.begin_stream("r2", prompt2)
    n_before = eng._streams["r2"].n
    sfx_before = eng._streams["r2"].sfx.n
    assert eng.propose("r2", n=5) is None  # only 2-gram at current tail
    ext = eng.propose_after("r2", [202, 203], n=4)
    assert ext is not None, "Expected propose_after match after [202, 203]"
    assert ext.tokens[:4].tolist() == [601, 602, 603, 604], ext.tokens.tolist()
    assert eng._streams["r2"].n == n_before and eng._streams["r2"].sfx.n == sfx_before

    # 1c. build_plan multi-hop chaining: first match gives 2 tokens near end, hop 2 extends further.
    # History: [300,301,302,303, 304,305, 801,802,803,804,805,806,807,808,809,810, 990,991,992, 300,301,302,303]
    # Wait: test partial match that ends at prompt boundary and chains into an earlier occurrence:
    # Occurrence A (early): [302,303,304,305] -> [801,802,803,804,805,806,807,808]
    # Occurrence B (late):  [300,301,302,303, 304,305] at the very end -> only 2 tokens [304,305] avail after [300..303]
    # Wait, if [300..303] is repeated at the very end again, Occurrence B only has [304,305] before another marker!
    # Simpler exact check: prompt ends with [400,401,402,403], earlier has [400,401,402,403, 404,405] right before EOF?
    # Let's construct: [302,303,304,305, 801,802,803,804,805,806,807,808,809,810, 950,951, 300,301,302,303,304,305]
    # Then extend with [960, 300, 301, 302, 303]:
    # Now [300,301,302,303] matches late occurrence -> cont=[304,305,960,...] (not bounded).
    # Instead, put [300,301,302,303] with only 2 tokens left before the end of hist:
    # hist = [302,303,304,305, 801,802,803,804,805,806,807,808,809,810, 300,301,302,303,304,305]
    # Actually at self_end=len-1, match of [300..303] hits idx 17 -> avail=2 -> cont=[304,305]!
    # Then hop 2 sees [...,300,301,302,303] + [304,305] -> tail ends in [302,303,304,305] -> hits idx 3 -> [801..808]!
    hop_prompt = [
        302, 303, 304, 305, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810,
        300, 301, 302, 303, 304, 305, 300, 301, 302, 303,
    ]
    req_hop = _FakeReq("r_hop", hop_prompt)
    plan, entries = runtime.build_plan(eng, [req_hop], gamma=5)
    assert plan[0, 0] >= 5, f"Expected multi-hop plan real>=5, got {plan[0, 0]}"
    # Anchor is 303 (plan[0, 2]), plan[0, 3:10] should start with [304, 305] then chain!
    assert plan[0, 2] == 303 and plan[0, 3:5].tolist() == [304, 305], plan[0, 2:10].tolist()
    assert plan[0, 5:10].tolist() == [300, 301, 302, 303, 304] or plan[0, 5:10].tolist() == [801, 802, 803, 804, 805], (
        f"Unexpected hop continuation: {plan[0, 2:12].tolist()}"
    )

    # 1d. XTIER_RECALL_SUFFIX_MIN=0 disables short-suffix matching cleanly.
    eng_off = recall_core.RecallEngine(
        recall_core.RecallRecord("", ident, backend="memory"),
        gamma=5,
        environ={"XTIER_RECALL_SUFFIX_MIN": "0"},
    )
    assert eng_off.suffix_min == 0
    eng_off.begin_stream("r_off", prompt)
    assert eng_off.propose("r_off", n=5) is None

    # 1e. Backward-compat shims in xtier.packs.deepseek_v4.recall.
    from xtier.packs.deepseek_v4 import recall as ds_recall
    from xtier.packs.deepseek_v4.recall import core as ds_core
    from xtier.packs.deepseek_v4.recall import dspark as ds_dspark
    from xtier.packs.deepseek_v4.recall import tree as ds_tree

    assert ds_recall.install is recall.install
    assert ds_core.RecallEngine is recall_core.RecallEngine
    assert ds_dspark.build_plan is runtime.build_plan
    assert hasattr(ds_tree, "propose")
    print("[PASS] 1. CPU _OwnSuffixIndex + propose_after + multi-hop + shims")


def test_gpu_suffix_pool_and_kernels():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA unavailable")
        return
    dev = torch.device("cuda")
    pool = gpu_suffix.GpuSuffixPool(dev, max_slots=4)

    # 2a. Self-clobber avoidance + p=0 >= 8-gram bar:
    # 8-gram [10..17] appears at index 7 AND at hlen-1.
    # Because _rebuild_slot inserts only end <= n-2, HT_POS keeps 7 and lag=0,p=0 hits [18..22]!
    orig = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 90, 91, 92, 10, 11, 12, 13, 14, 15, 16, 17]
    req0 = _FakeReq("g0", orig)
    assert pool.sync_batch([req0]) is True
    gamma = 5
    seq_lens = torch.tensor([len(orig) - 1], dtype=torch.int64, device=dev)  # lag = 0
    anchors = torch.tensor([17], dtype=torch.int32, device=dev)
    drafts = torch.tensor([[999, 998, 997, 996, 995]], dtype=torch.int32, device=dev)
    ow = torch.zeros((1, gamma), dtype=torch.uint8, device=dev)
    pool.propose_after(
        bs=1, gamma=gamma, seq_lens=seq_lens, anchors=anchors,
        draft_tokens=drafts, overwrite=ow, min_match=6,
    )
    torch.cuda.synchronize()
    assert drafts[0].tolist() == [18, 19, 20, 21, 22], f"Self-clobber test failed: {drafts[0].tolist()}"
    assert ow[0].tolist() == [1, 1, 1, 1, 1], ow[0].tolist()

    # 2b. Incremental decode append + lag=0 p=2 6-gram tail chaining + high-conf protection:
    # Append 6-gram [75..80] -> [81..85], then tail [500, 75, 76, 77, 78] (only 4-gram -> p=0 skipped).
    # Neural d0=79, d1=80 -> at p=2, [75,76,77,78,79,80] (6-gram!) matches and overwrites d2..d4 with [81,82,83].
    req0.output_ids.extend([75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 500, 75, 76, 77, 78])
    pool.sync_batch([req0])
    total_len = len(req0.origin_input_ids) + len(req0.output_ids)
    seq_lens = torch.tensor([total_len - 1], dtype=torch.int64, device=dev)
    anchors = torch.tensor([78], dtype=torch.int32, device=dev)
    drafts = torch.tensor([[79, 80, 991, 992, 993]], dtype=torch.int32, device=dev)
    conf = torch.tensor([[0.9, 0.8, 0.4, 0.1, 0.1]], dtype=torch.float32, device=dev)
    ow.zero_()
    pool.propose_after(
        bs=1, gamma=gamma, seq_lens=seq_lens, anchors=anchors,
        draft_tokens=drafts, overwrite=ow, confidence=conf, min_match=6, min_conf=0.35,
    )
    torch.cuda.synchronize()
    assert drafts[0].tolist() == [79, 80, 81, 82, 83], drafts[0].tolist()
    assert ow[0].tolist() == [0, 0, 1, 1, 1], ow[0].tolist()
    assert abs(conf[0, 0].item() - 0.9) < 1e-5 and conf[0, 2:].tolist() == [1.0, 1.0, 1.0], conf[0].tolist()

    # 2b-ii. High neural confidence (>=0.90) on target slot d2 protects it from short (<10-gram) overwrite!
    drafts_hc = torch.tensor([[79, 80, 991, 992, 993]], dtype=torch.int32, device=dev)
    conf_hc = torch.tensor([[0.95, 0.95, 0.95, 0.95, 0.95]], dtype=torch.float32, device=dev)
    ow.zero_()
    pool.propose_after(
        bs=1, gamma=gamma, seq_lens=seq_lens, anchors=anchors,
        draft_tokens=drafts_hc, overwrite=ow, confidence=conf_hc, min_match=6, min_conf=0.35,
    )
    torch.cuda.synchronize()
    assert drafts_hc[0].tolist() == [79, 80, 991, 992, 993], f"High-conf slot clobbered: {drafts_hc[0].tolist()}"
    assert ow[0].tolist() == [0, 0, 0, 0, 0]

    # 2c. Confidence gating: low conf[0] blocks p=2 chaining.
    drafts2 = torch.tensor([[79, 80, 991, 992, 993]], dtype=torch.int32, device=dev)
    conf2 = torch.tensor([[0.2, 0.9, 0.4, 0.1, 0.1]], dtype=torch.float32, device=dev)
    ow.zero_()
    pool.propose_after(
        bs=1, gamma=gamma, seq_lens=seq_lens, anchors=anchors,
        draft_tokens=drafts2, overwrite=ow, confidence=conf2, min_match=6, min_conf=0.35,
    )
    torch.cuda.synchronize()
    assert drafts2[0].tolist() == [79, 80, 991, 992, 993], drafts2[0].tolist()
    assert ow[0].tolist() == [0, 0, 0, 0, 0], ow[0].tolist()

    # 2d. Overlap lag=1: CPU hist ends at 77 (before anchor=78), seq_lens == blen -> lag = 1!
    req_lag1 = _FakeReq("g_lag1", [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 500, 75, 76, 77])
    pool.sync_batch([req_lag1])
    seq_l1 = torch.tensor([len(req_lag1.origin_input_ids)], dtype=torch.int64, device=dev)  # lag = 1
    anc_l1 = torch.tensor([78], dtype=torch.int32, device=dev)
    d_l1 = torch.tensor([[79, 80, 991, 992, 993]], dtype=torch.int32, device=dev)
    ow.zero_()
    pool.propose_after(
        bs=1, gamma=gamma, seq_lens=seq_l1, anchors=anc_l1,
        draft_tokens=d_l1, overwrite=ow, min_match=6,
    )
    torch.cuda.synchronize()
    assert d_l1[0].tolist() == [79, 80, 81, 82, 83], f"lag=1 failed: {d_l1[0].tolist()}"
    assert ow[0].tolist() == [0, 0, 1, 1, 1], ow[0].tolist()

    # 2e. Overlap lag=3 (multi-token lag): pure-GPU 6-gram [anchor, d0..d4] matches at p=5 (gamma=7),
    # AND backward extension must NOT touch stale hist (verified with min_match=7)!
    req_lag3 = _FakeReq("g_lag3", [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 500, 501, 67])
    pool.sync_batch([req_lag3])
    seq_l3 = torch.tensor([len(req_lag3.origin_input_ids) + 2], dtype=torch.int64, device=dev)  # lag = 3
    anc_l3 = torch.tensor([68], dtype=torch.int32, device=dev)
    d_l3 = torch.tensor([[69, 70, 71, 72, 73, 991, 992]], dtype=torch.int32, device=dev)
    ow7 = torch.zeros((1, 7), dtype=torch.uint8, device=dev)
    # min_match=7 MUST REJECT even though hist[-1]==67 because lag=3 means hist[-1] is NOT adjacent to anchor!
    pool.propose_after(
        bs=1, gamma=7, seq_lens=seq_l3, anchors=anc_l3,
        draft_tokens=d_l3, overwrite=ow7, min_match=7,
    )
    torch.cuda.synchronize()
    assert d_l3[0].tolist() == [69, 70, 71, 72, 73, 991, 992], f"lag=3 stale-hist leak! {d_l3[0].tolist()}"
    assert ow7[0].tolist() == [0] * 7
    # min_match=6 MUST ACCEPT at p=5 -> overwrites d5, d6 with [74, 75]!
    pool.propose_after(
        bs=1, gamma=7, seq_lens=seq_l3, anchors=anc_l3,
        draft_tokens=d_l3, overwrite=ow7, min_match=6,
    )
    torch.cuda.synchronize()
    assert d_l3[0].tolist() == [69, 70, 71, 72, 73, 74, 75], f"lag=3 p=5 failed: {d_l3[0].tolist()}"
    assert ow7[0].tolist() == [0, 0, 0, 0, 0, 1, 1], ow7[0].tolist()

    # 2f. skip_mask skips flagged requests cleanly.
    d_skip = torch.tensor([[69, 70, 71, 72, 73, 991, 992]], dtype=torch.int32, device=dev)
    ow7.zero_()
    pool.propose_after(
        bs=1, gamma=7, seq_lens=seq_l3, anchors=anc_l3,
        draft_tokens=d_skip, skip_mask=torch.tensor([1], dtype=torch.int32, device=dev),
        overwrite=ow7, min_match=6,
    )
    torch.cuda.synchronize()
    assert d_skip[0].tolist() == [69, 70, 71, 72, 73, 991, 992] and ow7[0].tolist() == [0] * 7

    # 2f-ii. Exact real-count masking in device_merge: NEVER overwrite slots k where o + k >= real!
    # Plan has real=4 continuation tokens [101, 102, 103, 104] followed by pad 104, with overlap o=2.
    # Only k=0 (103) and k=1 (104) are real; k=2,3,4 MUST preserve neural drafts [881, 882, 883]!
    plan_pad = torch.zeros((1, 2 + 5 + 7), dtype=torch.int64, device=dev)
    plan_pad[0, 0] = 4  # real = 4
    plan_pad[0, 1] = 10  # L = 10
    plan_pad[0, 2:] = torch.tensor([100, 101, 102, 103, 104, 104, 104, 104, 104, 104, 104, 104], device=dev)
    dt_pad = torch.tensor([[880, 881, 882, 883, 884]], dtype=torch.int32, device=dev)
    conf_pad = torch.full((1, 5), 0.75, dtype=torch.float32, device=dev)
    ok_pos = runtime.device_merge(
        plan_pad,
        torch.tensor([11], dtype=torch.int64, device=dev),  # o = 11 - 10 + 1 = 2
        torch.tensor([102], dtype=torch.int32, device=dev),
        dt_pad,
        conf_pad,
        gamma=5,
    )
    torch.cuda.synchronize()
    assert dt_pad[0].tolist() == [103, 104, 882, 883, 884], f"Pad token leaked into drafts! {dt_pad[0].tolist()}"
    assert ok_pos[0].tolist() == [True, True, False, False, False], ok_pos[0].tolist()
    assert conf_pad[0].tolist() == [1.0, 1.0, 0.75, 0.75, 0.75], conf_pad[0].tolist()

    # 2g. _onehot_masked_kernel: both probs (0/1) and logits (-inf/0) on fp32 & bf16; untouched rows bit-exact.
    for dtype in (torch.float32, torch.bfloat16):
        V = 8201  # non-multiple of BLOCK_V=4096
        orig_t = torch.randn((2, 4, V), dtype=dtype, device=dev)
        ow_m = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.uint8, device=dev)
        toks = torch.tensor([[3, 17, 8200, 9], [0, 5, 6, 4096]], dtype=torch.int32, device=dev)

        probs = orig_t.clone()
        gpu_suffix.onehot_masked(probs, ow_m, toks, is_probs=True)
        logits = orig_t.clone()
        gpu_suffix.onehot_masked(logits, ow_m, toks, is_probs=False)
        torch.cuda.synchronize()

        for b in range(2):
            for j in range(4):
                if ow_m[b, j].item() == 0:
                    assert torch.equal(probs[b, j], orig_t[b, j])
                    assert torch.equal(logits[b, j], orig_t[b, j])
                else:
                    t = toks[b, j].item()
                    assert probs[b, j, t].item() == 1.0 and probs[b, j].sum().item() == 1.0
                    assert logits[b, j, t].item() == 0.0
                    assert torch.isneginf(logits[b, j, (t + 1) % V]).item()
    print("[PASS] 2. GPU GpuSuffixPool + _suffix_propose_after_kernel + _onehot_masked_kernel")


def test_dspark_adapter():
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    _reset_runtime()
    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL_BACKEND"] = "memory"
    os.environ.pop("XTIER_RECALL_SUFFIX_MIN", None)  # test production default (CPU=8, GPU=6 greedy / 8 non-greedy)
    os.environ["XTIER_RECALL_CHAIN"] = "1"

    # req0: CPU anchor hit (8-gram [10..17] -> [18..22])
    r0 = _FakeReq("ds0", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17], req_pool_idx=0)
    # req1: CPU anchor miss (6-gram [20..25] < 8), GPU propose_after hit at p=2 with neural [26,27] -> 8-gram [20..27] -> [28,29,30]
    r1 = _FakeReq("ds1", [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 93, 94, 20, 21, 22, 23, 24, 25], req_pool_idx=1)
    # req2: complete miss
    r2 = _FakeReq("ds2", [300, 301, 302, 303, 304, 305, 306, 307], req_pool_idx=2)

    bs, gamma, V = 3, 5, 1024
    dt = torch.tensor(
        [
            [500, 501, 502, 503, 504],
            [26, 27, 510, 511, 512],
            [600, 601, 602, 603, 604],
        ],
        dtype=torch.int32,
        device=dev,
    )
    anchors = torch.tensor([17, 25, 307], dtype=torch.int32, device=dev)
    block_ids = torch.cat([anchors.unsqueeze(1), dt], dim=1).contiguous()
    verify_ids = block_ids.clone()
    epi_buf = torch.zeros((bs * gamma,), dtype=torch.int32, device=dev)
    conf = torch.full((bs, gamma), 0.8, dtype=torch.float32, device=dev)
    cl_orig = torch.randn((bs, gamma, V), dtype=torch.float32, device=dev)
    cl = cl_orig.clone()

    batch = types.SimpleNamespace(
        reqs=[r0, r1, r2],
        seq_lens=torch.tensor(
            [len(r0.origin_input_ids) - 1, len(r1.origin_input_ids) - 1, len(r2.origin_input_ids) - 1],
            dtype=torch.int64,
            device=dev,
        ),
        has_grammar=False,
    )
    out = types.SimpleNamespace(
        draft_block=types.SimpleNamespace(draft_tokens=dt, corrected_logits=cl),
        draft_block_ids=block_ids,
        verify_ids=verify_ids,
        confidence=conf,
    )
    worker = types.SimpleNamespace(_verify_epilogue=types.SimpleNamespace(draft_tokens_buf=epi_buf))
    kw = {"batch": batch, "bs": bs, "sampling_info": types.SimpleNamespace(is_all_greedy=False)}

    runtime._on_dspark_propose(worker, kw, out)
    torch.cuda.synchronize()

    assert dt[0].tolist() == [18, 19, 20, 21, 22], dt[0].tolist()
    assert dt[1].tolist() == [26, 27, 28, 29, 30], dt[1].tolist()
    assert dt[2].tolist() == [600, 601, 602, 603, 604], dt[2].tolist()
    assert torch.equal(epi_buf.view(bs, gamma), dt)
    assert torch.equal(verify_ids[:, 1:], dt)
    assert conf[0].tolist() == [1.0] * 5
    assert torch.allclose(conf[1, :2], torch.full((2,), 0.8, device=dev)) and conf[1, 2:].tolist() == [1.0, 1.0, 1.0]
    assert torch.allclose(conf[2], torch.full((5,), 0.8, device=dev))
    # Partial-tail one-hot check on corrected_logits:
    # r0: all 5 slots one-hot; r1: slots 0..1 UNTOUCHED bit-exact, slots 2..4 one-hot; r2: all UNTOUCHED!
    for j in range(5):
        assert cl[0, j, dt[0, j]].item() == 0.0 and torch.isneginf(cl[0, j, 0]).item()
    assert torch.equal(cl[1, :2], cl_orig[1, :2]), "Neural prefix logits on r1 were modified!"
    for j in range(2, 5):
        assert cl[1, j, dt[1, j]].item() == 0.0 and torch.isneginf(cl[1, j, 0]).item()
    assert torch.equal(cl[2], cl_orig[2]), "Miss row r2 logits were modified!"
    print("[PASS] 3. DSpark adapter (anchor Recall + GPU propose_after + partial-tail corrected_logits)")


def test_eagle_adapter_and_truncate():
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    _reset_runtime()
    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL_BACKEND"] = "memory"
    os.environ.pop("XTIER_RECALL_SUFFIX_MIN", None)
    os.environ["XTIER_RECALL_CHAIN"] = "1"

    # Test with GLM-5.3 style rejection sampling (draft_probs != None) + truncate.apply interplay!
    r0 = _FakeReq("e0", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17])  # CPU 8-gram hit
    r1 = _FakeReq("e1", [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 92, 93, 20, 21, 22, 23, 24, 25])  # GPU propose_after (ml=8)
    r2 = _FakeReq("e2", [400, 401, 402, 403, 404, 405])  # miss -> low-q tail should still truncate!

    bs, steps, V = 3, 4, 1024
    width = steps + 1
    dt_2d = torch.tensor(
        [
            [17, 500, 501, 502, 503],
            [25, 26, 27, 510, 511],
            [405, 600, 601, 602, 603],
        ],
        dtype=torch.int64,
        device=dev,
    )
    # Build draft_probs where r0 has low neural probs (0.05) -> Recall overwrites all to 1.0 -> truncate KEEPS!
    # r1 has high neural prob on steps 0,1 (0.7) and low on steps 2,3 (0.05) -> propose_after overwrites 2,3 to 1.0 -> KEEPS!
    # r2 has high on step 0 (0.8) and low on steps 1..3 (0.05) -> miss -> truncate TRUNCATES step 1..3!
    dp = torch.full((bs, steps, V), 1e-5, dtype=torch.float32, device=dev)
    for j in range(steps):
        dp[0, j, dt_2d[0, 1 + j]] = 0.05
    dp[1, 0, 26] = 0.70
    dp[1, 1, 27] = 0.70
    dp[1, 2, 510] = 0.05
    dp[1, 3, 511] = 0.05
    dp[2, 0, 600] = 0.80
    dp[2, 1, 601] = 0.05
    dp[2, 2, 602] = 0.05
    dp[2, 3, 603] = 0.05
    dp_r1_prefix = dp[1, :2].clone()

    vi = types.SimpleNamespace(
        topk=1,
        draft_token_num=width,
        draft_token=dt_2d.view(-1),
        draft_probs=dp,
        retrive_next_token=torch.tensor(
            [[1, 2, 3, 4, -1]] * bs, dtype=torch.int64, device=dev
        ),
        retrieve_next_token=torch.tensor(
            [[1, 2, 3, 4, -1]] * bs, dtype=torch.int64, device=dev
        ),
    )
    batch = types.SimpleNamespace(
        forward_mode=_FakeMode(False),
        has_grammar=False,
        spec_info=vi,
        reqs=[r0, r1, r2],
        seq_lens=torch.tensor(
            [len(r0.origin_input_ids) - 1, len(r1.origin_input_ids) - 1, len(r2.origin_input_ids) - 1],
            dtype=torch.int64,
            device=dev,
        ),
        sampling_info=types.SimpleNamespace(is_all_greedy=False),
    )

    assert recall.on_eagle_verify(None, batch) is True
    torch.cuda.synchronize()
    assert dt_2d[0, 1:].tolist() == [18, 19, 20, 21], dt_2d[0].tolist()
    assert dt_2d[1, 1:].tolist() == [26, 27, 28, 29], dt_2d[1].tolist()
    assert dt_2d[2, 1:].tolist() == [600, 601, 602, 603], dt_2d[2].tolist()
    assert torch.equal(dp[1, :2], dp_r1_prefix), "EAGLE neural prefix draft_probs modified!"
    assert dp[0, :, 18].tolist()[0] == 1.0 and dp[1, 2, 28].item() == 1.0 and dp[1, 3, 29].item() == 1.0

    # Now apply truncate.py (threshold=0.2) right after on_eagle_verify:
    truncate.enable(dev)
    applied = truncate.apply(batch, topk=1, threshold=0.2)
    try:
        assert applied is True
        torch.cuda.synchronize()
        flags = truncate._flags[0][: bs * width].view(bs, width).cpu().tolist()
        assert flags[0] == [1, 1, 1, 1, 1], f"r0 (recalled) should NOT be truncated: {flags[0]}"
        assert flags[1] == [1, 1, 1, 1, 1], f"r1 (chained) should NOT be truncated: {flags[1]}"
        assert flags[2] == [1, 1, 0, 0, 0], f"r2 (miss, low-q at step 1) SHOULD be truncated: {flags[2]}"
        assert dp[2, 1].abs().max().item() == 0.0, "r2 truncated q row should be zeroed"
    finally:
        if applied:
            truncate.reset()

    # Also test Qwen3.8 MTP target-only mode (draft_probs=None, fixed steps=3, greedy)
    dt_q = torch.tensor([[17, 500, 501, 502]], dtype=torch.int64, device=dev)
    vi_q = types.SimpleNamespace(topk=1, draft_token_num=4, draft_token=dt_q.view(-1), draft_probs=None)
    batch_q = types.SimpleNamespace(
        forward_mode=_FakeMode(False),
        has_grammar=False,
        spec_info=vi_q,
        reqs=[r0],
        seq_lens=torch.tensor([len(r0.origin_input_ids) - 1], dtype=torch.int64, device=dev),
        sampling_info=types.SimpleNamespace(is_all_greedy=True),
    )
    assert recall.on_eagle_verify(None, batch_q) is True
    torch.cuda.synchronize()
    assert dt_q[0, 1:].tolist() == [18, 19, 20], dt_q[0].tolist()
    print("[PASS] 4. EAGLE adapter (fixed/adaptive, target-only & rejection sampling + truncate.py)")


def test_dfly_adapter_inside_cuda_graph():
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda")
    _reset_runtime()
    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL_BACKEND"] = "memory"
    os.environ.pop("XTIER_RECALL_SUFFIX_MIN", None)
    os.environ["XTIER_RECALL_CHAIN"] = "1"

    bs, width = 3, 6  # gamma = 5
    gamma = width - 1
    dfly = types.SimpleNamespace(
        candidates_out=torch.zeros((bs * width,), dtype=torch.int32, device=dev),
        dfly_config=types.SimpleNamespace(block_size=width),
        _conf_probs=torch.full((bs, gamma), 0.75, dtype=torch.float32, device=dev),
    )
    recall.bind_dfly_buffers(dfly)
    assert dfly._recall_plan is not None and dfly._recall_gpu_pool is not None

    block_ids = torch.zeros((bs, width), dtype=torch.int32, device=dev)
    positions = torch.zeros((bs * width,), dtype=torch.int64, device=dev)

    # Warmup & capture merge_dfly_graph inside a REAL CUDA graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        recall.merge_dfly_graph(dfly, block_ids, positions, bs, width)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        recall.merge_dfly_graph(dfly, block_ids, positions, bs, width)
    torch.cuda.synchronize()

    # Replay 1 (greedy, min_match=6):
    # r0: CPU 8-gram anchor hit; r1: GPU 6-gram propose_after at p=1; r2: miss.
    r0 = _FakeReq("df0", [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 90, 91, 10, 11, 12, 13, 14, 15, 16, 17])
    r1 = _FakeReq("df1", [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 92, 93, 20, 21, 22, 23, 24])  # 6-gram [20..24, d0=25]
    r2 = _FakeReq("df2", [700, 701, 702, 703, 704, 705])
    batch = types.SimpleNamespace(
        reqs=[r0, r1, r2],
        has_grammar=False,
        sampling_info=types.SimpleNamespace(is_all_greedy=True),
    )
    recall.prepare_dfly_step(dfly, batch, gamma)
    block_ids.copy_(torch.tensor(
        [
            [17, 500, 501, 502, 503, 504],
            [24, 25, 510, 511, 512, 513],
            [705, 600, 601, 602, 603, 604],
        ],
        dtype=torch.int32,
        device=dev,
    ))
    pos_2d = positions.view(bs, width)
    pos_2d[0, 0] = len(r0.origin_input_ids) - 1
    pos_2d[1, 0] = len(r1.origin_input_ids) - 1
    pos_2d[2, 0] = len(r2.origin_input_ids) - 1
    dfly._conf_probs.fill_(0.75)

    g.replay()
    torch.cuda.synchronize()
    assert block_ids[0, 1:].tolist() == [18, 19, 20, 21, 22], block_ids[0].tolist()
    assert block_ids[1, 1:].tolist() == [25, 26, 27, 28, 29], block_ids[1].tolist()
    assert block_ids[2, 1:].tolist() == [600, 601, 602, 603, 604], block_ids[2].tolist()
    assert dfly._conf_probs[0].tolist() == [1.0] * 5
    assert dfly._conf_probs[1].tolist() == [0.75, 1.0, 1.0, 1.0, 1.0]
    assert dfly._conf_probs[2].tolist() == [0.75] * 5

    # Replay 2 (non-greedy -> min_match=8 updated dynamically via GPU _recall_cfg WITHOUT recapturing!):
    # r1 only has a 6-gram match (ml=6 < 8) -> MUST NOT overwrite!
    # r1b ends in 6-gram [29..34] (CPU miss < 8), neural d0=35,d1=36 gives 8-gram [29..36] -> MUST overwrite d2..d4!
    r1b = _FakeReq("df1b", [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 94, 95, 29, 30, 31, 32, 33, 34])
    batch_ng = types.SimpleNamespace(
        reqs=[r0, r1, r1b],
        has_grammar=False,
        sampling_info=types.SimpleNamespace(is_all_greedy=False),
    )
    recall.prepare_dfly_step(dfly, batch_ng, gamma)
    block_ids.copy_(torch.tensor(
        [
            [17, 500, 501, 502, 503, 504],
            [24, 25, 510, 511, 512, 513],
            [34, 35, 36, 610, 611, 612],
        ],
        dtype=torch.int32,
        device=dev,
    ))
    pos_2d[0, 0] = len(r0.origin_input_ids) - 1
    pos_2d[1, 0] = len(r1.origin_input_ids) - 1
    pos_2d[2, 0] = len(r1b.origin_input_ids) - 1
    dfly._conf_probs.fill_(0.75)

    g.replay()
    torch.cuda.synchronize()
    assert block_ids[0, 1:].tolist() == [18, 19, 20, 21, 22], block_ids[0].tolist()
    assert block_ids[1, 1:].tolist() == [25, 510, 511, 512, 513], (
        f"Non-greedy min_match=8 inside CUDA graph failed to reject 6-gram: {block_ids[1].tolist()}"
    )
    assert block_ids[2, 1:].tolist() == [35, 36, 37, 38, 39], block_ids[2].tolist()
    assert dfly._conf_probs[2].tolist() == [0.75, 0.75, 1.0, 1.0, 1.0], dfly._conf_probs[2].tolist()

    # Replay 3: Verify exact real-count masking inside CUDA graph (never writes pad tokens when o + k >= real!)
    dfly._recall_plan[:bs].zero_()
    dfly._recall_gpu_pool.disable_batch(bs)
    dfly._recall_plan[0, 0] = 4  # real = 4
    dfly._recall_plan[0, 1] = 10  # L = 10
    dfly._recall_plan[0, 2:14] = torch.tensor([100, 101, 102, 103, 104, 104, 104, 104, 104, 104, 104, 104], device=dev)
    block_ids.copy_(torch.tensor(
        [
            [102, 880, 881, 882, 883, 884],
            [24, 25, 510, 511, 512, 513],
            [34, 35, 36, 610, 611, 612],
        ],
        dtype=torch.int32,
        device=dev,
    ))
    pos_2d[0, 0] = 11  # o = 11 - 10 + 1 = 2 -> only k=0,1 (103, 104) are real; k=2,3,4 stay [882, 883, 884]!
    dfly._conf_probs.fill_(0.75)
    g.replay()
    torch.cuda.synchronize()
    assert block_ids[0, 1:].tolist() == [103, 104, 882, 883, 884], (
        f"DFly CUDA graph leaked pad tokens: {block_ids[0].tolist()}"
    )
    assert dfly._conf_probs[0].tolist() == [1.0, 1.0, 0.75, 0.75, 0.75], dfly._conf_probs[0].tolist()
    print("[PASS] 5. DFly2 adapter inside real captured torch.cuda.CUDAGraph (dynamic greedy/non-greedy)")


def test_install_idempotency_and_cleanup():
    calls = []

    class _FakeReg:
        def register(self, target, fn, htype):
            calls.append((target, htype))

    class _FakeHookType:
        AFTER = "AFTER"
        AROUND = "AROUND"

    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL"] = "1"
    reg = _FakeReg()
    recall.install(reg, _FakeHookType)
    n1 = len(calls)
    assert n1 == 5, f"Expected 5 registered hooks (2 dynamic gamma + 3 recall), got {n1}: {calls}"
    # Second install on same registry is a no-op
    recall.install(reg, _FakeHookType)
    assert len(calls) == n1

    # Opt-out via XTIER_RECALL=0 registers only dynamic gamma hooks (no recall hooks)
    calls2 = []

    class _FakeReg2:
        def register(self, target, fn, htype):
            calls2.append((target, htype))

    os.environ["XTIER_RECALL"] = "0"
    os.environ["RECALL_ACCEL"] = "0"
    recall.install(_FakeReg2(), _FakeHookType)
    assert len(calls2) == 2, f"Expected only 2 dynamic gamma hooks with XTIER_RECALL=0, got {calls2}"
    os.environ["XTIER_RECALL"] = "1"
    os.environ["RECALL_ACCEL"] = "1"

    # after_step releases finished stream from both engine and gpu_pool
    eng = runtime._state["engine"]
    pool = runtime._state["gpu_pool"]
    if eng is not None and pool is not None:
        rfin = _FakeReq("df0", [10, 11, 12, 13], out=[14], finished=True)
        runtime.after_step(eng, rfin, 4, 5)
        assert "df0" not in pool._rid_to_slot
    print("[PASS] 6. Idempotent install(), XTIER_RECALL=0 opt-out, and resource release")


def main():
    try:
        test_cpu_suffix_and_propose_after()
        test_gpu_suffix_pool_and_kernels()
        test_dspark_adapter()
        test_eagle_adapter_and_truncate()
        test_dfly_adapter_inside_cuda_graph()
        test_install_idempotency_and_cleanup()
        print("ALL recall_universal_check TESTS PASSED")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
