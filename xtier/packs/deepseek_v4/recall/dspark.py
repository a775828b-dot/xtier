"""Recall copy drafts inside SGLang's DSpark spec-v2 worker (adapted for SGLang 084a5e9aae + XTier TP1/TPN).

Fixes & optimizations over stock Yee-4.1:
1. Updates `proposal.verify_ids[:, 1:1+gamma]` alongside `draft_block.draft_tokens` and `epilogue.draft_tokens_buf`
   (required for SGLang `SGLANG_DSPARK_FOLDED_PROPOSAL=1`, where target verify reads `proposal.verify_ids`).
2. Zero-GPU-overhead fast path on TP=1 (`world_size == 1`): when no request in the step has an active recall plan,
   skips GPU tensor allocation, NCCL broadcast, `device_merge`, and `onehot_rows` (0 kernel launches).
3. Signature-agnostic `DeepseekV4ForCausalLMDSpark.compute_confidence` dynamic `gamma` patch (`draft_tokens` or
   `sampled_tokens`) for `DSPARK_BLOCK_SIZE > 5`.
4. Uses XTier's `registry` hook system and auto-resolves tokenizer path from XTier / SGLang runtime context.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import time

import numpy as np

from . import tree as _tree
from .core import ENV_NEW, ENV_OLD, _env

logger = logging.getLogger(__name__)

O_MAX = 6
PAD = 7
MAX_BS = 64
RING = 8
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")

_state = {
    "engine": None,
    "engine_failed": False,
    "group": None,
    "pinned": None,
    "slot": 0,
    "worker": None,
    "paused": False,
    "pause_t": 0.0,
    "log_t": 0.0,
    "recall_hot": frozenset(),
}
_fifo: dict = {}
_counts = collections.Counter()


def _sync_xtier_env() -> None:
    """Map XTIER_RECALL* / YEE_RECALL* environment aliases into RECALL_ACCEL* if unset."""
    on = os.environ.get("XTIER_RECALL") or os.environ.get("YEE_RECALL")
    if on is not None and not os.environ.get("RECALL_ACCEL"):
        os.environ["RECALL_ACCEL"] = on
    mapping = {
        "XTIER_RECALL_DIR": "RECALL_ACCEL_DIR",
        "XTIER_RECALL_BACKEND": "RECALL_ACCEL_BACKEND",
        "XTIER_RECALL_GIB": "RECALL_ACCEL_GIB",
        "XTIER_RECALL_MIN_MATCH": "RECALL_ACCEL_MIN_MATCH",
        "XTIER_RECALL_PRIOR": "RECALL_ACCEL_PRIOR",
        "XTIER_RECALL_QHASH": "RECALL_ACCEL_QHASH",
        "XTIER_RECALL_NONGREEDY": "RECALL_ACCEL_NONGREEDY",
    }
    for src, dst in mapping.items():
        val = os.environ.get(src)
        if val is not None and not os.environ.get(dst):
            os.environ[dst] = val
    # Default to memory backend unless a disk dir was explicitly given, avoiding unexpected disk writes.
    if not os.environ.get("RECALL_ACCEL_BACKEND") and not os.environ.get("RECALL_ACCEL_DIR"):
        os.environ["RECALL_ACCEL_BACKEND"] = "memory"


def enabled() -> bool:
    _sync_xtier_env()
    return _env(ENV_NEW["on"], ENV_OLD["on"], "1").strip().lower() in _TRUE


def nongreedy_enabled() -> bool:
    return str(os.environ.get("RECALL_ACCEL_NONGREEDY", "1")).strip().lower() not in _FALSE


def budget_hook_enabled() -> bool:
    val = os.environ.get("XTIER_RECALL_BUDGET") or os.environ.get("YEE_RECALL_BUDGET", "1")
    return str(val).strip().lower() in _TRUE


def _omax(gamma: int) -> int:
    return int(gamma) + 1


def _pad(gamma: int) -> int:
    return int(gamma) + 2


def build_plan(engine, reqs, gamma: int, paused: bool = False):
    W = gamma + _pad(gamma)
    bs = len(reqs)
    plan = np.zeros((bs, 2 + W), dtype=np.int64)
    entries = []
    for i, req in enumerate(reqs):
        rid = req.rid
        origin = req.origin_input_ids
        out = req.output_ids
        n_out = len(out)
        L = len(origin) + n_out
        flag, toks, real = 0, None, 0
        if engine is not None and L > 0 and not req.finished():
            k = engine.stream_out_len(rid)
            if k > n_out:
                engine.end_stream(rid, finished=False)
                k = -1
            if k < 0:
                engine.begin_stream(rid, origin, cached_len=int(getattr(req, "cached_tokens", 0) or 0))
                k = 0
            if k < n_out:
                engine.extend_stream(rid, out[k:])
            if not paused:
                prop = _tree.propose(engine, rid, n=W - 1)
                if prop is not None:
                    plan[i, 0] = 1
                    plan[i, 1] = L
                    plan[i, 2] = int(out[-1]) if n_out else int(origin[-1])
                    plan[i, 3:2 + W] = prop.tokens[:W - 1]
                    flag, real = 1, int(prop.real)
                    toks = plan[i, 2:].tolist()
        entries.append((rid, flag, L, toks, real))
    return plan, entries


def device_merge(plan, seq_lens, anchors, draft_tokens, confidence, gamma: int,
                 epilogue_buf=None, verify_ids=None):
    import torch

    flag = plan[:, 0]
    L = plan[:, 1]
    Wt = plan[:, 2:]
    o = seq_lens.to(torch.int64) - L + 1
    om = _omax(gamma)
    ok = (flag > 0) & (o >= 0) & (o <= om)
    oc = o.clamp(0, om)
    ok = ok & (Wt.gather(1, oc.unsqueeze(1)).squeeze(1) == anchors.to(torch.int64))
    idx = oc.unsqueeze(1) + 1 + torch.arange(gamma, device=Wt.device, dtype=torch.int64).unsqueeze(0)
    ext = Wt.gather(1, idx).to(draft_tokens.dtype)
    merged = torch.where(ok.unsqueeze(1), ext, draft_tokens)
    draft_tokens.copy_(merged)
    bs = draft_tokens.shape[0]
    if epilogue_buf is not None:
        epilogue_buf[: bs * gamma].view(bs, gamma).copy_(merged)
    if verify_ids is not None and verify_ids.dim() == 2 and verify_ids.shape[1] >= 1 + gamma:
        verify_ids[:bs, 1 : 1 + gamma].copy_(merged.to(verify_ids.dtype))
    if confidence is not None:
        confidence.masked_fill_(ok.unsqueeze(1), 1.0)
    return ok


def onehot_rows(logits, ok, tokens) -> None:
    import torch

    if logits is None or logits.dim() != 3:
        return
    bs, g = int(tokens.shape[0]), int(tokens.shape[1])
    lg = logits[:bs, :g]
    tok = tokens.to(torch.int64).clamp_(0, int(lg.shape[2]) - 1).unsqueeze(2)
    keep = lg.gather(2, tok)
    vals = torch.where(ok.view(bs, 1, 1), torch.zeros_like(keep), keep)
    lg.masked_fill_(ok.view(bs, 1, 1), float("-inf"))
    lg.scatter_(2, tok, vals)


def account(engine, entry, Lb: int, La: int, token_at, gamma: int) -> str:
    rid, flag, L, W, real = entry
    if W is not None:
        gamma = (len(W) - 2) // 2
    commit = La - Lb
    if not flag:
        if commit > 0:
            engine.observe_own(rid, commit - 1)
        return "own"
    o = Lb - L
    if 0 <= o <= _omax(gamma) and Lb >= 1 and W[o] == int(token_at(Lb - 1)):
        avail = max(0, min(gamma, real - o))
        landed = min(commit - 1, avail) if commit > 0 else 0
        engine.feedback(rid, landed)
        return "used"
    engine.feedback(rid, 0, used=False)
    if commit > 0:
        engine.observe_own(rid, commit - 1)
    return "fallback"


def after_step(engine, req, Lb: int, gamma: int) -> None:
    rid = req.rid
    dq = _fifo.get(rid)
    entry = dq.popleft() if dq else None
    if getattr(req, "is_retracted", False):
        engine.drop_pending(rid)
        _fifo.pop(rid, None)
        return
    origin = req.origin_input_ids
    out = req.output_ids
    n0 = len(origin)
    La = n0 + len(out)
    if entry is not None:
        kind = account(engine, entry, Lb, La, lambda i: origin[i] if i < n0 else out[i - n0], gamma)
        _counts[kind] += 1
        if kind == "used":
            _counts["landed_rows"] += 1
    if req.finished():
        name = type(getattr(req, "finished_reason", None)).__name__.upper()
        engine.end_stream(rid, out, finished="ABORT" not in name)
        _fifo.pop(rid, None)


def _digest_from_dir(d: str):
    h = hashlib.sha256()
    vocab = 0
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        found = True
        with open(p, "rb") as f:
            data = f.read()
        h.update(name.encode())
        h.update(data)
        if name == "tokenizer.json":
            try:
                js = json.loads(data)
                vocab = len((js.get("model") or {}).get("vocab") or {}) + len(js.get("added_tokens") or [])
            except ValueError:
                pass
    return h.hexdigest(), vocab, found


def _resolve_tokenizer_dir() -> str:
    d = _env(ENV_NEW["tokenizer_dir"], ENV_OLD["tokenizer_dir"], "")
    if d:
        return d
    if os.environ.get("XTIER_ENGRAM_MODEL"):
        return os.environ["XTIER_ENGRAM_MODEL"]
    try:
        from sglang.srt.runtime_context import get_model
        return get_model().model_path
    except Exception:
        return "/data/fast/cold1/models/DeepSeek-V4.1-Flash-1017"


def _identity() -> dict:
    d = _resolve_tokenizer_dir()
    digest, vocab, found = _digest_from_dir(d)
    if not found:
        logger.warning("xtier recall: no tokenizer files under %s; using placeholder identity", d)
    return {
        "tokenizer_digest": digest or "deepseek-v4.1-placeholder",
        "vocab_size": vocab,
        "model": _env(ENV_NEW["model_name"], ENV_OLD["model_name"], "deepseek-v4.1-flash"),
    }


def _get_engine(gamma: int):
    eng = _state["engine"]
    if eng is not None or _state["engine_failed"]:
        return eng
    try:
        _sync_xtier_env()
        from .core import engine_from_env

        eng = engine_from_env(gamma, _identity())
        if eng is None:
            raise RuntimeError("Recall is not enabled in this process")
        _state["engine"] = eng
        st = eng.record.stats()
        logger.info("xtier recall: engine ready dir=%s backend=%s ids=%d state=%s gamma=%d",
                    st["dir"], st.get("backend", "disk"), st["ids"], st["state"], gamma)
    except Exception as e:
        _state["engine_failed"] = True
        logger.warning("xtier recall: engine init failed, copies stay off: %r", e)
    return _state["engine"]


def _check_pause(eng) -> bool:
    now = time.monotonic()
    if now - _state["pause_t"] >= 2.0:
        _state["pause_t"] = now
        d = getattr(eng.record, "dir", None)
        _state["paused"] = bool(d) and os.path.exists(os.path.join(d, "paused"))
    return _state["paused"]


def _maybe_log(eng) -> None:
    now = time.monotonic()
    if now - _state["log_t"] < 30.0:
        return
    _state["log_t"] = now
    st = eng.stats()
    logger.info(
        "xtier recall: steps=%d rows=%d plan_rows=%d used=%d fallback=%d own=%d landed=%d offers=%d probes=%d "
        "record_ids=%d streams=%d budget_hot=%d",
        _counts["steps"], _counts["rows"], _counts["plan_rows"], _counts["used"], _counts["fallback"],
        _counts["own"], st["landed_tokens"], st["offers"], st["probes"], st["ids"], st["active_streams"],
        _counts["budget_hot_rows"],
    )


def _pool_idx(req):
    idx = getattr(req, "req_pool_idx", None)
    if idx is None:
        kv = getattr(req, "kv", None)
        if kv is not None:
            idx = getattr(kv, "req_pool_idx", None)
    return idx


def _hot_rows(reqs, flags) -> frozenset:
    hot = []
    for req, flag in zip(reqs, flags):
        if not flag:
            continue
        idx = _pool_idx(req)
        if idx is not None:
            try:
                hot.append(int(idx))
            except (TypeError, ValueError):
                pass
    return frozenset(hot)


def recall_aware_confidence(confidence, req_pool_indices_cpu, hot):
    if not hot or confidence is None or getattr(confidence, "dim", lambda: 0)() != 2:
        return confidence, 0
    try:
        pools = [int(x) for x in req_pool_indices_cpu.tolist()]
    except Exception:
        return confidence, 0
    rows = [i for i, p in enumerate(pools[: int(confidence.shape[0])]) if p in hot]
    if not rows:
        return confidence, 0
    out = confidence.clone()
    out[rows, :] = 1.0
    return out, len(rows)


def _participates(kw, out) -> bool:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    batch = kw.get("batch")
    bs = kw.get("bs")
    if batch is None or bs is None or not (0 < int(bs) <= MAX_BS):
        return False
    if getattr(batch, "has_grammar", False):
        return False
    dblk = getattr(out, "draft_block", None)
    si = kw.get("sampling_info")
    if si is not None and not getattr(si, "is_all_greedy", False):
        if not nongreedy_enabled():
            return False
        cl = getattr(dblk, "corrected_logits", None)
        if cl is None or cl.dim() != 3 or int(cl.shape[0]) < int(bs):
            return False
    dt = getattr(dblk, "draft_tokens", None)
    return dt is not None and dt.dim() == 2 and dt.shape[0] == int(bs)


def _pinned(width: int):
    import torch

    buf = _state["pinned"]
    if buf is None or buf.shape[2] != width:
        buf = _state["pinned"] = torch.zeros((RING, MAX_BS, width), dtype=torch.int64, pin_memory=True)
    slot = _state["slot"]
    _state["slot"] = (slot + 1) % RING
    return buf[slot]


def _on_propose(worker, kw, out) -> None:
    import torch

    batch = kw["batch"]
    bs = int(kw["bs"])
    dt = out.draft_block.draft_tokens
    gamma = int(dt.shape[1])
    width = 2 + gamma + _pad(gamma)
    group = _state["group"]
    if group is None:
        from sglang.srt.distributed import get_tp_group
        group = _state["group"] = get_tp_group()
    world_size = int(getattr(group, "world_size", 1))
    rank0 = int(getattr(group, "rank_in_group", 0)) == 0

    arr = None
    any_plan = False
    if rank0:
        eng = _get_engine(gamma)
        entries = None
        hot = frozenset()
        try:
            if eng is not None:
                arr, entries = build_plan(eng, batch.reqs, gamma, paused=_check_pause(eng))
                if arr.shape == (bs, width) and arr[:, 0].any():
                    any_plan = True
                    _counts["plan_rows"] += int(arr[:, 0].sum())
                    hot = _hot_rows(batch.reqs, arr[:, 0].tolist())
        except Exception as e:
            _counts["plan_errors"] += 1
            if _counts["plan_errors"] <= 3:
                logger.warning("xtier recall: plan error (zero plan this step): %r", e)
            entries = None
            hot = frozenset()
            any_plan = False
        _state["recall_hot"] = hot
        if eng is not None:
            if entries is None:
                entries = [(r.rid, 0, 0, None, 0) for r in batch.reqs]
            for e in entries:
                dq = _fifo.get(e[0])
                if dq is None:
                    dq = _fifo[e[0]] = collections.deque()
                dq.append(e)
        _counts["steps"] += 1
        _counts["rows"] += bs

    # TP=1 fast path: when no request has a recall plan, skip all GPU allocations & kernels!
    if world_size == 1 and not any_plan:
        return

    plan = torch.zeros((bs, width), dtype=torch.int64, device=dt.device)
    if rank0 and any_plan and arr is not None:
        host = _pinned(width)
        host[:bs].copy_(torch.from_numpy(arr))
        plan.copy_(host[:bs], non_blocking=True)
    if world_size > 1:
        group.broadcast(plan, src=0)

    epi = None
    if worker is not None:
        epi = getattr(worker, "_verify_epilogue", None)
        if epi is None:
            vex = getattr(worker, "_verify_executor", None)
            epi = getattr(vex, "verify_epilogue", None) if vex is not None else None
    ok = device_merge(
        plan, batch.seq_lens, out.draft_block_ids[:, 0], dt, out.confidence, gamma,
        epilogue_buf=getattr(epi, "draft_tokens_buf", None),
        verify_ids=getattr(out, "verify_ids", None),
    )
    si = kw.get("sampling_info")
    if si is not None and not getattr(si, "is_all_greedy", False):
        onehot_rows(getattr(out.draft_block, "corrected_logits", None), ok, dt)


def _make_propose(worker, inner):
    def propose(*args, **kw):
        out = inner(*args, **kw)
        if enabled() and _participates(kw, out):
            _on_propose(worker, kw, out)
        return out

    propose.__recall_wrapped__ = True
    return propose


def _after_worker_init(result, self, *args, **kwargs):
    prop = getattr(self, "_proposer", None)
    if prop is None:
        return result
    _state["worker"] = self
    if not getattr(prop.propose, "__recall_wrapped__", False):
        try:
            prop.propose = _make_propose(self, prop.propose)
        except (AttributeError, TypeError):
            pcls = type(prop)
            if not getattr(pcls.propose, "__recall_wrapped__", False):
                inner = pcls.propose

                def propose(self_, *a2, **kw2):
                    out = inner(self_, *a2, **kw2)
                    if enabled() and _participates(kw2, out):
                        _on_propose(_state["worker"], kw2, out)
                    return out

                propose.__recall_wrapped__ = True
                pcls.propose = propose
    return result


def _around_process_decode(original, self, batch, result, *args, **kwargs):
    eng = _state["engine"]
    if eng is None or batch.spec_algorithm.is_none():
        return original(self, batch, result, *args, **kwargs)
    reqs = list(batch.reqs)
    before = [len(r.origin_input_ids) + len(r.output_ids) for r in reqs]
    ret = original(self, batch, result, *args, **kwargs)
    try:
        for req, Lb in zip(reqs, before):
            after_step(eng, req, Lb, eng.gamma)
        eng.tick()
        _maybe_log(eng)
    except Exception as e:
        _counts["account_errors"] += 1
        if _counts["account_errors"] <= 3:
            logger.warning("xtier recall: accounting error: %r", e)
    return ret


def _around_compute_budget(original, self, *args, **kwargs):
    _counts["budget_calls"] += 1
    hot = _state["recall_hot"]
    if hot and budget_hook_enabled() and not args and "confidence" in kwargs and "req_pool_indices_cpu" in kwargs:
        conf, n = recall_aware_confidence(kwargs["confidence"], kwargs["req_pool_indices_cpu"], hot)
        if n:
            _counts["budget_hot_rows"] += n
            kwargs = dict(kwargs, confidence=conf)
    return original(self, *args, **kwargs)


def _around_compute_confidence(original, self, *args, **kwargs):
    tok = kwargs.get("draft_tokens")
    if tok is None:
        tok = kwargs.get("sampled_tokens")
    if tok is not None and getattr(tok, "dim", lambda: 0)() == 2:
        g = int(tok.shape[1])
        if g != self.gamma:
            logger.info("xtier dspark: compute_confidence gamma %d -> %d", self.gamma, g)
            self.gamma = g
            ch = getattr(self, "confidence_head", None)
            if ch is not None and hasattr(ch, "gamma"):
                ch.gamma = g
    return original(self, *args, **kwargs)


def _after_dspark_model_init(result, self, *args, **kwargs):
    try:
        from sglang.srt.runtime_context import get_spec
        ndt = getattr(get_spec(), "speculative_num_draft_tokens", None)
        if ndt is not None and int(ndt) >= 2:
            g = int(ndt) - 1
            if g != getattr(self, "gamma", g):
                logger.info("xtier dspark: draft model init gamma %s -> %d", getattr(self, "gamma", None), g)
                self.gamma = g
                self.block_size = g
                ch = getattr(self, "confidence_head", None)
                if ch is not None and hasattr(ch, "gamma"):
                    ch.gamma = g
    except Exception:
        pass
    return result


def install(registry, hook_type) -> None:
    # Dynamic gamma support for DSPARK_BLOCK_SIZE != 5 is always safe and lossless:
    if os.environ.get("XTIER_DSPARK_DYNAMIC_GAMMA", "1") != "0":
        registry.register(
            "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.__init__",
            _after_dspark_model_init,
            hook_type.AFTER,
        )
        registry.register(
            "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark.compute_confidence",
            _around_compute_confidence,
            hook_type.AROUND,
        )
    if not enabled():
        return
    registry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.DSparkWorkerV2.__init__",
        _after_worker_init,
        hook_type.AFTER,
    )
    registry.register(
        "sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_decode",
        _around_process_decode,
        hook_type.AROUND,
    )
    registry.register(
        "sglang.srt.speculative.dspark_components.dspark_planner.HostConfidenceBudgetPlanner.compute_budget",
        _around_compute_budget,
        hook_type.AROUND,
    )
    logger.info("xtier recall: hooks registered (backend=%s, budget=%s)",
                os.environ.get("RECALL_ACCEL_BACKEND", "memory"), budget_hook_enabled())
