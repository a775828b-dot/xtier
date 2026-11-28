"""Universal lossless Recall + GPU propose_after runtime across DSpark, EAGLE/MTP, and DFly2.

Adapters:
  1. DSparkWorkerV2 (DeepSeek V4.1, Inkling-2-Small, MiniMax-M3.1): hooks ``_proposer.propose``,
     merges CPU Recall anchor plan + zero-sync GPU ``propose_after`` tail continuation, updates
     ``draft_tokens`` / ``epilogue.draft_tokens_buf`` / ``proposal.verify_ids`` / ``confidence``,
     and one-hots ONLY overwritten ``corrected_logits`` rows (preserving neural prefix logits).
  2. EAGLEWorkerV2 (Qwen3.8 MTP, GLM-5.3 EAGLE, fixed or adaptive steps, topk == 1): invoked from
     ``xtier.packs.speculative._verify`` before confidence truncation; merges CPU Recall anchor plan +
     GPU ``propose_after`` into ``verify_input.draft_token`` and one-hots ONLY overwritten
     ``verify_input.draft_probs`` rows (100% lossless for greedy, target-only, and rejection sampling).
  3. DFlyWorkerV2 / HYV4FlashDFly (Hunyuan-4-Flash 5-layer DFly2): prepares static GPU plan + suffix
     metadata before target verify and executes a single CUDA-graph-capturable merge + ``propose_after``
     kernel inside ``HYV4FlashDFly.draft_into_verify()``.

All three share ONE ``RecallEngine``, ONE ``GpuSuffixPool``, and ONE ``process_batch_result_decode`` hook.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import time
from typing import Optional

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
    "gpu_pool": None,
    "gpu_pool_failed": False,
}
_fifo: dict = {}
_counts = collections.Counter()
_installed_registries: set[int] = set()


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
    if not os.environ.get("RECALL_ACCEL_BACKEND") and not os.environ.get("RECALL_ACCEL_DIR"):
        os.environ["RECALL_ACCEL_BACKEND"] = "memory"


def enabled() -> bool:
    _sync_xtier_env()
    return _env(ENV_NEW["on"], ENV_OLD["on"], "1").strip().lower() in _TRUE


def nongreedy_enabled() -> bool:
    return str(os.environ.get("RECALL_ACCEL_NONGREEDY", "1")).strip().lower() not in _FALSE


def chain_enabled() -> bool:
    return str(os.environ.get("XTIER_RECALL_CHAIN", "1")).strip().lower() not in _FALSE


def chain_min_match(*, greedy: bool = True) -> int:
    raw = os.environ.get("XTIER_RECALL_SUFFIX_MIN")
    base = 6
    if raw is not None and str(raw).strip():
        try:
            v = int(str(raw).strip())
            if v <= 0:
                return 0
            base = max(6, v)
        except ValueError:
            base = 6
    return base if greedy else max(base, 8)


def chain_conf_min() -> float:
    raw = os.environ.get("XTIER_RECALL_CHAIN_CONF", "0.35")
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.35


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
                    cont = np.asarray(prop.tokens[:W - 1], dtype=np.int64).copy()
                    real = int(min(prop.real, W - 1))
                    # Multi-hop CPU propose_after when a primary match ends before filling the plan window
                    if 0 < real < W - 1 and chain_enabled() and hasattr(engine, "propose_after"):
                        ext = engine.propose_after(rid, cont[:real], n=(W - 1) - real)
                        if ext is not None and ext.real > 0:
                            take = min(int(ext.real), (W - 1) - real)
                            cont[real:real + take] = ext.tokens[:take]
                            real += take
                            if real < W - 1:
                                cont[real:] = cont[real - 1]
                    plan[i, 0] = max(1, real)
                    plan[i, 1] = L
                    plan[i, 2] = int(out[-1]) if n_out else int(origin[-1])
                    plan[i, 3:2 + W] = cont[:W - 1]
                    flag = 1
                    toks = plan[i, 2:].tolist()
        entries.append((rid, flag, L, toks, real))
    return plan, entries


def device_merge(
    plan,
    seq_lens,
    anchors,
    draft_tokens,
    confidence,
    gamma: int,
    epilogue_buf=None,
    verify_ids=None,
):
    """Merge CPU Recall plan into draft_tokens, overwriting ONLY slots backed by real continuation tokens.

    ``plan[:, 0]`` carries ``real`` (count of valid continuation tokens after ``plan[:, 2]``).
    Slots with ``o + k >= real`` preserve their neural draft tokens and logits (zero pad garbage).
    Returns 2D bool mask ``ok_pos`` of shape ``(bs, gamma)``.
    """
    import torch

    real_cnt = plan[:, 0]
    L = plan[:, 1]
    Wt = plan[:, 2:]
    o = seq_lens.to(torch.int64) - L + 1
    om = _omax(gamma)
    ok_row = (real_cnt > 0) & (o >= 0) & (o <= om)
    oc = o.clamp(0, om)
    ok_row = ok_row & (Wt.gather(1, oc.unsqueeze(1)).squeeze(1) == anchors.to(torch.int64))
    k_idx = torch.arange(gamma, device=Wt.device, dtype=torch.int64).unsqueeze(0)
    ok_pos = ok_row.unsqueeze(1) & ((oc.unsqueeze(1) + k_idx) < real_cnt.unsqueeze(1))
    idx = oc.unsqueeze(1) + 1 + k_idx
    ext = Wt.gather(1, idx).to(draft_tokens.dtype)
    merged = torch.where(ok_pos, ext, draft_tokens)
    draft_tokens.copy_(merged)
    bs = draft_tokens.shape[0]
    if epilogue_buf is not None:
        epilogue_buf[: bs * gamma].view(bs, gamma).copy_(merged)
    if verify_ids is not None and verify_ids.dim() == 2 and verify_ids.shape[1] >= 1 + gamma:
        verify_ids[:bs, 1 : 1 + gamma].copy_(merged.to(verify_ids.dtype))
    if confidence is not None:
        confidence.masked_fill_(ok_pos, 1.0)
    return ok_pos


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
        pool = _state["gpu_pool"]
        if pool is not None:
            pool.release(rid)
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
        pool = _state["gpu_pool"]
        if pool is not None:
            pool.release(rid)


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
        return ""


def _identity() -> dict:
    d = _resolve_tokenizer_dir()
    digest, vocab, found = _digest_from_dir(d) if d else ("", 0, False)
    if not found and d:
        logger.warning("xtier recall: no tokenizer files under %s; using placeholder identity", d)
    default_model = os.path.basename(d.rstrip("/")) if d else "xtier-generic"
    return {
        "tokenizer_digest": digest or f"{default_model}-placeholder",
        "vocab_size": vocab,
        "model": _env(ENV_NEW["model_name"], ENV_OLD["model_name"], default_model),
    }


def _get_engine(gamma: int):
    eng = _state["engine"]
    if eng is not None or _state["engine_failed"]:
        return eng
    try:
        _sync_xtier_env()
        from .core import engine_from_env

        eng = engine_from_env(max(1, int(gamma)), _identity())
        if eng is None:
            raise RuntimeError("Recall is not enabled in this process")
        _state["engine"] = eng
        st = eng.record.stats()
        logger.info(
            "xtier recall: engine ready dir=%s backend=%s ids=%d state=%s gamma=%d suffix_min=%d chain=%s",
            st["dir"], st.get("backend", "disk"), st["ids"], st["state"], eng.gamma,
            getattr(eng, "suffix_min", 0), chain_enabled(),
        )
    except Exception as e:
        _state["engine_failed"] = True
        logger.warning("xtier recall: engine init failed, copies stay off: %r", e)
    return _state["engine"]


def _get_gpu_pool(device):
    pool = _state["gpu_pool"]
    if pool is not None or _state["gpu_pool_failed"] or not chain_enabled():
        return pool
    try:
        import torch
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            return None
        from .gpu_suffix import GpuSuffixPool

        pool = GpuSuffixPool(device)
        _state["gpu_pool"] = pool
        logger.info("xtier recall: GPU suffix pool ready (slots=%d, ~%.1f MiB)",
                    pool.max_slots, pool.max_slots * 0.95)
    except Exception as e:
        _state["gpu_pool_failed"] = True
        logger.warning("xtier recall: GPU suffix pool init skipped: %r", e)
    return _state["gpu_pool"]


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
        "record_ids=%d streams=%d budget_hot=%d eagle_steps=%d dfly_steps=%d",
        _counts["steps"], _counts["rows"], _counts["plan_rows"], _counts["used"], _counts["fallback"],
        _counts["own"], st["landed_tokens"], st["offers"], st["probes"], st["ids"], st["active_streams"],
        _counts["budget_hot_rows"], _counts["eagle_steps"], _counts["dfly_steps"],
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


def _pinned(width: int):
    import torch

    buf = _state["pinned"]
    if buf is None or buf.shape[2] < width:
        alloc_w = max(width, 32)
        buf = _state["pinned"] = torch.zeros((RING, MAX_BS, alloc_w), dtype=torch.int64, pin_memory=True)
    slot = _state["slot"]
    _state["slot"] = (slot + 1) % RING
    return buf[slot, :, :width]


def _tp_info():
    group = _state["group"]
    if group is None:
        try:
            from sglang.srt.distributed import get_tp_group
            group = _state["group"] = get_tp_group()
        except Exception:
            return None, 1, True
    world_size = int(getattr(group, "world_size", 1))
    rank0 = int(getattr(group, "rank_in_group", 0)) == 0
    return group, world_size, rank0


def _plan_on_rank0(batch, bs: int, gamma: int):
    width = 2 + gamma + _pad(gamma)
    eng = _get_engine(gamma)
    arr = None
    entries = None
    hot = frozenset()
    any_plan = False
    try:
        if eng is not None:
            arr, entries = build_plan(eng, batch.reqs, gamma, paused=_check_pause(eng))
            if arr.shape == (bs, width) and (arr[:, 0] > 0).any():
                any_plan = True
                _counts["plan_rows"] += int((arr[:, 0] > 0).sum())
                hot = _hot_rows(batch.reqs, (arr[:, 0] > 0).tolist())
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
    return arr, any_plan, width


# ==================== Adapter 1: DSparkWorkerV2 (DS4.1, Inkling-2, MiniMax) ====================

def _participates_dspark(kw, out) -> bool:
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


def _on_dspark_propose(worker, kw, out) -> None:
    import torch
    from .gpu_suffix import onehot_masked

    batch = kw["batch"]
    bs = int(kw["bs"])
    dt = out.draft_block.draft_tokens
    gamma = int(dt.shape[1])
    width = 2 + gamma + _pad(gamma)
    group, world_size, rank0 = _tp_info()

    arr = None
    any_plan = False
    if rank0:
        arr, any_plan, width = _plan_on_rank0(batch, bs, gamma)

    si = kw.get("sampling_info")
    is_greedy = si is None or bool(getattr(si, "is_all_greedy", True))
    mm = chain_min_match(greedy=is_greedy)
    gpu_pool = _get_gpu_pool(dt.device) if (mm > 0 and chain_enabled()) else None
    any_gpu_chain = False
    if rank0 and gpu_pool is not None:
        any_gpu_chain = gpu_pool.sync_batch(batch.reqs)

    # TP=1 fast path: skip all GPU allocations & kernels when neither CPU plan nor GPU suffix is active.
    if world_size == 1 and not any_plan and not any_gpu_chain:
        return

    epi = None
    if worker is not None:
        epi = getattr(worker, "_verify_epilogue", None)
        if epi is None:
            vex = getattr(worker, "_verify_executor", None)
            epi = getattr(vex, "verify_epilogue", None) if vex is not None else None
    epilogue_buf = getattr(epi, "draft_tokens_buf", None)
    verify_ids = getattr(out, "verify_ids", None)

    ok_pos = torch.zeros((bs, gamma), dtype=torch.bool, device=dt.device)
    if any_plan or world_size > 1:
        plan = torch.zeros((bs, width), dtype=torch.int64, device=dt.device)
        if rank0 and any_plan and arr is not None:
            host = _pinned(width)
            host[:bs].copy_(torch.from_numpy(arr))
            plan.copy_(host[:bs], non_blocking=True)
        if world_size > 1:
            group.broadcast(plan, src=0)
        ok_pos = device_merge(
            plan,
            batch.seq_lens,
            out.draft_block_ids[:, 0],
            dt,
            out.confidence,
            gamma,
            epilogue_buf=epilogue_buf,
            verify_ids=verify_ids,
        )

    overwrite = ok_pos.to(torch.uint8).contiguous()
    if gpu_pool is not None and (any_gpu_chain or world_size > 1):
        if rank0 and any_gpu_chain:
            gpu_pool.propose_after(
                bs=bs,
                gamma=gamma,
                seq_lens=batch.seq_lens,
                anchors=out.draft_block_ids[:, 0],
                draft_tokens=dt,
                skip_mask=ok_pos.all(dim=1).to(torch.int32),
                overwrite=overwrite,
                confidence=out.confidence,
                min_match=mm,
                min_conf=chain_conf_min(),
            )
        if world_size > 1:
            group.broadcast(dt, src=0)
            group.broadcast(overwrite, src=0)
            if out.confidence is not None:
                group.broadcast(out.confidence, src=0)
        if epilogue_buf is not None:
            epilogue_buf[: bs * gamma].view(bs, gamma).copy_(dt)
        if verify_ids is not None and verify_ids.dim() == 2 and verify_ids.shape[1] >= 1 + gamma:
            verify_ids[:bs, 1 : 1 + gamma].copy_(dt.to(verify_ids.dtype))

    if not is_greedy:
        onehot_masked(getattr(out.draft_block, "corrected_logits", None), overwrite, dt, is_probs=False)


def _make_dspark_propose(worker, inner):
    def propose(*args, **kw):
        out = inner(*args, **kw)
        if enabled() and _participates_dspark(kw, out):
            _on_dspark_propose(worker, kw, out)
        return out

    propose.__recall_wrapped__ = True
    return propose


def _after_dspark_worker_init(result, self, *args, **kwargs):
    prop = getattr(self, "_proposer", None)
    if prop is None:
        return result
    _state["worker"] = self
    if not getattr(prop.propose, "__recall_wrapped__", False):
        try:
            prop.propose = _make_dspark_propose(self, prop.propose)
        except (AttributeError, TypeError):
            pcls = type(prop)
            if not getattr(pcls.propose, "__recall_wrapped__", False):
                inner = pcls.propose

                def propose(self_, *a2, **kw2):
                    out = inner(self_, *a2, **kw2)
                    if enabled() and _participates_dspark(kw2, out):
                        _on_dspark_propose(_state["worker"], kw2, out)
                    return out

                propose.__recall_wrapped__ = True
                pcls.propose = propose
    return result


# ==================== Adapter 2: EAGLEWorkerV2 (Qwen3.8 MTP, GLM-5.3 EAGLE) ====================

def on_eagle_verify(worker, batch) -> bool:
    """Apply Recall + GPU propose_after to EagleVerifyInput before target verify & truncate.py.

    Supports both fixed steps (Qwen3.8 MTP, steps=3) and adaptive steps (GLM-5.3 EAGLE, steps=1..5),
    greedy, target-only sampling (draft_probs=None), and full rejection sampling (draft_probs!=None).
    Returns True when Recall participated on this batch.
    """
    import torch
    from .gpu_suffix import onehot_masked

    if not enabled() or batch is None:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    if batch.forward_mode.is_idle() or getattr(batch, "has_grammar", False):
        return False
    vi = getattr(batch, "spec_info", None)
    if vi is None or int(getattr(vi, "topk", 0) or 0) != 1:
        return False
    reqs = getattr(batch, "reqs", None)
    if not reqs or not (0 < len(reqs) <= MAX_BS):
        return False
    bs = len(reqs)
    draft_token = getattr(vi, "draft_token", None)
    width = int(getattr(vi, "draft_token_num", 0) or 0)
    if draft_token is None or width < 2 or int(draft_token.numel()) != bs * width:
        return False
    steps = width - 1

    si = getattr(batch, "sampling_info", None)
    is_greedy = si is None or bool(getattr(si, "is_all_greedy", True))
    draft_probs = getattr(vi, "draft_probs", None)
    if not is_greedy:
        if not nongreedy_enabled():
            return False
        if draft_probs is not None and (
            draft_probs.dim() != 3 or int(draft_probs.shape[0]) < bs or int(draft_probs.shape[1]) < steps
        ):
            return False

    group, world_size, rank0 = _tp_info()
    arr = None
    any_plan = False
    width_plan = 2 + steps + _pad(steps)
    if rank0:
        arr, any_plan, width_plan = _plan_on_rank0(batch, bs, steps)
        _counts["eagle_steps"] += 1

    mm = chain_min_match(greedy=is_greedy)
    gpu_pool = _get_gpu_pool(draft_token.device) if (mm > 0 and chain_enabled()) else None
    any_gpu_chain = False
    if rank0 and gpu_pool is not None:
        any_gpu_chain = gpu_pool.sync_batch(reqs)

    if world_size == 1 and not any_plan and not any_gpu_chain:
        return False

    dt_2d = draft_token.view(bs, width)
    anchors = dt_2d[:, 0]
    drafts = dt_2d[:, 1:]  # stride(0) == width, stride(1) == 1

    ok_pos = torch.zeros((bs, steps), dtype=torch.bool, device=draft_token.device)
    if any_plan or world_size > 1:
        plan = torch.zeros((bs, width_plan), dtype=torch.int64, device=draft_token.device)
        if rank0 and any_plan and arr is not None:
            host = _pinned(width_plan)
            host[:bs].copy_(torch.from_numpy(arr))
            plan.copy_(host[:bs], non_blocking=True)
        if world_size > 1:
            group.broadcast(plan, src=0)
        ok_pos = device_merge(
            plan,
            batch.seq_lens,
            anchors,
            drafts,
            None,
            steps,
        )

    overwrite = ok_pos.to(torch.uint8).contiguous()
    if gpu_pool is not None and (any_gpu_chain or world_size > 1):
        if rank0 and any_gpu_chain:
            gpu_pool.propose_after(
                bs=bs,
                gamma=steps,
                seq_lens=batch.seq_lens,
                anchors=anchors,
                draft_tokens=drafts,
                skip_mask=ok_pos.all(dim=1).to(torch.int32),
                overwrite=overwrite,
                confidence=None,
                min_match=mm,
                min_conf=0.0,
            )
        if world_size > 1:
            group.broadcast(dt_2d, src=0)
            group.broadcast(overwrite, src=0)

    if draft_probs is not None:
        if draft_probs.stride(-1) != 1:
            draft_probs = vi.draft_probs = draft_probs.contiguous()
        onehot_masked(draft_probs, overwrite, drafts, is_probs=True)
    return True


# ==================== Adapter 3: DFlyWorkerV2 / HYV4FlashDFly (Hy4-Flash) ====================

def bind_dfly_buffers(dfly) -> None:
    """Allocate fixed-address GPU Recall plan & suffix pool handles on HYV4FlashDFly for CUDA-graph safety."""
    import torch

    if not enabled():
        dfly._recall_plan = None
        dfly._recall_gpu_pool = None
        dfly._recall_plan_dirty = False
        return
    dev = dfly.candidates_out.device
    gamma = max(1, int(dfly.dfly_config.block_size) - 1)
    plan_w = 2 + gamma + _pad(gamma)
    dfly._recall_gamma = gamma
    dfly._recall_plan_width = plan_w
    dfly._recall_plan = torch.zeros((MAX_BS, plan_w), dtype=torch.int64, device=dev)
    dfly._recall_plan_dirty = False
    dfly._recall_min_match = chain_min_match(greedy=True)
    dfly._recall_cfg = torch.tensor([max(6, dfly._recall_min_match)], dtype=torch.int32, device=dev)
    dfly._recall_cfg_host = torch.full((RING, 1), max(6, dfly._recall_min_match), dtype=torch.int32, pin_memory=True)
    dfly._recall_cfg_slot = 0
    dfly._recall_gpu_pool = _get_gpu_pool(dev) if (chain_enabled() and dfly._recall_min_match > 0) else None


def prepare_dfly_step(dfly, batch, gamma: int) -> None:
    """Host-side preparation in DFlyWorkerV2._forward_decode before target verify / CUDA graph replay."""
    import torch

    plan_buf = getattr(dfly, "_recall_plan", None)
    if plan_buf is None:
        return
    gpu_pool = getattr(dfly, "_recall_gpu_pool", None)
    reqs = getattr(batch, "reqs", None)
    bs = len(reqs) if reqs else 0
    si = getattr(batch, "sampling_info", None)
    is_greedy = si is None or bool(getattr(si, "is_all_greedy", True))

    if (
        not enabled()
        or bs <= 0
        or bs > MAX_BS
        or getattr(batch, "has_grammar", False)
        or (not is_greedy and not nongreedy_enabled())
    ):
        if getattr(dfly, "_recall_plan_dirty", False) and bs > 0:
            plan_buf[:bs, 0].zero_()
            dfly._recall_plan_dirty = False
        if gpu_pool is not None and bs > 0:
            gpu_pool.disable_batch(bs)
        return

    group, world_size, rank0 = _tp_info()
    arr = None
    any_plan = False
    width_plan = int(dfly._recall_plan_width)
    if rank0:
        arr, any_plan, width_plan = _plan_on_rank0(batch, bs, gamma)
        _counts["dfly_steps"] += 1

    if any_plan:
        if rank0 and arr is not None:
            host = _pinned(width_plan)
            host[:bs].copy_(torch.from_numpy(arr))
            plan_buf[:bs, :width_plan].copy_(host[:bs], non_blocking=True)
        if world_size > 1:
            group.broadcast(plan_buf[:bs, :width_plan], src=0)
        dfly._recall_plan_dirty = True
    elif world_size > 1:
        plan_buf[:bs, 0].zero_()
        group.broadcast(plan_buf[:bs, :width_plan], src=0)
        dfly._recall_plan_dirty = True
    elif getattr(dfly, "_recall_plan_dirty", False):
        plan_buf[:bs, 0].zero_()
        dfly._recall_plan_dirty = False

    if gpu_pool is not None:
        mm = chain_min_match(greedy=is_greedy)
        dfly._recall_min_match = mm
        if mm > 0 and world_size == 1:
            cfg = getattr(dfly, "_recall_cfg", None)
            cfg_host = getattr(dfly, "_recall_cfg_host", None)
            if cfg is not None and cfg_host is not None:
                cslot = getattr(dfly, "_recall_cfg_slot", 0)
                dfly._recall_cfg_slot = (cslot + 1) % RING
                cfg_host[cslot, 0] = max(6, mm)
                cfg.copy_(cfg_host[cslot], non_blocking=True)
            if not gpu_pool.sync_batch(reqs):
                gpu_pool.disable_batch(bs)
        else:
            gpu_pool.disable_batch(bs)


def merge_dfly_graph(dfly, block_ids, positions, bs: int, width: int) -> None:
    """Single CUDA-graph-capturable kernel inside HYV4FlashDFly.draft_into_verify."""
    plan_buf = getattr(dfly, "_recall_plan", None)
    if plan_buf is None or bs <= 0 or width < 2:
        return
    from .gpu_suffix import (
        HIST_CAP,
        HT_CAP,
        HT_MASK,
        PROBE_LIMIT,
        _dfly_recall_merge_kernel,
    )

    gpu_pool = getattr(dfly, "_recall_gpu_pool", None)
    has_chain = gpu_pool is not None
    conf = getattr(dfly, "_conf_probs", None)
    has_conf = conf is not None
    cfg = getattr(dfly, "_recall_cfg", None)
    dummy = plan_buf
    _dfly_recall_merge_kernel[(bs,)](
        block_ids,
        positions,
        plan_buf,
        conf if has_conf else dummy,
        cfg if cfg is not None else dummy,
        gpu_pool.slot_ids if has_chain else dummy,
        gpu_pool.hist_lens if has_chain else dummy,
        gpu_pool.base_lens if has_chain else dummy,
        gpu_pool.hist if has_chain else dummy,
        gpu_pool.ht_keys if has_chain else dummy,
        gpu_pool.ht_pos if has_chain else dummy,
        int(bs),
        int(width),
        int(plan_buf.stride(0)),
        int(conf.stride(0)) if has_conf else 1,
        float(chain_conf_min()),
        HIST_CAP_C=HIST_CAP,
        HT_CAP_C=HT_CAP,
        HT_MASK_C=HT_MASK,
        PROBE_C=PROBE_LIMIT,
        HAS_CONF=has_conf,
        ENABLE_CHAIN=has_chain,
        num_warps=1,
    )


# ==================== Shared Hooks & Dynamic Gamma ====================

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
    """Idempotently register universal Recall & dynamic-gamma hooks on ``registry``."""
    reg_id = id(registry)
    if reg_id in _installed_registries:
        return
    _installed_registries.add(reg_id)

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
        _after_dspark_worker_init,
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
    logger.info(
        "xtier recall: universal hooks registered (backend=%s, chain=%s, budget=%s)",
        os.environ.get("RECALL_ACCEL_BACKEND", "memory"),
        chain_enabled(),
        budget_hook_enabled(),
    )
