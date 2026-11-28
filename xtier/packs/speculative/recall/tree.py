"""Multi-branch tree drafts (Yee 2.3.0, P4).

A round verifies several candidate continuations at once: the head's own chain (or its own top-k tree) and the
record's recalled copies. Every candidate is merged into one token trie under the verified token ``t1`` (the root);
the target scores every node in one forward with an ancestor mask, and the walk keeps the target's own picks along
the branches that agree with them.

Lossless: with point-mass drafts (recalled copies, a greedy head) the walk draws one target sample a node and follows
the child whose token equals it, which is exactly serial sampling. With a head that sampled its children from ``q``
(EAGLE top-k sampling, rejection-sampling mode) :func:`verify_specinfer` accepts child ``c`` with ``min(1, p/q)`` and
moves the residual ``norm(max(p - q, 0))`` to the next sibling (SpecInfer's multi-round speculative sampling).

Bases that verify only a chain take :func:`choose_chain` instead: the candidate chain (the head's own included) that
promises the most tokens fills the chain's slots, and with the head's chain alone nothing changes.

Pure numpy; the engine adapters (``yee.integrations.*``) convert to and from their own layouts
(:func:`to_sglang` / :func:`from_sglang` for SGLang's EAGLE tree, :func:`ngram_insert` for its n-gram trie).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# The verify ladder (ms by rows, linear between its points), as mlx-serve's REL_COST: relative to one row.
_LADDER = ((1, 17.7), (2, 25.5), (3, 29.6), (4, 33.6), (5, 36.8), (6, 42.4), (7, 44.8), (8, 47.9), (9, 51.1),
           (12, 60.5), (16, 69.1))
DEFAULT_STATE_COST = 0.08
DEFAULT_K = 2
MIN_COPY = 2

# --------------------------------------------------------------------------------------------- environment

_TRUE = ("1", "true", "yes", "on")


def env_value(name: str, default: str = "", environ=None) -> str:
    """``YEE_<name>`` first, then the recall aliases ``RECALL_ACCEL_<name>``, ``SGLANG_D_RECALL_<name>``,
    ``VLLM_D_RECALL_<name>`` (the same order as every other Yee switch). Blank counts as unset."""
    env = os.environ if environ is None else environ
    for prefix in ("YEE_", "RECALL_ACCEL_", "SGLANG_D_RECALL_", "VLLM_D_RECALL_"):
        v = env.get(prefix + name)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def tree_enabled(environ=None) -> bool:
    """YEE_TREE (default 1): candidates merge into the head's tree where it has one, else the best chain is picked."""
    return env_value("TREE", "1", environ).lower() in _TRUE


def tree_k(environ=None) -> int:
    """YEE_TREE_K (default 2, 0..3): recalled branches a round may add."""
    try:
        return max(0, min(3, int(float(env_value("TREE_K", str(DEFAULT_K), environ)))))
    except ValueError:
        return DEFAULT_K


def tree_nodes(default: int, environ=None) -> int:
    """YEE_TREE_NODES (default: the head's own num_draft_tokens): nodes a merged tree keeps, the root included."""
    raw = env_value("TREE_NODES", "", environ)
    try:
        return max(1, int(raw)) if raw else int(default)
    except ValueError:
        return int(default)


# -------------------------------------------------------------------------------------------------- costs

def rel_cost(rows: int) -> float:
    """Verify cost of ``rows`` rows relative to one row; past 16 the last segment's slope continues."""
    r = max(1, int(rows))
    base = _LADDER[0][1]
    if r >= _LADDER[-1][0]:
        (x0, y0), (x1, y1) = _LADDER[-2], _LADDER[-1]
        return (y1 + (y1 - y0) / (x1 - x0) * (r - x1)) / base
    for (x0, y0), (x1, y1) in zip(_LADDER, _LADDER[1:]):
        if x0 <= r <= x1:
            return (y0 + (y1 - y0) * (r - x0) / (x1 - x0)) / base
    return _LADDER[-1][1] / base


def chain_tokens(p: Sequence[float], n: Optional[int] = None) -> float:
    """Expected tokens of a chain round: 1 + sum of prod(p[:k]) for k = 1..n (p = per-position acceptance given the
    previous one landed)."""
    e, alive = 1.0, 1.0
    for x in list(p)[: len(p) if n is None else n]:
        alive *= min(max(float(x), 0.0), 1.0)
        e += alive
    return e


def best_copy_len(alpha: float, lo: int, hi: int) -> int:
    """The cut in lo..hi with the best tokens per unit of verify; a longer one must beat the shorter by 2%."""
    best, rate = lo, chain_tokens([alpha] * lo) / rel_cost(lo + 1)
    for n in range(lo + 1, hi + 1):
        r = chain_tokens([alpha] * n) / rel_cost(n + 1)
        if r > rate * 1.02:
            best, rate = n, r
    return best


# ------------------------------------------------------------------------------------------------ candidates

@dataclass
class Candidate:
    """One continuation offered to a round: ``tokens`` after t1, ``p`` its per-position acceptance estimate."""
    tokens: np.ndarray
    p: float
    source: str = "record"  # "own" (the head) | "record" | "context"
    tier: int = 0
    match_len: int = 0
    pos: int = -1  # the record position it continues from (heat credit), -1 for none

    def probs(self, n: Optional[int] = None) -> List[float]:
        m = len(self.tokens) if n is None else min(n, len(self.tokens))
        return [float(self.p)] * m


def dedupe(cands: Sequence[Candidate]) -> List[Candidate]:
    """Drops a candidate whose tokens start another one's (the longer stays, with the better estimate), keeping
    first-seen order."""
    out: List[Candidate] = []
    for c in cands:
        t = np.asarray(c.tokens, dtype=np.int64)
        if t.size == 0:
            continue
        dup = False
        for i, o in enumerate(out):
            ot = np.asarray(o.tokens, dtype=np.int64)
            m = min(ot.size, t.size)
            if np.array_equal(ot[:m], t[:m]):
                dup = True
                if t.size > ot.size or c.p > o.p:
                    keep = t if t.size >= ot.size else ot
                    best = c if c.p >= o.p else o
                    out[i] = Candidate(tokens=keep, p=max(c.p, o.p), source=best.source, tier=best.tier,
                                       match_len=max(c.match_len, o.match_len), pos=best.pos)
                break
        if not dup:
            out.append(Candidate(tokens=t, p=c.p, source=c.source, tier=c.tier, match_len=c.match_len, pos=c.pos))
    return out


# ------------------------------------------------------------------------------------------------------ trees

@dataclass
class Tree:
    """A token tree under the root ``tokens[0]`` (the verified token t1). Nodes are in parents-first order:
    ``parents[i] < i`` and ``parents[0] == -1``. ``score[i]`` is the chance the target walks through node i."""
    tokens: np.ndarray
    parents: np.ndarray
    depth: np.ndarray
    score: np.ndarray
    source: np.ndarray  # per node: index of the chain that first put it there (-1 for the root)

    @property
    def n(self) -> int:
        return int(self.tokens.size)

    def children(self, i: int) -> List[int]:
        return [int(j) for j in np.flatnonzero(self.parents == i)]

    def ancestor_mask(self) -> np.ndarray:
        """[n, n] bool: row i sees column j when j is i or one of its ancestors."""
        n = self.n
        m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            j = i
            while j >= 0:
                m[i, j] = True
                j = int(self.parents[j])
        return m

    def positions(self, prefix_len: int) -> np.ndarray:
        """Absolute positions: the root sits at ``prefix_len``, a node at ``prefix_len + depth``."""
        return int(prefix_len) + self.depth.astype(np.int64)

    def path(self, i: int) -> List[int]:
        out = []
        while i >= 0:
            out.append(int(i))
            i = int(self.parents[i])
        return out[::-1]

    def expected_tokens(self) -> float:
        """1 + the sum of every non-root node's chance: the expected tokens the round emits (t1 and kept drafts)."""
        return 1.0 + float(self.score[1:].sum())


def build_tree(root: int, chains: Sequence[Tuple[Sequence[int], Sequence[float]]], budget: int) -> Tree:
    """Merges ``chains`` (tokens after the root, per-position acceptance) into one trie, keeps the ``budget`` nodes
    (root included) with the best chance, ancestors always kept with a node. A node several chains share is one
    event (its chance the best chain estimate); siblings are exclusive (one target pick per node), so when their
    chances sum past 1 they are scaled to 1."""
    toks, par, dep, q, src = [int(root)], [-1], [0], [1.0], [-1]
    for ci, (ct, cp) in enumerate(chains):
        cur = 0
        for d, t in enumerate(list(ct)[: len(cp)]):
            t = int(t)
            nxt = -1
            for j in range(len(toks)):
                if par[j] == cur and toks[j] == t:
                    nxt = j
                    break
            pj = min(max(float(cp[d]), 0.0), 1.0)
            if nxt >= 0:
                q[nxt] = max(q[nxt], pj)
            else:
                toks.append(t)
                par.append(cur)
                dep.append(d + 1)
                q.append(pj)
                src.append(ci)
                nxt = len(toks) - 1
            cur = nxt
    n = len(toks)
    score = np.zeros(n)
    score[0] = 1.0
    for i in range(1, n):
        sib = sum(q[j] for j in range(1, n) if par[j] == par[i])
        scale = 1.0 / sib if sib > 1.0 else 1.0
        score[i] = score[par[i]] * q[i] * scale
    keep = _prune(np.asarray(par), score, max(1, int(budget)))
    return _subtree(toks, par, dep, score, src, keep)


def _prune(par: np.ndarray, score: np.ndarray, budget: int) -> List[int]:
    """The best ``budget`` nodes, each with its ancestors (a node's chance never exceeds its parent's, so best-first
    with ancestors pulled in keeps the set closed)."""
    order = sorted(range(1, par.size), key=lambda i: (-score[i], i))
    keep = {0}
    for i in order:
        if len(keep) >= budget:
            break
        need = []
        j = i
        while j not in keep:
            need.append(j)
            j = int(par[j])
        if len(keep) + len(need) <= budget:
            keep.update(need)
    return sorted(keep)


def _subtree(toks, par, dep, score, src, keep: List[int]) -> Tree:
    # Breadth first, children in the order they were added: parents before children, siblings in score order.
    kept = set(keep)
    order = [0]
    head = 0
    while head < len(order):
        i = order[head]
        head += 1
        kids = [j for j in range(len(toks)) if par[j] == i and j in kept]
        kids.sort(key=lambda j: (-score[j], j))
        order.extend(kids)
    new = {old: k for k, old in enumerate(order)}
    return Tree(tokens=np.asarray([toks[i] for i in order], dtype=np.int64),
                parents=np.asarray([new[par[i]] if par[i] >= 0 else -1 for i in order], dtype=np.int64),
                depth=np.asarray([dep[i] for i in order], dtype=np.int64),
                score=np.asarray([score[i] for i in order], dtype=np.float64),
                source=np.asarray([src[i] for i in order], dtype=np.int64))


def tree_from_parents(tokens: Sequence[int], parents: Sequence[int], score: Optional[Sequence[float]] = None) -> Tree:
    """A tree from node tokens and parents in parents-first order (an engine's own tree)."""
    t = np.asarray(tokens, dtype=np.int64)
    p = np.asarray(parents, dtype=np.int64)
    if t.size == 0 or p.size != t.size or p[0] != -1 or any(p[i] >= i or p[i] < 0 for i in range(1, p.size)):
        raise ValueError("parents must be in parents-first order with the root at 0")
    d = np.zeros(t.size, dtype=np.int64)
    for i in range(1, t.size):
        d[i] = d[p[i]] + 1
    s = np.ones(t.size) if score is None else np.asarray(score, dtype=np.float64)
    return Tree(tokens=t, parents=p, depth=d, score=s, source=np.zeros(t.size, dtype=np.int64))


def merge_chains(tree: Tree, chains: Sequence[Tuple[Sequence[int], Sequence[float]]], budget: Optional[int] = None,
                 own_p: float = 0.5) -> Tree:
    """An engine's own tree with candidate chains merged in, ``budget`` nodes kept (default: the tree's own count, so
    the verify keeps its shape). The tree's own nodes take ``own_p`` per position unless it carries scores."""
    budget = tree.n if budget is None else int(budget)
    own: List[Tuple[List[int], List[float]]] = []
    for leaf in [i for i in range(tree.n) if not tree.children(i)]:
        path = tree.path(leaf)[1:]
        if not path:
            continue
        sc = []
        for k, node in enumerate(path):
            parent_s = tree.score[tree.parents[node]] if tree.parents[node] >= 0 else 1.0
            s = tree.score[node] / parent_s if parent_s > 0 and tree.score[node] <= parent_s and tree.score[node] < 1 else own_p
            sc.append(float(s))
        own.append(([int(tree.tokens[j]) for j in path], sc))
    return build_tree(int(tree.tokens[0]), own + [(list(c), list(p)) for c, p in chains], budget)


# ---------------------------------------------------------------------------------------------- engine layouts

def to_sglang(tree: Tree, prefix_len: int) -> dict:
    """One request's arrays as SGLang's ``build_tree_kernel_efficient`` lays them out (draft token 0 = the root):
    ``draft_tokens`` [n], ``positions`` [n] (prefix + depth), ``retrive_index`` [n] (0..n-1, the caller adds the
    request's offset), ``retrive_next_token`` [n] (first child or -1), ``retrive_next_sibling`` [n] (next sibling
    or -1), ``tree_mask`` [n, n] (ancestor-or-self; FULL_MASK prepends ``prefix_len`` True columns a row,
    :func:`sglang_full_mask`)."""
    n = tree.n
    nxt = np.full(n, -1, dtype=np.int64)
    sib = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        kids = tree.children(i)
        if kids:
            nxt[i] = kids[0]
            for a, b in zip(kids, kids[1:]):
                sib[a] = b
    return {"draft_tokens": tree.tokens.copy(), "positions": tree.positions(prefix_len),
            "retrive_index": np.arange(n, dtype=np.int64), "retrive_next_token": nxt,
            "retrive_next_sibling": sib, "tree_mask": tree.ancestor_mask()}


def sglang_full_mask(tree: Tree, prefix_len: int) -> np.ndarray:
    """FULL_MASK for one request, flattened as SGLang concatenates it: row i = prefix_len True + its tree row."""
    m = tree.ancestor_mask()
    return np.concatenate([np.ones((tree.n, int(prefix_len)), dtype=bool), m], axis=1).reshape(-1)


def from_sglang(draft_tokens, retrive_next_token, retrive_next_sibling) -> Tree:
    """Back from one request's SGLang arrays (draft token 0 the root) to a :class:`Tree`."""
    t = np.asarray(draft_tokens, dtype=np.int64).reshape(-1)
    nt = np.asarray(retrive_next_token, dtype=np.int64).reshape(-1)
    ns = np.asarray(retrive_next_sibling, dtype=np.int64).reshape(-1)
    n = t.size
    parent = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        c = int(nt[i])
        while c >= 0:
            parent[c] = i
            c = int(ns[c])
    order = [0]
    head = 0
    while head < len(order):
        i = order[head]
        head += 1
        c = int(nt[i])
        while c >= 0:
            order.append(c)
            c = int(ns[c])
    new = {old: k for k, old in enumerate(order)}
    return tree_from_parents([t[i] for i in order], [new[parent[i]] if parent[i] >= 0 else -1 for i in order])


def ngram_insert(trie_children: dict, root: int, chain: Sequence[int]) -> int:
    """Inserts ``chain`` under ``root`` into a dict trie ``{node_id: {token: node_id}}`` with ``node_id -> token``
    implied by the caller; returns how many nodes were added. Node ids are allocated as ``len(trie_children)``."""
    added = 0
    cur = root
    for t in chain:
        kids = trie_children.setdefault(cur, {})
        if int(t) in kids:
            cur = kids[int(t)]
            continue
        nid = len(trie_children)
        while nid in trie_children:
            nid += 1
        kids[int(t)] = nid
        trie_children[nid] = {}
        cur = nid
        added += 1
    return added


# --------------------------------------------------------------------------------------------- verification

@dataclass
class TreeVerdict:
    path: List[int]  # accepted node indices, the root first
    next_token: int

    @property
    def accepted(self) -> int:
        return len(self.path) - 1


def verify_walk(tree: Tree, pick_at: Callable[[int], int], cap: Optional[int] = None) -> TreeVerdict:
    """Greedy (``pick_at`` = the target's argmax at a node) or point-mass sampled (``pick_at`` = one target draw at
    a node): from the root, the child whose token is the pick is kept, until none is (or ``cap`` drafts)."""
    cur, path = 0, [0]
    while True:
        y = int(pick_at(cur))
        if cap is not None and len(path) - 1 >= cap:
            return TreeVerdict(path=path, next_token=y)
        nxt = next((c for c in tree.children(cur) if int(tree.tokens[c]) == y), None)
        if nxt is None:
            return TreeVerdict(path=path, next_token=y)
        cur = nxt
        path.append(cur)


def verify_specinfer(tree: Tree, p_at: Callable[[int], np.ndarray], q_at: Callable[[int], np.ndarray],
                     rng: np.random.Generator, cap: Optional[int] = None) -> TreeVerdict:
    """Multi-round speculative sampling over children drawn i.i.d. from the head's ``q_at(node)``: child c is kept
    with ``min(1, p[c]/q[c])``; on a rejection ``p <- norm(max(p - q, 0))`` and the next child is tried; with every
    child rejected the next token is drawn from the residual. Lossless for any ``q``."""
    cur, path = 0, [0]
    while True:
        p = np.asarray(p_at(cur), dtype=np.float64).copy()
        p /= p.sum()
        if cap is not None and len(path) - 1 >= cap:
            return TreeVerdict(path=path, next_token=int(rng.choice(p.size, p=p)))
        q = np.asarray(q_at(cur), dtype=np.float64)
        q = q / q.sum()
        moved = False
        for c in tree.children(cur):
            x = int(tree.tokens[c])
            if q[x] <= 0:
                continue
            if rng.random() < min(1.0, p[x] / q[x]):
                cur = c
                path.append(c)
                moved = True
                break
            r = np.maximum(p - q, 0.0)
            s = r.sum()
            if s <= 0:
                break
            p = r / s
        if not moved:
            return TreeVerdict(path=path, next_token=int(rng.choice(p.size, p=p)))


# ------------------------------------------------------------------------------------------------- selection

def propose(engine, rid, n: Optional[int] = None):
    """What a chain-only head's plan asks the engine for: ``propose_select`` (the best of every distinct candidate)
    under YEE_TREE, else ``propose``; an engine without ``propose_select`` (an older core, a test double) keeps
    ``propose``."""
    sel = getattr(engine, "propose_select", None)
    overridden = "propose" in getattr(engine, "__dict__", {})  # a propose replaced on the instance is what runs
    if sel is not None and not overridden and tree_enabled():
        return sel(rid, n=n, k=max(1, tree_k()) + 2)
    return engine.propose(rid, n=n)


def choose_chain(chains: Sequence[Sequence[float]], slots: Optional[int] = None) -> int:
    """For a chain-only verify: the index of the chain (per-position acceptance lists, chain 0 the head's own) that
    promises the most tokens in ``slots`` drafts (default: each its own length). Ties keep the lower index, so the
    head's chain alone (or as good as any copy) is what runs."""
    best, best_e = 0, -1.0
    for i, p in enumerate(chains):
        e = chain_tokens(p, slots)
        if e > best_e + 1e-12:
            best, best_e = i, e
    return best


@dataclass
class Plan:
    """Which branches a tree round verifies and how far (``lens[b]`` drafts, 0 = left out)."""
    lens: List[int]
    rows: int
    tokens: float
    rate: float

    @property
    def branches(self) -> int:
        return sum(1 for x in self.lens if x > 0)


def _tree_tokens(branches, lens, cap) -> float:
    chains = [(list(t)[:L], list(p)[:L]) for (t, p), L in zip(branches, lens) if L > 0]
    if not chains:
        return 1.0
    tree = build_tree(-1, chains, 10 ** 6)
    s = tree.score[1:][tree.depth[1:] <= cap] if cap is not None else tree.score[1:]
    return 1.0 + float(s.sum())


def choose_plan(branches: Sequence[Tuple[Sequence[int], Sequence[float]]], max_recall: int = DEFAULT_K,
                rows: int = 16, state_cost: float = DEFAULT_STATE_COST, stage: float = 0.3,
                cap: Optional[int] = None) -> Tuple[Plan, Plan]:
    """(own, mixed) for a batched-branch verify (mlx-serve's tree rounds, this module's reference): branch 0 the
    head's chain at any cut, the others recalled at a few cuts, at most ``max_recall`` of them, every plan within
    ``rows`` verify rows; priced by expected tokens per unit of verify (rows on the ladder, each extra branch state,
    the draft forward). ``mixed`` is the best plan with a recalled branch (``rows == 0`` when none fits)."""
    from itertools import combinations, product

    def rate_of(lens):
        r = sum(L + 1 for L in lens if L > 0) or 1
        if r > rows:
            return None
        nb = sum(1 for L in lens if L > 0)
        tok = _tree_tokens(branches, lens, cap)
        return Plan(lens=list(lens), rows=r, tokens=tok, rate=tok / (rel_cost(r) + max(0, nb - 1) * state_cost + stage))

    nb = len(branches)
    own_max = min(len(branches[0][0]), len(branches[0][1]), rows - 1)
    own = Plan(lens=[0] * nb, rows=1, tokens=1.0, rate=0.0)
    for l0 in range(own_max + 1):
        p = rate_of([l0] + [0] * (nb - 1))
        if p is not None and p.rate > own.rate:
            own = p
    mixed = Plan(lens=[0] * nb, rows=0, tokens=1.0, rate=0.0)
    for k in range(1, min(max_recall, nb - 1) + 1):
        for members in combinations(range(1, nb), k):
            cuts = []
            for b in members:
                n = min(len(branches[b][0]), len(branches[b][1]))
                cuts.append(sorted({c for c in (n, (3 * n + 3) // 4, (n + 1) // 2, MIN_COPY) if 0 < c <= n}))
            for pick in product(*cuts):
                for l0 in range(own_max + 1):
                    lens = [0] * nb
                    lens[0] = l0
                    for b, c in zip(members, pick):
                        lens[b] = c
                    p = rate_of(lens)
                    if p is not None and p.rate > mixed.rate:
                        mixed = p
    return own, mixed


__all__ = [
    "Candidate", "Tree", "TreeVerdict", "Plan", "env_value", "tree_enabled", "tree_k", "tree_nodes", "propose",
    "rel_cost", "chain_tokens", "best_copy_len", "dedupe", "build_tree", "tree_from_parents",
    "merge_chains", "to_sglang", "sglang_full_mask", "from_sglang", "ngram_insert", "verify_walk",
    "verify_specinfer", "choose_chain", "choose_plan",
]
