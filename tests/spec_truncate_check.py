"""Verify truncation of the speculative pack (packs/speculative/truncate.py) and the core's decode-path token flags:

1. SGLang's chain sampler (chain_speculative_sampling_triton) on a chain whose q row L is zero returns what it returns
   on the chain cut to L drafts, same coins: accept count, accepted tokens, final token (bitwise; with block
   verification up to the float rounding of a residual sum);
2. the truncation kernel: kept drafts, token flags and the zeroed q row against a reference;
3. the mailbox packing kernel with token flags: routes of flagged tokens -1, the rest unchanged.

  python tests/spec_truncate_check.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from sglang.kernels.ops.speculative.reject_sampling import chain_speculative_sampling_triton  # noqa: E402

from xtier.backend import kernels  # noqa: E402
from xtier.packs.speculative import truncate  # noqa: E402

torch.manual_seed(20261023)
DEV = 'cuda'
V = 6000


def sample_chain(bs, S, sharp):
    """Target p [bs, S + 1, V], draft q [bs, S, V] (softmax of noisy shared logits), drafts drawn from q."""
    base = torch.randn(bs, S + 1, V, device=DEV) * sharp
    p = torch.softmax(base + 0.5 * torch.randn_like(base), -1)
    q = torch.softmax(base[:, :S] + 0.5 * torch.randn(bs, S, V, device=DEV), -1)
    drafts = torch.multinomial(q.reshape(-1, V), 1).reshape(bs, S)
    root = torch.randint(0, V, (bs, 1), device=DEV)
    return p, q, torch.cat([root, drafts], 1)


def run(p, q, cand, coins, final, block):
    bs, slots = cand.shape
    predicts = torch.full((bs * slots,), -1, dtype=torch.int32, device=DEV)
    accept_index = torch.full((bs, slots), -1, dtype=torch.int32, device=DEV)
    accept_num = torch.zeros(bs, dtype=torch.int32, device=DEV)
    index = torch.arange(bs * slots, device=DEV).reshape(bs, slots)
    chain_speculative_sampling_triton(predicts, accept_index, accept_num, cand, index, None, None, coins.contiguous(),
                                      final, p.contiguous(), q.contiguous(), 1.0, 1.0, True, block_verification=block)
    n = accept_num.tolist()
    return [(n[b], predicts[b * slots:b * slots + n[b] + 1].tolist()) for b in range(bs)]


ok = True
for block in (False, True):
    for sharp in (1.0, 3.0, 8.0):
        bs, S, trials = 16, 5, 40
        same = 0
        for _ in range(trials):
            p, q, cand = sample_chain(bs, S, sharp)
            coins = torch.rand(bs, S + 1, device=DEV)
            final = torch.rand(bs, device=DEV)
            keep = torch.randint(1, S + 1, (bs,)).tolist()
            qz = q.clone()
            for b, L in enumerate(keep):
                if L < S:
                    qz[b, L] = 0
            got = run(p, qz, cand, coins, final, block)
            for b, L in enumerate(keep):
                ref = run(p[b:b + 1, :L + 1], q[b:b + 1, :L], cand[b:b + 1, :L + 1], coins[b:b + 1, :L + 1],
                          final[b:b + 1], block)[0]
                same += got[b] == ref
        # classic: bitwise; block verification: the same decisions up to float rounding of a residual sum
        ok &= same == bs * trials if not block else same >= 0.99 * bs * trials
        print(f'zero q row == cut chain (block verification {block}, sharpness {sharp}): {same}/{bs * trials}')

# 2. the truncation kernel
bs, S, TH = 6, 5, 0.2
q = torch.softmax(torch.randn(bs, S, V, device=DEV) * 4, -1)
tokens = torch.randint(0, V, (bs, S + 1), device=DEV)
for b in range(bs):   # confident drafts first, then a weak one at a different place per request
    for j in range(S):
        q[b, j, tokens[b, j + 1]] = 0.05 if j == b else 0.6
top_ks = torch.tensor([50, 50, 1, 50, 50, 50], dtype=torch.int32, device=DEV)
probs = q.clone()


batch = NS(forward_mode=NS(is_idle=lambda: False), sampling_info=NS(top_ks=top_ks),
           spec_info=NS(draft_probs=probs, draft_token=tokens.reshape(-1).clone()))
truncate.enable(DEV)
flags = truncate._flags[0]
assert truncate.apply(batch, 1, TH)
torch.cuda.synchronize()
good = True
for b in range(bs):
    first_weak = next((j for j in range(S) if q[b, j, tokens[b, j + 1]] < TH), S)
    L = S if top_ks[b] == 1 else max(first_weak, 1)
    want_flags = [1 if j <= L else 0 for j in range(S + 1)]
    got_flags = flags[b * (S + 1):(b + 1) * (S + 1)].tolist()
    rows_zero = [bool((probs[b, j] == 0).all()) for j in range(S)]
    want_zero = [j == L for j in range(S)]
    rows_same = all(torch.equal(probs[b, j], q[b, j]) for j in range(S) if j != L)
    good &= got_flags == want_flags and rows_zero == want_zero and rows_same
    print(f'request {b}: keep {L}, flags {got_flags}, zero rows {[j for j in range(S) if rows_zero[j]]}')
truncate.reset()
torch.cuda.synchronize()
good &= bool((flags == 1).all())
ok &= good
print(f'truncation kernel: {"ok" if good else "WRONG"}')

# 3. mailbox packing with token flags
m, k, H = 6, 8, 256
x = torch.randn(m, H, device=DEV).bfloat16()
ids = torch.randint(0, 288, (m, k), dtype=torch.int32, device=DEV)
routing = torch.rand(m, k, device=DEV)
act = torch.tensor([1, 0, 1, 1, 0, 0], dtype=torch.int32, device=DEV)
slot = (torch.empty(m, H, dtype=torch.bfloat16, device=DEV), torch.empty(m, k, dtype=torch.int32, device=DEV),
        torch.empty(m, k, device=DEV))
kernels.pack(tuple(t.data_ptr() for t in slot), m, k, x.data_ptr(), ids.data_ptr(), routing.data_ptr(), H,
             act.data_ptr())
want = torch.where(act[:, None] != 0, ids, -1)
packed = torch.equal(slot[1], want) and torch.equal(slot[0], x) and torch.equal(slot[2], routing)
kernels.pack(tuple(t.data_ptr() for t in slot), m, k, x.data_ptr(), ids.data_ptr(), routing.data_ptr(), H)
packed &= torch.equal(slot[1], ids)
ok &= packed
print(f'mailbox packing with token flags: {"ok" if packed else "WRONG"}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
