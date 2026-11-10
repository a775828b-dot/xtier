"""The elastic cache between the chunks of a long prompt (xtier/cache/elastic.py Governor.after_decode, _fills_chunk):
with --prefill-decode-interval N another request decodes N steps after every chunk. Extra slots added in those steps go
back before the next chunk (2026-11-10, DeepSeek V4.1, N = 150: 8.4 GiB released and uploaded again at each of ten
chunks), so after a prefill forward that filled its chunk they are added only after XTIER_CACHE_ELASTIC_IDLE_CHUNK decode
forwards, after any other allocating forward after XTIER_CACHE_ELASTIC_IDLE as before. No GPU: the pool manager, the
adaptive policy and the device's free memory are stood in for.

  python tests/elastic_chunk_check.py
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['XTIER_CACHE_ELASTIC'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from xtier.cache import elastic, pool  # noqa: E402

GIB = 2 ** 30
pool.started = lambda: True
torch.cuda.empty_cache = lambda: None
torch.cuda.mem_get_info = lambda *a: (40 * GIB, 72 * GIB)
bad = 0


def check(name, ok, detail=''):
    global bad
    print(f'{name}: {"ok" if ok else "FAIL"} {detail}'.rstrip(), flush=True)
    bad += not ok


def governor():
    """40 layers of 50 slots with room for 19 more, none of them held; a growth is only recorded."""
    pools = [SimpleNamespace(S=69, base=50, active=50, arena=[torch.zeros(69, 64, dtype=torch.uint8) for _ in range(4)])
             for _ in range(40)]
    g = elastic.Governor(pools, SimpleNamespace(), commit=lambda: None)
    g.seen[False] = {(1, 1): 1}   # a request's prefill has been seen
    grown = []
    g._grow = lambda target, step: grown.append(g.since)
    return g, grown


def decode(g, grown, forwards):
    """`forwards` decode forwards; the number of them after which a growth started (None: none did)."""
    for _ in range(forwards):
        g.after_decode()
        if g.grower is not None:
            g.grower.join()
    return grown[0] if grown else None


runner = SimpleNamespace(server_args=SimpleNamespace(chunked_prefill_size=22528))
check('a forward of the chunk size fills its chunk', elastic._fills_chunk(runner, 22528) and elastic._fills_chunk(runner, 21120)
      and not elastic._fills_chunk(runner, 10836) and not elastic._fills_chunk(runner, 21119))
check('no chunked prefill, no server arguments: never', not elastic._fills_chunk(SimpleNamespace(), 22528)
      and not elastic._fills_chunk(SimpleNamespace(server_args=SimpleNamespace(chunked_prefill_size=-1)), 22528)
      and not elastic._fills_chunk(SimpleNamespace(server_args=SimpleNamespace(chunked_prefill_size=None)), 22528))

g, grown = governor()
g.before_alloc(10836, False, 213588, False)
at = decode(g, grown, 64)
check('after a prompt\'s last chunk slots are added after XTIER_CACHE_ELASTIC_IDLE decode forwards', at is not None
      and g.idle <= at < g.idle + g.every, f'(after {at})')

g, grown = governor()
g.before_alloc(22528, False, 45056, True)
at = decode(g, grown, 600)
check('between two chunks (600 decode forwards of another request) none are added', at is None, f'(after {at})')
g.before_alloc(22528, True, 45056, False)   # the draft model's forward of the same chunk
at = decode(g, grown, 600)
check('the draft model\'s forward does not end the prompt', at is None, f'(after {at})')
g.before_alloc(22528, False, 67584, True)
g.before_alloc(10836, False, 78420, False)
at = decode(g, grown, 64)
check('after the chunk that ends the prompt they are', at is not None and at < g.idle + g.every, f'(after {at})')

g, grown = governor()
g.before_alloc(22528, False, 22528, True)   # a prompt of exactly one chunk: nothing follows
at = decode(g, grown, g.idle_chunk + 64)
check('a prompt that ends on a chunk boundary: after XTIER_CACHE_ELASTIC_IDLE_CHUNK decode forwards', at is not None
      and g.idle_chunk <= at < g.idle_chunk + g.every, f'(after {at})')
print('ELASTIC-CHUNK-CHECK', 'PASS' if not bad else f'FAIL ({bad})')
sys.exit(1 if bad else 0)
