"""Port the GPU prefill of lkqmoe 0.5.6 (source commit 93edcc1, python/lkqmoe/prefill.py) into xtier/backend/prefill.py.

Same code, renamed: environment switches LKQMOE_* -> XTIER_*, lkqmoe -> xtier in comments and messages. The kernels,
tiles and the order of every FP32 addition are unchanged, so outputs stay bitwise identical. The plan it uploads from
is an ``xtier.engine.CpuPlan`` (shards, localize, scale_format; ``pool`` = the layer's expert cache or None).
Usage: port_prefill.py LKQMOE_SRC XTIER_REPO"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
RENAMES = [
    ('LKQMOE_', 'XTIER_'),
    ('lkqmoe_shard', 'xtier_shard'),
    ("lkqmoe's", "the engine's"),
    ('lkqmoe', 'xtier'),
]
FIRST = '"""Independent NVFP4 grouped GPU prefill. No LKMoE imports or binary calls.\n'
HEADER = ('"""Grouped GPU prefill of a MoE layer whose experts live in host memory (the CPU engine\'s NUMA shards).\n'
          'Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_prefill.py: names changed only.\n')
text = (src / 'python/lkqmoe/prefill.py').read_text()
assert text.startswith(FIRST)
text = text[len(FIRST):]
for a, b in RENAMES:
    text = text.replace(a, b)
assert 'lkqmoe' not in text.lower() and 'lkmoe' not in text.lower()
text = HEADER + text
out = dst / 'xtier/backend/prefill.py'
out.write_text(text, encoding='utf-8')
print(f'prefill.py -> {out.relative_to(dst)} ({text.count(chr(10))} lines)')
