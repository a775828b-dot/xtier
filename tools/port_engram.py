"""Port the Engram row store of the DeepSeek V4.1 1017 pipeline (ds41-1017/adapter/row_store.cpp, 2026-10-22 version
with the multi-GPU fixes) into csrc/engram/rows.cpp: same code, renamed (exported row_store_* -> xtier_rows_*, the
environment DS41_ENGRAM_IO_THREADS -> XTIER_ENGRAM_IO_THREADS, OFFLOAD_MODE -> XTIER_ENGRAM_MODE).
Usage: port_engram.py ROW_STORE_CPP XTIER_REPO"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
RENAMES = [
    ('row_store_', 'xtier_rows_'),
    ('DS41_ENGRAM_IO_THREADS', 'XTIER_ENGRAM_IO_THREADS'),
    ('"OFFLOAD_MODE"', '"XTIER_ENGRAM_MODE"'),
]
HEADER = ('// Ported from the DeepSeek V4.1 1017 pipeline (adapter/row_store.cpp, 2026-10-22) by tools/port_engram.py:\n'
          '// exported names and environment switches renamed only.\n')
text = src.read_text()
for a, b in RENAMES:
    assert a in text, a
    text = text.replace(a, b)
assert 'row_store' not in text and 'DS41_' not in text and 'OFFLOAD_MODE' not in text
out = dst / 'csrc/engram/rows.cpp'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(HEADER + text, encoding='utf-8')
print(f'row_store.cpp -> {out.relative_to(dst)} ({text.count(chr(10))} lines)')
