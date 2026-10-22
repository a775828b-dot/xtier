"""Port the CPU engine of lkqmoe 0.5.6 (source commit 93edcc1) into xtier/csrc: same code, renamed.

Only identifiers, strings and comments change (lkqmoe_* -> xtier_*, LKQMOE_* -> XTIER_*, the expert-pool
environment LKQSTRATA_* -> XTIER_CACHE_*, library names), and the Qwen-only compatibility entry point
lkqmoe_create(experts, threads, ...) is removed (create2 / create3 take the shape). The kernels and the runtime
stay one translation unit, so the compiler sees the same code and the outputs must stay bitwise identical.
Usage: port_engine.py LKQMOE_SRC XTIER_CSRC"""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
RENAMES = [
    ('"../include/lkqmoe_cuda.h"', '"xtier_mailbox.h"'),
    ('"../include/lkqmoe.h"', '"xtier_engine.h"'),
    ('"/liblkqmoe_"', '"/libxtier_engine_"'),
    ('lkqmoe_cuda_', 'xtier_mailbox_'),
    ('LKQMOE_CUDA_', 'XTIER_MAILBOX_'),
    ('lkqmoe_', 'xtier_'),
    ('LKQMOE_', 'XTIER_'),
    ('LKQSTRATA_', 'XTIER_CACHE_'),
    ('LKQstrata', 'XTier cache'),
    ('LKQSTRATA', 'XTIER_CACHE'),
    ('lkqstrata', 'xtier_cache'),
    ('[lkqmoe]', '[xtier]'),
    ('lkqmoe', 'xtier'),
]
REMOVE = {
    'src/lkqmoe.cpp': ['void *LKQMOE_API(lkqmoe_create)(int experts,int threads,const uint8_t *w13,const uint8_t *w2,\n'
                       '                   const uint8_t *s13,const uint8_t *s2,const float *g13,const float *g2) {\n'
                       '    return LKQMOE_API(lkqmoe_create2)(2560,640,experts,threads,0.f,w13,w2,s13,s2,g13,g2);\n'
                       '}\n'],
    'src/dispatch.cpp': ['    void *(*create)(int,int,const uint8_t*,const uint8_t*,const uint8_t*,const uint8_t*,const float*,const float*);\n',
                         '&&bind(h,p,"lkqmoe_create",m.create)',
                         'void *lkqmoe_create(int experts,int threads,const uint8_t *w13,const uint8_t *w2,\n'
                         '                    const uint8_t *s13,const uint8_t *s2,const float *g13,const float *g2) {\n'
                         '    auto p=impl();return p?p->create(experts,threads,w13,w2,s13,s2,g13,g2):nullptr;\n'
                         '}\n'],
    'include/lkqmoe.h': ['// Compatibility entry point: H=2560, I=640, no clamp (Qwen3.8 Flash Next).\n'
                         'void *lkqmoe_create(int experts, int threads, const uint8_t *w13, const uint8_t *w2,\n'
                         '                   const uint8_t *s13, const uint8_t *s2, const float *g13, const float *g2);\n'],
}
FILES = {'src/lkqmoe.cpp': 'engine/engine.cpp', 'src/dispatch.cpp': 'engine/dispatch.cpp',
         'src/cuda_doorbell.cpp': 'mailbox/mailbox.cpp', 'include/lkqmoe.h': 'include/xtier_engine.h',
         'include/lkqmoe_cuda.h': 'include/xtier_mailbox.h'}
HEADER = ('// Ported from lkqmoe 0.5.6 (commit 93edcc1) by tools/port_engine.py: identifiers, strings and comments\n'
          '// renamed only, so the kernels compile to the same code.\n')
for rel, out in FILES.items():
    text = (src / rel).read_text()
    for snippet in REMOVE.get(rel, []):
        assert text.count(snippet) == 1, (rel, snippet[:60])
        text = text.replace(snippet, '')
    for a, b in RENAMES:
        text = text.replace(a, b)
    assert 'lkqmoe' not in text.lower() and 'lkqstrata' not in text.lower(), rel
    lead = '#pragma once\n' if text.startswith('#pragma once\n') else ''
    path = dst / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lead + HEADER + text[len(lead):], encoding='utf-8')
    print(f'{rel} -> {out} ({text.count(chr(10))} lines)')
