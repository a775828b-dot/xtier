"""DeepSeek V4.1's FP8 steps in software below SM89 (RTX 30 series) against SGLang's float8e4nv kernels on a GPU with
float8e4nv, bit for bit: the Engram row gather (xtier/packs/deepseek_v4/engram_gather.py) and the FP4 fake
quantization of compressed KV with its E4M3 block scales (fake_quant_fp8.py, in a child process each way: Triton keeps
a kernel compiled with the row function it first saw).

  python tests/ds_fp8_soft_check.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('XTIER_FP8_DECODE', 'soft')   # the RTX 30 series decode on this GPU too (children inherit it)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402


def fake_quant(out, rebuilt):
    from sglang.kernels.ops.attention.dsv4 import fp4_rope_fake_quant as fq
    if rebuilt:
        from xtier.packs.deepseek_v4 import fake_quant_fp8
        fake_quant_fp8._rebuild()
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(20261029)
    x = torch.randn(8192, 512, device=dev, generator=g) * torch.logspace(-4, 2, 8192, device=dev)[:, None]
    x = x.to(torch.bfloat16)
    angles = torch.rand(8192, 32, device=dev, generator=g) * 6.3
    freqs = torch.polar(torch.ones_like(angles), angles)
    result = {('E4M3' if kv else 'UE8M0'): fq.rope_tail_fake_quant_fp4(x, freqs, 64, compressed_kv=kv)
              for kv in (True, False)}
    torch.save({k: v.cpu() for k, v in result.items()}, out)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--fake-quant':
        return fake_quant(sys.argv[2], sys.argv[3] == '1')
    dev = torch.device('cuda')
    ok = True
    # Engram rows: random FP8 payloads (NaN codes left out) and E8M0 scales, ids also outside the table's rows
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather as sglang_gather
    from xtier.packs.deepseek_v4.engram_gather import engram_gather
    g = torch.Generator(device=dev).manual_seed(20261029)
    rows, dim, block = 200000, 256, 32
    w = torch.randint(0, 256, (rows, dim), dtype=torch.uint8, device=dev, generator=g)
    w[(w & 0x7F) == 0x7F] = 0
    s = torch.randint(0, 256, (rows, dim // block), dtype=torch.uint8, device=dev, generator=g)
    s[s > 160] = 127
    ids = torch.randint(-1000, rows + 1000, (65536,), device=dev, generator=g)
    for lo, hi in ((0, rows), (50000, 150000)):
        a = sglang_gather(w.data_ptr(), s.data_ptr(), ids, torch.empty(ids.numel(), dim, dtype=torch.bfloat16,
                                                                          device=dev), dim, block, lo, hi)
        b = engram_gather(w.data_ptr(), s.data_ptr(), ids, torch.empty(ids.numel(), dim, dtype=torch.bfloat16,
                                                                        device=dev), dim, block, lo, hi)
        same = torch.equal(a.view(torch.int16), b.view(torch.int16))
        ok &= same
        print(f'Engram gather, rows [{lo}, {hi}): {ids.numel()} ids x {dim}: '
              f'{"bitwise equal" if same else "DIFFER"}')
    with tempfile.TemporaryDirectory() as tmp:
        outs = []
        for rebuilt in (0, 1):
            out = f'{tmp}/fq{rebuilt}.pt'
            subprocess.run([sys.executable, __file__, '--fake-quant', out, str(rebuilt)], check=True)
            outs.append(torch.load(out))
        for key in outs[0]:
            same = torch.equal(outs[0][key].view(torch.int16), outs[1][key].view(torch.int16))
            ok &= same
            print(f'FP4 fake quant, {key} block scales: {"bitwise equal" if same else "DIFFER"}')
    print('DS-FP8-SOFT-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
