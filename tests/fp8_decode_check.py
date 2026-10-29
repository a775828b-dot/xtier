"""FP8 (E4M3) decode of XTier's kernels: the software path (RTX 30 series, no float8e4nv in Triton) against torch's
float8_e4m3fn values and, on GPUs with the conversion (SM89 and newer), against the hardware path bit for bit: all 256
bytes, then kernels that read NVFP4 block scales (the GPU prefill's expert unpacking, the pool's gate/up and down on
Qwen3.8 / GLM-5.3 decode shapes), run once with XTIER_FP8_DECODE=hw and once with =soft in child processes.

  python tests/fp8_decode_check.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SHAPES = {'qwen38': (2560, 640), 'glm53': (4096, 2048)}


def worker(out):
    import torch
    import triton
    import triton.language as tl

    from xtier import arch
    from xtier.backend import prefill
    from xtier.cache import pool

    @triton.jit
    def decode_bytes(SRC, DST):
        i = tl.arange(0, 256)
        tl.store(DST + i, arch.e4m3_f32(tl.load(SRC + i)).to(tl.int32, bitcast=True))

    dev = torch.device('cuda')
    src = torch.arange(256, dtype=torch.uint8, device=dev)
    bits = torch.empty(256, dtype=torch.int32, device=dev)
    decode_bytes[(1,)](src, bits)
    result = {'bits': bits.cpu(), 'hw': bool(arch.HW_FP8)}
    for name, (H, I) in SHAPES.items():
        g = torch.Generator(device='cpu').manual_seed(20261029)
        # the GPU prefill's unpacking of a few experts' down projection: [E, H, I / 2] codes, [E, H, I / 16] scales
        E = 2
        w = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g).to(dev)
        s = torch.randint(0, 256, (E, H, I // 16), dtype=torch.uint8, generator=g)
        s[s & 0x7F == 0x7F] = 0x38   # NaN codes do not occur in checkpoints
        s = s.to(dev)
        d = torch.empty((E, H, I), dtype=torch.float32, device=dev)
        B = 1024
        prefill._unpack_all[(E, H * I // B)](w, s, d, H, I, B)
        # the pool's decode kernels on random NVFP4 slots (as tests/fp4_decode_check.py)
        S, m, k = 4, 4, 2
        q13 = torch.randint(0, 256, (S, 2 * I, H // 2), dtype=torch.uint8, generator=g).to(dev)
        q2 = torch.randint(0, 256, (S, H, I // 2), dtype=torch.uint8, generator=g).to(dev)
        s13 = torch.randint(0x28, 0x48, (S, 2 * I, H // 16), dtype=torch.uint8, generator=g).to(dev)
        s2 = torch.randint(0x28, 0x48, (S, H, I // 16), dtype=torch.uint8, generator=g).to(dev)
        g13 = (torch.rand((8, 2), generator=g) * 0.02 + 0.01).to(dev)
        g2 = (torch.rand((8,), generator=g) * 0.02 + 0.01).to(dev)
        x = torch.randn((m, H), generator=g).to(torch.bfloat16).to(dev)
        slot_list = torch.tensor([0, 1, 2, 3, 1, 2, 0, 3], dtype=torch.int32, device=dev)[: m * k]
        slot_exp = torch.tensor([5, 1, 7, 2, 1, 7, 5, 2], dtype=torch.int32, device=dev)[: m * k]
        mid = torch.zeros((pool.MAX_ROUTES, pool.MAX_M, I), dtype=torch.bfloat16, device=dev)
        part = torch.zeros((pool.MAX_ROUTES, pool.MAX_M, H), dtype=torch.float32, device=dev)
        cfg, nk = pool.CONFIG, m * k
        pool._gate_up[(nk, I // cfg['gbn'])](x, slot_list, slot_exp, q13, s13, g13, mid, m, H, I, 0.0,
                                             pool.MAX_M, cfg['gbn'], pool._tile_k(H),
                                             num_warps=cfg['gw'], num_stages=cfg['gs'])
        pool._down[(nk, H // cfg['dbn'])](mid, slot_list, slot_exp, q2, s2, g2, part, m, H, I, pool.MAX_M, cfg['dbn'],
                                          pool._tile_k(I), num_warps=cfg['dw'], num_stages=cfg['ds'])
        torch.cuda.synchronize()
        result[name] = (d.cpu(), mid[:nk, :m].cpu(), part[:nk, :m].cpu())
    torch.save(result, out)


def main():
    import torch
    with tempfile.TemporaryDirectory() as d:
        runs = {}
        for mode in ('soft', 'hw'):
            path = os.path.join(d, mode + '.pt')
            env = dict(os.environ, XTIER_FP8_DECODE=mode)
            r = subprocess.run([sys.executable, __file__, '--worker', path], env=env, capture_output=True, text=True)
            if r.returncode != 0:
                if mode == 'hw':
                    print('hardware path not available on this GPU:', (r.stderr.strip().splitlines() or ['?'])[-1][:200])
                    break
                print(r.stdout, r.stderr)
                raise SystemExit('FP8-DECODE-CHECK FAIL: software path did not run')
            runs[mode] = torch.load(path)
    soft = runs['soft']
    # 1. the software decode against torch's E4M3 values (NaN for 0x7F / 0xFF, -0.0 for 0x80), bit for bit
    want = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    got = soft['bits'].view(torch.float32)
    nan = torch.isnan(want)
    bad = int((torch.isnan(got) != nan).sum()) + int((got[~nan].view(torch.int32) != want[~nan].view(torch.int32)).sum())
    print(f'software E4M3 -> FP32 over 256 bytes: {bad} differences from torch.float8_e4m3fn')
    ok = bad == 0
    # 2. hardware against software, bit for bit
    if 'hw' in runs:
        hw = runs['hw']
        same = torch.equal(hw['bits'][~nan], soft['bits'][~nan])
        print(f'hardware float8e4nv vs software over 256 bytes: {"identical" if same else "DIFFERENT"}')
        ok &= same
        for name in SHAPES:
            (d_hw, m_hw, p_hw), (d_sw, m_sw, p_sw) = hw[name], soft[name]
            same = (torch.equal(d_hw.view(torch.int32), d_sw.view(torch.int32))
                    and torch.equal(m_hw.view(torch.int16), m_sw.view(torch.int16))
                    and torch.equal(p_hw.view(torch.int32), p_sw.view(torch.int32)))
            print(f'{name}: prefill unpacking, pool gate/up and down, hardware vs software decode: '
                  f'{"bitwise identical" if same else "DIFFERENT"} (|part| max {p_sw.abs().max().item():.3g})')
            ok &= same
    print('FP8-DECODE-CHECK', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(sys.argv[2])
    else:
        main()
