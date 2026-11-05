"""Kimi K3.1 Pro: a checkpoint variant whose large BF16 projections are 4-bit, compressed-tensors layout, loaded by
SGLang's W4A16 Marlin linear methods (the kimi_k3 pack names the dense parameter as the model loader expects and drops
the BF16 originals):
- nvfp4 (default): E2M1, E4M3 scale per 16, one FP32 global scale (divisor) per fused group (q/k/v/g, gate/up: SGLang
  requires the parts of a fused projection to share it), ``X.weight_packed`` / ``X.weight_scale`` /
  ``X.weight_global_scale``, in a second config group matched by name (names win over the MXFP4 group's "Linear");
  about 9.5% weight error on Kimi's projections;
- mxfp4: E2M1, UE8M0 scale per 32, as the routed experts (``X.weight_packed`` uint8 [out, in/2], low nibble first;
  ``X.weight_scale`` uint8 [out, in/32]), the existing MXFP4 group; about 11.3% weight error.

Converted: KDA q/k/v/g/o, the DSA layers' g/o, the shared experts, layer 0's dense MLP (with --latent also the
latent MoE projections, routed_expert_up/down_proj). Kept in BF16: embeddings, lm_head, the router, MLA q_a/q_b/kv_a/
kv_b, the DSA indexer, KDA's small projections (b, f_a, f_b), norms, the vision tower. Not quantization-aware trained:
compare against BF16 before serving (tools/kimi_mxfp4_dense.py prints the weight error per module kind).

Per block of 32, the shared exponent is the better (lower squared error) of two: the smallest that clips nothing,
and one below it (clips the largest values to 6). Values round to the nearest E2M1 magnitude.

The variant directory holds the new config, index and the MXFP4 files; everything else is a symlink to the source
(the experts stay in the source shards, the BF16 originals there are not loaded).

  python tools/kimi_mxfp4_dense.py SRC DST [--format nvfp4|mxfp4] [--latent] [--device cuda]
"""
import argparse
import collections
import json
import os
import re
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DENSE = (r'\.self_attn\.(q_proj|k_proj|v_proj|g_proj|o_proj)$', r'\.block_sparse_moe\.shared_experts\.(gate_proj|up_proj|down_proj)$',
         r'\.layers\.0\.mlp\.(gate_proj|up_proj|down_proj)$')
LATENT = r'\.block_sparse_moe\.routed_expert_(up|down)_proj$'
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def mxfp4(w: torch.Tensor):
    """BF16 [R, C] -> (packed uint8 [R, C/2], E8M0 uint8 [R, C/32], relative Frobenius error)."""
    R, C = w.shape
    x = w.float().reshape(R, C // 32, 32)
    amax = x.abs().amax(-1, keepdim=True)
    e_hi = torch.ceil(torch.log2(amax.clamp_min(2.0 ** -120) / 6.0))
    bounds, grid = BOUNDS.to(w.device), E2M1.to(w.device)
    best_err = best_q = best_e = None
    for e in (e_hi - 1, e_hi):
        scale = torch.exp2(e)
        mag = (x / scale).abs()
        code = torch.bucketize(mag.clamp(max=6.0), bounds)
        q = grid[code] * torch.sign(x) * scale
        err = (q - x).square().sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_e, best_code = err, q, e, code
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_q = torch.where(take, q, best_q)
            best_e = torch.where(take, e, best_e)
            best_code = torch.where(take, code, best_code)
    code = (best_code | ((x < 0) & (best_code > 0)).to(best_code.dtype) << 3).to(torch.uint8).reshape(R, C)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    scale = (best_e.reshape(R, C // 32) + 127).clamp(0, 254).to(torch.uint8)
    rel = float((best_q.reshape(R, C) - w.float()).norm() / w.float().norm().clamp_min(1e-30))
    return packed.contiguous(), scale.contiguous(), rel


def nvfp4(w: torch.Tensor, g: float):
    """BF16 [R, C] with global multiplier g (amax of its fused group / (6 * 448)) -> (packed uint8 [R, C/2], E4M3
    [R, C/16] as float8, relative Frobenius error). w ~ e2m1 x scale x g; the checkpoint stores 1 / g."""
    R, C = w.shape
    x = w.float().reshape(R, C // 16, 16)
    s = (x.abs().amax(-1) / (6 * g)).clamp(max=448).to(torch.float8_e4m3fn)
    d = (s.float() * g).clamp_min(1e-30).unsqueeze(-1)
    code = torch.bucketize((x / d).abs().clamp(max=6.0), BOUNDS.to(w.device))
    q = E2M1.to(w.device)[code] * torch.sign(x) * d
    code = (code | ((x < 0) & (code > 0)).to(code.dtype) << 3).to(torch.uint8).reshape(R, C)
    rel = float((q.reshape(R, C) - w.float()).norm() / w.float().norm().clamp_min(1e-30))
    return (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous(), s.contiguous(), rel


GROUP = re.compile(r'\.(q_proj|k_proj|v_proj|g_proj|gate_proj|up_proj)$')


def group_of(m):
    """Fused group key: q/k/v/g of a layer's attention, gate/up of an MLP; other projections alone."""
    hit = GROUP.search(m)
    if hit is None:
        return m
    return m[:hit.start()] + ('.qkvg' if hit.group(1) in ('q_proj', 'k_proj', 'v_proj', 'g_proj') else '.gate_up')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--latent', action='store_true', help='also the latent MoE projections')
    ap.add_argument('--format', choices=('nvfp4', 'mxfp4'), default='nvfp4')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    pats = [re.compile(p) for p in DENSE + ((LATENT,) if a.latent else ())]
    os.makedirs(a.dst, exist_ok=True)
    index = json.load(open(os.path.join(a.src, 'model.safetensors.index.json')))
    wm = index['weight_map']
    convert = {n[:-len('.weight')]: f for n, f in wm.items()
               if n.endswith('.weight') and any(p.search(n[:-len('.weight')]) for p in pats)}
    by_file = collections.defaultdict(list)
    for m, f in convert.items():
        by_file[f].append(m)
    stats = collections.defaultdict(list)
    t0 = time.time()
    for i, (f, mods) in enumerate(sorted(by_file.items())):
        out = {}
        with safe_open(os.path.join(a.src, f), framework='pt', device='cpu') as h:
            ws = {m: h.get_tensor(m + '.weight').to(a.device) for m in sorted(mods)}
        amax = collections.defaultdict(float)
        for m, w in ws.items():
            assert w.dim() == 2 and w.shape[1] % 32 == 0 and w.dtype == torch.bfloat16, (m, w.shape, w.dtype)
            amax[group_of(m)] = max(amax[group_of(m)], float(w.float().abs().amax()))
        for m, w in ws.items():
            if a.format == 'mxfp4':
                packed, scale, rel = mxfp4(w)
            else:
                g = max(amax[group_of(m)], 1e-12) / (6 * 448)
                packed, scale, rel = nvfp4(w, g)
                out[m + '.weight_global_scale'] = torch.tensor([1.0 / g], dtype=torch.float32)
            out[m + '.weight_packed'], out[m + '.weight_scale'] = packed.cpu(), scale.cpu()
            stats[re.sub(r'\.layers\.\d+\.', '.layers.N.', m).split('.', 3)[-1]].append(rel)
        del ws
        name = f'model-{a.format}-from-' + f.replace('model-', '').replace('.safetensors', '') + '.safetensors'
        save_file(out, os.path.join(a.dst, name), metadata={'format': 'pt'})
        for m in mods:
            del wm[m + '.weight']
            wm[m + '.weight_packed'] = wm[m + '.weight_scale'] = name
            if a.format == 'nvfp4':
                wm[m + '.weight_global_scale'] = name
        print(f'[{i + 1}/{len(by_file)}] {f}: {len(mods)} projections -> {name} ({time.time() - t0:.0f} s)', flush=True)
    json.dump(index, open(os.path.join(a.dst, 'model.safetensors.index.json'), 'w'), indent=2)

    cfg = json.load(open(os.path.join(a.src, 'config.json')))
    qc = cfg['text_config']['quantization_config']
    keep_attn = r'(q_proj|k_proj|v_proj|g_proj|o_proj|fused_qkvg_proj|qkv_proj)$'
    new_ignore = []
    for pat in qc['ignore']:
        if pat == 're:.*self_attn.*':
            new_ignore.append(r're:.*self_attn\.(?!' + keep_attn + ').*')
        elif pat in ('re:.*shared_experts.*', r're:.*mlp\.(gate|up|gate_up|down)_proj.*'):
            continue
        elif a.latent and pat == r're:.*block_sparse_moe\.routed_expert_(up|down)_proj.*':
            continue
        else:
            new_ignore.append(pat)
    qc['ignore'] = new_ignore
    if a.format == 'nvfp4':
        targets = [r're:.*self_attn\.(q_proj|k_proj|v_proj|g_proj|o_proj|fused_qkvg_proj|qkv_proj)$',
                   r're:.*shared_experts\.(gate_proj|up_proj|down_proj|gate_up_proj)$',
                   r're:.*layers\.0\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$']
        if a.latent:
            targets.append(r're:.*routed_expert_(up|down)_proj$')
        weights = dict(qc['config_groups']['group_0']['weights'], strategy='tensor_group', group_size=16,
                       scale_dtype=None)
        qc['config_groups']['xtier_dense_nvfp4'] = {'format': 'nvfp4-pack-quantized', 'targets': targets,
                                                    'weights': weights, 'input_activations': None,
                                                    'output_activations': None}
    qc['xtier_dense_mxfp4'] = {'latent': a.latent, 'format': a.format}   # read by the kimi_k3 pack
    json.dump(cfg, open(os.path.join(a.dst, 'config.json'), 'w'), indent=2)

    for name in os.listdir(a.src):
        if name in ('config.json', 'model.safetensors.index.json'):
            continue
        link = os.path.join(a.dst, name)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(a.src, name)), link)
    print(f'{a.dst}: {len(convert)} projections in {a.format}, ignore {new_ignore}')
    for kind, v in sorted(stats.items()):
        print(f'  {kind:55s} n={len(v):3d}  weight rel error mean {sum(v) / len(v):.4f} max {max(v):.4f}')


if __name__ == '__main__':
    main()
