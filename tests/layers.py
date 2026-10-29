"""One real MoE layer of each production model, as the CPU engine's inputs (shared by the engine tests)."""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

# name: model dir, expert key, (gate, up, down) names, H, I, E, top-k, SwiGLU limit, layer, verify width, format,
#       production kernel switches (without prefix)
PRESETS = {
    'glm53': ('/data/fast/nvme1/models/huggingface/LibertAIDAI/GLM-5.3-Flash-NVFP4', 'model.language_model.layers.{layer}.mlp.experts.{e}.',
              ('gate_proj', 'up_proj', 'down_proj'), 4096, 2048, 288, 8, 10.0, 20, 4, 'nvfp4', {'DYNAMIC': '2', 'DOWN_BF16': '1', 'SPIN_COUNT': '1048576'}),
    'qwen38': ('/data/fast/nvme2/models/huggingface/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4',
               'model.language_model.layers.{layer}.mlp.experts.{e}.', ('gate_proj', 'up_proj', 'down_proj'), 2560, 640, 512, 10, 0.0, 36, 4,
               'nvfp4', {'SPIN_COUNT': '1048576'}),
    'ds41_1017': ('/data/fast/nvme1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017', 'layers.{layer}.ffn.experts.{e}.', ('w1', 'w3', 'w2'),
                  5120, 2304, 384, 6, 10.0, 20, 6, 'mxfp4', {'DYNAMIC': '2', 'DOWN_BF16': '1', 'DOWN_GRAIN': '24', 'SPIN_COUNT': '1048576', 'THP': '0'}),
    # compressed-tensors NVFP4 W4A4 (weight_packed, weight_global_scale = 1 / scale_2)
    'minimax_m31': ('/data/fast/nvme2/models/modelscope/MiniMax/MiniMax-M3.1-Flash', 'language_model.model.layers.{layer}.block_sparse_moe.experts.{e}.',
                    ('w1', 'w3', 'w2'), 6144, 3072, 128, 4, 7.0, 20, 8, 'nvfp4', {'SPIN_COUNT': '1048576'}),
}
# GPT-OSS SwiGLU (alpha, beta) of a preset: gate * sigmoid(alpha * gate) * (up + beta); the others (1, 0) = SiLU
SWIGLU = {'minimax_m31': (1.702, 1.0)}


def load(preset, layer=None):
    """(w13, w2, s13, s2, g13, g2) host tensors of one layer, in the engine's layout. XTIER_LAYERS_<PRESET>=DIR[:LAYER]
    (e.g. XTIER_LAYERS_DS41_1017) reads another copy of the checkpoint, and its default layer, instead of the preset's:
    a machine that downloaded only some shards."""
    MODEL, KEY, NAMES, H, I, E, K, LIMIT, LAYER, VM, FMT, SWITCHES = PRESETS[preset]
    MODEL, _, alt = os.environ.get(f'XTIER_LAYERS_{preset.upper()}', MODEL).partition(':')
    LAYER = int(alt) if alt else LAYER
    layer = LAYER if layer is None else layer
    G = 32 if FMT == 'mxfp4' else 16
    root = Path(MODEL)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    w13 = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8); w2 = torch.empty((E, H, I // 2), dtype=torch.uint8)
    s13 = torch.empty((E, 2 * I, H // G), dtype=torch.uint8); s2 = torch.empty((E, H, I // G), dtype=torch.uint8)
    g13 = torch.ones((E, 2), dtype=torch.float32); g2 = torch.ones((E,), dtype=torch.float32)
    scale_suffix = '.scale' if FMT == 'mxfp4' else '.weight_scale'
    packed = KEY.format(layer=layer, e=0) + NAMES[0] + '.weight_packed' in index   # compressed-tensors naming
    weight_suffix = '.weight_packed' if packed else '.weight'
    grouped = {}
    for e in range(E):
        p = KEY.format(layer=layer, e=e)
        for role, n in zip(('gate', 'up', 'down'), NAMES):
            grouped.setdefault(index[p + n + weight_suffix], []).append((e, p, n, role))
    for file, items in grouped.items():
        with safe_open(root / file, framework='pt', device='cpu') as f:
            for e, p, n, role in items:
                w, s, g, rows = {'gate': (w13[e], s13[e], g13[e, 0], slice(0, I)), 'up': (w13[e], s13[e], g13[e, 1], slice(I, 2 * I)),
                                 'down': (w2[e], s2[e], g2[e], slice(None))}[role]
                w[rows].copy_(f.get_tensor(p + n + weight_suffix).view(torch.uint8))
                s[rows].copy_(f.get_tensor(p + n + scale_suffix).view(torch.uint8))
                if packed:
                    g.copy_(f.get_tensor(p + n + '.weight_global_scale').float().reciprocal().reshape(()))
                elif FMT == 'nvfp4':
                    g.copy_(f.get_tensor(p + n + '.weight_scale_2'))
    return w13, w2, s13, s2, g13, g2
