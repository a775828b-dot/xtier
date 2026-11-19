"""Bitwise equality check for xtier.packs.deepseek_v4.mhc_sm120.mhc_post_exact against SGLang's post_torch_impl."""
import torch

from xtier.packs.deepseek_v4.mhc_sm120 import mhc_post_exact


def _ref(x, residual, post_mix, comb):
    return (
        post_mix.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def main():
    torch.manual_seed(20261119)
    for m in (385, 743, 1024, 4096, 16384):
        x = torch.randn((m, 5120), dtype=torch.bfloat16, device='cuda')
        res = torch.randn((m, 4, 5120), dtype=torch.bfloat16, device='cuda')
        post_mix = 2.0 * torch.sigmoid(torch.randn((m, 4), dtype=torch.float32, device='cuda'))
        comb = torch.softmax(torch.randn((m, 4, 4), dtype=torch.float32, device='cuda'), dim=-1)
        got = mhc_post_exact(x, res, post_mix, comb)
        exp = _ref(x, res, post_mix, comb)
        assert torch.equal(got, exp), f'mismatch at m={m}: max_abs={(got.float() - exp.float()).abs().max().item()}'
    print('PASS: ds_mhc_post_check (m=385..16384 bitwise identical)')


if __name__ == '__main__':
    main()
