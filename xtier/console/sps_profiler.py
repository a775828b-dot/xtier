"""SGLang SPS profiler with token-resolution fitting for small verification batches."""

from functools import partial


def main():
    """Preserve measured token counts instead of merging budgets into 64-token bins."""
    from sglang.benchmark import dspark_sps_profiler as profiler

    profiler.ols_resid_backfit = partial(profiler.ols_resid_backfit, mbin_w=1)
    profiler.cli_main()


if __name__ == '__main__':
    main()
