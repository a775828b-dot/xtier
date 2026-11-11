"""The SPS cost table DSpark's compact (variable-length) verification needs, measured on this machine by SGLang's
dspark_sps_profiler. Without a measured table, or with a flat one, SGLANG_RAGGED_VERIFY_MODE=compact verifies full width
every step, the same as static (docs/reports/2026-11-06-ds-compact-verify.md).

1. A server to measure, here DeepSeek V4.1 Flash 1017 (the profiler checks all three settings):
     SGLANG_RAGGED_VERIFY_MODE=compact SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_SIMULATE_ACC_LEN=1.0 DSPARK=1 \\
       configs/ds41-pro5000-nps4/run-ds-xtier.sh
2. python tools/sps_profile.py --base-url http://127.0.0.1:18190 --out my-sps-table.json   (a few minutes)
3. Stop that server; serve with the table:
     SGLANG_RAGGED_VERIFY_MODE=compact SPS_TABLE=$PWD/my-sps-table.json DSPARK=1 configs/ds41-pro5000-nps4/run-ds-xtier.sh

  python tools/sps_profile.py [all | run | fit] [dspark_sps_profiler options]

Added when not given: the server http://127.0.0.1:18190 (run-ds-xtier.sh's port), budget fractions 0.2 to 1.0 (--fracs:
the two-dimensional table compact uses), batch sizes 1 to 4 (--batch-size: up to the server's --max-running-requests,
4 in run-ds-xtier.sh), no plot. The fit keeps every measured token count (mbin_w=1) instead of SGLang's 64-token
bins, as verify budgets here are a few tokens (from cyrus-xc's PR #13).
"""
import sys
from functools import partial

RUN = {'--base-url': ['http://127.0.0.1:18190'], '--fracs': ['0.2', '0.4', '0.6', '0.8', '1.0'],
       '--batch-size': ['1', '2', '3', '4']}
FIT = {'--no-plot': []}


def main():
    from sglang.benchmark import dspark_sps_profiler as profiler

    profiler.ols_resid_backfit = partial(profiler.ols_resid_backfit, mbin_w=1)
    args = sys.argv[1:]
    if not args or args[0] not in ('all', 'run', 'fit'):
        args = ['all'] + args
    added = {**(RUN if args[0] in ('all', 'run') else {}), **(FIT if args[0] in ('all', 'fit') else {})}
    for flag, values in added.items():
        if flag not in args:
            args += [flag, *values]
    sys.argv = [sys.argv[0]] + args
    profiler.cli_main()


if __name__ == '__main__':
    main()
