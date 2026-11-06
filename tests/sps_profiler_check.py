"""Small-batch SPS fitting keeps measured token budgets distinct."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sglang.benchmark import dspark_sps_profiler as profiler
from xtier.console.sps_profiler import main


class SpsProfilerTests(unittest.TestCase):
    def test_small_budget_costs_survive_fitting(self):
        cells = [{'bs': bs, 'M': m, 'T': 0.01 + 0.005 * bs + 0.002 * m}
                 for bs in (1, 2) for m in range(bs, bs * 6 + 1)]

        def fit():
            table = profiler.build_additive_table_from_cells(cells=cells)
            self.assertEqual(table.m_probes, list(range(1, 13)))
            for cell in cells:
                self.assertAlmostEqual(
                    table.step_time(num_reqs=cell['bs'], budget=cell['M'] - cell['bs']), cell['T'],
                )

        with patch.object(profiler, 'ols_resid_backfit', profiler.ols_resid_backfit), \
                patch.object(profiler, 'cli_main', side_effect=fit):
            main()


if __name__ == '__main__':
    unittest.main()
