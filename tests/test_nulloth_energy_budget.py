from __future__ import annotations

import unittest

from causal_repair.grid import PeriodicGrid
from experiments.close_nulloth_coordinate_energy_budget import _geometric_work_rate
from experiments.measure_nulloth_weak_field_backreaction import _solver


class NullothEnergyBudgetTests(unittest.TestCase):
    def test_flat_vacuum_has_zero_geometric_work(self) -> None:
        solver = _solver(
            PeriodicGrid((4,), (4.0,), method="spectral"),
            1.0e-3,
        )
        state = solver.vacuum_state()

        self.assertAlmostEqual(
            _geometric_work_rate(solver, solver._pack(state), state.time),
            0.0,
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
