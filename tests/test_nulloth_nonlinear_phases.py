from __future__ import annotations

import unittest

import numpy as np

from experiments.map_nulloth_homogeneous_nonlinear_phases import (
    _evolve,
    _mass_squared,
    _predicted_turnover,
)


class NullothNonlinearPhaseTests(unittest.TestCase):
    def test_growing_seed_turns_at_conserved_quartic_energy_root(self) -> None:
        mass_squared = _mass_squared(0.18)
        amplitude = 0.005
        measured = _evolve(
            0.18,
            amplitude,
            -np.sqrt(-mass_squared) * amplitude,
            30.0,
            0.002,
        )[0]
        predicted = _predicted_turnover(amplitude, mass_squared)

        self.assertAlmostEqual(measured, predicted, places=6)

    def test_zero_momentum_seeds_remain_in_opposite_sign_basins(self) -> None:
        negative = _evolve(0.18, -0.001, 0.0, 20.0, 0.002)
        positive = _evolve(0.18, 0.001, 0.0, 20.0, 0.002)

        self.assertLess(negative[6], 0.0)
        self.assertGreater(positive[5], 0.0)


if __name__ == "__main__":
    unittest.main()
