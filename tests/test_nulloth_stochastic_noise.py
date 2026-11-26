from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.audit_nulloth_brake_stochastic_noise import stochastic_profile


class NullothStochasticNoiseTests(unittest.TestCase):
    def test_profile_is_reproducible_normalized_and_has_declared_rms(self) -> None:
        coordinate = np.arange(32, dtype=float)
        wavenumber = 2.0 * np.pi / 32.0
        first, fraction, norm = stochastic_profile(coordinate, wavenumber, 1103)
        second, repeated_fraction, repeated_norm = stochastic_profile(
            coordinate, wavenumber, 1103
        )
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.max(np.abs(first))), 1.0)
        self.assertAlmostEqual(fraction, 0.10)
        self.assertEqual(fraction, repeated_fraction)
        self.assertEqual(norm, repeated_norm)

    def test_distinct_seeds_produce_distinct_profiles(self) -> None:
        coordinate = np.arange(32, dtype=float)
        wavenumber = 2.0 * np.pi / 32.0
        first = stochastic_profile(coordinate, wavenumber, 1103)[0]
        second = stochastic_profile(coordinate, wavenumber, 2207)[0]
        self.assertGreater(float(np.max(np.abs(first - second))), 1.0e-3)


if __name__ == "__main__":
    unittest.main()
