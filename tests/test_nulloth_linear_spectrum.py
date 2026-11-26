from __future__ import annotations

import unittest

import numpy as np

from experiments.audit_nulloth_linear_spectrum import (
    _ordered,
    _scalar_expected,
    _scalar_generator,
    _vector_expected,
    _vector_generator,
)


class NullothLinearSpectrumTests(unittest.TestCase):
    def test_scalar_fourier_generator_matches_condensed_dispersion(self) -> None:
        amplitude = np.sqrt((12.0 * 0.18**2 - 0.4**2) / 50.0)
        curvature = -2.0 * (0.4**2 - 12.0 * 0.18**2)
        numerical = _ordered(
            np.linalg.eigvals(_scalar_generator(16, 32.0, 2, 0.18, amplitude))
        )
        expected = _scalar_expected(4.0 * np.pi / 32.0, curvature)

        self.assertLess(np.max(np.abs(numerical - expected)), 1.0e-8)

    def test_mixed_proca_generator_includes_mass_and_cleaning_branches(self) -> None:
        numerical_generator, parameters = _vector_generator(16, 2, 1.0)
        numerical = _ordered(np.linalg.eigvals(numerical_generator))
        expected = _vector_expected(2, 1.0, parameters)

        self.assertLess(np.max(np.abs(numerical - expected)), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
