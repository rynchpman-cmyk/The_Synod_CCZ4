from __future__ import annotations

import unittest

import numpy as np

from experiments.audit_nulloth_mixing_and_relaxation import (
    _mass_crossing_audit,
)
from experiments.map_nulloth_linear_growth_atlas import (
    _expected_scalar,
    _multidimensional_scalar_spectrum,
    _ordered,
)


class NullothLinearAtlasTests(unittest.TestCase):
    def test_spatial_diagonal_scalar_mode_is_isotropic(self) -> None:
        numerical = _ordered(_multidimensional_scalar_spectrum(8, (1, 1, 1)))
        expected = _expected_scalar((1, 1, 1))

        self.assertLess(np.max(np.abs(numerical - expected)), 1.0e-9)

    def test_vector_mixing_opens_the_exact_mass_crossing(self) -> None:
        uncoupled_gap, coupled_gap, mixing = _mass_crossing_audit()

        self.assertLess(uncoupled_gap, 1.0e-12)
        self.assertGreater(coupled_gap, 0.09)
        self.assertGreater(mixing, 0.49)


if __name__ == "__main__":
    unittest.main()
