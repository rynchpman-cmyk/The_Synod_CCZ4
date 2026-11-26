from __future__ import annotations

import unittest

import numpy as np

from experiments.derive_nulloth_boundary_characteristics import (
    _clustered_projectors,
    _sign_invariant_projectors,
)


class NullothBoundaryInventoryTests(unittest.TestCase):
    def test_clustered_projectors_resolve_repeated_physical_modes(self) -> None:
        symbol = np.diag([-1.0, -1.0, 0.0, 1.0, 1.0])
        result = _clustered_projectors(symbol)

        self.assertEqual(result[0], (2, 1, 2))
        self.assertLess(result[3], 1.0e-13)
        self.assertLess(result[4], 1.0e-13)
        self.assertLess(result[5], 1.0e-13)

    def test_sign_projectors_ignore_internal_repeated_mode_basis(self) -> None:
        similarity = np.array(
            [
                [1.0, 0.2, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        diagonal = np.diag([-1.0, -1.0, 0.0, 1.0, 1.0])
        symbol = similarity @ diagonal @ np.linalg.inv(similarity)
        result = _sign_invariant_projectors(symbol)

        self.assertEqual(result[0], (2, 1, 2))
        self.assertLess(result[3], 1.0e-13)
        self.assertLess(result[4], 1.0e-13)
        self.assertLess(result[5], 1.0e-13)


if __name__ == "__main__":
    unittest.main()
