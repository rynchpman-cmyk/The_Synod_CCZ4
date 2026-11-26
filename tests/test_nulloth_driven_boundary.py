from __future__ import annotations

import unittest

import numpy as np

from causal_repair.driven_finite_nulloth import DrivenNullothScalarBoundary
from causal_repair.grid import CartesianGrid
from causal_repair.nulloth_scalar import NullothScalarState


class DrivenNullothBoundaryTests(unittest.TestCase):
    def test_zero_drive_reduces_to_homogeneous_penalty(self) -> None:
        grid = CartesianGrid((8,), (1.0,), method="fd2")
        boundary = DrivenNullothScalarBoundary(grid, lambda side, time: (0.0, 0.0))
        h = np.broadcast_to(np.eye(3)[:, :, None], (3, 3, 8)).copy()
        state = NullothScalarState(grid.zeros(), grid.zeros(), grid.zeros((3,)))
        rhs = (grid.zeros(), grid.zeros(), grid.zeros((3,)))
        result = boundary.scalar_rhs(h, state, rhs, np.ones(8), grid.zeros((3,)))
        self.assertEqual(max(float(np.max(np.abs(value))) for value in result), 0.0)

    def test_left_drive_changes_only_boundary_scalar_rhs(self) -> None:
        grid = CartesianGrid((8,), (1.0,), method="fd2")
        boundary = DrivenNullothScalarBoundary(
            grid, lambda side, time: ((1.0, 0.0) if side == -1 else (0.0, 0.0))
        )
        h = np.broadcast_to(np.eye(3)[:, :, None], (3, 3, 8)).copy()
        state = NullothScalarState(grid.zeros(), grid.zeros(), grid.zeros((3,)))
        rhs = (grid.zeros(), grid.zeros(), grid.zeros((3,)))
        _, momentum, gradient = boundary.scalar_rhs(
            h, state, rhs, np.ones(8), grid.zeros((3,))
        )
        self.assertGreater(abs(float(momentum[0])), 0.0)
        self.assertGreater(abs(float(gradient[0, 0])), 0.0)
        self.assertEqual(float(momentum[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
