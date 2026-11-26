from __future__ import annotations

import unittest

import numpy as np

from causal_repair.adm import flat_metric
from causal_repair.finite_nulloth import (
    FiniteNullothProductionSolver,
    NullothScalarCharacteristicBoundary,
)
from causal_repair.grid import CartesianGrid
from causal_repair.nulloth_scalar import NullothScalarSystem


class FiniteNullothTests(unittest.TestCase):
    def test_scalar_boundary_preserves_outgoing_and_damps_incoming_mode(self) -> None:
        grid = CartesianGrid((8,), (1.0,), origin=(0.0,), method="fd2")
        system = NullothScalarSystem(grid)
        state = system.initialize(np.zeros(grid.shape))
        state.momentum[0] = 3.0
        state.gradient[0, 0] = 1.0
        raw = (grid.zeros(), grid.zeros(), grid.zeros((3,)))

        _, dmomentum, dgradient = NullothScalarCharacteristicBoundary(
            grid, damping=2.0
        ).scalar_rhs(
            flat_metric(grid),
            state,
            raw,
            np.ones(grid.shape),
            grid.zeros((3,)),
        )

        # At the lower face Phi_s=-Phi_x. Incoming Pi-Phi_s=4 and
        # outgoing Pi+Phi_s=2; preserve d(outgoing)=0 and set d(incoming)=-8.
        self.assertAlmostEqual(dmomentum[0], -4.0)
        self.assertAlmostEqual(dgradient[0, 0], -4.0)

    def test_finite_vacuum_is_fixed_point(self) -> None:
        grid = CartesianGrid((8,), (1.0,), origin=(0.0,), method="fd2")
        solver = FiniteNullothProductionSolver(grid)
        state = solver.vacuum_state()

        evolved = solver.step(state, 1.0e-5)

        for before, after in zip(solver._pack(state), solver._pack(evolved)):
            np.testing.assert_allclose(after, before, atol=3.0e-13)

    def test_multidimensional_finite_solver_is_not_yet_claimed(self) -> None:
        grid = CartesianGrid((8, 8), (1.0, 1.0), method="fd2")

        with self.assertRaises(NotImplementedError):
            FiniteNullothProductionSolver(grid)

    def test_finite_backend_exposes_boundary_entropy_account(self) -> None:
        grid = CartesianGrid((8,), (1.0,), origin=(-0.5,), method="fd2")
        solver = FiniteNullothProductionSolver(grid)
        state = solver.step(solver.vacuum_state(), 1.0e-5)

        self.assertAlmostEqual(solver.last_entropy_boundary_outflow, 0.0)
        self.assertAlmostEqual(solver.last_entropy_balance_residual, 0.0)
        self.assertEqual(state.time, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
