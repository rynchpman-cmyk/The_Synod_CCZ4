from __future__ import annotations

import unittest

import numpy as np

from causal_repair.adm import inverse_metric
from causal_repair.grid import PeriodicGrid
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from experiments.audit_nulloth_noether_accounting import run_audit


class NullothCanonicalScalarTests(unittest.TestCase):
    def test_curved_spatial_scalar_stress_matches_canonical_projection(self) -> None:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        system = NullothScalarSystem(grid)
        h_cell = np.array(
            [
                [1.2, 0.07, -0.02],
                [0.07, 0.9, 0.04],
                [-0.02, 0.04, 1.1],
            ]
        )
        h = np.broadcast_to(h_cell[:, :, None], (3, 3) + grid.shape).copy()
        momentum = np.array([0.21, -0.13, 0.08, -0.03])
        gradient = np.array(
            [
                [0.11, 0.04, -0.07, 0.02],
                [-0.03, 0.09, 0.01, -0.05],
                [0.06, -0.02, 0.03, 0.08],
            ]
        )
        potential = np.array([0.17, 0.12, 0.09, 0.2])
        state = NullothScalarState(grid.zeros(), momentum, gradient)

        actual = system.stress(h, state, potential)
        h_inv = inverse_metric(h)[0]
        gradient_squared = np.einsum(
            "ij...,i...,j...->...", h_inv, gradient, gradient
        )
        expected_rho = 0.5 * (momentum**2 + gradient_squared) + potential
        expected_momentum = momentum[None, ...] * gradient
        expected_stress = (
            gradient[:, None, ...] * gradient[None, :, ...]
            + h
            * (
                0.5 * (momentum**2 - gradient_squared) - potential
            )[None, None, ...]
        )
        expected_trace = np.einsum(
            "ij...,ij...->...", h_inv, expected_stress
        )

        np.testing.assert_allclose(actual.rho, expected_rho)
        np.testing.assert_allclose(actual.momentum, expected_momentum)
        np.testing.assert_allclose(actual.stress, expected_stress)
        np.testing.assert_allclose(actual.trace, expected_trace)

    def test_scalar_momentum_sign_pair_is_preserved(self) -> None:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        system = NullothScalarSystem(grid)
        h = np.zeros((3, 3) + grid.shape)
        for axis in range(3):
            h[axis, axis] = 1.0
        state = system.initialize(varphi=0.3, momentum=-0.4)
        equation_source = np.full(grid.shape, 0.2)

        dvarphi, dmomentum, dgradient = system.rhs(
            h,
            grid.zeros(),
            np.ones(grid.shape),
            grid.zeros((3,)),
            state,
            equation_source,
        )

        np.testing.assert_allclose(dvarphi, 0.4)
        np.testing.assert_allclose(dmomentum, equation_source)
        np.testing.assert_allclose(dgradient, 0.0)


class NullothAccountingTests(unittest.TestCase):
    def test_internal_exchanges_and_periodic_currents_close(self) -> None:
        audit = run_audit(seed=20260724, scalar_exchange_samples=16)

        self.assertLessEqual(
            audit.maximum_scalar_material_energy_exchange_residual, 1.0e-13
        )
        self.assertLessEqual(audit.proca_energy_exchange_residual, 2.0e-10)
        self.assertLessEqual(
            audit.maximum_proca_momentum_exchange_residual, 2.0e-10
        )
        self.assertLessEqual(audit.maximum_initial_gauss_constraint, 1.0e-12)
        self.assertLessEqual(audit.periodic_baryon_rhs_integral, 1.0e-12)
        self.assertLessEqual(
            audit.periodic_carrier_charge_rhs_integral, 1.0e-12
        )
        self.assertLessEqual(audit.periodic_total_energy_rhs_integral, 1.0e-12)
        self.assertLessEqual(
            audit.maximum_periodic_total_momentum_rhs_integral, 1.0e-12
        )


if __name__ == "__main__":
    unittest.main()
