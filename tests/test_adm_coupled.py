import unittest

import numpy as np

from causal_repair.adm import ADMParameters, ADMSolver, christoffel, inverse_metric, ricci_tensor
from causal_repair.coupled import CoupledADMProcaSolver
from causal_repair.curved_proca import CurvedProcaSystem
from causal_repair.grid import PeriodicGrid
from causal_repair.proca import ProcaParameters, ProcaSolver, ProcaState


class ADMCoupledTests(unittest.TestCase):
    def setUp(self):
        self.grid = PeriodicGrid((8,), (2.0 * np.pi,), "spectral")
        self.adm = ADMSolver(self.grid)

    def test_flat_geometry_has_zero_curvature(self):
        state = self.adm.flat_state()
        inverse, _, _ = inverse_metric(state.h)
        gamma = christoffel(self.grid, state.h, inverse)
        np.testing.assert_allclose(gamma, 0.0)
        np.testing.assert_allclose(ricci_tensor(self.grid, state.h, inverse, gamma), 0.0)

    def test_minkowski_vacuum_is_exact_fixed_point(self):
        coupled = CoupledADMProcaSolver(self.adm, CurvedProcaSystem(self.grid))
        state = coupled.flat_state()
        state = coupled.step(state, 0.01)
        np.testing.assert_allclose(state.h, self.adm.flat_state().h)
        np.testing.assert_allclose(state.K, 0.0)
        self.assertEqual(coupled.constraint_norms(state)["hamiltonian_l2"], 0.0)

    def test_homogeneous_active_initial_data_satisfy_constraints(self):
        kappa = 2.0
        adm = ADMSolver(self.grid, ADMParameters(kappa=kappa))
        curved = CurvedProcaSystem(self.grid, ProcaParameters())
        coupled = CoupledADMProcaSolver(adm, curved)
        state = coupled.flat_state()
        state.a[0] = 1.0e-2
        rho, _, _ = curved.stress_energy(state.h, state.a, state.pi)
        H = np.sqrt(kappa * rho / 3.0)
        for i in range(3):
            state.K[i, i] = -H
        norms = coupled.constraint_norms(state)
        self.assertLess(norms["hamiltonian_linf"], 1e-14)
        self.assertLess(norms["momentum_linf"], 1e-14)

    def test_curved_stress_reduces_to_flat_implementation(self):
        parameters = ProcaParameters(Z=1.3, mass=0.7)
        flat = ProcaSolver(self.grid, parameters)
        curved = CurvedProcaSystem(self.grid, parameters)
        geometry = self.adm.flat_state()
        (x,) = self.grid.coordinates()
        state = ProcaState(self.grid.zeros((3,)), self.grid.zeros((3,)))
        state.a[0] = np.cos(x)
        state.a[1] = np.sin(2.0 * x)
        state.pi[2] = np.sin(x)
        expected = flat.stress_energy_3p1(state)
        actual = curved.stress_energy(geometry.h, state.a, state.pi)
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left, right, atol=1e-11)


if __name__ == "__main__":
    unittest.main()

