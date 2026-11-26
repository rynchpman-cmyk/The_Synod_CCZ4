import unittest

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.proca import ProcaParameters, ProcaSolver, ProcaState


class ProcaTests(unittest.TestCase):
    def setUp(self):
        self.grid = PeriodicGrid((32,), (2.0 * np.pi,), "spectral")
        self.parameters = ProcaParameters(Z=1.7, mass=1.2)
        self.solver = ProcaSolver(self.grid, self.parameters)
        (self.x,) = self.grid.coordinates()

    def _one_period(self, component):
        mode = 2
        omega = np.sqrt(mode**2 + self.parameters.mass**2)
        a = self.grid.zeros((3,))
        a[component] = 0.2 * np.cos(mode * self.x)
        state = ProcaState(a, self.grid.zeros((3,)))
        initial = state.copy()
        initial_energy = self.solver.diagnostics(state).energy
        period = 2.0 * np.pi / omega
        steps = 400
        for _ in range(steps):
            state = self.solver.step(state, period / steps)
        return initial, state, initial_energy

    def test_transverse_dispersion_and_energy(self):
        initial, final, energy = self._one_period(1)
        np.testing.assert_allclose(final.a, initial.a, atol=2e-8)
        self.assertLess(abs(self.solver.diagnostics(final).energy / energy - 1.0), 2e-10)

    def test_longitudinal_dispersion_and_energy(self):
        initial, final, energy = self._one_period(0)
        np.testing.assert_allclose(final.a, initial.a, atol=2e-8)
        self.assertLess(abs(self.solver.diagnostics(final).energy / energy - 1.0), 2e-10)

    def test_source_enters_secondary_constraint(self):
        rho = np.sin(self.x)
        source = lambda _time: (rho, self.grid.zeros((3,)))
        solver = ProcaSolver(self.grid, self.parameters, source)
        state = solver.zero_state()
        expected = rho / (self.parameters.Z * self.parameters.mass**2)
        np.testing.assert_allclose(solver.phi(state), expected)

    def test_flat_stress_matches_energy_density(self):
        state = self.solver.zero_state()
        state.a[0] = np.cos(self.x)
        state.pi[1] = np.sin(self.x)
        rho, momentum, stress = self.solver.stress_energy_3p1(state)
        self.assertEqual(momentum.shape, (3,) + self.grid.shape)
        self.assertEqual(stress.shape, (3, 3) + self.grid.shape)
        np.testing.assert_allclose(rho, self.solver.energy_density(state)["total"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()

