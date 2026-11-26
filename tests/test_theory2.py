import unittest

import numpy as np

from causal_repair.adm import flat_metric
from causal_repair.boundary import prevent_inflow, sommerfeld_rhs, zero_incoming_characteristic
from causal_repair.curved_proca import Theory2ProcaSystem
from causal_repair.coupled_theory2 import Theory2ADMState, Theory2PrescribedMatterADM
from causal_repair.adm import ADMSolver
from causal_repair.fixed_theory2 import Theory2FixedSolver, Theory2FixedState
from causal_repair.gauge import MovingPunctureGaugeState, moving_puncture_rhs
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_data import TwoLobeParameters, rotating_two_lobe_data
from causal_repair.matter import FluidPrimitive, IdealGasEOS, perfect_fluid_stress_energy
from causal_repair.theory2 import (
    Theory2Parameters,
    equilibrium_target,
    interaction_stress_energy,
    source_data,
    target_data,
)


class Theory2Tests(unittest.TestCase):
    def setUp(self):
        self.grid = PeriodicGrid((32,), (2.0 * np.pi,), "spectral")
        self.h = flat_metric(self.grid)
        self.parameters = Theory2Parameters(
            Z=2.0, mass=0.7, coupling=0.3, tau_star=0.8, ell_sigma=1.2
        )
        density = np.full(self.grid.shape, 0.4)
        internal = np.full(self.grid.shape, 0.2)
        velocity = self.grid.zeros((3,))
        sigma = np.zeros(self.grid.shape)
        self.fluid = FluidPrimitive(density, internal, velocity, sigma)

    def test_source_contract(self):
        astar = self.grid.zeros((3,))
        astar[1] = 0.5
        data = source_data(self.h, self.fluid, astar, self.parameters)
        coefficient = self.parameters.coupling * 0.4 / self.parameters.Z
        np.testing.assert_allclose(data.effective_mass_squared, self.parameters.mass**2 + coefficient)
        np.testing.assert_allclose(data.J_star_space_lower[1], coefficient * 0.5)
        np.testing.assert_allclose(data.rho_star, 0.0)

    def test_equilibrium_target_has_asinh_magnitude(self):
        (x,) = self.grid.coordinates()
        self.fluid.sigma[:] = np.sin(x)
        Aeq, _, q = equilibrium_target(
            self.grid, self.h, self.fluid, self.parameters.ell_sigma
        )
        magnitude = np.sqrt(np.sum(Aeq * Aeq, axis=0))
        np.testing.assert_allclose(magnitude, np.arcsinh(q), atol=1e-12)

    def test_target_phi_reconstruction_enforces_orthogonality(self):
        self.fluid.velocity[0] = 0.2
        astar = self.grid.zeros((3,))
        astar[0] = np.linspace(-1.0, 1.0, self.grid.shape[0])
        data = target_data(self.grid, self.h, self.fluid, astar, self.parameters)
        np.testing.assert_allclose(data.Phi_star, 0.2 * astar[0])
        np.testing.assert_allclose(data.target_constraint, 0.0)

    def test_interaction_stress_vanishes_at_match(self):
        astar = self.grid.zeros((3,))
        astar[2] = 0.3
        phi = np.zeros(self.grid.shape)
        rho, momentum, stress = interaction_stress_energy(
            self.h, self.fluid, astar.copy(), phi, astar, self.parameters
        )
        np.testing.assert_allclose(rho, 0.0)
        np.testing.assert_allclose(momentum, 0.0)
        np.testing.assert_allclose(stress, 0.0)

    def test_generalized_gauss_constraint_is_solved(self):
        astar = self.grid.zeros((3,))
        astar[0] = 0.1
        pi = self.grid.zeros((3,))
        (x,) = self.grid.coordinates()
        pi[0] = np.sin(2.0 * x)
        system = Theory2ProcaSystem(self.grid, self.parameters)
        phi = system.constrained_phi(self.h, pi, self.fluid, astar)
        residual = system.gauss_constraint(
            self.h, astar, pi, self.fluid, astar, phi
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_uniform_driven_proca_equilibrium(self):
        astar = self.grid.zeros((3,))
        astar[1] = 0.4
        sources = source_data(self.h, self.fluid, astar, self.parameters)
        a = (
            sources.target_coupling
            / (self.parameters.Z * sources.effective_mass_squared)
        )[None, ...] * astar
        system = Theory2ProcaSystem(self.grid, self.parameters)
        da, dpi = system.rhs_theory2(
            self.h,
            a,
            self.grid.zeros((3,)),
            self.fluid,
            astar,
            np.ones(self.grid.shape),
            self.grid.zeros((3,)),
        )
        np.testing.assert_allclose(da, 0.0, atol=1e-14)
        np.testing.assert_allclose(dpi, 0.0, atol=1e-14)

    def test_two_lobe_data_solve_gauss_and_current_is_causal(self):
        data = rotating_two_lobe_data(
            self.grid, self.parameters, TwoLobeParameters(peak_density=1e-3)
        )
        solver = Theory2FixedSolver(
            self.grid, data.fluid, self.parameters, h=data.geometry.h
        )
        residual = solver.proca.gauss_constraint(
            data.geometry.h,
            data.repair.a,
            data.repair.pi,
            data.fluid,
            data.repair.astar,
            data.phi,
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-14)
        _, margin = solver.minkowski_current(data.repair)
        self.assertTrue(np.all(margin >= -1e-15))
        self.assertFalse(data.gravitational_constraints_solved)

    def test_fluid_stress_at_rest(self):
        rho, momentum, stress = perfect_fluid_stress_energy(
            self.grid, self.h, self.fluid, IdealGasEOS(self.parameters.gamma_ad)
        )
        pressure = (self.parameters.gamma_ad - 1.0) * 0.4 * 0.2
        np.testing.assert_allclose(rho, 0.4 * 1.2)
        np.testing.assert_allclose(momentum, 0.0)
        for i in range(3):
            np.testing.assert_allclose(stress[i, i], pressure)

    def test_gauge_and_boundary_contracts(self):
        gauge = MovingPunctureGaugeState(
            np.ones(self.grid.shape), self.grid.zeros((3,)), self.grid.zeros((3,))
        )
        rhs = moving_puncture_rhs(
            self.grid,
            gauge,
            np.ones(self.grid.shape) * 0.2,
            self.grid.zeros((3,)),
        )
        np.testing.assert_allclose(rhs.lapse, -0.4)
        np.testing.assert_allclose(prevent_inflow(np.array([-0.2, 0.1]), 1), [0.0, 0.1])
        np.testing.assert_allclose(
            sommerfeld_rhs(np.array([2.0]), np.array([0.5]), np.array([4.0]), 1.0),
            [-0.75],
        )
        outgoing, incoming = zero_incoming_characteristic(np.array([1.0]), np.array([2.0]))
        np.testing.assert_allclose(outgoing, 1.0)
        np.testing.assert_allclose(incoming, 0.0)

    def test_prescribed_matter_adm_vacuum_fixed_point(self):
        vacuum = FluidPrimitive(
            np.zeros(self.grid.shape),
            np.zeros(self.grid.shape),
            self.grid.zeros((3,)),
            np.zeros(self.grid.shape),
        )
        bridge = Theory2PrescribedMatterADM(
            ADMSolver(self.grid), vacuum, self.parameters
        )
        geometry = bridge.adm.flat_state()
        state = Theory2ADMState(
            geometry.h,
            geometry.K,
            self.grid.zeros((3,)),
            self.grid.zeros((3,)),
            self.grid.zeros((3,)),
        )
        evolved = bridge.step(state, 0.01)
        np.testing.assert_allclose(evolved.h, state.h)
        np.testing.assert_allclose(evolved.K, 0.0)
        self.assertEqual(bridge.constraints(evolved)["hamiltonian_l2"], 0.0)


if __name__ == "__main__":
    unittest.main()
