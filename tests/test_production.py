import unittest
import tempfile
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.production import (
    ProductionSolver,
    load_production_state,
    save_production_state,
)
from causal_repair.theory2 import Theory2Parameters


class ProductionTests(unittest.TestCase):
    def test_independent_longitudinal_variable_reconstructs_gauss_solution(self):
        grid = PeriodicGrid((32,), (1.0,), "fd2")
        solver = ProductionSolver(
            grid, theory=Theory2Parameters(mass=0.5, coupling=30.0)
        )
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            np.where(x < 0.5, 1.0, 0.125),
            np.full(grid.shape, 0.2), grid.zeros((3,)), grid.zeros(),
        )
        astar = grid.zeros((3,))
        astar[0] = 0.1
        pi = grid.zeros((3,))
        pi[0] = 0.02*np.sin(2.0*np.pi*x)
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, astar.copy(), pi, astar
        )
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        recovered = solver.recover(state)
        old_phi = solver.proca.constrained_phi(h, pi, recovered.fluid, astar)
        new_phi = solver.proca.phi_from_longitudinal(
            h, state.longitudinal, recovered.fluid, astar
        )
        np.testing.assert_allclose(new_phi, old_phi, atol=2e-13)
        np.testing.assert_allclose(
            solver.proca.hyperbolic_gauss_constraint(
                h, pi, state.longitudinal
            ),
            0.0, atol=2e-13,
        )

    def test_gauss_cleaner_damps_uniform_violation(self):
        grid = PeriodicGrid((16,), (1.0,), "fd2")
        solver = ProductionSolver(grid)
        state = solver.vacuum_state()
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        state.longitudinal.fill(0.2)
        state.gauss_cleaning.fill(0.3)
        recovery = solver.recover(state)
        _, _, dlongitudinal, dcleaning = solver.proca.rhs_hyperbolic(
            h, state.a, state.pi, state.longitudinal,
            state.gauss_cleaning, recovery.fluid, state.astar,
            state.geometry.lapse, state.geometry.shift,
            cleaning_damping=2.0,
        )
        np.testing.assert_allclose(dlongitudinal, 0.0, atol=1e-14)
        np.testing.assert_allclose(dcleaning, -0.8, atol=1e-14)

    def test_coupled_speed_bound_increases_with_target_mismatch(self):
        grid = PeriodicGrid((16,), (1.0,), "fd2")
        solver = ProductionSolver(
            grid, theory=Theory2Parameters(coupling=30.0)
        )
        state = solver.vacuum_state()
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        recovery = solver.recover(state)
        baseline = solver._coupled_speed_bound(
            recovery, h, state.geometry.lapse, state.geometry.shift, 0,
            state.a, state.astar, state.longitudinal,
        )
        mismatched = state.astar.copy()
        mismatched[0] = 1.0
        deformed = solver._coupled_speed_bound(
            recovery, h, state.geometry.lapse, state.geometry.shift, 0,
            state.a, mismatched, state.longitudinal,
        )
        self.assertGreater(float(np.min(deformed)), float(np.max(baseline)))

    def test_coupled_discontinuity_preserves_recovery_and_gauss_constraint(self):
        grid = PeriodicGrid((32,), (1.0,), "fd2")
        solver = ProductionSolver(
            grid,
            theory=Theory2Parameters(mass=0.5, coupling=30.0, tau_star=0.5),
            production=None,
        )
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            np.where(x < 0.5, 1.0, 0.125),
            np.where(x < 0.5, 1.5, 1.2),
            grid.zeros((3,)), grid.zeros(),
        )
        astar = grid.zeros((3,))
        astar[0] = 0.5
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, 0.5*astar,
            grid.zeros((3,)), astar,
        )
        initial_baryons = float(np.sum(state.matter.D)*grid.cell_volume)
        for _ in range(3):
            state = solver.step(state, min(2.0e-5, solver.recommended_dt(state)))
        diagnostics = solver.diagnostics(state)
        self.assertEqual(diagnostics["recovery_failures"], 0)
        self.assertLess(diagnostics["gauss_l2"], 1.0e-10)
        self.assertAlmostEqual(
            float(np.sum(state.matter.D)*grid.cell_volume), initial_baryons,
            places=12,
        )

    def test_atmosphere_vacuum_is_exact_fixed_point(self):
        grid = PeriodicGrid((8,), (1.0,), "fd4")
        solver = ProductionSolver(grid)
        state = solver.vacuum_state()
        evolved = solver.step(state, 1.0e-4)
        diagnostics = solver.diagnostics(evolved)
        self.assertEqual(diagnostics["hamiltonian_l2"], 0.0)
        self.assertEqual(diagnostics["momentum_l2"], 0.0)
        self.assertEqual(diagnostics["gauss_l2"], 0.0)
        self.assertEqual(diagnostics["recovery_failures"], 0)

    def test_initial_total_state_recovers_original_fluid(self):
        grid = PeriodicGrid((8,), (1.0,), "fd4")
        solver = ProductionSolver(grid)
        density = np.full(grid.shape, 1.0e-3)
        internal = np.full(grid.shape, 0.1)
        velocity = grid.zeros((3,))
        velocity[0] = 0.1
        fluid = FluidPrimitive(density, internal, velocity, grid.zeros())
        astar = grid.zeros((3,))
        astar[1] = 0.02
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, astar.copy(), grid.zeros((3,)), astar
        )
        recovered = solver.recover(state)
        np.testing.assert_allclose(recovered.fluid.baryon_density, density, atol=1e-14)
        np.testing.assert_allclose(recovered.fluid.specific_internal_energy, internal, atol=1e-13)
        np.testing.assert_allclose(recovered.fluid.velocity, velocity, atol=1e-13)
        self.assertEqual(recovered.report.failed_cells, 0)

    def test_total_matter_flux_is_periodically_conservative(self):
        grid = PeriodicGrid((16,), (1.0,), "fd4")
        solver = ProductionSolver(grid)
        (x,) = grid.coordinates()
        velocity = grid.zeros((3,))
        velocity[0] = 0.1
        fluid = FluidPrimitive(
            1.0e-3 * (1.0 + 0.1 * np.sin(2.0 * np.pi * x)),
            np.full(grid.shape, 0.1),
            velocity,
            grid.zeros(),
        )
        astar = grid.zeros((3,))
        astar[1] = 0.01 * np.cos(2.0 * np.pi * x)
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, astar.copy(), grid.zeros((3,)), astar
        )
        recovery = solver.recover(state)
        h, K = solver.ccz4.physical_geometry(state.geometry)
        dD, dmomentum, denergy, dtracer = solver._total_matter_rhs(
            state, recovery, h, K
        )
        for derivative in (dD, dmomentum, denergy, dtracer):
            self.assertLess(abs(np.sum(derivative)), 1e-12)

    def test_checkpoint_round_trip_is_exact(self):
        grid = PeriodicGrid((8,), (1.0,), "fd4")
        solver = ProductionSolver(grid)
        state = solver.vacuum_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restart.npz"
            save_production_state(path, state, metadata={"case": "vacuum"})
            loaded, metadata = load_production_state(path)
        self.assertEqual(metadata["case"], "vacuum")
        for original, recovered in zip(solver._pack(state), solver._pack(loaded)):
            np.testing.assert_array_equal(original, recovered)


if __name__ == "__main__":
    unittest.main()
