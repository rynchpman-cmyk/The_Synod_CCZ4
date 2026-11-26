from __future__ import annotations

import unittest

import numpy as np

from causal_repair.grid import PeriodicGrid
from experiments.audit_nulloth_gauge_robustness import (
    _relative_spread,
    _solver as _gauge_solver,
)
from experiments.converge_nulloth_ccz4_constraints import _gravity_norm
from experiments.converge_nulloth_elevated_curvature import (
    _solver as _elevated_curvature_solver,
)
from experiments.audit_nulloth_elevated_curvature_gauge import (
    _solver as _elevated_gauge_solver,
)
from experiments.measure_nulloth_weak_field_backreaction import _l2, _solver
from experiments.measure_nulloth_periodic_tensor_response import _tt_l2
from experiments.measure_nulloth_source_pulses import _mode_amplitude


class NullothGravityTests(unittest.TestCase):
    def test_backreaction_norm_is_component_averaged_l2(self) -> None:
        field = np.array([3.0, 4.0])

        self.assertAlmostEqual(_l2(field), np.sqrt(12.5))

    def test_weak_field_solver_uses_registered_gravity_and_gauge(self) -> None:
        solver = _solver(
            PeriodicGrid((4,), (4.0,), method="spectral"),
            5.0e-4,
        )

        self.assertEqual(solver.ccz4.parameters.kappa_gravity, 5.0e-4)
        self.assertEqual(solver.ccz4.parameters.slicing, "harmonic")
        self.assertFalse(solver.ccz4.parameters.evolve_shift)
        self.assertFalse(solver.ccz4.parameters.radiative_boundaries)

    def test_gauge_trial_uses_one_plus_log_and_relative_spread(self) -> None:
        solver = _gauge_solver(
            PeriodicGrid((4,), (4.0,), method="spectral"),
            "1+log",
        )

        self.assertEqual(solver.ccz4.parameters.slicing, "1+log")
        self.assertAlmostEqual(_relative_spread(2.0, 1.5), 0.25)

    def test_constraint_ladder_uses_the_larger_ccz4_norm(self) -> None:
        diagnostic = {"hamiltonian_l2": 2.0e-6, "momentum_l2": 3.0e-6}

        self.assertEqual(_gravity_norm(diagnostic), 3.0e-6)

    def test_periodic_tt_projector_removes_conformal_scalar_mode(self) -> None:
        grid = PeriodicGrid((4, 4, 4), (4.0, 4.0, 4.0), method="spectral")
        x, _, _ = grid.coordinates()
        conformal_mode = grid.zeros((3, 3))
        for axis in range(3):
            conformal_mode[axis, axis] = np.cos(2.0 * np.pi * x / 4.0)
        tensor_mode = grid.zeros((3, 3))
        tensor_mode[1, 1] = np.cos(2.0 * np.pi * x / 4.0)
        tensor_mode[2, 2] = -tensor_mode[1, 1]

        self.assertLess(_tt_l2(conformal_mode, grid), 1.0e-14)
        self.assertGreater(_tt_l2(tensor_mode, grid), 0.5)

    def test_source_pulse_amplitude_uses_fourier_coefficient(self) -> None:
        grid = PeriodicGrid((8,), (4.0,), method="spectral")
        (coordinate,) = grid.coordinates()
        wavenumber = 2.0 * np.pi / 4.0
        field = 0.012 * np.cos(wavenumber * coordinate)

        self.assertAlmostEqual(
            _mode_amplitude(field, coordinate, wavenumber),
            0.012,
            places=14,
        )

    def test_elevated_curvature_solver_uses_registered_coupling(self) -> None:
        solver = _elevated_curvature_solver(
            PeriodicGrid((4,), (32.0,), method="spectral")
        )

        self.assertEqual(solver.ccz4.parameters.kappa_gravity, 0.5)
        self.assertEqual(solver.ccz4.parameters.slicing, "harmonic")

    def test_elevated_gauge_solver_accepts_one_plus_log(self) -> None:
        solver = _elevated_gauge_solver(
            PeriodicGrid((4,), (32.0,), method="spectral"),
            "1+log",
        )

        self.assertEqual(solver.ccz4.parameters.kappa_gravity, 0.5)
        self.assertEqual(solver.ccz4.parameters.slicing, "1+log")


if __name__ == "__main__":
    unittest.main()
