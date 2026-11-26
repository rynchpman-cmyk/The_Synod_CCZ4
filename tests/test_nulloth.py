from __future__ import annotations

import unittest

import numpy as np

from experiments.run_nulloth_homogeneous_growth import run_homogeneous_growth
from experiments.audit_nulloth_coupled_symbol import run_symbol_scan
from experiments.discover_nulloth_linear_modes import discover_linear_modes
from experiments.qualify_nulloth_discovery_domain import (
    qualify_discovery_domain,
)

from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction, NullothParameters
from causal_repair.nulloth_scalar import NullothScalarSystem
from causal_repair.grid import PeriodicGrid
from causal_repair.production33 import Theory33ProductionSolver
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import (
    CarrierPrimitive,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


class _TinyPeriodicGrid:
    def __init__(self, points: int = 16, length: float = 2.0 * np.pi) -> None:
        self.shape = (points,)
        self.ndim = 1
        self.spacing = (length / points,)
        self.cell_volume = self.spacing[0]
        self.is_periodic = True

    def zeros(self, prefix: tuple[int, ...] = ()) -> np.ndarray:
        return np.zeros(prefix + self.shape)

    def derivative(self, value: np.ndarray, axis: int) -> np.ndarray:
        if axis >= self.ndim:
            return np.zeros_like(value)
        array_axis = value.ndim - self.ndim + axis
        return (
            np.roll(value, -1, axis=array_axis)
            - np.roll(value, 1, axis=array_axis)
        ) / (2.0 * self.spacing[axis])

    def gradient(self, value: np.ndarray) -> np.ndarray:
        result = self.zeros((3,))
        result[0] = self.derivative(value, 0)
        return result

    def divergence(self, vector: np.ndarray) -> np.ndarray:
        return self.derivative(vector[0], 0)


class NullothConstitutiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = np.eye(3).reshape(3, 3, 1)
        self.fluid = FluidPrimitive(
            np.array([0.8]),
            np.array([0.2]),
            np.array([[0.08], [0.0], [0.0]]),
            np.array([0.0]),
        )
        self.carrier = CarrierPrimitive(
            np.array([0.2]),
            np.array([[-0.04], [0.0], [0.0]]),
        )

    def test_phi_zero_is_theory33_regression(self) -> None:
        base_parameters = Theory33MasterParameters(
            entrainment_linear=0.03,
            entrainment_quadratic=0.02,
        )
        nulloth_parameters = NullothParameters(
            entrainment_linear=0.03,
            entrainment_quadratic=0.02,
            scalar_baryon_coupling=0.7,
            scalar_carrier_coupling=1.3,
            scalar_counterflow_coupling=0.9,
        )
        base = Theory33MasterFunction(base_parameters).evaluate(
            self.h, self.fluid, self.carrier
        )
        nulloth = NullothMasterFunction(nulloth_parameters).evaluate(
            self.h, self.fluid, self.carrier, np.array([0.0])
        )
        names = (
            "lambda_mf",
            "generalized_pressure",
            "B_N",
            "B_D",
            "entrainment",
            "chemical_N",
            "chemical_D",
            "thermodynamic_hessian_nn",
            "thermodynamic_hessian_nd",
            "thermodynamic_hessian_dd",
            "carrier_momentum_down",
            "carrier_momentum_normal",
            "source_charge_eulerian",
            "source_current_up",
        )
        for name in names:
            np.testing.assert_allclose(getattr(nulloth, name), getattr(base, name))
        for name in ("rho", "momentum", "stress", "trace"):
            np.testing.assert_allclose(
                getattr(nulloth.stress, name), getattr(base.stress, name)
            )

    def test_scalar_action_derivative_matches_finite_difference(self) -> None:
        master = NullothMasterFunction(
            NullothParameters(
                scalar_baryon_coupling=0.2,
                scalar_carrier_coupling=0.8,
                scalar_counterflow_coupling=0.4,
            )
        )
        state = master.evaluate(
            self.h, self.fluid, self.carrier, np.array([0.3])
        )
        n2 = self.fluid.baryon_density**2
        d2 = self.carrier.number_density**2
        gamma = state.relative_lorentz_factor
        x2 = self.fluid.baryon_density * self.carrier.number_density * gamma
        entropy = master.specific_entropy(
            self.fluid.baryon_density, self.fluid.specific_internal_energy
        )
        epsilon = 1.0e-6
        plus = master.lambda_from_invariants(
            n2, d2, x2, entropy, np.array([0.3 + epsilon])
        )
        minus = master.lambda_from_invariants(
            n2, d2, x2, entropy, np.array([0.3 - epsilon])
        )
        numerical = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(
            numerical, state.scalar_action_derivative, rtol=2.0e-7, atol=2.0e-9
        )

    def test_homogeneous_condensed_stationary_point(self) -> None:
        parameters = NullothParameters(
            scalar_mass=0.4,
            scalar_quartic=2.0,
            scalar_carrier_coupling=12.0,
            scalar_carrier_scale=1.0,
            scalar_baryon_coupling=0.0,
            scalar_counterflow_coupling=0.0,
        )
        master = NullothMasterFunction(parameters)
        zero = master.evaluate(
            self.h, self.fluid, self.carrier, np.array([0.0])
        )
        self.assertLess(float(zero.effective_mass_squared[0]), 0.0)
        amplitude = np.sqrt(
            -float(zero.effective_mass_squared[0]) / parameters.scalar_quartic
        )
        branch = master.evaluate(
            self.h, self.fluid, self.carrier, np.array([amplitude])
        )
        self.assertAlmostEqual(float(branch.scalar_equation_source[0]), 0.0, places=11)

    def test_homogeneous_scalar_exchange_conserves_frozen_energy(self) -> None:
        master = NullothMasterFunction(
            NullothParameters(
                scalar_mass=0.4,
                scalar_quartic=1.7,
                scalar_carrier_coupling=4.0,
                scalar_counterflow_coupling=0.0,
            )
        )
        phi = np.array([0.23])
        pi = np.array([-0.11])
        state = master.evaluate(self.h, self.fluid, self.carrier, phi)
        phi_dot = -pi
        pi_dot = state.scalar_equation_source
        interaction_derivative = -state.scalar_action_derivative
        energy_derivative = (
            pi * pi_dot
            + master.scalar_potential_derivative(phi) * phi_dot
            + interaction_derivative * phi_dot
        )
        np.testing.assert_allclose(energy_derivative, 0.0, atol=2.0e-14)

    def test_cell_prediction_matches_tensor_projection(self) -> None:
        master = NullothMasterFunction(
            NullothParameters(
                scalar_baryon_coupling=0.2,
                scalar_carrier_coupling=0.5,
                scalar_counterflow_coupling=0.3,
            )
        )
        h = np.array([[1.2, 0.03, 0.0], [0.03, 0.9, 0.0], [0.0, 0.0, 1.1]])
        primitive = np.array(
            [np.log(0.8), 0.1, 0.02, 0.0, np.log(0.2), -0.04, 0.01, 0.0]
        )
        entropy = 0.3
        varphi = 0.25
        sqrt_h = float(np.sqrt(np.linalg.det(h)))
        predicted = master.cell_predictions(
            primitive, h, sqrt_h, entropy, varphi
        )
        self.assertEqual(predicted.shape, (9,))
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_full_coupled_principal_symbol_has_complete_causal_basis(self) -> None:
        master = NullothMasterFunction(
            NullothParameters(
                scalar_baryon_coupling=0.2,
                scalar_carrier_coupling=0.5,
                scalar_counterflow_coupling=0.3,
            )
        )
        audit = master.coupled_characteristic_audit(
            0.8,
            0.2,
            np.array([0.08, 0.01, 0.0]),
            0.2,
            np.array([-0.04, 0.0, 0.01]),
            0.1,
            np.array([1.0, 1.0, 0.2]),
        )
        self.assertEqual(audit.time_matrix.shape, (25, 25))
        self.assertEqual(audit.space_matrix.shape, (25, 25))
        self.assertEqual(audit.eigenvector_rank, 25)
        self.assertTrue(audit.complete_eigenbasis)
        self.assertTrue(audit.strongly_hyperbolic)
        self.assertTrue(audit.causal)
        self.assertGreater(audit.scalar_material_coupling_norm, 0.0)
        self.assertEqual(audit.constraint_reduced_eigenvector_rank, 25)
        self.assertLess(audit.constraint_reduction_spectrum_error, 1.0e-10)
        self.assertEqual(np.count_nonzero(np.isclose(audit.speeds, -1.0)), 7)
        self.assertEqual(np.count_nonzero(np.isclose(audit.speeds, 1.0)), 7)

    def test_full_symbol_scalar_coupling_column_vanishes_on_symmetric_branch(
        self,
    ) -> None:
        master = NullothMasterFunction(
            NullothParameters(
                scalar_baryon_coupling=0.2,
                scalar_carrier_coupling=0.5,
                scalar_counterflow_coupling=0.3,
            )
        )
        audit = master.coupled_characteristic_audit(
            0.8,
            0.2,
            np.zeros(3),
            0.2,
            np.zeros(3),
            0.0,
        )
        self.assertLess(audit.scalar_material_coupling_norm, 1.0e-10)
        self.assertTrue(audit.causal)


class NullothScalarTests(unittest.TestCase):
    def test_homogeneous_rhs_uses_effective_source(self) -> None:
        grid = _TinyPeriodicGrid()
        system = NullothScalarSystem(grid)
        state = system.initialize(0.2, 0.0)
        h = np.zeros((3, 3) + grid.shape)
        for index in range(3):
            h[index, index] = 1.0
        source = np.full(grid.shape, -0.3)
        dphi, dpi, dgradient = system.rhs(
            h,
            np.zeros(grid.shape),
            np.ones(grid.shape),
            np.zeros((3,) + grid.shape),
            state,
            source,
        )
        np.testing.assert_allclose(dphi, 0.0)
        np.testing.assert_allclose(dpi, source)
        np.testing.assert_allclose(dgradient, 0.0)


class NullothProductionTests(unittest.TestCase):
    @staticmethod
    def _parameters() -> tuple[Theory33MasterParameters, NullothParameters]:
        shared = dict(
            entrainment_linear=0.01,
            entrainment_quadratic=0.005,
            relaxation_time=1.0,
        )
        return (
            Theory33MasterParameters(**shared),
            NullothParameters(
                **shared,
                scalar_baryon_coupling=0.0,
                scalar_carrier_coupling=0.0,
                scalar_counterflow_coupling=0.0,
            ),
        )

    @staticmethod
    def _initial_data(
        grid: PeriodicGrid,
    ) -> tuple[FluidPrimitive, CarrierPrimitive]:
        (x,) = grid.coordinates()
        fluid = FluidPrimitive(
            0.8 * (1.0 + 0.01 * np.sin(np.pi * x)),
            np.full(grid.shape, 0.2),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(
            0.2 * (1.0 + 0.01 * np.cos(np.pi * x)),
            grid.zeros((3,)),
        )
        return fluid, carrier

    def _solvers(
        self, grid: PeriodicGrid
    ) -> tuple[Theory33ProductionSolver, NullothProductionSolver]:
        base_parameters, nulloth_parameters = self._parameters()
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=base_parameters.relaxation_time,
            gamma_ad=base_parameters.gamma_ad,
        )
        return (
            Theory33ProductionSolver(grid, base_parameters, vector),
            NullothProductionSolver(grid, nulloth_parameters, vector),
        )

    def test_smallest_real_periodic_grid_initializes_and_recovers(self) -> None:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        _, solver = self._solvers(grid)
        fluid, carrier = self._initial_data(grid)
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=0.0
        )
        recovery = solver.recover(state)
        self.assertEqual(recovery.report.failed_cells, 0)
        self.assertEqual(len(solver._pack(state)), 27)
        self.assertEqual(state.scalar.gradient.shape, (3,) + grid.shape)

    def test_zero_scalar_first_24_rhs_groups_match_theory33(self) -> None:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        base, nulloth = self._solvers(grid)
        fluid, carrier = self._initial_data(grid)
        base_state = base.initialize(
            base.ccz4.flat_state(), fluid, carrier=carrier
        )
        nulloth_state = nulloth.initialize(
            nulloth.ccz4.flat_state(), fluid, carrier=carrier, varphi=0.0
        )
        nulloth_recovery = nulloth.recover(nulloth_state)
        self.assertEqual(nulloth_recovery.report.failed_cells, 0)
        base_rhs = base.rhs(base_state.time, base._pack(base_state))
        nulloth_rhs = nulloth.rhs(
            nulloth_state.time, nulloth._pack(nulloth_state)
        )
        self.assertEqual(len(base_rhs), 24)
        self.assertEqual(len(nulloth_rhs), 27)
        for group, (expected, actual) in enumerate(
            zip(base_rhs, nulloth_rhs[:24])
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=2.0e-11,
                atol=2.0e-12,
                err_msg=f"RHS group {group}",
            )

    def test_small_step_first_24_groups_converge_as_scalar_vanishes(self) -> None:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        base, _ = self._solvers(grid)
        fluid, carrier = self._initial_data(grid)
        dt = 1.0e-6
        base_state = base.initialize(
            base.ccz4.flat_state(), fluid, carrier=carrier
        )
        base_after = base.step(base_state, dt)
        base_values = base._pack(base_after)
        errors = []
        for amplitude in (2.0e-3, 1.0e-3, 5.0e-4):
            _, nulloth = self._solvers(grid)
            state = nulloth.initialize(
                nulloth.ccz4.flat_state(),
                fluid,
                carrier=carrier,
                varphi=amplitude,
            )
            after = nulloth.step(state, dt)
            values = nulloth._pack(after)
            errors.append(
                max(
                    float(np.max(np.abs(actual - expected)))
                    for expected, actual in zip(base_values, values[:24])
                )
            )
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[1] / errors[0], 0.3)
        self.assertLess(errors[2] / errors[1], 0.3)

    def test_homogeneous_scalarization_growth_rate(self) -> None:
        result = run_homogeneous_growth()
        self.assertLess(result.effective_mass_squared, 0.0)
        self.assertGreater(result.final_amplitude, result.initial_amplitude)
        self.assertLess(result.relative_error, 1.0e-6)

    def test_quick_full_symbol_scan_has_complete_repeated_eigenspaces(self) -> None:
        result = run_symbol_scan(quick=True)
        self.assertEqual(result.samples, 12)
        self.assertEqual(result.failures, 0)
        self.assertTrue(result.sampled_domain_passed)
        self.assertIsNone(result.first_failure)
        self.assertGreater(result.maximum_scalar_material_coupling_norm, 0.0)

    def test_discovery_domain_quick_qualification(self) -> None:
        result = qualify_discovery_domain(random_samples=8, quick=True)
        self.assertEqual(result.failures, 0)
        self.assertGreater(result.activated_samples, 0)
        self.assertTrue(result.sampled_domain_qualified)
        self.assertLess(result.maximum_eigenspace_residual, 1.0e-10)
        self.assertLess(result.maximum_reduction_spectrum_error, 1.0e-10)

    def test_linear_mode_discovery_accepts_predicted_growing_modes(self) -> None:
        result = discover_linear_modes(
            points=16,
            duration=0.2,
            dt=1.0e-2,
            carrier_densities=(0.16,),
            mode_numbers=(0, 1),
        )
        self.assertEqual(result.tested_states, 2)
        self.assertEqual(result.unstable_modes, 2)
        self.assertEqual(result.accepted_candidates, 2)
        self.assertTrue(all(candidate.accepted for candidate in result.candidates))


if __name__ == "__main__":
    unittest.main()
