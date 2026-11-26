"""Audit Nulloth covariance, discrete symmetries, and random recovery round trips."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
    NullothParameters,
)
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters, Theory3System
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class NullothSymmetryAudit:
    covariance_samples: int
    recovery_samples: int
    recovery_failures: int
    maximum_rotation_scalar_error: float
    maximum_rotation_vector_error: float
    maximum_rotation_tensor_error: float
    maximum_rotation_characteristic_error: float
    maximum_boost_stress_error: float
    maximum_scalar_sign_even_error: float
    maximum_scalar_sign_odd_error: float
    maximum_scalar_sign_characteristic_error: float
    maximum_material_time_reversal_error: float
    maximum_scalar_rhs_time_reversal_error: float
    maximum_proca_rhs_time_reversal_error: float
    maximum_recovery_primitive_error: float


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0.0:
        matrix[:, 0] *= -1.0
    return matrix


def _random_velocity(
    rng: np.random.Generator, maximum: float, minimum: float = 0.01
) -> np.ndarray:
    direction = rng.normal(size=3)
    return direction * rng.uniform(minimum, maximum) / np.linalg.norm(direction)


def _maximum(*values: np.ndarray | float) -> float:
    return max(float(np.max(np.abs(value))) for value in values)


def _lorentz_matrix(beta: np.ndarray) -> np.ndarray:
    beta_squared = float(beta @ beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    matrix = np.eye(4)
    matrix[0, 0] = gamma
    matrix[0, 1:] = -gamma * beta
    matrix[1:, 0] = -gamma * beta
    if beta_squared > 0.0:
        matrix[1:, 1:] += (gamma - 1.0) * np.outer(beta, beta) / beta_squared
    return matrix


def _material_symmetry_audit(
    rng: np.random.Generator, samples: int
) -> tuple[float, ...]:
    parameters = NullothParameters(
        entrainment_linear=0.03,
        entrainment_quadratic=0.02,
        scalar_baryon_coupling=0.7,
        scalar_carrier_coupling=1.3,
        scalar_counterflow_coupling=0.9,
        scalar_quartic=3.0,
    )
    master = NullothMasterFunction(parameters)
    h = np.eye(3).reshape(3, 3, 1)
    maxima = np.zeros(10)
    scalar_names = (
        "lambda_mf",
        "generalized_pressure",
        "B_N",
        "B_D",
        "entrainment",
        "chemical_N",
        "chemical_D",
        "coupling_coefficient",
        "effective_mass_squared",
    )
    even_names = scalar_names + (
        "thermodynamic_hessian_nn",
        "thermodynamic_hessian_nd",
        "thermodynamic_hessian_dd",
    )

    for _ in range(samples):
        density = rng.uniform(0.45, 0.85)
        carrier_density = rng.uniform(0.13, 0.23)
        internal = rng.uniform(0.12, 0.32)
        varphi = rng.uniform(0.03, 0.14)
        velocity = _random_velocity(rng, 0.18)
        carrier_velocity = _random_velocity(rng, 0.25)
        direction = rng.normal(size=3)
        rotation = _random_rotation(rng)
        fluid = FluidPrimitive(
            np.array([density]),
            np.array([internal]),
            velocity.reshape(3, 1),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([carrier_density]), carrier_velocity.reshape(3, 1)
        )
        state = master.evaluate(h, fluid, carrier, np.array([varphi]))
        rotated = master.evaluate(
            h,
            FluidPrimitive(
                fluid.baryon_density,
                fluid.specific_internal_energy,
                (rotation @ velocity).reshape(3, 1),
                fluid.sigma,
            ),
            CarrierPrimitive(
                carrier.number_density,
                (rotation @ carrier_velocity).reshape(3, 1),
            ),
            np.array([varphi]),
        )
        scalar_error = max(
            _maximum(getattr(rotated, name) - getattr(state, name))
            for name in scalar_names
        )
        vector_error = max(
            _maximum(
                rotated.stress.momentum[:, 0]
                - rotation @ state.stress.momentum[:, 0]
            ),
            _maximum(
                rotated.carrier_momentum_down[:, 0]
                - rotation @ state.carrier_momentum_down[:, 0]
            ),
            _maximum(
                rotated.source_current_up[:, 0]
                - rotation @ state.source_current_up[:, 0]
            ),
        )
        tensor_error = _maximum(
            rotated.stress.stress[:, :, 0]
            - rotation @ state.stress.stress[:, :, 0] @ rotation.T
        )
        audit = master.coupled_characteristic_audit(
            density,
            internal,
            velocity,
            carrier_density,
            carrier_velocity,
            varphi,
            direction,
        )
        rotated_audit = master.coupled_characteristic_audit(
            density,
            internal,
            rotation @ velocity,
            carrier_density,
            rotation @ carrier_velocity,
            varphi,
            rotation @ direction,
        )
        characteristic_rotation_error = _maximum(
            np.sort_complex(audit.speeds)
            - np.sort_complex(rotated_audit.speeds)
        )

        negative = master.evaluate(h, fluid, carrier, np.array([-varphi]))
        sign_even_error = max(
            _maximum(getattr(negative, name) - getattr(state, name))
            for name in even_names
        )
        sign_even_error = max(
            sign_even_error,
            _maximum(negative.stress.rho - state.stress.rho),
            _maximum(negative.stress.momentum - state.stress.momentum),
            _maximum(negative.stress.stress - state.stress.stress),
        )
        sign_odd_error = max(
            _maximum(
                negative.scalar_action_derivative
                + state.scalar_action_derivative
            ),
            _maximum(
                negative.scalar_equation_source + state.scalar_equation_source
            ),
        )
        negative_audit = master.coupled_characteristic_audit(
            density,
            internal,
            velocity,
            carrier_density,
            carrier_velocity,
            -varphi,
            direction,
        )
        sign_characteristic_error = _maximum(
            np.sort_complex(audit.speeds)
            - np.sort_complex(negative_audit.speeds)
        )

        reversed_state = master.evaluate(
            h,
            FluidPrimitive(
                fluid.baryon_density,
                fluid.specific_internal_energy,
                (-velocity).reshape(3, 1),
                fluid.sigma,
            ),
            CarrierPrimitive(
                carrier.number_density, (-carrier_velocity).reshape(3, 1)
            ),
            np.array([varphi]),
        )
        material_time_error = max(
            _maximum(reversed_state.stress.rho - state.stress.rho),
            _maximum(reversed_state.stress.stress - state.stress.stress),
            _maximum(reversed_state.stress.momentum + state.stress.momentum),
            _maximum(
                reversed_state.carrier_momentum_down
                + state.carrier_momentum_down
            ),
            _maximum(
                reversed_state.source_current_up + state.source_current_up
            ),
        )

        four_stress = np.empty((4, 4))
        four_stress[0, 0] = float(state.stress.rho[0])
        four_stress[0, 1:] = state.stress.momentum[:, 0]
        four_stress[1:, 0] = state.stress.momentum[:, 0]
        four_stress[1:, 1:] = state.stress.stress[:, :, 0]
        boost = _random_velocity(rng, 0.12)
        lorentz = _lorentz_matrix(boost)
        W_N = 1.0 / np.sqrt(1.0 - velocity @ velocity)
        W_D = 1.0 / np.sqrt(1.0 - carrier_velocity @ carrier_velocity)
        current_N = density * np.concatenate(([W_N], W_N * velocity))
        current_D = carrier_density * np.concatenate(
            ([W_D], W_D * carrier_velocity)
        )
        transformed_N = lorentz @ current_N
        transformed_D = lorentz @ current_D
        boosted = master.evaluate(
            h,
            FluidPrimitive(
                np.array([density]),
                np.array([internal]),
                (transformed_N[1:] / transformed_N[0]).reshape(3, 1),
                np.zeros(1),
            ),
            CarrierPrimitive(
                np.array([carrier_density]),
                (transformed_D[1:] / transformed_D[0]).reshape(3, 1),
            ),
            np.array([varphi]),
        )
        boosted_stress = np.empty((4, 4))
        boosted_stress[0, 0] = float(boosted.stress.rho[0])
        boosted_stress[0, 1:] = boosted.stress.momentum[:, 0]
        boosted_stress[1:, 0] = boosted.stress.momentum[:, 0]
        boosted_stress[1:, 1:] = boosted.stress.stress[:, :, 0]
        expected_boosted_stress = lorentz @ four_stress @ lorentz.T
        boost_error = _maximum(boosted_stress - expected_boosted_stress)

        maxima = np.maximum(
            maxima,
            (
                scalar_error,
                vector_error,
                tensor_error,
                characteristic_rotation_error,
                boost_error,
                sign_even_error,
                sign_odd_error,
                sign_characteristic_error,
                material_time_error,
                0.0,
            ),
        )
    return tuple(map(float, maxima[:9]))


def _rhs_time_reversal_audit() -> tuple[float, float]:
    grid = PeriodicGrid((16,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    h = np.eye(3).reshape(3, 3, 1)
    scalar_system = NullothScalarSystem(grid)
    scalar = scalar_system.initialize(
        0.03 * np.cos(coordinate),
        0.02 * np.sin(coordinate),
        gradient=grid.gradient(0.03 * np.cos(coordinate)),
    )
    source = 0.16 * scalar.varphi + 2.0 * scalar.varphi**3

    def scalar_rhs(state: NullothScalarState) -> tuple[np.ndarray, ...]:
        return scalar_system.rhs(
            h,
            grid.zeros(),
            np.ones(grid.shape),
            grid.zeros((3,)),
            state,
            0.16 * state.varphi + 2.0 * state.varphi**3,
        )

    original_scalar_rhs = scalar_rhs(scalar)
    reversed_scalar = NullothScalarState(
        scalar.varphi, -scalar.momentum, scalar.gradient
    )
    reversed_scalar_rhs = scalar_rhs(reversed_scalar)
    scalar_parities = (1.0, -1.0, 1.0)
    scalar_error = max(
        _maximum(reversed_value + parity * original_value)
        for reversed_value, original_value, parity in zip(
            reversed_scalar_rhs, original_scalar_rhs, scalar_parities
        )
    )

    vector_parameters = Theory3Parameters(
        Z_A=1.0,
        Z_B=1.0,
        mass_A=0.5,
        mass_B=0.6,
        mixing_mass_squared=0.2,
        target_speed=0.9,
        target_strength=0.0,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    vector_system = Theory3System(grid, vector_parameters)
    a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_A = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    a[1] = 0.03 * np.cos(coordinate)
    b[2] = -0.02 * np.sin(coordinate)
    pi_A[1] = 0.01 * np.sin(coordinate)
    pi_B[2] = 0.015 * np.cos(coordinate)
    longitudinal_A, longitudinal_B = vector_system.initial_longitudinal_charges(
        h, pi_A, pi_B
    )
    cleaning_A = 0.004 * np.cos(coordinate)
    cleaning_B = -0.003 * np.sin(coordinate)

    def vector_rhs(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return vector_system.rhs_fields(
            h,
            *values,
            grid.zeros(),
            grid.zeros((3,)),
            np.ones(grid.shape),
            grid.zeros((3,)),
            0.0,
        )

    vector_state = (
        a,
        pi_A,
        b,
        pi_B,
        longitudinal_A,
        longitudinal_B,
        cleaning_A,
        cleaning_B,
    )
    parities = (-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0)
    original_vector_rhs = vector_rhs(vector_state)
    reversed_vector_rhs = vector_rhs(
        tuple(parity * value for parity, value in zip(parities, vector_state))
    )
    vector_error = max(
        _maximum(reversed_value + parity * original_value)
        for reversed_value, original_value, parity in zip(
            reversed_vector_rhs, original_vector_rhs, parities
        )
    )
    return scalar_error, vector_error


def _recovery_round_trip(
    rng: np.random.Generator, samples: int
) -> tuple[int, float]:
    grid = PeriodicGrid((4,), (2.0,), method="fd2")
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
    )
    solver = NullothProductionSolver(grid, parameters, vector)
    failures = 0
    maximum_error = 0.0
    for _ in range(samples):
        density = rng.uniform(0.45, 0.85)
        carrier_density = rng.uniform(0.13, 0.23)
        internal = rng.uniform(0.12, 0.32)
        varphi = rng.uniform(-0.12, 0.12)
        velocity = _random_velocity(rng, 0.16, 0.0)
        carrier_velocity = _random_velocity(rng, 0.24, 0.0)
        fluid = FluidPrimitive(
            np.full(grid.shape, density),
            np.full(grid.shape, internal),
            np.broadcast_to(velocity[:, None], (3,) + grid.shape).copy(),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(
            np.full(grid.shape, carrier_density),
            np.broadcast_to(
                carrier_velocity[:, None], (3,) + grid.shape
            ).copy(),
        )
        try:
            state = solver.initialize(
                solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=varphi
            )
            recovered = solver.recover(state)
        except (FloatingPointError, ValueError):
            failures += 1
            continue
        comparisons = (
            (recovered.fluid.baryon_density, fluid.baryon_density),
            (
                recovered.fluid.specific_internal_energy,
                fluid.specific_internal_energy,
            ),
            (recovered.fluid.velocity, fluid.velocity),
            (recovered.carrier.number_density, carrier.number_density),
            (recovered.carrier.velocity, carrier.velocity),
        )
        maximum_error = max(
            maximum_error,
            max(
                _maximum(actual - expected)
                / max(_maximum(expected), 1.0)
                for actual, expected in comparisons
            ),
        )
    return failures, maximum_error


def run_symmetry_audit(
    *, seed: int = 20260723, covariance_samples: int = 32, recovery_samples: int = 64
) -> NullothSymmetryAudit:
    rng = np.random.default_rng(seed)
    material = _material_symmetry_audit(rng, covariance_samples)
    scalar_time_error, proca_time_error = _rhs_time_reversal_audit()
    failures, recovery_error = _recovery_round_trip(rng, recovery_samples)
    return NullothSymmetryAudit(
        covariance_samples,
        recovery_samples,
        failures,
        *material[:9],
        scalar_time_error,
        proca_time_error,
        recovery_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--covariance-samples", type=int, default=32)
    parser.add_argument("--recovery-samples", type=int, default=64)
    args = parser.parse_args()
    result = run_symmetry_audit(
        seed=args.seed,
        covariance_samples=args.covariance_samples,
        recovery_samples=args.recovery_samples,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

