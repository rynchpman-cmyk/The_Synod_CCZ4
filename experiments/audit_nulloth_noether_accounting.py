"""Audit Nulloth's internal exchanges and periodic Noether balances."""

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
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters, Theory3System
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class NullothNoetherAudit:
    scalar_exchange_samples: int
    maximum_scalar_material_energy_exchange_residual: float
    proca_energy_exchange_residual: float
    maximum_proca_momentum_exchange_residual: float
    maximum_initial_gauss_constraint: float
    periodic_baryon_rhs_integral: float
    periodic_carrier_charge_rhs_integral: float
    periodic_total_energy_rhs_integral: float
    maximum_periodic_total_momentum_rhs_integral: float
    maximum_periodic_passive_current_rhs_integral: float


def _maximum(value: np.ndarray | float) -> float:
    return float(np.max(np.abs(value)))


def _scalar_exchange_audit(
    rng: np.random.Generator, samples: int
) -> float:
    maximum_residual = 0.0
    h = np.eye(3).reshape(3, 3, 1)
    for _ in range(samples):
        parameters = NullothParameters(
            scalar_mass=rng.uniform(0.15, 0.7),
            scalar_quartic=rng.uniform(0.2, 4.0),
            scalar_baryon_coupling=rng.uniform(0.0, 1.2),
            scalar_carrier_coupling=rng.uniform(0.0, 7.0),
            scalar_counterflow_coupling=rng.uniform(0.0, 1.5),
        )
        master = NullothMasterFunction(parameters)
        velocity = rng.normal(size=3)
        velocity *= rng.uniform(0.0, 0.18) / max(np.linalg.norm(velocity), 1.0e-300)
        carrier_velocity = rng.normal(size=3)
        carrier_velocity *= rng.uniform(0.0, 0.24) / max(
            np.linalg.norm(carrier_velocity), 1.0e-300
        )
        fluid = FluidPrimitive(
            np.array([rng.uniform(0.45, 0.85)]),
            np.array([rng.uniform(0.12, 0.32)]),
            velocity.reshape(3, 1),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([rng.uniform(0.13, 0.23)]),
            carrier_velocity.reshape(3, 1),
        )
        phi = np.array([rng.uniform(-0.18, 0.18)])
        pi = np.array([rng.uniform(-0.2, 0.2)])
        state = master.evaluate(h, fluid, carrier, phi)
        phi_dot = -pi
        pi_dot = state.scalar_equation_source
        scalar_energy_dot = (
            pi * pi_dot + master.scalar_potential_derivative(phi) * phi_dot
        )
        material_interaction_energy_dot = -state.scalar_action_derivative * phi_dot
        maximum_residual = max(
            maximum_residual,
            _maximum(scalar_energy_dot + material_interaction_energy_dot),
        )
    return maximum_residual


def _integrated_vector_exchange() -> tuple[float, float, float]:
    grid = PeriodicGrid((32,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    h = np.eye(3).reshape(3, 3, 1)
    parameters = Theory3Parameters(
        Z_A=1.1,
        Z_B=0.9,
        mass_A=0.4,
        mass_B=0.7,
        mixing_mass_squared=0.06,
        target_speed=0.9,
        target_strength=0.0,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    system = Theory3System(grid, parameters)
    a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_A = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    a[1] = 0.02 * np.sin(coordinate)
    a[2] = 0.01 * np.cos(2.0 * coordinate)
    b[1] = -0.015 * np.cos(coordinate)
    b[2] = 0.012 * np.sin(2.0 * coordinate)
    pi_A[0] = 0.004 * np.sin(coordinate)
    pi_A[1] = 0.008 * np.cos(2.0 * coordinate)
    pi_B[0] = -0.006 * np.cos(coordinate)
    pi_B[2] = 0.007 * np.sin(2.0 * coordinate)
    charge = 0.18 + 0.01 * np.cos(coordinate)
    current = grid.zeros((3,))
    current[0] = 0.013 * np.sin(coordinate)
    current[1] = -0.011 * np.cos(coordinate)
    current[2] = 0.009 * np.sin(2.0 * coordinate)
    longitudinal_A, longitudinal_B = system.initial_longitudinal_charges(
        h, pi_A, pi_B
    )
    zero = grid.zeros()
    field_rhs = system.rhs_fields(
        h,
        a,
        pi_A,
        b,
        pi_B,
        longitudinal_A,
        longitudinal_B,
        zero,
        zero,
        charge,
        current,
        np.ones(grid.shape),
        grid.zeros((3,)),
        0.0,
    )
    charge_rhs = -grid.divergence(current)

    def stress(offset: float):
        fields = (a, pi_A, b, pi_B, longitudinal_A, longitudinal_B, zero, zero)
        advanced = tuple(
            field + offset * derivative
            for field, derivative in zip(fields, field_rhs)
        )
        return system.vector_stress(
            h, *advanced[:6], charge + offset * charge_rhs
        )

    epsilon = 1.0e-6
    plus = stress(epsilon)
    minus = stress(-epsilon)
    vector_energy_dot = (plus.rho - minus.rho) / (2.0 * epsilon)
    vector_momentum_dot = (plus.momentum - minus.momentum) / (2.0 * epsilon)
    drive_power, drive_force = system.drive_exchange(
        h, b, pi_B, charge, current
    )
    gauss = system.gauss_constraints(
        h, pi_A, pi_B, longitudinal_A, longitudinal_B
    )
    energy_residual = abs(float(grid.integrate(vector_energy_dot + drive_power)))
    momentum_residual = max(
        abs(float(grid.integrate(vector_momentum_dot[i] + drive_force[i])))
        for i in range(3)
    )
    gauss_residual = max(_maximum(value) for value in gauss)
    return energy_residual, momentum_residual, gauss_residual


def _periodic_total_balance() -> tuple[float, ...]:
    grid = PeriodicGrid((16,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    solver = NullothProductionSolver(grid, parameters, vector)
    velocity = grid.zeros((3,))
    velocity[0] = 0.035 * np.sin(coordinate)
    velocity[1] = 0.012 * np.cos(2.0 * coordinate)
    carrier_velocity = grid.zeros((3,))
    carrier_velocity[0] = -0.05 * np.cos(coordinate)
    carrier_velocity[2] = 0.016 * np.sin(2.0 * coordinate)
    fluid = FluidPrimitive(
        0.65 + 0.03 * np.cos(coordinate),
        0.2 + 0.015 * np.sin(coordinate),
        velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        0.18 + 0.012 * np.sin(coordinate), carrier_velocity
    )
    b = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    b[1] = 0.012 * np.sin(coordinate)
    pi_B[2] = 0.008 * np.cos(coordinate)
    state = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        b=b,
        pi_B=pi_B,
        varphi=0.035 * np.cos(coordinate) + 0.006 * np.sin(2.0 * coordinate),
        scalar_momentum=0.014 * np.sin(coordinate),
    )
    rhs = solver.rhs(state.time, solver._pack(state))

    def integral(group: np.ndarray) -> float:
        return abs(float(grid.integrate(group)))

    passive = max(integral(rhs[index]) for index in (12, 13))
    momentum = max(integral(rhs[10][component]) for component in range(3))
    return (
        integral(rhs[9]),
        integral(rhs[22]),
        integral(rhs[11]),
        momentum,
        passive,
    )


def run_audit(*, seed: int, scalar_exchange_samples: int) -> NullothNoetherAudit:
    rng = np.random.default_rng(seed)
    scalar_residual = _scalar_exchange_audit(rng, scalar_exchange_samples)
    proca_energy, proca_momentum, gauss = _integrated_vector_exchange()
    baryon, carrier, energy, momentum, passive = _periodic_total_balance()
    return NullothNoetherAudit(
        scalar_exchange_samples,
        scalar_residual,
        proca_energy,
        proca_momentum,
        gauss,
        baryon,
        carrier,
        energy,
        momentum,
        passive,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--scalar-exchange-samples", type=int, default=128)
    arguments = parser.parse_args()
    print(
        json.dumps(
            asdict(
                run_audit(
                    seed=arguments.seed,
                    scalar_exchange_samples=arguments.scalar_exchange_samples,
                )
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
