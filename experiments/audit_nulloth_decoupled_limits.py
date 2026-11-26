"""Audit the vacuum, canonical-scalar, and single-Proca limits of Nulloth."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction, NullothParameters
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters, Theory3System
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class DecoupledLimitAudit:
    vacuum_carrier_floors: tuple[float, ...]
    vacuum_ccz4_rhs_errors: tuple[float, ...]
    vacuum_error_ratios: tuple[float, ...]
    vacuum_minimum_scaling_order: float
    vacuum_smallest_floor_error: float
    scalar_frequency: float
    scalar_field_max_error: float
    scalar_momentum_max_error: float
    scalar_energy_relative_drift: float
    scalar_time_reversal_max_error: float
    condensed_static_source_error: float
    proca_A_to_B_cross_excitation: float
    proca_B_to_A_cross_excitation: float
    proca_relabeling_max_error: float


def _maximum_group_error(
    expected: tuple[np.ndarray, ...], actual: tuple[np.ndarray, ...]
) -> float:
    return max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(expected, actual)
    )


def _vacuum_ccz4_limit() -> tuple[tuple[float, ...], tuple[float, ...]]:
    grid = PeriodicGrid((8,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    floors = (4.0e-4, 2.0e-4, 1.0e-4)
    errors = []
    for floor in floors:
        parameters = NullothParameters(
            target_fraction=0.0,
            carrier_floor=floor,
            scalar_baryon_coupling=0.0,
            scalar_carrier_coupling=0.0,
            scalar_counterflow_coupling=0.0,
        )
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=parameters.relaxation_time,
            target_strength=0.0,
            gamma_ad=parameters.gamma_ad,
        )
        ccz4 = CCZ4Parameters(
            slicing="harmonic",
            evolve_shift=False,
            radiative_boundaries=False,
        )
        solver = NullothProductionSolver(grid, parameters, vector, ccz4)
        geometry = solver.ccz4.flat_state()
        geometry.conformal_factor = 1.0 + 0.01 * np.cos(coordinate)
        geometry.lapse = 1.0 + 0.02 * np.sin(coordinate)
        fluid = FluidPrimitive(
            np.full(grid.shape, solver.grhd.parameters.density_floor),
            np.full(grid.shape, solver.grhd.parameters.internal_energy_floor),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier = CarrierPrimitive(
            np.full(grid.shape, floor), grid.zeros((3,))
        )
        state = solver.initialize(geometry, fluid, carrier=carrier, varphi=0.0)
        actual = solver.rhs(state.time, solver._pack(state))[:9]
        zero_sources = (
            grid.zeros(),
            grid.zeros((3,)),
            grid.zeros((3, 3)),
        )
        expected = solver.ccz4.rhs(
            state.time, solver._pack(state)[:9], zero_sources
        )
        errors.append(_maximum_group_error(expected, actual))
    return floors, tuple(errors)


def _scalar_rhs(
    system: NullothScalarSystem,
    h: np.ndarray,
    mass_squared: float,
    state: NullothScalarState,
) -> NullothScalarState:
    dphi, dpi, dgradient = system.rhs(
        h,
        system.grid.zeros(),
        np.ones(system.grid.shape),
        system.grid.zeros((3,)),
        state,
        mass_squared * state.varphi,
    )
    return NullothScalarState(dphi, dpi, dgradient)


def _add_scaled(
    state: NullothScalarState, slope: NullothScalarState, scale: float
) -> NullothScalarState:
    return NullothScalarState(
        state.varphi + scale * slope.varphi,
        state.momentum + scale * slope.momentum,
        state.gradient + scale * slope.gradient,
    )


def _scalar_step(
    system: NullothScalarSystem,
    h: np.ndarray,
    mass_squared: float,
    state: NullothScalarState,
    dt: float,
) -> NullothScalarState:
    k1 = _scalar_rhs(system, h, mass_squared, state)
    k2 = _scalar_rhs(
        system, h, mass_squared, _add_scaled(state, k1, 0.5 * dt)
    )
    k3 = _scalar_rhs(
        system, h, mass_squared, _add_scaled(state, k2, 0.5 * dt)
    )
    k4 = _scalar_rhs(system, h, mass_squared, _add_scaled(state, k3, dt))
    return NullothScalarState(
        state.varphi
        + dt * (k1.varphi + 2.0 * k2.varphi + 2.0 * k3.varphi + k4.varphi) / 6.0,
        state.momentum
        + dt
        * (k1.momentum + 2.0 * k2.momentum + 2.0 * k3.momentum + k4.momentum)
        / 6.0,
        state.gradient
        + dt
        * (k1.gradient + 2.0 * k2.gradient + 2.0 * k3.gradient + k4.gradient)
        / 6.0,
    )


def _scalar_energy(
    grid: PeriodicGrid,
    state: NullothScalarState,
    mass_squared: float,
) -> float:
    return float(
        grid.integrate(
            0.5
            * (
                state.momentum**2
                + np.sum(state.gradient**2, axis=0)
                + mass_squared * state.varphi**2
            )
        )
    )


def _canonical_scalar_limit() -> tuple[float, float, float, float, float]:
    grid = PeriodicGrid((32,), (2.0 * np.pi,), method="spectral")
    system = NullothScalarSystem(grid)
    (coordinate,) = grid.coordinates()
    mode = 2
    mass = 0.4
    mass_squared = mass**2
    frequency = float(np.sqrt(mode**2 + mass_squared))
    amplitude = 1.0e-3
    initial = system.initialize(amplitude * np.cos(mode * coordinate), 0.0)
    state = initial.copy()
    initial_energy = _scalar_energy(grid, state, mass_squared)
    dt = 1.0e-3
    duration = 0.5
    for _ in range(round(duration / dt)):
        state = _scalar_step(system, np.eye(3).reshape(3, 3, 1), mass_squared, state, dt)
    expected_phi = amplitude * np.cos(frequency * duration) * np.cos(
        mode * coordinate
    )
    expected_pi = amplitude * frequency * np.sin(frequency * duration) * np.cos(
        mode * coordinate
    )
    field_error = float(np.max(np.abs(state.varphi - expected_phi)))
    momentum_error = float(np.max(np.abs(state.momentum - expected_pi)))
    energy_drift = abs(_scalar_energy(grid, state, mass_squared) - initial_energy) / initial_energy

    reversed_state = NullothScalarState(
        state.varphi.copy(), -state.momentum.copy(), state.gradient.copy()
    )
    for _ in range(round(duration / dt)):
        reversed_state = _scalar_step(
            system,
            np.eye(3).reshape(3, 3, 1),
            mass_squared,
            reversed_state,
            dt,
        )
    reversal_error = max(
        float(np.max(np.abs(reversed_state.varphi - initial.varphi))),
        float(np.max(np.abs(reversed_state.momentum + initial.momentum))),
        float(np.max(np.abs(reversed_state.gradient - initial.gradient))),
    )
    return frequency, field_error, momentum_error, energy_drift, reversal_error


def _condensed_static_error() -> float:
    parameters = NullothParameters(
        scalar_mass=0.4,
        scalar_quartic=2.0,
        scalar_carrier_coupling=12.0,
        scalar_baryon_coupling=0.0,
        scalar_counterflow_coupling=0.0,
    )
    master = NullothMasterFunction(parameters)
    h = np.eye(3).reshape(3, 3, 1)
    fluid = FluidPrimitive(
        np.array([0.8]), np.array([0.2]), np.zeros((3, 1)), np.zeros(1)
    )
    carrier = CarrierPrimitive(np.array([0.2]), np.zeros((3, 1)))
    symmetric = master.evaluate(h, fluid, carrier, np.zeros(1))
    amplitude = np.sqrt(
        -float(symmetric.effective_mass_squared[0]) / parameters.scalar_quartic
    )
    condensed = master.evaluate(h, fluid, carrier, np.array([amplitude]))
    return abs(float(condensed.scalar_equation_source[0]))


def _proca_decoupling_limit() -> tuple[float, float, float]:
    grid = PeriodicGrid((16,), (2.0 * np.pi,), method="spectral")
    parameters = Theory3Parameters(
        Z_A=1.0,
        Z_B=1.0,
        mass_A=0.5,
        mass_B=0.5,
        mixing_mass_squared=0.0,
        target_speed=0.9,
        target_strength=0.0,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    system = Theory3System(grid, parameters)
    (coordinate,) = grid.coordinates()
    h = np.eye(3).reshape(3, 3, 1)
    potential = grid.zeros((3,))
    potential[1] = 0.02 * np.sin(coordinate)
    momentum = grid.zeros((3,))
    momentum[1] = 0.01 * np.cos(coordinate)
    zero_vector = grid.zeros((3,))
    zero_scalar = grid.zeros()
    common = (
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_scalar,
        zero_vector,
        np.ones(grid.shape),
        zero_vector,
        0.0,
    )
    rhs_A = system.rhs_fields(
        h,
        potential,
        momentum,
        zero_vector,
        zero_vector,
        *common,
    )
    rhs_B = system.rhs_fields(
        h,
        zero_vector,
        zero_vector,
        potential,
        momentum,
        *common,
    )
    A_to_B = max(float(np.max(np.abs(rhs_A[index]))) for index in (2, 3, 5, 7))
    B_to_A = max(float(np.max(np.abs(rhs_B[index]))) for index in (0, 1, 4, 6))
    relabel_error = max(
        _maximum_group_error(
            (rhs_A[0], rhs_A[1], rhs_A[4], rhs_A[6]),
            (rhs_B[2], rhs_B[3], rhs_B[5], rhs_B[7]),
        ),
        _maximum_group_error(
            (rhs_A[2], rhs_A[3], rhs_A[5], rhs_A[7]),
            (rhs_B[0], rhs_B[1], rhs_B[4], rhs_B[6]),
        ),
    )
    return A_to_B, B_to_A, relabel_error


def run_decoupled_limit_audit() -> DecoupledLimitAudit:
    floors, vacuum_errors = _vacuum_ccz4_limit()
    vacuum_ratios = tuple(
        right / left for left, right in zip(vacuum_errors, vacuum_errors[1:])
    )
    vacuum_orders = tuple(-float(np.log2(ratio)) for ratio in vacuum_ratios)
    scalar = _canonical_scalar_limit()
    proca = _proca_decoupling_limit()
    return DecoupledLimitAudit(
        floors,
        vacuum_errors,
        vacuum_ratios,
        min(vacuum_orders),
        vacuum_errors[-1],
        *scalar,
        _condensed_static_error(),
        *proca,
    )


def main() -> None:
    print(json.dumps(asdict(run_decoupled_limit_audit()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
