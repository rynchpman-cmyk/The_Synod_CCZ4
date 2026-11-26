"""Close Nulloth's periodic coordinate-energy balance against geometric work."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.adm import inverse_metric
from causal_repair.grid import PeriodicGrid

try:
    from experiments.map_nulloth_production_regimes import _initial_state
    from experiments.measure_nulloth_weak_field_backreaction import _solver
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _initial_state
    from measure_nulloth_weak_field_backreaction import _solver


@dataclass(frozen=True)
class EnergyCase:
    name: str
    points: int
    kappa_gravity: float
    coordinate_energy_change: float
    rk_geometric_work: float
    absolute_energy_balance_error: float
    relative_energy_balance_error: float
    maximum_stage_flux_cancellation_error: float
    maximum_post_operator_energy_change: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class EnergyAtlas:
    duration: float
    dt: float
    cases: tuple[EnergyCase, ...]
    maximum_absolute_energy_balance_error: float
    maximum_relative_energy_balance_error: float
    maximum_stage_flux_cancellation_error: float
    maximum_post_operator_energy_change: float
    elevated_work_monotone_to_fine: int
    coarse_to_medium_work_error_reduction: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _geometric_work_rate(solver, values: tuple[np.ndarray, ...], time: float) -> float:
    state = solver._unpack(values, time)
    recovery = solver.recover(state)
    h, K = solver.ccz4.physical_geometry(state.geometry)
    h_inv, _, sqrt_h = inverse_metric(h)
    momentum_up = np.einsum(
        "ij...,j...->i...", h_inv, recovery.total_stress.momentum
    )
    stress_up = np.einsum(
        "ik...,jl...,kl...->ij...",
        h_inv,
        h_inv,
        recovery.total_stress.stress,
    )
    gradient_lapse = solver.grid.gradient(state.geometry.lapse)
    work_density = sqrt_h * (
        state.geometry.lapse
        * np.einsum("ij...,ij...->...", K, stress_up)
        - np.einsum("i...,i...->...", momentum_up, gradient_lapse)
    )
    return solver.grid.integrate(work_density)


def _accounted_step(solver, state, dt: float):
    values = solver._pack(state)
    time = state.time

    def stage(base, slope, factor, stage_time):
        candidate = solver._unpack(
            tuple(y + factor * dt * k for y, k in zip(base, slope)),
            stage_time,
        )
        candidate.geometry = solver.ccz4.project_algebraic(candidate.geometry)
        candidate.geometry.time = stage_time
        candidate.scalar.gradient = solver.grid.gradient(candidate.scalar.varphi)
        candidate = solver._project_to_master_manifold(candidate)
        return solver._pack(candidate)

    k1 = tuple(solver.rhs(time, values))
    y2 = stage(values, k1, 0.5, time + 0.5 * dt)
    k2 = tuple(solver.rhs(time + 0.5 * dt, y2))
    y3 = stage(values, k2, 0.5, time + 0.5 * dt)
    k3 = tuple(solver.rhs(time + 0.5 * dt, y3))
    y4 = stage(values, k3, 1.0, time + dt)
    k4 = tuple(solver.rhs(time + dt, y4))
    stage_values = (values, y2, y3, y4)
    stage_times = (time, time + 0.5 * dt, time + 0.5 * dt, time + dt)
    slopes = (k1, k2, k3, k4)
    weights = (1.0, 2.0, 2.0, 1.0)
    work_rates = tuple(
        _geometric_work_rate(solver, current, current_time)
        for current, current_time in zip(stage_values, stage_times)
    )
    flux_errors = tuple(
        abs(solver.grid.integrate(slope[11]) - work)
        for slope, work in zip(slopes, work_rates)
    )
    predicted_change = dt * sum(
        weight * work for weight, work in zip(weights, work_rates)
    ) / 6.0
    energy_before = solver.grid.integrate(state.matter.energy)
    result = solver.step(state, dt)
    actual_change = solver.grid.integrate(result.matter.energy) - energy_before
    return result, actual_change, predicted_change, max(flux_errors)


def _run_case(
    name: str,
    points: int,
    kappa_gravity: float,
    duration: float,
    dt: float,
) -> EnergyCase:
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid, kappa_gravity)
    state = _initial_state(solver, 0.18, 0.02, "growing")
    initial = solver.diagnostics(state)
    total_actual = 0.0
    total_work = 0.0
    maximum_flux_error = 0.0
    maximum_post_change = 0.0
    maximum_gravity = 0.0
    maximum_field = 0.0
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermodynamic = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    for _ in range(int(round(duration / dt))):
        state, actual, work, flux_error = _accounted_step(solver, state, dt)
        total_actual += actual
        total_work += work
        maximum_flux_error = max(maximum_flux_error, flux_error)
        maximum_post_change = max(maximum_post_change, abs(actual - work))
        diagnostic = solver.diagnostics(state)
        maximum_gravity = max(
            maximum_gravity,
            float(diagnostic["hamiltonian_l2"]),
            float(diagnostic["momentum_l2"]),
        )
        maximum_field = max(
            maximum_field,
            float(diagnostic["gauss_A_l2"]),
            float(diagnostic["gauss_B_l2"]),
            float(diagnostic["scalar_constraint_l2"]),
        )
        minimum_legendre = min(
            minimum_legendre,
            float(diagnostic["minimum_legendre_eigenvalue"]),
        )
        minimum_thermodynamic = min(
            minimum_thermodynamic,
            float(diagnostic["minimum_thermodynamic_eigenvalue"]),
        )
        maximum_recovery = max(
            maximum_recovery,
            float(diagnostic["recovery_maximum_residual"]),
        )
    final = solver.diagnostics(state)
    balance_error = abs(total_actual - total_work)
    number_drift = max(
        abs(float(final[key] - initial[key]))
        / max(abs(float(initial[key])), 1.0e-300)
        for key in ("baryon_mass", "carrier_number")
    )
    return EnergyCase(
        name,
        points,
        kappa_gravity,
        total_actual,
        total_work,
        balance_error,
        balance_error / max(abs(total_actual), abs(total_work), 1.0e-300),
        maximum_flux_error,
        maximum_post_change,
        maximum_gravity,
        maximum_field,
        number_drift,
        float(final["entropy_integral"] - initial["entropy_integral"]),
        minimum_legendre,
        minimum_thermodynamic,
        maximum_recovery,
    )


def run_atlas(*, duration: float = 0.05, dt: float = 0.01) -> EnergyAtlas:
    cases = (
        _run_case("weak-fine", 16, 1.0e-3, duration, dt),
        _run_case("elevated-coarse", 4, 0.5, duration, dt),
        _run_case("elevated-medium", 8, 0.5, duration, dt),
        _run_case("elevated-fine", 16, 0.5, duration, dt),
    )
    elevated = cases[1:]
    coarse, medium, fine = elevated
    coarse_error = abs(coarse.rk_geometric_work - fine.rk_geometric_work)
    medium_error = abs(medium.rk_geometric_work - fine.rk_geometric_work)
    return EnergyAtlas(
        duration,
        dt,
        cases,
        max(case.absolute_energy_balance_error for case in cases),
        max(case.relative_energy_balance_error for case in cases),
        max(case.maximum_stage_flux_cancellation_error for case in cases),
        max(case.maximum_post_operator_energy_change for case in cases),
        int(
            abs(coarse.rk_geometric_work - fine.rk_geometric_work)
            > abs(medium.rk_geometric_work - fine.rk_geometric_work)
        ),
        coarse_error / max(medium_error, 1.0e-300),
        max(case.maximum_gravitational_constraint_l2 for case in cases),
        max(case.maximum_field_constraint_l2 for case in cases),
        max(case.maximum_number_relative_drift for case in cases),
        min(case.entropy_change for case in cases),
        min(case.minimum_legendre for case in cases),
        min(case.minimum_thermodynamic for case in cases),
        max(case.maximum_recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(duration=arguments.duration, dt=arguments.dt)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
