"""Converge an elevated-curvature periodic Nulloth CMC trajectory."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters

try:
    from experiments.map_nulloth_production_regimes import _initial_state
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _initial_state


@dataclass(frozen=True)
class CurvatureCase:
    points: int
    maximum_gravitational_constraint_l2: float
    final_gravitational_constraint_l2: float
    final_scalar_amplitude: float
    final_metric_deviation_l2: float
    final_extrinsic_curvature_l2: float
    minimum_lapse: float
    minimum_metric_eigenvalue: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class CurvatureAtlas:
    kappa_gravity: float
    duration: float
    dt: float
    cases: tuple[CurvatureCase, ...]
    gravitational_constraint_monotone_with_resolution: int
    coarse_to_medium_constraint_reduction: float
    medium_to_fine_constraint_reduction: float
    coarse_to_medium_scalar_error_reduction: float
    coarse_to_medium_metric_error_reduction: float
    finest_gravitational_constraint_l2: float
    finest_metric_deviation_l2: float
    finest_extrinsic_curvature_l2: float
    minimum_lapse: float
    minimum_metric_eigenvalue: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _solver(grid: PeriodicGrid) -> NullothProductionSolver:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
    )
    geometry = CCZ4Parameters(
        kappa_gravity=0.5,
        slicing="harmonic",
        evolve_shift=False,
        radiative_boundaries=False,
    )
    return NullothProductionSolver(grid, parameters, vector, geometry)


def _gravity_norm(diagnostic: dict[str, float]) -> float:
    return max(
        float(diagnostic["hamiltonian_l2"]),
        float(diagnostic["momentum_l2"]),
    )


def _run_case(points: int, duration: float, dt: float) -> CurvatureCase:
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid)
    state = _initial_state(solver, 0.18, 0.02, "growing")
    initial = solver.diagnostics(state)
    maximum_gravity = _gravity_norm(initial)
    maximum_field = max(
        float(initial["gauss_A_l2"]),
        float(initial["gauss_B_l2"]),
        float(initial["scalar_constraint_l2"]),
    )
    minimum_lapse = float(np.min(state.geometry.lapse))
    minimum_metric_eigenvalue = np.inf
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermodynamic = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    for _ in range(int(round(duration / dt))):
        state = solver.step(state, dt)
        diagnostic = solver.diagnostics(state)
        physical_metric, _ = solver.ccz4.physical_geometry(state.geometry)
        metric_cells = np.moveaxis(physical_metric, (0, 1), (-2, -1))
        minimum_metric_eigenvalue = min(
            minimum_metric_eigenvalue,
            float(np.min(np.linalg.eigvalsh(metric_cells))),
        )
        minimum_lapse = min(minimum_lapse, float(np.min(state.geometry.lapse)))
        maximum_gravity = max(maximum_gravity, _gravity_norm(diagnostic))
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
    physical_metric, extrinsic_curvature = solver.ccz4.physical_geometry(
        state.geometry
    )
    flat_metric = np.zeros_like(physical_metric)
    for axis in range(3):
        flat_metric[axis, axis] = 1.0
    number_drift = max(
        abs(float(final[key] - initial[key]))
        / max(abs(float(initial[key])), 1.0e-300)
        for key in ("baryon_mass", "carrier_number")
    )
    return CurvatureCase(
        points,
        maximum_gravity,
        _gravity_norm(final),
        float(np.max(np.abs(state.scalar.varphi))),
        float(np.sqrt(np.mean((physical_metric - flat_metric) ** 2))),
        float(np.sqrt(np.mean(extrinsic_curvature**2))),
        minimum_lapse,
        minimum_metric_eigenvalue,
        maximum_field,
        number_drift,
        float(final["entropy_integral"] - initial["entropy_integral"]),
        minimum_legendre,
        minimum_thermodynamic,
        maximum_recovery,
    )


def run_atlas(*, duration: float = 0.05, dt: float = 0.01) -> CurvatureAtlas:
    cases = tuple(_run_case(points, duration, dt) for points in (4, 8, 16))
    coarse, medium, fine = cases
    coarse_scalar_error = abs(coarse.final_scalar_amplitude - fine.final_scalar_amplitude)
    medium_scalar_error = abs(medium.final_scalar_amplitude - fine.final_scalar_amplitude)
    coarse_metric_error = abs(coarse.final_metric_deviation_l2 - fine.final_metric_deviation_l2)
    medium_metric_error = abs(medium.final_metric_deviation_l2 - fine.final_metric_deviation_l2)
    return CurvatureAtlas(
        0.5,
        duration,
        dt,
        cases,
        int(
            coarse.maximum_gravitational_constraint_l2
            > medium.maximum_gravitational_constraint_l2
            > fine.maximum_gravitational_constraint_l2
        ),
        coarse.maximum_gravitational_constraint_l2
        / medium.maximum_gravitational_constraint_l2,
        medium.maximum_gravitational_constraint_l2
        / fine.maximum_gravitational_constraint_l2,
        coarse_scalar_error / max(medium_scalar_error, 1.0e-300),
        coarse_metric_error / max(medium_metric_error, 1.0e-300),
        fine.maximum_gravitational_constraint_l2,
        fine.final_metric_deviation_l2,
        fine.final_extrinsic_curvature_l2,
        min(case.minimum_lapse for case in cases),
        min(case.minimum_metric_eigenvalue for case in cases),
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
