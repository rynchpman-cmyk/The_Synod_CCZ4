"""Compare short-horizon Nulloth backreaction under two CCZ4 slicings."""

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
class GaugeCase:
    slicing: str
    final_active_scalar_amplitude: float
    final_physical_metric_contrast_l2: float
    final_lapse_contrast_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class GaugeAtlas:
    points: int
    duration: float
    dt: float
    cases: tuple[GaugeCase, ...]
    scalar_amplitude_relative_spread: float
    physical_metric_contrast_relative_spread: float
    lapse_contrast_relative_spread: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _l2(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(field) ** 2)))


def _relative_spread(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second), 1.0e-300)


def _solver(grid: PeriodicGrid, slicing: str) -> NullothProductionSolver:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
    )
    geometry = CCZ4Parameters(
        kappa_gravity=1.0e-3,
        slicing=slicing,
        evolve_shift=False,
        radiative_boundaries=False,
    )
    return NullothProductionSolver(grid, parameters, vector, geometry)


def _run_case(points: int, duration: float, dt: float, slicing: str) -> GaugeCase:
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    control_solver = _solver(grid, slicing)
    active_solver = _solver(grid, slicing)
    control = _initial_state(control_solver, 0.18, 0.0, "zero")
    active = _initial_state(active_solver, 0.18, 0.02, "growing")
    control_initial = control_solver.diagnostics(control)
    active_initial = active_solver.diagnostics(active)
    maximum_gravity = 0.0
    maximum_field = 0.0
    steps = int(round(duration / dt))
    for step in range(steps):
        control = control_solver.step(control, dt)
        active = active_solver.step(active, dt)
        if (step + 1) % 5 == 0:
            for solver, state in ((control_solver, control), (active_solver, active)):
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
    control_final = control_solver.diagnostics(control)
    active_final = active_solver.diagnostics(active)
    control_h, _ = control_solver.ccz4.physical_geometry(control.geometry)
    active_h, _ = active_solver.ccz4.physical_geometry(active.geometry)
    number_drifts = []
    entropy_changes = []
    for initial, final in (
        (control_initial, control_final),
        (active_initial, active_final),
    ):
        for key in ("baryon_mass", "carrier_number"):
            number_drifts.append(
                abs(float(final[key] - initial[key]))
                / max(abs(float(initial[key])), 1.0e-300)
            )
        entropy_changes.append(float(final["entropy_integral"] - initial["entropy_integral"]))
    diagnostics = (control_final, active_final)
    return GaugeCase(
        slicing,
        float(np.max(np.abs(active.scalar.varphi))),
        _l2(active_h - control_h),
        _l2(active.geometry.lapse - control.geometry.lapse),
        maximum_gravity,
        maximum_field,
        max(number_drifts),
        min(entropy_changes),
        min(float(item["minimum_legendre_eigenvalue"]) for item in diagnostics),
        min(float(item["minimum_thermodynamic_eigenvalue"]) for item in diagnostics),
        max(float(item["recovery_maximum_residual"]) for item in diagnostics),
    )


def run_atlas(*, points: int = 4, duration: float = 0.25, dt: float = 0.01) -> GaugeAtlas:
    cases = tuple(
        _run_case(points, duration, dt, slicing) for slicing in ("harmonic", "1+log")
    )
    harmonic, one_plus_log = cases
    return GaugeAtlas(
        points,
        duration,
        dt,
        cases,
        _relative_spread(
            harmonic.final_active_scalar_amplitude,
            one_plus_log.final_active_scalar_amplitude,
        ),
        _relative_spread(
            harmonic.final_physical_metric_contrast_l2,
            one_plus_log.final_physical_metric_contrast_l2,
        ),
        _relative_spread(
            harmonic.final_lapse_contrast_l2,
            one_plus_log.final_lapse_contrast_l2,
        ),
        max(case.maximum_gravitational_constraint_l2 for case in cases),
        max(case.maximum_field_constraint_l2 for case in cases),
        max(case.maximum_number_relative_drift for case in cases),
        min(case.minimum_entropy_change for case in cases),
        min(case.minimum_legendre for case in cases),
        min(case.minimum_thermodynamic for case in cases),
        max(case.maximum_recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(
        points=arguments.points,
        duration=arguments.duration,
        dt=arguments.dt,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
