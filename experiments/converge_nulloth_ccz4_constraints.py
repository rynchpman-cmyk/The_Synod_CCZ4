"""Converge weak-field Nulloth observables and CCZ4 constraints separately."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid

try:
    from experiments.map_nulloth_production_regimes import _initial_state, _solver
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _initial_state, _solver


@dataclass(frozen=True)
class ResolutionCase:
    points: int
    initial_gravitational_constraint_l2: float
    maximum_gravitational_constraint_l2: float
    final_gravitational_constraint_l2: float
    final_scalar_amplitude: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class ConstraintAtlas:
    duration: float
    dt: float
    cases: tuple[ResolutionCase, ...]
    gravitational_constraint_monotone_with_resolution: int
    coarse_to_medium_constraint_reduction: float
    medium_to_fine_constraint_reduction: float
    coarse_to_medium_scalar_error_reduction: float
    maximum_finest_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _gravity_norm(diagnostic: dict[str, float]) -> float:
    return max(
        float(diagnostic["hamiltonian_l2"]),
        float(diagnostic["momentum_l2"]),
    )


def _run_case(points: int, duration: float, dt: float) -> ResolutionCase:
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
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermodynamic = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    steps = int(round(duration / dt))
    for step in range(steps):
        state = solver.step(state, dt)
        if (step + 1) % 5 == 0:
            diagnostic = solver.diagnostics(state)
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
    number_drifts = [
        abs(float(final[key] - initial[key]))
        / max(abs(float(initial[key])), 1.0e-300)
        for key in ("baryon_mass", "carrier_number")
    ]
    return ResolutionCase(
        points,
        _gravity_norm(initial),
        maximum_gravity,
        _gravity_norm(final),
        float(np.max(np.abs(state.scalar.varphi))),
        maximum_field,
        max(number_drifts),
        float(final["entropy_integral"] - initial["entropy_integral"]),
        minimum_legendre,
        minimum_thermodynamic,
        maximum_recovery,
    )


def run_atlas(*, duration: float = 0.25, dt: float = 0.01) -> ConstraintAtlas:
    cases = tuple(_run_case(points, duration, dt) for points in (4, 8, 16))
    coarse, medium, fine = cases
    coarse_error = abs(coarse.final_scalar_amplitude - fine.final_scalar_amplitude)
    medium_error = abs(medium.final_scalar_amplitude - fine.final_scalar_amplitude)
    return ConstraintAtlas(
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
        coarse_error / max(medium_error, 1.0e-300),
        fine.maximum_gravitational_constraint_l2,
        max(case.maximum_field_constraint_l2 for case in cases),
        max(case.maximum_number_relative_drift for case in cases),
        min(case.entropy_change for case in cases),
        min(case.minimum_legendre for case in cases),
        min(case.minimum_thermodynamic for case in cases),
        max(case.maximum_recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.25)
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
