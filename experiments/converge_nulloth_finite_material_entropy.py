"""Converge the finite material entropy remainder at fixed CFL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from causal_repair.grid import CartesianGrid

try:
    from experiments.audit_nulloth_finite_scalar_outflow import _solver
    from experiments.audit_nulloth_finite_vector_material_outflow import (
        _initial_state,
    )
except ModuleNotFoundError:
    from audit_nulloth_finite_scalar_outflow import _solver
    from audit_nulloth_finite_vector_material_outflow import _initial_state


@dataclass(frozen=True)
class FiniteEntropyCase:
    points: int
    dt: float
    steps: int
    minimum_step_entropy_remainder: float
    maximum_step_entropy_remainder: float
    cumulative_entropy_remainder: float
    cumulative_raw_projection_change: float
    cumulative_boundary_entropy_outflow: float
    measured_entropy_change: float
    balance_identity_error: float
    maximum_carrier_number_balance_residual: float
    gravitational_constraint_l2: float
    proca_constraint_l2: float
    recovery_residual: float


@dataclass(frozen=True)
class FiniteEntropyConvergence:
    duration: float
    cfl_ratio: float
    cases: tuple[FiniteEntropyCase, ...]
    minimum_step_entropy_remainder: float
    cumulative_remainder_monotone_to_fine: int
    maximum_step_remainder_monotone_to_fine: int
    coarse_to_fine_cumulative_reduction: float
    coarse_to_fine_maximum_step_reduction: float
    fine_raw_projection_to_retained_remainder_ratio: float
    maximum_balance_identity_error: float
    maximum_carrier_number_balance_residual: float
    maximum_gravitational_constraint_l2: float
    maximum_proca_constraint_l2: float
    maximum_recovery_residual: float


def _run_case(points: int, duration: float, cfl_ratio: float) -> FiniteEntropyCase:
    grid = CartesianGrid((points,), (1.0,), origin=(-0.5,), method="fd2")
    solver = _solver(grid)
    state = _initial_state(solver, "material")
    dt = cfl_ratio * grid.spacing[0]
    steps = int(round(duration / dt))
    if steps < 1 or abs(steps * dt - duration) > 1.0e-14:
        raise ValueError("duration must close exactly on every fixed-CFL ladder")
    minimum_remainder = float("inf")
    maximum_remainder = float("-inf")
    cumulative_remainder = 0.0
    cumulative_projection = 0.0
    cumulative_outflow = 0.0
    measured_change = 0.0
    maximum_carrier_residual = 0.0
    for _ in range(steps):
        state = solver.step(state, dt)
        remainder = solver.last_entropy_balance_residual
        minimum_remainder = min(minimum_remainder, remainder)
        maximum_remainder = max(maximum_remainder, remainder)
        cumulative_remainder += remainder
        cumulative_projection += solver.last_projection_entropy_change
        cumulative_outflow += solver.last_entropy_boundary_outflow
        measured_change += solver.last_step_entropy_change
        maximum_carrier_residual = max(
            maximum_carrier_residual,
            abs(solver.last_carrier_number_balance_residual),
        )
    diagnostics = solver.diagnostics(state)
    return FiniteEntropyCase(
        points,
        dt,
        steps,
        minimum_remainder,
        maximum_remainder,
        cumulative_remainder,
        cumulative_projection,
        cumulative_outflow,
        measured_change,
        abs(cumulative_remainder - measured_change - cumulative_outflow),
        maximum_carrier_residual,
        max(
            float(diagnostics["hamiltonian_l2"]),
            float(diagnostics["momentum_l2"]),
        ),
        max(
            float(diagnostics["gauss_A_l2"]),
            float(diagnostics["gauss_B_l2"]),
        ),
        float(diagnostics["recovery_maximum_residual"]),
    )


def run_convergence(
    *, duration: float = 0.024, cfl_ratio: float = 0.024
) -> FiniteEntropyConvergence:
    cases = tuple(
        _run_case(points, duration, cfl_ratio) for points in (12, 18, 24)
    )
    coarse, medium, fine = cases
    return FiniteEntropyConvergence(
        duration,
        cfl_ratio,
        cases,
        min(case.minimum_step_entropy_remainder for case in cases),
        int(
            coarse.cumulative_entropy_remainder
            > medium.cumulative_entropy_remainder
            > fine.cumulative_entropy_remainder
        ),
        int(
            coarse.maximum_step_entropy_remainder
            > medium.maximum_step_entropy_remainder
            > fine.maximum_step_entropy_remainder
        ),
        coarse.cumulative_entropy_remainder
        / max(fine.cumulative_entropy_remainder, 1.0e-300),
        coarse.maximum_step_entropy_remainder
        / max(fine.maximum_step_entropy_remainder, 1.0e-300),
        fine.cumulative_raw_projection_change
        / max(fine.cumulative_entropy_remainder, 1.0e-300),
        max(case.balance_identity_error for case in cases),
        max(case.maximum_carrier_number_balance_residual for case in cases),
        max(case.gravitational_constraint_l2 for case in cases),
        max(case.proca_constraint_l2 for case in cases),
        max(case.recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.024)
    parser.add_argument("--cfl-ratio", type=float, default=0.024)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_convergence(
        duration=arguments.duration,
        cfl_ratio=arguments.cfl_ratio,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
