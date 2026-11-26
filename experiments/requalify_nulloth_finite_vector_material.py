"""Requalify finite vector/material passage with full-horizon entropy ladders."""

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
class FullHorizonEntropyCase:
    source: str
    points: int
    dt: float
    steps: int
    minimum_step_entropy_remainder: float
    maximum_step_entropy_remainder: float
    cumulative_entropy_remainder: float
    cumulative_boundary_entropy_outflow: float
    measured_entropy_change: float
    balance_identity_error: float
    maximum_carrier_number_balance_residual: float
    gravitational_constraint_l2: float
    proca_constraint_l2: float
    scalar_constraint_l2: float
    recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float


@dataclass(frozen=True)
class FullHorizonRequalification:
    duration: float
    cfl_ratio: float
    cases: tuple[FullHorizonEntropyCase, ...]
    original_vector_norm_loss_fraction: float
    original_material_excess_baryon_loss_fraction: float
    original_material_excess_carrier_loss_fraction: float
    material_minimum_step_entropy_remainder: float
    material_cumulative_remainder_monotone_to_fine: int
    material_coarse_to_fine_cumulative_reduction: float
    vector_minimum_step_entropy_remainder: float
    fine_vector_absolute_cumulative_entropy_remainder: float
    fine_vector_maximum_step_entropy_remainder: float
    fine_material_cumulative_entropy_remainder: float
    maximum_balance_identity_error: float
    maximum_carrier_number_balance_residual: float
    maximum_gravitational_constraint_l2: float
    maximum_proca_constraint_l2: float
    maximum_scalar_constraint_l2: float
    maximum_recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float


def _run_case(
    source: str, points: int, duration: float, cfl_ratio: float
) -> FullHorizonEntropyCase:
    grid = CartesianGrid((points,), (1.0,), origin=(-0.5,), method="fd2")
    solver = _solver(grid)
    state = _initial_state(solver, source)
    dt = cfl_ratio * grid.spacing[0]
    steps = int(round(duration / dt))
    if steps < 1 or abs(steps * dt - duration) > 1.0e-14:
        raise ValueError("full horizon must close on every fixed-CFL ladder")
    minimum_remainder = float("inf")
    maximum_remainder = float("-inf")
    cumulative_remainder = 0.0
    cumulative_outflow = 0.0
    measured_change = 0.0
    maximum_carrier_residual = 0.0
    for _ in range(steps):
        state = solver.step(state, dt)
        remainder = solver.last_entropy_balance_residual
        minimum_remainder = min(minimum_remainder, remainder)
        maximum_remainder = max(maximum_remainder, remainder)
        cumulative_remainder += remainder
        cumulative_outflow += solver.last_entropy_boundary_outflow
        measured_change += solver.last_step_entropy_change
        maximum_carrier_residual = max(
            maximum_carrier_residual,
            abs(solver.last_carrier_number_balance_residual),
        )
    diagnostics = solver.diagnostics(state)
    return FullHorizonEntropyCase(
        source,
        points,
        dt,
        steps,
        minimum_remainder,
        maximum_remainder,
        cumulative_remainder,
        cumulative_outflow,
        measured_change,
        abs(cumulative_remainder - cumulative_outflow - measured_change),
        maximum_carrier_residual,
        max(
            float(diagnostics["hamiltonian_l2"]),
            float(diagnostics["momentum_l2"]),
        ),
        max(
            float(diagnostics["gauss_A_l2"]),
            float(diagnostics["gauss_B_l2"]),
        ),
        float(diagnostics["scalar_constraint_l2"]),
        float(diagnostics["recovery_maximum_residual"]),
        float(diagnostics["minimum_legendre_eigenvalue"]),
        float(diagnostics["minimum_thermodynamic_eigenvalue"]),
    )


def run_requalification(
    record_path: Path,
    *,
    duration: float = 0.12,
    cfl_ratio: float = 0.024,
) -> FullHorizonRequalification:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    original = record["parsed_stdout"]
    cases = tuple(
        _run_case(source, points, duration, cfl_ratio)
        for source in ("vector", "material")
        for points in (12, 18, 24)
    )
    vector = tuple(case for case in cases if case.source == "vector")
    material = tuple(case for case in cases if case.source == "material")
    return FullHorizonRequalification(
        duration,
        cfl_ratio,
        cases,
        original["vector_norm_loss_fraction"],
        original["material_excess_baryon_loss_fraction"],
        original["material_excess_carrier_loss_fraction"],
        min(case.minimum_step_entropy_remainder for case in material),
        int(
            material[0].cumulative_entropy_remainder
            > material[1].cumulative_entropy_remainder
            > material[2].cumulative_entropy_remainder
        ),
        material[0].cumulative_entropy_remainder
        / max(material[2].cumulative_entropy_remainder, 1.0e-300),
        min(case.minimum_step_entropy_remainder for case in vector),
        abs(vector[2].cumulative_entropy_remainder),
        max(
            abs(vector[2].minimum_step_entropy_remainder),
            abs(vector[2].maximum_step_entropy_remainder),
        ),
        material[2].cumulative_entropy_remainder,
        max(case.balance_identity_error for case in cases),
        max(case.maximum_carrier_number_balance_residual for case in cases),
        max(case.gravitational_constraint_l2 for case in cases),
        max(case.proca_constraint_l2 for case in cases),
        max(case.scalar_constraint_l2 for case in cases),
        max(case.recovery_residual for case in cases),
        min(case.minimum_legendre_eigenvalue for case in cases),
        min(case.minimum_thermodynamic_eigenvalue for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--cfl-ratio", type=float, default=0.024)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_requalification(
        arguments.record,
        duration=arguments.duration,
        cfl_ratio=arguments.cfl_ratio,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
