"""Refine a Carrier Coherence Brake threshold against a sealed control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from search_nulloth_brake_controls import _run_branch

from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS


def _accepted(path: Path) -> bool:
    record = json.loads(path.read_text(encoding="utf-8"))
    return bool(record["scientific_acceptance"]["passed"])


def _source_resolution(record_path: Path, points: int) -> tuple[float, float]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    matches = [
        result
        for result in record["parsed_stdout"]["resolution_results"]
        if result["active"]["points"] == points
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one sealed {points}-point source result")
    result = matches[0]
    return (
        float(result["control"]["carrier_kinetic_proxy"]),
        float(result["active"]["carrier_kinetic_proxy"]),
    )


def _suppression(kinetic: float, control_kinetic: float) -> float:
    return (control_kinetic - kinetic) / max(abs(control_kinetic), 1.0e-300)


def converge(
    *,
    source_record: Path,
    assessment_record: Path,
    points: int = 16,
    duration: float = 2.0,
    dt: float = 1.0e-2,
    target_suppression: float = 0.02,
    lower_amplitude: float = 0.008,
    upper_amplitude: float = 0.02,
    refinement_steps: int = 4,
    reference_eight_point_threshold: float = 0.01025,
) -> dict[str, object]:
    control_kinetic, sealed_upper_kinetic = _source_resolution(source_record, points)
    upper_suppression = _suppression(sealed_upper_kinetic, control_kinetic)
    wavenumber = 2.0 * 3.141592653589793 / 32.0
    peak_effective_mass = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass**2
        - NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling * 0.22**2
    )
    growing_rate = (-peak_effective_mass - wavenumber**2) ** 0.5
    branches = []

    def evaluate(amplitude: float):
        branch = _run_branch(
            branch=f"active_{points}",
            parameters=NULLOTH_DISCOVERY_PARAMETERS,
            points=points,
            duration=duration,
            dt=dt,
            amplitude=amplitude,
            growing_rate=growing_rate,
        )
        branches.append(branch)
        suppression = _suppression(branch.carrier_kinetic_proxy, control_kinetic)
        return suppression

    lower = lower_amplitude
    upper = upper_amplitude
    lower_suppression = evaluate(lower)
    initial_lower_passes = int(lower_suppression >= target_suppression)
    source_upper_passes = int(upper_suppression >= target_suppression)
    if initial_lower_passes or not source_upper_passes:
        raise RuntimeError("registered source values do not bracket the target")

    samples = [
        {
            "amplitude": lower,
            "relative_suppression": lower_suppression,
            "source": "new production run",
        },
        {
            "amplitude": upper,
            "relative_suppression": upper_suppression,
            "source": "sealed NUL-SPL-001 production run",
        },
    ]
    for _ in range(refinement_steps):
        midpoint = 0.5 * (lower + upper)
        suppression = evaluate(midpoint)
        samples.append(
            {
                "amplitude": midpoint,
                "relative_suppression": suppression,
                "source": "new production run",
            }
        )
        if suppression >= target_suppression:
            upper = midpoint
            upper_suppression = suppression
        else:
            lower = midpoint
            lower_suppression = suppression

    relative_shift = (
        upper - reference_eight_point_threshold
    ) / reference_eight_point_threshold
    return {
        "source_assessment_accepted": int(_accepted(assessment_record)),
        "points": points,
        "new_production_evaluations": len(branches),
        "initial_lower_passes": initial_lower_passes,
        "source_upper_passes": source_upper_passes,
        "final_lower_amplitude": lower,
        "final_upper_amplitude": upper,
        "final_bracket_width": upper - lower,
        "final_lower_suppression": lower_suppression,
        "final_upper_suppression": upper_suppression,
        "relative_threshold_shift_from_eight_points": relative_shift,
        "threshold_converged_with_eight_points": int(abs(relative_shift) <= 0.10),
        "all_new_branches_admissible": int(
            all(branch.admissible for branch in branches)
        ),
        "maximum_number_relative_drift": max(
            max(branch.baryon_relative_drift, branch.carrier_relative_drift)
            for branch in branches
        ),
        "minimum_entropy_change": min(branch.entropy_change for branch in branches),
        "maximum_gauss_l2": max(branch.gauss_max_l2 for branch in branches),
        "maximum_scalar_constraint_l2": max(
            branch.scalar_constraint_l2 for branch in branches
        ),
        "maximum_gravitational_constraint_l2": max(
            max(branch.hamiltonian_l2, branch.momentum_l2) for branch in branches
        ),
        "maximum_recovery_residual": max(
            branch.recovery_residual for branch in branches
        ),
        "minimum_legendre_eigenvalue": min(
            branch.minimum_legendre_eigenvalue for branch in branches
        ),
        "minimum_thermodynamic_eigenvalue": min(
            branch.minimum_thermodynamic_eigenvalue for branch in branches
        ),
        "samples": samples,
        "new_branches": [asdict(branch) for branch in branches],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_record", type=Path)
    parser.add_argument("assessment_record", type=Path)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--refinement-steps", type=int, default=4)
    arguments = parser.parse_args()
    result = converge(
        source_record=arguments.source_record,
        assessment_record=arguments.assessment_record,
        points=arguments.points,
        refinement_steps=arguments.refinement_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
