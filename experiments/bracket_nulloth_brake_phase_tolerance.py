"""Bracket the phase-target tolerance of the periodic Carrier Coherence Brake."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from audit_nulloth_brake_robustness import _run_variant, _source_values


def _accepted(path: Path) -> int:
    record = json.loads(path.read_text(encoding="utf-8"))
    return int(record["scientific_acceptance"]["passed"])


def bracket(
    spell_record: Path,
    spell_assessment: Path,
    robustness_record: Path,
    robustness_assessment: Path,
    *,
    refinement_steps: int = 3,
) -> dict[str, object]:
    control_kinetic, nominal_kinetic, spell_accepted = _source_values(
        spell_record, spell_assessment
    )
    nominal_suppression = (control_kinetic - nominal_kinetic) / abs(control_kinetic)
    robustness = json.loads(robustness_record.read_text(encoding="utf-8"))
    phase_plus = next(
        variant
        for variant in robustness["parsed_stdout"]["variants"]
        if variant["name"] == "phase_plus_pi_over_8"
    )
    lower = 0.0
    lower_suppression = nominal_suppression
    upper = np.pi / 8.0
    upper_suppression = float(phase_plus["relative_suppression"])
    target = 0.02
    if lower_suppression < target or upper_suppression >= target:
        raise RuntimeError("sealed phase endpoints do not bracket the objective")

    branches = []
    samples = [
        {
            "phase_radians": lower,
            "relative_suppression": lower_suppression,
            "source": "sealed nominal production run",
        },
        {
            "phase_radians": upper,
            "relative_suppression": upper_suppression,
            "source": "sealed NUL-CTL-005 phase run",
        },
    ]
    for step in range(refinement_steps):
        midpoint = 0.5 * (lower + upper)
        branch = _run_variant(
            name=f"phase_bisection_{step + 1}",
            amplitude=0.02,
            phase=midpoint,
            noise_fraction=0.0,
            momentum_scale=1.0,
            control_kinetic=control_kinetic,
            points=32,
            duration=2.0,
            dt=0.01,
            target=target,
        )
        branches.append(branch)
        samples.append(
            {
                "phase_radians": midpoint,
                "relative_suppression": branch.relative_suppression,
                "source": "new production run",
            }
        )
        if branch.meets_objective:
            lower = midpoint
            lower_suppression = branch.relative_suppression
        else:
            upper = midpoint
            upper_suppression = branch.relative_suppression

    return {
        "spell_source_accepted": spell_accepted,
        "robustness_assessment_accepted": _accepted(robustness_assessment),
        "points": 32,
        "new_production_evaluations": len(branches),
        "phase_tolerance_lower_radians": lower,
        "phase_tolerance_upper_radians": upper,
        "phase_tolerance_lower_degrees": float(np.degrees(lower)),
        "phase_tolerance_upper_degrees": float(np.degrees(upper)),
        "phase_bracket_width_radians": upper - lower,
        "lower_bound_suppression": lower_suppression,
        "upper_bound_suppression": upper_suppression,
        "lower_bound_meets_objective": int(lower_suppression >= target),
        "upper_bound_meets_objective": int(upper_suppression >= target),
        "source_phase_pair_relative_asymmetry": robustness["parsed_stdout"][
            "phase_pair_relative_asymmetry"
        ],
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
    parser.add_argument("spell_record", type=Path)
    parser.add_argument("spell_assessment", type=Path)
    parser.add_argument("robustness_record", type=Path)
    parser.add_argument("robustness_assessment", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            bracket(
                arguments.spell_record,
                arguments.spell_assessment,
                arguments.robustness_record,
                arguments.robustness_assessment,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
