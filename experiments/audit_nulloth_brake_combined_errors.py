"""Audit combined targeting errors for the periodic Carrier Coherence Brake."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from audit_nulloth_brake_robustness import _run_variant, _source_values


PHASE_BOUND = 0.3436116964863836
VARIANTS = (
    ("negative_second_harmonic", 0.020, 0.0, -0.10, 1.0),
    ("phase_plus_low_amplitude", 0.018, PHASE_BOUND, 0.0, 1.0),
    ("phase_plus_momentum_underdrive", 0.020, PHASE_BOUND, 0.0, 0.8),
    ("phase_plus_negative_shape", 0.020, PHASE_BOUND, -0.10, 1.0),
    ("combined_detrimental_corner", 0.018, PHASE_BOUND, -0.10, 0.8),
)


def audit(source_record: Path, assessment_record: Path) -> dict[str, object]:
    control_kinetic, _, source_accepted = _source_values(
        source_record, assessment_record
    )
    target = 0.02
    variants = [
        _run_variant(
            name=name,
            amplitude=amplitude,
            phase=phase,
            noise_fraction=noise,
            momentum_scale=momentum,
            control_kinetic=control_kinetic,
            points=32,
            duration=2.0,
            dt=0.01,
            target=target,
        )
        for name, amplitude, phase, noise, momentum in VARIANTS
    ]
    by_name = {variant.name: variant for variant in variants}
    paired = [
        by_name["phase_plus_low_amplitude"],
        by_name["phase_plus_momentum_underdrive"],
        by_name["phase_plus_negative_shape"],
    ]
    corner = by_name["combined_detrimental_corner"]
    negative_shape = by_name["negative_second_harmonic"]
    return {
        "source_assessment_accepted": source_accepted,
        "points": 32,
        "variant_count": len(variants),
        "target_relative_suppression": target,
        "variants_meeting_objective": sum(v.meets_objective for v in variants),
        "minimum_variant_relative_suppression": min(
            v.relative_suppression for v in variants
        ),
        "negative_shape_meets_objective": int(negative_shape.meets_objective),
        "negative_shape_relative_suppression": negative_shape.relative_suppression,
        "paired_phase_failures": sum(not variant.meets_objective for variant in paired),
        "combined_corner_meets_objective": int(corner.meets_objective),
        "combined_corner_relative_suppression": corner.relative_suppression,
        "all_variants_admissible": int(all(v.admissible for v in variants)),
        "maximum_number_relative_drift": max(
            max(v.baryon_relative_drift, v.carrier_relative_drift)
            for v in variants
        ),
        "minimum_entropy_change": min(v.entropy_change for v in variants),
        "maximum_gauss_l2": max(v.gauss_max_l2 for v in variants),
        "maximum_scalar_constraint_l2": max(v.scalar_constraint_l2 for v in variants),
        "maximum_gravitational_constraint_l2": max(
            max(v.hamiltonian_l2, v.momentum_l2) for v in variants
        ),
        "maximum_recovery_residual": max(v.recovery_residual for v in variants),
        "minimum_legendre_eigenvalue": min(
            v.minimum_legendre_eigenvalue for v in variants
        ),
        "minimum_thermodynamic_eigenvalue": min(
            v.minimum_thermodynamic_eigenvalue for v in variants
        ),
        "variants": [asdict(variant) for variant in variants],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_record", type=Path)
    parser.add_argument("assessment_record", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.source_record, arguments.assessment_record),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
