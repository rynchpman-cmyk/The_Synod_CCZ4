"""Classify the immutable combined-error campaign and its shape-sign denial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_RAW_FAILURES = {
    "negative_shape_meets_objective",
    "negative_shape_relative_suppression",
}


def assess(combined_path: Path, robustness_path: Path) -> dict[str, float | int | str]:
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    robustness = json.loads(robustness_path.read_text(encoding="utf-8"))
    if combined.get("campaign_id") != "NUL-CTL-007":
        raise ValueError("combined source is not NUL-CTL-007")
    if robustness.get("campaign_id") != "NUL-CTL-005":
        raise ValueError("robustness source is not NUL-CTL-005")
    if combined.get("termination") != "completed" or combined.get("return_code") != 0:
        raise ValueError("combined source process did not complete")
    criteria = combined["scientific_acceptance"]["criteria"]
    failed = [criterion for criterion in criteria if not criterion["passed"]]
    failed_names = {criterion["observable"] for criterion in failed}
    parsed = combined["parsed_stdout"]
    variants = {variant["name"]: variant for variant in parsed["variants"]}
    positive_shape = next(
        variant
        for variant in robustness["parsed_stdout"]["variants"]
        if variant["name"] == "second_harmonic_10_percent"
    )
    negative_shape = variants["negative_second_harmonic"]
    return {
        "source_process_completed": 1,
        "raw_scientific_acceptance_passed": int(
            combined["scientific_acceptance"]["passed"]
        ),
        "raw_failed_criteria": len(failed),
        "failed_criteria_are_negative_shape_prediction": int(
            failed_names == EXPECTED_RAW_FAILURES
        ),
        "all_other_raw_criteria_passed": int(
            all(
                criterion["passed"]
                for criterion in criteria
                if criterion["observable"] not in EXPECTED_RAW_FAILURES
            )
        ),
        "positive_shape_meets_objective": int(positive_shape["meets_objective"]),
        "positive_shape_relative_suppression": positive_shape[
            "relative_suppression"
        ],
        "negative_shape_meets_objective": int(negative_shape["meets_objective"]),
        "negative_shape_relative_suppression": negative_shape[
            "relative_suppression"
        ],
        "shape_sign_suppression_difference": (
            positive_shape["relative_suppression"]
            - negative_shape["relative_suppression"]
        ),
        "phase_low_amplitude_relative_suppression": variants[
            "phase_plus_low_amplitude"
        ]["relative_suppression"],
        "phase_momentum_underdrive_relative_suppression": variants[
            "phase_plus_momentum_underdrive"
        ]["relative_suppression"],
        "phase_negative_shape_relative_suppression": variants[
            "phase_plus_negative_shape"
        ]["relative_suppression"],
        "paired_phase_failures": parsed["paired_phase_failures"],
        "combined_corner_meets_objective": parsed[
            "combined_corner_meets_objective"
        ],
        "combined_corner_relative_suppression": parsed[
            "combined_corner_relative_suppression"
        ],
        "all_variants_admissible": parsed["all_variants_admissible"],
        "maximum_number_relative_drift": parsed["maximum_number_relative_drift"],
        "minimum_entropy_change": parsed["minimum_entropy_change"],
        "maximum_gauss_l2": parsed["maximum_gauss_l2"],
        "maximum_scalar_constraint_l2": parsed["maximum_scalar_constraint_l2"],
        "maximum_gravitational_constraint_l2": parsed[
            "maximum_gravitational_constraint_l2"
        ],
        "maximum_recovery_residual": parsed["maximum_recovery_residual"],
        "minimum_legendre_eigenvalue": parsed["minimum_legendre_eigenvalue"],
        "minimum_thermodynamic_eigenvalue": parsed[
            "minimum_thermodynamic_eigenvalue"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("combined_record", type=Path)
    parser.add_argument("robustness_record", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            assess(arguments.combined_record, arguments.robustness_record),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
