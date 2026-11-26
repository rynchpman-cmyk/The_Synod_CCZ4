"""Assess the 32-point threshold run with explicit floating-point equality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def assess(path: Path) -> dict[str, float | int | str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("run_record_schema") != "nulloth.run.v1":
        raise ValueError("source is not a Nulloth run record")
    if record.get("campaign_id") != "NUL-CTL-004":
        raise ValueError("source is not NUL-CTL-004")
    if record.get("termination") != "completed" or record.get("return_code") != 0:
        raise ValueError("source production process did not complete")
    parsed = record.get("parsed_stdout")
    if not isinstance(parsed, dict):
        raise ValueError("source record has no parsed result")
    criteria = record["scientific_acceptance"]["criteria"]
    failed = [criterion for criterion in criteria if not criterion["passed"]]
    declared_width = 0.0005
    observed_width = float(parsed["final_bracket_width"])
    excess = max(observed_width - declared_width, 0.0)
    allowance = 8.0 * math.ulp(declared_width)
    only_width = int(
        len(failed) == 1
        and failed[0]["observable"] == "final_bracket_width"
        and failed[0]["operator"] == "<="
        and float(failed[0]["expected"]) == declared_width
    )
    return {
        "source_run_id": record["run_id"],
        "source_process_completed": 1,
        "raw_scientific_acceptance_passed": int(
            record["scientific_acceptance"]["passed"]
        ),
        "raw_failed_criteria": len(failed),
        "all_nonwidth_criteria_passed": int(
            all(
                criterion["passed"]
                for criterion in criteria
                if criterion["observable"] != "final_bracket_width"
            )
        ),
        "only_width_roundoff_failure": only_width,
        "declared_bracket_width": declared_width,
        "observed_bracket_width": observed_width,
        "bracket_width_excess": excess,
        "roundoff_allowance": allowance,
        "bracket_width_equal_within_eight_ulps": int(excess <= allowance),
        "final_lower_amplitude": parsed["final_lower_amplitude"],
        "final_upper_amplitude": parsed["final_upper_amplitude"],
        "final_lower_suppression": parsed["final_lower_suppression"],
        "final_upper_suppression": parsed["final_upper_suppression"],
        "relative_threshold_shift_from_sixteen_upper": parsed[
            "relative_threshold_shift_from_sixteen_upper"
        ],
        "threshold_converged_with_sixteen_points": parsed[
            "threshold_converged_with_sixteen_points"
        ],
        "all_new_branches_admissible": parsed["all_new_branches_admissible"],
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
    parser.add_argument("record", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(assess(arguments.record), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
