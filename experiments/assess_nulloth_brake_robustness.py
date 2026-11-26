"""Classify the immutable Carrier Coherence Brake robustness denial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_FAILURES = {
    "variants_meeting_objective",
    "minimum_variant_relative_suppression",
}


def assess(path: Path) -> dict[str, float | int | str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("run_record_schema") != "nulloth.run.v1":
        raise ValueError("source is not a Nulloth run record")
    if record.get("campaign_id") != "NUL-CTL-005":
        raise ValueError("source is not NUL-CTL-005")
    if record.get("termination") != "completed" or record.get("return_code") != 0:
        raise ValueError("source robustness process did not complete")
    parsed = record.get("parsed_stdout")
    if not isinstance(parsed, dict):
        raise ValueError("source record has no parsed result")
    criteria = record["scientific_acceptance"]["criteria"]
    failed = [criterion for criterion in criteria if not criterion["passed"]]
    failed_names = {criterion["observable"] for criterion in failed}
    variants = parsed["variants"]
    phase = [variant for variant in variants if variant["name"].startswith("phase_")]
    nonphase = [variant for variant in variants if not variant["name"].startswith("phase_")]
    phase_suppressions = [float(variant["relative_suppression"]) for variant in phase]
    return {
        "source_run_id": record["run_id"],
        "source_process_completed": 1,
        "raw_scientific_acceptance_passed": int(
            record["scientific_acceptance"]["passed"]
        ),
        "raw_failed_criteria": len(failed),
        "failed_criteria_are_target_only": int(failed_names == TARGET_FAILURES),
        "all_nontarget_criteria_passed": int(
            all(
                criterion["passed"]
                for criterion in criteria
                if criterion["observable"] not in TARGET_FAILURES
            )
        ),
        "phase_variants": len(phase),
        "phase_variants_meeting_objective": sum(
            bool(variant["meets_objective"]) for variant in phase
        ),
        "nonphase_variants": len(nonphase),
        "nonphase_variants_meeting_objective": sum(
            bool(variant["meets_objective"]) for variant in nonphase
        ),
        "minimum_phase_relative_suppression": min(phase_suppressions),
        "maximum_phase_relative_suppression": max(phase_suppressions),
        "phase_pair_relative_asymmetry": parsed["phase_pair_relative_asymmetry"],
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
    parser.add_argument("record", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(assess(arguments.record), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
