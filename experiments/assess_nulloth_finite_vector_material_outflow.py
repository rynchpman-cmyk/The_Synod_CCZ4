"""Assess the denied NUL-BND-003 finite source-passage record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(record_path: Path) -> dict[str, float | int]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    acceptance = record["scientific_acceptance"]
    result = record["parsed_stdout"]
    criteria = acceptance["criteria"]
    entropy = [
        item
        for item in criteria
        if "entropy_balance_residual" in item["observable"]
    ]
    nonentropy = [item for item in criteria if item not in entropy]
    return {
        "original_run_completed": int(
            record["termination"] == "completed" and record["return_code"] == 0
        ),
        "original_scientific_acceptance_passed": int(acceptance["passed"]),
        "all_nonentropy_criteria_passed": int(
            all(item["passed"] for item in nonentropy)
        ),
        "failed_entropy_criteria": sum(not item["passed"] for item in entropy),
        "vector_norm_loss_fraction": result["vector_norm_loss_fraction"],
        "material_excess_baryon_loss_fraction": result[
            "material_excess_baryon_loss_fraction"
        ],
        "material_excess_carrier_loss_fraction": result[
            "material_excess_carrier_loss_fraction"
        ],
        "maximum_step_entropy_balance_residual": result[
            "maximum_step_entropy_balance_residual"
        ],
        "maximum_absolute_cumulative_entropy_balance_residual": result[
            "maximum_absolute_cumulative_entropy_balance_residual"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = assess(arguments.record)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
