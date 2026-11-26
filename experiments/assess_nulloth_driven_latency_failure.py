"""Classify the immutable failed NUL-BND-008 driven-boundary run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assess(record_path: Path) -> dict[str, int | float]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    stderr_path = record_path.with_name("stderr.txt")
    stderr = stderr_path.read_text(encoding="utf-8")
    acceptance = record["scientific_acceptance"]
    observed = [item["observed"] for item in acceptance["criteria"]]
    return {
        "original_run_completed": int(
            record["termination"] == "completed" and record["return_code"] == 0
        ),
        "original_scientific_acceptance_passed": int(acceptance["passed"]),
        "decisive_observables_produced": int(any(value is not None for value in observed)),
        "singular_recovery_jacobian_reported": int(
            "numpy.linalg.LinAlgError: Singular matrix" in stderr
            and "_recover_cells_from_energy" in stderr
        ),
        "failed_return_code": int(record["return_code"] != 0),
        "duration_seconds": float(record["duration_seconds"]),
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
