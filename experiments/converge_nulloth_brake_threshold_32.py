"""Compare the 32-point brake threshold with the accepted 16-point bracket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from converge_nulloth_brake_threshold import converge


def compare(
    source_record: Path,
    assessment_record: Path,
    sixteen_point_record: Path,
) -> dict[str, object]:
    previous = json.loads(sixteen_point_record.read_text(encoding="utf-8"))
    previous_accepted = bool(previous["scientific_acceptance"]["passed"])
    reference = float(previous["parsed_stdout"]["final_upper_amplitude"])
    result = converge(
        source_record=source_record,
        assessment_record=assessment_record,
        points=32,
        lower_amplitude=0.012,
        upper_amplitude=0.02,
        refinement_steps=4,
        reference_eight_point_threshold=reference,
    )
    shift = result.pop("relative_threshold_shift_from_eight_points")
    converged = result.pop("threshold_converged_with_eight_points")
    result.update(
        {
            "sixteen_point_source_accepted": int(previous_accepted),
            "sixteen_point_reference_upper": reference,
            "relative_threshold_shift_from_sixteen_upper": shift,
            "threshold_converged_with_sixteen_points": converged,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_record", type=Path)
    parser.add_argument("assessment_record", type=Path)
    parser.add_argument("sixteen_point_record", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            compare(
                arguments.source_record,
                arguments.assessment_record,
                arguments.sixteen_point_record,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
