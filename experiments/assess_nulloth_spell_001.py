"""Assess the raw Spell 001 campaign using its correctly partitioned gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFERRED_LONG_RUN_GATES = {
    "coherence did not reach the nonlinear condensed branch",
    "coherence did not demonstrate nonlinear turnover",
}


def assess_spell_001(record_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("run_record_schema") != "nulloth.run.v1":
        raise ValueError("source is not a Nulloth run record")
    if record.get("campaign_id") != "NUL-SPL-001":
        raise ValueError("source record is not the raw Spell 001 campaign")
    if record.get("termination") != "completed" or record.get("return_code") != 0:
        raise ValueError("raw Spell 001 solver execution did not complete")
    raw = record.get("parsed_stdout")
    if not isinstance(raw, dict):
        raise ValueError("raw Spell 001 record has no parsed result")

    short_results = raw["resolution_results"]
    short_gate_passes = 0
    for result in short_results:
        rejection_reasons = set(result["rejection_reasons"])
        if rejection_reasons.issubset(DEFERRED_LONG_RUN_GATES):
            short_gate_passes += 1

    return {
        "source_run_id": record["run_id"],
        "source_process_completed": 1,
        "short_resolution_runs": len(short_results),
        "short_resolution_gate_passes": short_gate_passes,
        "long_run_promoted": raw["long_run_promoted"],
        "nonlinear_turnover": raw["nonlinear_turnover"],
        "scalar_amplitude_relative_change_16_to_32": raw[
            "scalar_amplitude_relative_change_16_to_32"
        ],
        "carrier_velocity_effect_relative_change_16_to_32": raw[
            "carrier_velocity_effect_relative_change_16_to_32"
        ],
        "high_resolution_kinetic_effect": raw[
            "high_resolution_kinetic_effect"
        ],
        "suppressive_sign_consistent_16_to_32": raw[
            "suppressive_sign_consistent_16_to_32"
        ],
        "maximum_number_relative_drift": raw["maximum_number_relative_drift"],
        "maximum_gauss_l2": raw["maximum_gauss_l2"],
        "maximum_scalar_constraint_l2": raw["maximum_scalar_constraint_l2"],
        "maximum_gravitational_constraint_l2": raw[
            "maximum_gravitational_constraint_l2"
        ],
        "minimum_legendre_eigenvalue": raw["minimum_legendre_eigenvalue"],
        "minimum_thermodynamic_eigenvalue": raw[
            "minimum_thermodynamic_eigenvalue"
        ],
        "maximum_restart_error": raw["maximum_restart_error"],
        "long_peak_scalar_amplitude": raw["long_peak_scalar_amplitude"],
        "long_peak_to_final_drop": raw["long_peak_to_final_drop"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(assess_spell_001(args.record), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

