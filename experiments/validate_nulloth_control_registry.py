"""Validate the machine-readable Synodic control grammar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_OBJECTIVES = {
    "suppression",
    "shielding",
    "redirection",
    "confinement",
    "transport",
    "stabilization",
    "sensing",
    "release",
}
REQUIRED_COSTS = {
    "energy",
    "heat",
    "carrier",
    "latency",
    "infrastructure",
    "admissibility",
}
PROHIBITED_CHANGES = {
    "metric",
    "extrinsic curvature",
    "lapse",
    "shift",
    "canonical coupling",
    "production cap",
    "constraint damping",
}


def validate_registry(path: Path) -> dict[str, int]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    channels = registry["channels"]
    objectives = registry["objectives"]
    channel_ids = {channel["id"] for channel in channels}
    duplicate_channel_ids = len(channels) - len(channel_ids)
    duplicate_objective_ids = len(objectives) - len(
        {objective["id"] for objective in objectives}
    )
    unknown_channel_references = sum(
        channel not in channel_ids
        for objective in objectives
        for channel in objective["channels"]
    )
    missing_channel_cost_links = sum(
        not REQUIRED_COSTS.issubset(set(channel["costs"]))
        for channel in channels
    )
    direct_state_overrides = sum(
        change.lower() in PROHIBITED_CHANGES
        for channel in channels
        for change in channel["changes"]
    )
    objective_names = {objective["name"] for objective in objectives}
    objectives_without_controls = sum(
        not objective["control"] for objective in objectives
    )
    objectives_without_abort = sum(not objective["abort"] for objective in objectives)
    return {
        "canonical_name_correct": int(registry["canonical_name"] == "Nulloth"),
        "channel_count": len(channels),
        "objective_count": len(objectives),
        "required_objectives_present": int(REQUIRED_OBJECTIVES == objective_names),
        "accounting_dimension_count": len(registry["accounting"]),
        "required_accounting_dimensions_present": int(
            REQUIRED_COSTS == set(registry["accounting"])
        ),
        "decision_state_count": len(registry["decision_states"]),
        "global_forbidden_shortcut_count": len(
            registry["global_forbidden_shortcuts"]
        ),
        "duplicate_channel_ids": duplicate_channel_ids,
        "duplicate_objective_ids": duplicate_objective_ids,
        "unknown_channel_references": unknown_channel_references,
        "missing_channel_cost_links": missing_channel_cost_links,
        "direct_state_overrides": direct_state_overrides,
        "objectives_without_controls": objectives_without_controls,
        "objectives_without_abort": objectives_without_abort,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(validate_registry(arguments.registry), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
