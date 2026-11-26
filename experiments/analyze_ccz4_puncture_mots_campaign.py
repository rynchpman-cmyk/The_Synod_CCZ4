"""Summarize and plot a dynamic CCZ4 puncture/MOTS trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def physical_common(record: dict[str, object]) -> dict[str, object | None]:
    classified = record.get("common_classified")
    if isinstance(classified, dict):
        return {
            "common_outer": classified.get("common_outer"),
            "common_inner": classified.get("common_inner"),
        }
    result: dict[str, object | None] = {
        "common_outer": None,
        "common_inner": None,
    }
    # Backward-compatible classification for trajectories written before the
    # stability-sign record was added.
    for value in record.get("common", {}).values():
        if not value.get("found", False):
            continue
        stability = value.get("normal_stability")
        if not isinstance(stability, dict) or not stability.get("valid", False):
            continue
        principal = float(stability["principal_eigenvalue_real"])
        name = "common_outer" if principal > 0.0 else "common_inner"
        current = result[name]
        if current is None or value["spectral_residual_norm"] < current["spectral_residual_norm"]:
            result[name] = value
    return result


def first_time(records, predicate):
    return next((float(record["time"]) for record in records if predicate(record)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    records = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("trajectory must contain a non-empty JSON record list")
    physical = [physical_common(record) for record in records]

    any_common = [
        index
        for index, values in enumerate(physical)
        if values["common_outer"] is not None or values["common_inner"] is not None
    ]
    first_index = any_common[0] if any_common else None
    formation_bracket = None
    if first_index is not None:
        formation_bracket = {
            "last_snapshot_without_common": (
                None if first_index == 0 else float(records[first_index - 1]["time"])
            ),
            "first_snapshot_with_common": float(records[first_index]["time"]),
        }

    def branch_values(name: str):
        return [values[name] for values in physical if values[name] is not None]

    outer = branch_values("common_outer")
    inner = branch_values("common_inner")
    sign_qualified = []
    for values in physical:
        for name in ("common_outer", "common_inner"):
            branch = values[name]
            if branch is not None and branch["cartesian_qualification"]["principal_mode_sign_gate_passed"]:
                sign_qualified.append((name, branch))

    final_physical = physical[-1]
    interior_hamiltonian = [
        float(record["constraints"]["hamiltonian_l2_interior"])
        for record in records
        if "hamiltonian_l2_interior" in record["constraints"]
    ]
    if not interior_hamiltonian:
        interior_hamiltonian = [
            float(record["constraints"]["hamiltonian_l2_masked"])
            for record in records
        ]
    payload: dict[str, object] = {
        "format": "causal_repair.ccz4_puncture_mots_analysis.v1",
        "records": len(records),
        "initial_time": float(records[0]["time"]),
        "final_time": float(records[-1]["time"]),
        "formation_bracket": formation_bracket,
        "common_inner_first_found_time": first_time(
            records,
            lambda record: physical_common(record)["common_inner"] is not None,
        ),
        "common_outer_first_found_time": first_time(
            records,
            lambda record: physical_common(record)["common_outer"] is not None,
        ),
        "both_common_branches_first_resolved_time": first_time(
            records,
            lambda record: all(
                physical_common(record)[name] is not None
                for name in ("common_outer", "common_inner")
            ),
        ),
        "outer_observations": len(outer),
        "inner_observations": len(inner),
        "qualified_sign_observations": len(sign_qualified),
        "interior_constraint_records": len(interior_hamiltonian),
        "minimum_interior_hamiltonian_l2": float(min(interior_hamiltonian)),
        "maximum_interior_hamiltonian_l2": float(max(interior_hamiltonian)),
        "final": {
            "puncture_coordinate_separation": float(records[-1]["coordinate_separation"]),
            "minimum_lapse": float(records[-1]["health"]["minimum_lapse"]),
            "common_outer": final_physical["common_outer"],
            "common_inner": final_physical["common_inner"],
        },
        "interpretation": (
            "Branch names are assigned by the sign of the physical-normal "
            "principal mode. A single-resolution formation time is a sampled "
            "bracket, and zero qualified sign observations means the run is a pilot."
        ),
    }
    output = args.output or args.trajectory.with_name("analysis.json")
    atomic_json(output, payload)

    plot = args.plot
    if plot is not None:
        import matplotlib.pyplot as plt

        times = np.asarray([record["time"] for record in records], dtype=float)
        hamiltonian = np.asarray(
            [
                record["constraints"].get("hamiltonian_l2_interior", np.nan)
                for record in records
            ],
            dtype=float,
        )
        separation = np.asarray(
            [record["coordinate_separation"] for record in records], dtype=float
        )
        figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
        for name, color in (("common_outer", "tab:blue"), ("common_inner", "tab:orange")):
            branch_times, eigenvalues, radii = [], [], []
            for record, values in zip(records, physical):
                branch = values[name]
                if branch is None:
                    continue
                branch_times.append(record["time"])
                eigenvalues.append(branch["normal_stability"]["principal_eigenvalue_real"])
                radii.append(branch["coordinate_radius_mean"])
            axes[0].plot(branch_times, eigenvalues, "o-", color=color, label=name)
            axes[1].plot(branch_times, radii, "o-", color=color, label=name)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_ylabel(r"$\lambda_0$")
        axes[0].legend()
        axes[1].set_ylabel("mean radius")
        axes[2].semilogy(times, hamiltonian, "o-", label="interior Hamiltonian L2")
        twin = axes[2].twinx()
        twin.plot(times, separation, "--", color="tab:green", label="puncture separation")
        axes[2].set_ylabel("constraint L2")
        twin.set_ylabel("coordinate separation")
        axes[2].set_xlabel("CCZ4 time")
        figure.suptitle("Dynamic CCZ4 puncture/MOTS pilot")
        figure.tight_layout()
        plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(plot, dpi=180)
        plt.close(figure)
    print(f"wrote {output}")
    if plot is not None:
        print(f"wrote {plot}")


if __name__ == "__main__":
    main()
