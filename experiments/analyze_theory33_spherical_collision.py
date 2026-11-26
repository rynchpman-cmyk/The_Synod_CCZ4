"""Analyze the isolated Theory 3.3 spherical collision campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_case(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    archive = np.load(path / "trajectory.npz")
    trajectory = {key: archive[key] for key in archive.files}
    return summary, trajectory


def resolution_result(
    fast_path: Path, slow_path: Path
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fast_summary, fast = load_case(fast_path)
    slow_summary, slow = load_case(slow_path)
    np.testing.assert_allclose(fast["time"], slow["time"], atol=0.0)
    np.testing.assert_allclose(
        fast["extraction_radii"], slow["extraction_radii"], atol=0.0
    )
    mass_split = slow["hawking_mass"] - fast["hawking_mass"]
    growth_mask = (fast["time"] >= 0.004) & (mass_split[:, 0] > 0.0)
    mass_growth_power = float(
        np.polyfit(
            np.log(fast["time"][growth_mask]),
            np.log(mass_split[growth_mask, 0]),
            1,
        )[0]
    )
    initial_mismatch = float(
        np.max(np.abs(slow["hawking_mass"][0] - fast["hawking_mass"][0]))
    )
    result = {
        "points": int(fast_summary["points"]),
        "initial_mass_mismatch": initial_mismatch,
        "final_hawking_mass_split": mass_split[-1].tolist(),
        "final_relative_hawking_mass_split": (
            mass_split[-1]
            / np.maximum(np.abs(fast["hawking_mass"][-1]), 1.0e-300)
        ).tolist(),
        "inner_hawking_mass_time_power": mass_growth_power,
        "final_relative_lorentz_factor_split": float(
            slow["relative_lorentz_factor"][-1]
            - fast["relative_lorentz_factor"][-1]
        ),
        "final_peak_baryon_density_split": float(
            slow["peak_baryon_density"][-1] - fast["peak_baryon_density"][-1]
        ),
        "final_peak_carrier_density_split": float(
            slow["peak_carrier_density"][-1] - fast["peak_carrier_density"][-1]
        ),
        "final_inner_baryon_mass_split": float(
            slow["inner_baryon_mass"][-1] - fast["inner_baryon_mass"][-1]
        ),
        "final_inner_carrier_number_split": float(
            slow["inner_carrier_number"][-1] - fast["inner_carrier_number"][-1]
        ),
        "final_collision_scale_split": float(
            slow["collision_scale"][-1] - fast["collision_scale"][-1]
        ),
        "drag_entropy_ratio_fast_to_slow": float(
            fast_summary["cumulative_drag_entropy_change"]
            / slow_summary["cumulative_drag_entropy_change"]
        ),
        "maximum_relative_baryon_change": max(
            abs(float(fast_summary["relative_baryon_change"])),
            abs(float(slow_summary["relative_baryon_change"])),
        ),
        "maximum_relative_carrier_change": max(
            abs(float(fast_summary["relative_carrier_change"])),
            abs(float(slow_summary["relative_carrier_change"])),
        ),
        "maximum_recovery_failures": max(
            int(fast_summary["recovery_failures"]),
            int(slow_summary["recovery_failures"]),
        ),
        "fast_final_core_hamiltonian_l2": float(
            fast_summary["final_core_hamiltonian_l2"]
        ),
        "slow_final_core_hamiltonian_l2": float(
            slow_summary["final_core_hamiltonian_l2"]
        ),
    }
    return result, {
        "time": fast["time"],
        "radii": fast["extraction_radii"],
        "mass_split": mass_split,
        "fast_gamma": fast["relative_lorentz_factor"],
        "slow_gamma": slow["relative_lorentz_factor"],
        "carrier_split": slow["inner_carrier_number"]
        - fast["inner_carrier_number"],
        "peak_carrier_split": slow["peak_carrier_density"]
        - fast["peak_carrier_density"],
        "fast_carrier_slice": fast["carrier_density_slice"],
        "slow_carrier_slice": slow["carrier_density_slice"],
        "fast_baryon_slice": fast["baryon_density_slice"],
        "slow_baryon_slice": slow["baryon_density_slice"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n10-fast", type=Path, default=Path("runs/theory33_spheres/fast")
    )
    parser.add_argument(
        "--n10-slow", type=Path, default=Path("runs/theory33_spheres/slow")
    )
    parser.add_argument(
        "--n12-fast", type=Path, default=Path("runs/theory33_spheres/n012_fast")
    )
    parser.add_argument(
        "--n12-slow", type=Path, default=Path("runs/theory33_spheres/n012_slow")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/theory33_spherical_collision.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("runs/theory33_spherical_collision.png"),
    )
    parser.add_argument(
        "--output-slices",
        type=Path,
        default=Path("runs/theory33_spherical_collision_slices.png"),
    )
    args = parser.parse_args()

    n10, trajectory10 = resolution_result(args.n10_fast, args.n10_slow)
    n12, trajectory12 = resolution_result(args.n12_fast, args.n12_slow)
    resolutions = [n10, n12]
    trajectories = [trajectory10, trajectory12]
    inner_splits = np.array(
        [float(row["final_hawking_mass_split"][0]) for row in resolutions]
    )
    outer_splits = np.array(
        [float(row["final_hawking_mass_split"][-1]) for row in resolutions]
    )
    resolution_difference = float(
        abs(inner_splits[1] - inner_splits[0])
        / max(abs(inner_splits[1]), 1.0e-300)
    )
    material_response_pass = all(
        float(row["final_relative_lorentz_factor_split"]) > 0.05
        and float(row["final_inner_carrier_number_split"]) > 2.0e-5
        for row in resolutions
    )
    inner_mass_response_pass = bool(
        np.all(inner_splits > 0.0) and resolution_difference < 0.25
    )
    outer_control_pass = bool(
        np.all(np.abs(outer_splits) < 0.05 * np.abs(inner_splits))
    )
    numerical_gates_pass = all(
        float(row["initial_mass_mismatch"]) < 1.0e-14
        and int(row["maximum_recovery_failures"]) == 0
        and float(row["maximum_relative_baryon_change"]) < 2.0e-6
        and float(row["maximum_relative_carrier_change"]) < 2.0e-6
        for row in resolutions
    )
    result = {
        "hypothesis": (
            "fast and slow carrier relaxation deform the inner material flow "
            "while leaving the outer Hawking mass unchanged"
        ),
        "observable": "Hawking quasi-local mass on nested coordinate spheres",
        "resolutions": resolutions,
        "inner_mass_split_resolution_difference": resolution_difference,
        "material_response_pass": material_response_pass,
        "inner_mass_response_pass": inner_mass_response_pass,
        "outer_mass_control_pass": outer_control_pass,
        "numerical_gates_pass": numerical_gates_pass,
        "qualification_boundary": (
            "N=8 does not remain inside the Theory 3.3 counterflow domain for "
            "the full interval; N=10 is the minimum completed resolution"
        ),
        "interpretation": (
            "the material/current split is large, but the early-time invariant "
            "mass response is seven to eight orders below the enclosed mass"
        ),
        "passed": bool(
            material_response_pass
            and inner_mass_response_pass
            and outer_control_pass
            and numerical_gates_pass
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    colors = ("tab:blue", "tab:orange")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, trajectory, color in zip(
        resolutions, trajectories, colors, strict=True
    ):
        label = f"N={row['points']}"
        axes[0, 0].plot(
            trajectory["time"], trajectory["mass_split"][:, 0], "o-", color=color, label=label
        )
        axes[0, 1].plot(
            trajectory["radii"], trajectory["mass_split"][-1], "o-", color=color, label=label
        )
        axes[1, 0].plot(
            trajectory["time"], trajectory["fast_gamma"] - 1.0, "-", color=color, label=label + " fast"
        )
        axes[1, 0].plot(
            trajectory["time"], trajectory["slow_gamma"] - 1.0, "--", color=color, label=label + " slow"
        )
        axes[1, 1].plot(
            trajectory["time"], trajectory["carrier_split"], "o-", color=color, label=label
        )
    axes[0, 0].set_ylabel(r"inner $M_H^{slow}-M_H^{fast}$")
    axes[0, 1].set_ylabel(r"final $M_H^{slow}-M_H^{fast}$")
    axes[1, 0].set_ylabel(r"maximum $\gamma_{rel}-1$")
    axes[1, 1].set_ylabel("inner carrier-number split")
    axes[0, 0].set_xlabel("time")
    axes[0, 1].set_xlabel("coordinate extraction radius")
    axes[1, 0].set_xlabel("time")
    axes[1, 1].set_xlabel("time")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Isolated two-current sphere collision: matter deforms before mass does")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    data = trajectory12
    extent = (-4.0, 4.0, -4.0, 4.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fields = (
        data["fast_baryon_slice"][-1],
        data["slow_baryon_slice"][-1],
        data["slow_baryon_slice"][-1] - data["fast_baryon_slice"][-1],
        data["fast_carrier_slice"][-1],
        data["slow_carrier_slice"][-1],
        data["slow_carrier_slice"][-1] - data["fast_carrier_slice"][-1],
    )
    titles = (
        "fast baryons",
        "slow baryons",
        "slow - fast baryons",
        "fast carriers",
        "slow carriers",
        "slow - fast carriers",
    )
    for axis, field, title in zip(axes.flat, fields, titles, strict=True):
        image = axis.imshow(field.T, origin="lower", extent=extent, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.suptitle("N=12 final equatorial collision slices")
    fig.savefig(args.output_slices, dpi=180)
    plt.close(fig)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
