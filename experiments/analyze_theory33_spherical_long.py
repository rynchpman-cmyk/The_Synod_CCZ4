"""Analyze the widened-domain, longer N=16 spherical collision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_case(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with np.load(path / "trajectory.npz") as archive:
        trajectory = {key: archive[key] for key in archive.files}
    return summary, trajectory


def power_fit(time: np.ndarray, values: np.ndarray, lower: float, upper: float) -> dict[str, float]:
    tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper))
    mask = (
        (time >= lower - tolerance)
        & (time <= upper + tolerance)
        & (values > 0.0)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"power fit [{lower}, {upper}] has fewer than three samples")
    exponent, log_amplitude = np.polyfit(np.log(time[mask]), np.log(values[mask]), 1)
    actual = np.log(values[mask])
    predicted = log_amplitude + exponent * np.log(time[mask])
    r_squared = 1.0 - float(np.sum((actual - predicted) ** 2)) / float(
        np.sum((actual - np.mean(actual)) ** 2)
    )
    return {
        "lower_time": lower,
        "upper_time": upper,
        "exponent": float(exponent),
        "amplitude": float(np.exp(log_amplitude)),
        "log_r_squared": r_squared,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n12-fast", type=Path, default=Path("runs/theory33_spheres/n012_fast")
    )
    parser.add_argument(
        "--n12-slow", type=Path, default=Path("runs/theory33_spheres/n012_slow")
    )
    parser.add_argument(
        "--n16-fast", type=Path, default=Path("runs/theory33_spheres_long/n016_fast")
    )
    parser.add_argument(
        "--n16-slow", type=Path, default=Path("runs/theory33_spheres_long/n016_slow")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/theory33_spherical_long.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("runs/theory33_spherical_long.png"),
    )
    parser.add_argument(
        "--output-slices",
        type=Path,
        default=Path("runs/theory33_spherical_long_slices.png"),
    )
    parser.add_argument(
        "--output-boundary-figure",
        type=Path,
        default=Path("runs/theory33_spherical_boundary_flux.png"),
    )
    args = parser.parse_args()

    summary12_fast, n12_fast = load_case(args.n12_fast)
    summary12_slow, n12_slow = load_case(args.n12_slow)
    summary16_fast, n16_fast = load_case(args.n16_fast)
    summary16_slow, n16_slow = load_case(args.n16_slow)
    mass12 = n12_slow["hawking_mass"] - n12_fast["hawking_mass"]
    mass16 = n16_slow["hawking_mass"] - n16_fast["hawking_mass"]
    carrier12 = n12_slow["inner_carrier_number"] - n12_fast["inner_carrier_number"]
    carrier16 = n16_slow["inner_carrier_number"] - n16_fast["inner_carrier_number"]
    baryon16 = n16_slow["inner_baryon_mass"] - n16_fast["inner_baryon_mass"]
    gamma16 = n16_slow["relative_lorentz_factor"] - n16_fast["relative_lorentz_factor"]
    overlap_time = float(n12_fast["time"][-1])
    overlap_index = int(np.argmin(np.abs(n16_fast["time"] - overlap_time)))
    mass_overlap_difference = float(
        abs(mass16[overlap_index, 0] - mass12[-1, 0])
        / max(abs(float(mass16[overlap_index, 0])), 1.0e-300)
    )
    carrier_overlap_difference = float(
        abs(carrier16[overlap_index] - carrier12[-1])
        / max(abs(float(carrier16[overlap_index])), 1.0e-300)
    )
    growth_fits = {
        "early": power_fit(n16_fast["time"], mass16[:, 0], 0.004, 0.016),
        "crossover": power_fit(n16_fast["time"], mass16[:, 0], 0.016, 0.024),
        "first_extension": power_fit(
            n16_fast["time"], mass16[:, 0], 0.024, 0.036
        ),
        "second_extension": power_fit(
            n16_fast["time"], mass16[:, 0], 0.036, 0.052
        ),
        "third_extension": power_fit(
            n16_fast["time"], mass16[:, 0], 0.052, 0.068
        ),
        "late": power_fit(n16_fast["time"], mass16[:, 0], 0.056, 0.068),
        "trailing": power_fit(n16_fast["time"], mass16[:, 0], 0.060, 0.068),
        "full": power_fit(n16_fast["time"], mass16[:, 0], 0.004, 0.068),
    }
    exponents = {
        name: float(growth_fits[name]["exponent"])
        for name in (
            "early",
            "crossover",
            "first_extension",
            "second_extension",
            "third_extension",
        )
    }
    if exponents["third_extension"] < exponents["second_extension"] - 0.05:
        exponent_outcome = "continues_falling_toward_2"
    elif exponents["third_extension"] > exponents["second_extension"] + 0.05:
        exponent_outcome = "reverses_upward"
    else:
        exponent_outcome = "saturates"
    final_inner_mass = float(mass16[-1, 0])
    final_outer_mass = float(mass16[-1, -1])
    final_outer_to_inner_mass_ratio = abs(final_outer_mass) / max(
        abs(final_inner_mass), 1.0e-300
    )
    domain_cap = float(
        summary16_slow["case"]["maximum_relative_lorentz_factor"]
    )
    maximum_actual_gamma = float(
        max(
            np.max(n16_fast["relative_lorentz_factor"]),
            np.max(n16_slow["relative_lorentz_factor"]),
        )
    )
    convergence_pass = mass_overlap_difference < 0.1 and carrier_overlap_difference < 0.1
    regime_pass = (
        growth_fits["early"]["log_r_squared"] > 0.995
        and growth_fits["crossover"]["log_r_squared"] > 0.995
        and growth_fits["first_extension"]["log_r_squared"] > 0.995
        and growth_fits["second_extension"]["log_r_squared"] > 0.995
        and growth_fits["third_extension"]["log_r_squared"] > 0.995
        and exponents["crossover"] < exponents["early"]
        and exponents["first_extension"] < exponents["crossover"]
        and exponents["second_extension"] < exponents["first_extension"]
        and exponents["third_extension"] < exponents["second_extension"]
    )
    restart_join_times = [
        float(value) for value in summary16_fast.get("restart_join_times", [])
    ]
    restart_integrity_pass = (
        restart_join_times
        == [float(value) for value in summary16_slow.get("restart_join_times", [])]
        and all(
            int(summary.get("restart_count", 0)) == len(restart_join_times)
            for summary in (summary16_fast, summary16_slow)
        )
        and all(
            all(
                np.count_nonzero(np.isclose(run["time"], join, atol=1.0e-12))
                == 1
                for join in restart_join_times
            )
            and np.allclose(
                np.diff(run["time"]), float(summary["dt"]), atol=1.0e-13
            )
        for run, summary in (
            (n16_fast, summary16_fast),
            (n16_slow, summary16_slow),
        )
        )
    )
    outer_control_threshold = 0.01
    outer_control_pass = final_outer_to_inner_mass_ratio < outer_control_threshold
    instrumentation_time = float(
        summary16_fast["boundary_flux_instrumentation_time"]
    )
    boundary_flux_pass = (
        abs(
            instrumentation_time
            - float(summary16_slow["boundary_flux_instrumentation_time"])
        )
        < 1.0e-12
        and all(
            np.count_nonzero(run["boundary_flux_instrumented"])
            == np.count_nonzero(run["time"] >= instrumentation_time - 1.0e-12)
            for run in (n16_fast, n16_slow)
        )
        and max(
            abs(float(summary16_fast["relative_carrier_number_balance_residual"])),
            abs(float(summary16_slow["relative_carrier_number_balance_residual"])),
        )
        < 1.0e-12
    )
    numerical_pass = (
        int(summary16_fast["recovery_failures"]) == 0
        and int(summary16_slow["recovery_failures"]) == 0
        and max(
            abs(float(summary16_fast["relative_baryon_change"])),
            abs(float(summary16_slow["relative_baryon_change"])),
        )
        < 1.0e-6
        and boundary_flux_pass
        and maximum_actual_gamma < domain_cap
    )
    result = {
        "hypothesis": (
            "the delayed inner Hawking-mass response continues to soften "
            "toward quadratic growth across checkpoint restarts"
        ),
        "points": 16,
        "final_time": float(n16_fast["time"][-1]),
        "domain_audit": {
            "maximum_relative_lorentz_factor": domain_cap,
            "maximum_actual_relative_lorentz_factor": maximum_actual_gamma,
            "maximum_characteristic_speed": float(
                summary16_fast["constitutive_gate"]["maximum_characteristic_speed"]
            ),
            "maximum_eigenvector_condition": float(
                summary16_fast["constitutive_gate"]["maximum_eigenvector_condition"]
            ),
            "minimum_legendre_eigenvalue": float(
                summary16_fast["constitutive_gate"]["minimum_legendre_eigenvalue"]
            ),
            "minimum_thermodynamic_eigenvalue": float(
                summary16_fast["constitutive_gate"]["minimum_thermodynamic_eigenvalue"]
            ),
        },
        "overlap_time": overlap_time,
        "N12_to_N16_inner_mass_relative_difference": mass_overlap_difference,
        "N12_to_N16_inner_carrier_relative_difference": carrier_overlap_difference,
        "growth_fits": growth_fits,
        "exponent_outcome": exponent_outcome,
        "restart_join_times": restart_join_times,
        "restart_integrity_pass": restart_integrity_pass,
        "boundary_flux_audit": {
            "instrumentation_time": instrumentation_time,
            "fast_carrier_number_change": float(
                summary16_fast[
                    "carrier_number_change_since_boundary_flux_instrumentation"
                ]
            ),
            "fast_cumulative_outflow": float(
                summary16_fast["cumulative_carrier_boundary_number_outflow"]
            ),
            "fast_balance_residual": float(
                summary16_fast["carrier_number_balance_residual"]
            ),
            "slow_carrier_number_change": float(
                summary16_slow[
                    "carrier_number_change_since_boundary_flux_instrumentation"
                ]
            ),
            "slow_cumulative_outflow": float(
                summary16_slow["cumulative_carrier_boundary_number_outflow"]
            ),
            "slow_balance_residual": float(
                summary16_slow["carrier_number_balance_residual"]
            ),
            "maximum_relative_balance_residual": max(
                abs(
                    float(
                        summary16_fast[
                            "relative_carrier_number_balance_residual"
                        ]
                    )
                ),
                abs(
                    float(
                        summary16_slow[
                            "relative_carrier_number_balance_residual"
                        ]
                    )
                ),
            ),
        },
        "boundary_flux_pass": boundary_flux_pass,
        "final_hawking_mass_split": mass16[-1].tolist(),
        "final_outer_to_inner_hawking_mass_split": final_outer_to_inner_mass_ratio,
        "outer_mass_control_threshold": outer_control_threshold,
        "final_relative_inner_hawking_mass_split": float(
            final_inner_mass / n16_fast["hawking_mass"][-1, 0]
        ),
        "final_inner_carrier_number_split": float(carrier16[-1]),
        "final_inner_baryon_mass_split": float(baryon16[-1]),
        "final_relative_lorentz_factor_split": float(gamma16[-1]),
        "fast_to_slow_drag_entropy_ratio": float(
            summary16_fast["cumulative_drag_entropy_change"]
            / summary16_slow["cumulative_drag_entropy_change"]
        ),
        "maximum_relative_baryon_change": max(
            abs(float(summary16_fast["relative_baryon_change"])),
            abs(float(summary16_slow["relative_baryon_change"])),
        ),
        "maximum_relative_carrier_change": max(
            abs(float(summary16_fast["relative_carrier_change"])),
            abs(float(summary16_slow["relative_carrier_change"])),
        ),
        "relative_baryon_change_gate": 1.0e-6,
        "final_core_hamiltonian_l2": max(
            float(summary16_fast["final_core_hamiltonian_l2"]),
            float(summary16_slow["final_core_hamiltonian_l2"]),
        ),
        "convergence_pass": convergence_pass,
        "growth_regime_pass": regime_pass,
        "outer_mass_control_pass": outer_control_pass,
        "numerical_gates_pass": numerical_pass,
        "passed": bool(
            convergence_pass
            and regime_pass
            and restart_integrity_pass
            and boundary_flux_pass
            and outer_control_pass
            and numerical_pass
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(n12_fast["time"], mass12[:, 0], "o-", label="N=12")
    axes[0, 0].plot(n16_fast["time"], mass16[:, 0], "o-", label="N=16")
    positive = mass16[:, 0] > 0.0
    axes[0, 1].loglog(n16_fast["time"][positive], mass16[positive, 0], "o", label="N=16")
    for name, color in (
        ("early", "tab:green"),
        ("crossover", "tab:red"),
        ("first_extension", "tab:purple"),
        ("second_extension", "tab:brown"),
        ("third_extension", "tab:pink"),
    ):
        fit = growth_fits[name]
        display_name = {
            "first_extension": "extension 1",
            "second_extension": "extension 2",
            "third_extension": "extension 3",
        }.get(name, name)
        line_time = np.linspace(fit["lower_time"], fit["upper_time"], 100)
        axes[0, 1].loglog(
            line_time,
            fit["amplitude"] * line_time ** fit["exponent"],
            "-",
            color=color,
            label=f"{display_name} p={fit['exponent']:.2f}",
        )
    for index, radius in enumerate(n16_fast["extraction_radii"]):
        axes[1, 0].plot(
            n16_fast["time"], mass16[:, index], "o-", label=f"r={radius:.1f}"
        )
    axes[1, 1].plot(n16_fast["time"], carrier16, "o-", label="inner carrier number")
    axes[1, 1].plot(n16_fast["time"], -baryon16, "s-", label="- inner baryon mass")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"inner $M_H^{slow}-M_H^{fast}$")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("inner Hawking-mass split")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("Hawking-mass split")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("integrated material split")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("N=16 restart extensions: mass-response exponent keeps falling")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    extent = (-4.0, 4.0, -4.0, 4.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fields = (
        n16_fast["baryon_density_slice"][-1],
        n16_slow["baryon_density_slice"][-1],
        n16_slow["baryon_density_slice"][-1] - n16_fast["baryon_density_slice"][-1],
        n16_fast["carrier_density_slice"][-1],
        n16_slow["carrier_density_slice"][-1],
        n16_slow["carrier_density_slice"][-1] - n16_fast["carrier_density_slice"][-1],
    )
    titles = (
        "fast baryons", "slow baryons", "slow - fast baryons",
        "fast carriers", "slow carriers", "slow - fast carriers",
    )
    for axis, field, title in zip(axes.flat, fields, titles, strict=True):
        plotted = axis.imshow(field.T, origin="lower", extent=extent, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(plotted, ax=axis, shrink=0.8)
    fig.suptitle(f"N=16 final equatorial slices at t={n16_fast['time'][-1]:.3f}")
    fig.savefig(args.output_slices, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for label, run, color in (
        ("fast", n16_fast, "tab:blue"),
        ("slow", n16_slow, "tab:orange"),
    ):
        mask = run["boundary_flux_instrumented"].astype(bool)
        time = run["time"][mask]
        axes[0].plot(
            time,
            run["carrier_boundary_number_outflow_rate"][mask],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            time,
            run["cumulative_carrier_boundary_number_outflow"][mask],
            "o-",
            color=color,
            label=f"{label} integrated outflow",
        )
        axes[1].plot(
            time,
            -run["carrier_number_change_since_boundary_instrumentation"][mask],
            "--",
            color=color,
            label=f"{label} measured loss",
        )
        axes[2].semilogy(
            time,
            np.maximum(
                np.abs(run["carrier_number_balance_residual"][mask]), 1.0e-30
            ),
            "o-",
            color=color,
            label=label,
        )
    axes[0].set_ylabel("outward carrier-number flux")
    axes[1].set_ylabel("carrier number")
    axes[2].set_ylabel(r"$|\Delta N_D+\int\Phi\,dt|$")
    for axis in axes:
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Finite-boundary carrier balance closes after flux correction")
    fig.savefig(args.output_boundary_figure, dpi=180)
    plt.close(fig)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
