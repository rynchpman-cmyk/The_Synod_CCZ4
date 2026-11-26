"""Continuum extrapolation of the Theory 3.3 K_D coefficient campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COEFFICIENTS = np.array([0.05, 0.10, 0.20, 0.40])
FRACTIONS = np.array([0.10, 0.20, 0.50, 1.00, 2.00])
RESOLUTIONS = np.array([16.0, 24.0, 32.0])


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    exponent, log_amplitude = np.polyfit(np.log(x), np.log(y), 1)
    actual = np.log(y)
    predicted = log_amplitude + exponent * np.log(x)
    r_squared = 1.0 - float(np.sum((actual - predicted) ** 2)) / float(
        np.sum((actual - np.mean(actual)) ** 2)
    )
    return float(exponent), float(np.exp(log_amplitude)), r_squared


def continuum_fit(
    resolutions: np.ndarray, values: np.ndarray, order: float
) -> dict[str, object]:
    coordinate = resolutions ** (-order)
    slope, continuum = np.polyfit(coordinate, values, 1)
    fitted = continuum + slope * coordinate
    scale = max(abs(float(continuum)), 1.0e-300)
    return {
        "continuum": float(continuum),
        "coefficient": float(slope),
        "order": float(order),
        "fitted": fitted.tolist(),
        "maximum_normalized_residual": float(np.max(np.abs(values - fitted)) / scale),
        "fine_to_continuum_relative_shift": float(
            (continuum - values[-1]) / max(abs(float(values[-1])), 1.0e-300)
        ),
    }


def shared_observed_order(sequences: np.ndarray) -> float:
    """Fit one convergence order shared by a family of three-level sequences."""
    best_score = np.inf
    best_order = 2.0
    for order in np.linspace(0.5, 6.0, 1101):
        coordinate = RESOLUTIONS ** (-order)
        score = 0.0
        for values in sequences.reshape(-1, len(RESOLUTIONS)):
            fitted = np.polyval(np.polyfit(coordinate, values, 1), coordinate)
            score += float(
                np.sum(
                    ((values - fitted) / max(abs(float(values[-1])), 1.0e-300))
                    ** 2
                )
            )
        if score < best_score:
            best_score = score
            best_order = float(order)
    return best_order


def load_scan(path: Path) -> dict[tuple[float, float], dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {
        (
            float(row["case"]["target_fraction"]),
            float(row["case"]["relaxation_time"]),
        ): row
        for row in rows
    }


def scan_paths(args: argparse.Namespace) -> dict[int, dict[float, Path]]:
    coarse = {
        0.05: args.coarse_kd005,
        0.10: args.coarse_kd010,
        0.20: args.coarse_kd020,
        0.40: args.coarse_kd040,
    }
    return {
        16: coarse,
        24: {
            coefficient: args.medium_root
            / f"kd{int(round(100 * coefficient)):03d}"
            / "scan_summary.json"
            for coefficient in COEFFICIENTS
        },
        32: {
            coefficient: args.fine_root
            / f"kd{int(round(100 * coefficient)):03d}"
            / "scan_summary.json"
            for coefficient in COEFFICIENTS
        },
    }


def fit_power_families(values: np.ndarray) -> tuple[list[dict[str, float]], dict[str, float]]:
    families = []
    amplitudes = []
    for coefficient, row in zip(COEFFICIENTS, values, strict=True):
        exponent, amplitude, r_squared = power_fit(FRACTIONS, row)
        families.append(
            {
                "carrier_K": float(coefficient),
                "fraction_exponent": exponent,
                "amplitude": amplitude,
                "log_r_squared": r_squared,
            }
        )
        amplitudes.append(amplitude)
    exponent, amplitude, r_squared = power_fit(COEFFICIENTS, np.asarray(amplitudes))
    return families, {
        "K_exponent": exponent,
        "amplitude": amplitude,
        "log_r_squared": r_squared,
    }


def separable_power_fit(values: np.ndarray) -> dict[str, float]:
    design = []
    actual = []
    for ki, coefficient in enumerate(COEFFICIENTS):
        for fi, fraction in enumerate(FRACTIONS):
            design.append([1.0, np.log(coefficient), np.log(fraction)])
            actual.append(np.log(values[ki, fi]))
    design_array = np.asarray(design)
    actual_array = np.asarray(actual)
    intercept, K_exponent, fraction_exponent = np.linalg.lstsq(
        design_array, actual_array, rcond=None
    )[0]
    predicted = design_array @ np.array([intercept, K_exponent, fraction_exponent])
    r_squared = 1.0 - float(np.sum((actual_array - predicted) ** 2)) / float(
        np.sum((actual_array - np.mean(actual_array)) ** 2)
    )
    return {
        "amplitude": float(np.exp(intercept)),
        "K_exponent": float(K_exponent),
        "fraction_exponent": float(fraction_exponent),
        "log_r_squared": r_squared,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coarse-kd005",
        type=Path,
        default=Path("runs/theory33_kd_005/scan_summary.json"),
    )
    parser.add_argument(
        "--coarse-kd010",
        type=Path,
        default=Path("runs/theory33_kd_010/scan_summary.json"),
    )
    parser.add_argument(
        "--coarse-kd020",
        type=Path,
        default=Path("runs/theory33_eos_cs070/scan_summary.json"),
    )
    parser.add_argument(
        "--coarse-kd040",
        type=Path,
        default=Path("runs/theory33_kd_040/scan_summary.json"),
    )
    parser.add_argument(
        "--medium-root",
        type=Path,
        default=Path("runs/theory33_kd_continuum/n024"),
    )
    parser.add_argument(
        "--fine-root",
        type=Path,
        default=Path("runs/theory33_kd_continuum/n032"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/theory33_kd_continuum.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("runs/theory33_kd_continuum.png"),
    )
    parser.add_argument(
        "--output-convergence-figure",
        type=Path,
        default=Path("runs/theory33_kd_continuum_convergence.png"),
    )
    args = parser.parse_args()

    paths = scan_paths(args)
    scans = {
        resolution: {
            coefficient: load_scan(path)
            for coefficient, path in coefficient_paths.items()
        }
        for resolution, coefficient_paths in paths.items()
    }
    shape = (len(COEFFICIENTS), len(FRACTIONS), len(RESOLUTIONS))
    compression = np.empty(shape)
    mean_lapse = np.empty(shape)
    point_minimum_lapse = np.empty(shape)
    gate_failures = []
    maximum_current_error = 0.0
    recovery_failures = 0
    minimum_drag_entropy = np.inf
    minimum_projection_entropy = np.inf
    minimum_step_entropy = np.inf

    for ni, resolution_value in enumerate(RESOLUTIONS.astype(int)):
        for ki, coefficient in enumerate(COEFFICIENTS):
            scan = scans[resolution_value][float(coefficient)]
            base = paths[resolution_value][float(coefficient)].parent
            for fi, fraction in enumerate(FRACTIONS):
                fast = scan[(float(fraction), 0.03)]
                slow = scan[(float(fraction), 0.30)]
                compression[ki, fi, ni] = float(fast["peak_compression_ratio"]) - float(
                    slow["peak_compression_ratio"]
                )
                point_minimum_lapse[ki, fi, ni] = float(slow["minimum_lapse"]) - float(
                    fast["minimum_lapse"]
                )
                fast_lapse = np.load(base / str(fast["name"]) / "trajectory.npz")[
                    "lapse"
                ][-1]
                slow_lapse = np.load(base / str(slow["name"]) / "trajectory.npz")[
                    "lapse"
                ][-1]
                mean_lapse[ki, fi, ni] = float(np.mean(slow_lapse) - np.mean(fast_lapse))
                for row in (fast, slow):
                    if not bool(row["constitutive_gate"]["passed"]) or not bool(
                        row["ctt"]["converged"]
                    ):
                        gate_failures.append(str(row["name"]))
                    recovery_failures += int(row["recovery_failures"])
                    minimum_drag_entropy = min(
                        minimum_drag_entropy,
                        float(row["cumulative_drag_entropy_change"]),
                    )
                    minimum_projection_entropy = min(
                        minimum_projection_entropy,
                        float(row["cumulative_projection_entropy_change"]),
                    )
                    minimum_step_entropy = min(
                        minimum_step_entropy,
                        float(row["cumulative_step_entropy_change"]),
                    )
                    maximum_current_error = max(
                        maximum_current_error,
                        abs(float(row["relative_baryon_error"])),
                        abs(float(row["relative_carrier_error"])),
                    )

    central_order = 2.0
    observables = {
        "compression": compression,
        "mean_final_lapse": mean_lapse,
        "point_minimum_lapse": point_minimum_lapse,
    }
    continuum: dict[str, np.ndarray] = {}
    fit_records: dict[str, list[list[dict[str, object]]]] = {}
    for name, sequences in observables.items():
        continuum[name] = np.empty((len(COEFFICIENTS), len(FRACTIONS)))
        fit_records[name] = []
        for ki in range(len(COEFFICIENTS)):
            row_records = []
            for fi in range(len(FRACTIONS)):
                record = continuum_fit(RESOLUTIONS, sequences[ki, fi], central_order)
                continuum[name][ki, fi] = float(record["continuum"])
                record["raw"] = sequences[ki, fi].tolist()
                row_records.append(record)
            fit_records[name].append(row_records)

    observed_orders = {
        name: shared_observed_order(sequences)
        for name, sequences in (
            ("compression", compression),
            ("mean_final_lapse", mean_lapse),
        )
    }
    shared_order_continuum = {}
    for name, sequences in (
        ("compression", compression),
        ("mean_final_lapse", mean_lapse),
    ):
        values = np.empty((len(COEFFICIENTS), len(FRACTIONS)))
        for ki in range(len(COEFFICIENTS)):
            for fi in range(len(FRACTIONS)):
                values[ki, fi] = continuum_fit(
                    RESOLUTIONS, sequences[ki, fi], observed_orders[name]
                )["continuum"]
        shared_order_continuum[name] = values

    central_family_fits = {}
    fine_family_fits = {}
    shared_order_family_fits = {}
    for name in ("compression", "mean_final_lapse"):
        families, amplitude_fit = fit_power_families(continuum[name])
        central_family_fits[name] = {
            "families": families,
            "amplitude_scaling": amplitude_fit,
        }
        families, amplitude_fit = fit_power_families(observables[name][:, :, -1])
        fine_family_fits[name] = {
            "families": families,
            "amplitude_scaling": amplitude_fit,
        }
        families, amplitude_fit = fit_power_families(shared_order_continuum[name])
        shared_order_family_fits[name] = {
            "families": families,
            "amplitude_scaling": amplitude_fit,
        }
    separable_continuum_fits = {
        name: separable_power_fit(continuum[name])
        for name in ("compression", "mean_final_lapse")
    }

    def maximum_fit_residual(name: str) -> float:
        return max(
            float(record["maximum_normalized_residual"])
            for row in fit_records[name]
            for record in row
        )

    def shift_range(name: str) -> list[float]:
        shifts = [
            abs(float(record["fine_to_continuum_relative_shift"]))
            for row in fit_records[name]
            for record in row
        ]
        return [float(min(shifts)), float(max(shifts))]

    nonmonotone_minimum_count = sum(
        1
        for values in point_minimum_lapse.reshape(-1, len(RESOLUTIONS))
        if np.diff(values)[0] * np.diff(values)[1] < 0.0
    )
    predicted_exponent = 1.0 + 0.7**2
    accepted = ("compression", "mean_final_lapse")
    positivity_pass = all(np.all(continuum[name] > 0.0) for name in accepted)
    convergence_pass = all(maximum_fit_residual(name) < 0.01 for name in accepted)
    physical_fit_pass = all(
        min(row["log_r_squared"] for row in central_family_fits[name]["families"])
        > 0.995
        and max(
            abs(row["fraction_exponent"] - predicted_exponent)
            for row in central_family_fits[name]["families"]
        )
        < 0.1
        and abs(
            central_family_fits[name]["amplitude_scaling"]["K_exponent"] - 1.0
        )
        < 0.1
        for name in accepted
    )
    result = {
        "hypothesis": "continuum response factorizes as K_D f_D^(1+c_D^2)",
        "carrier_sound_speed": 0.7,
        "predicted_fraction_exponent": predicted_exponent,
        "resolutions": RESOLUTIONS.tolist(),
        "time_steps": [0.002, 0.12 / 90.0, 0.001],
        "carrier_coefficients": COEFFICIENTS.tolist(),
        "carrier_fractions": FRACTIONS.tolist(),
        "central_extrapolation_order": central_order,
        "shared_observed_orders": observed_orders,
        "continuum_values": {
            name: values.tolist() for name, values in continuum.items()
        },
        "continuum_records": fit_records,
        "central_power_fits": central_family_fits,
        "separable_continuum_power_fits": separable_continuum_fits,
        "fine_grid_power_fits": fine_family_fits,
        "shared_order_sensitivity_power_fits": shared_order_family_fits,
        "convergence": {
            "compression_maximum_normalized_residual": maximum_fit_residual(
                "compression"
            ),
            "mean_lapse_maximum_normalized_residual": maximum_fit_residual(
                "mean_final_lapse"
            ),
            "point_minimum_lapse_maximum_normalized_residual": maximum_fit_residual(
                "point_minimum_lapse"
            ),
            "compression_fine_to_continuum_absolute_shift_range": shift_range(
                "compression"
            ),
            "mean_lapse_fine_to_continuum_absolute_shift_range": shift_range(
                "mean_final_lapse"
            ),
            "point_minimum_lapse_nonmonotone_sequences": nonmonotone_minimum_count,
        },
        "point_minimum_lapse_status": (
            "rejected for continuum power fit because the discrete spatial "
            "minimum is grid-phase sensitive"
        ),
        "gates": {
            "failed_cases": gate_failures,
            "recovery_failures": recovery_failures,
            "maximum_relative_current_error": maximum_current_error,
            "minimum_cumulative_drag_entropy_change": float(minimum_drag_entropy),
            "minimum_cumulative_projection_entropy_change": float(
                minimum_projection_entropy
            ),
            "minimum_cumulative_step_entropy_change": float(minimum_step_entropy),
        },
        "operational_criteria": {
            "positive_extrapolated_splits": positivity_pass,
            "maximum_normalized_continuum_residual": 0.01,
            "minimum_power_log_r_squared": 0.995,
            "maximum_fraction_exponent_error": 0.1,
            "maximum_K_exponent_error": 0.1,
        },
        "convergence_pass": convergence_pass,
        "physical_fit_pass": physical_fit_pass,
        "passed": bool(
            positivity_pass
            and convergence_pass
            and physical_fit_pass
            and not gate_failures
            and recovery_failures == 0
            and minimum_drag_entropy > 0.0
            and minimum_projection_entropy > 0.0
            and minimum_step_entropy > 0.0
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(COEFFICIENTS)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ki, (coefficient, color) in enumerate(
        zip(COEFFICIENTS, colors, strict=True)
    ):
        for axis, name in zip(axes[0], accepted, strict=True):
            axis.loglog(
                FRACTIONS,
                observables[name][ki, :, -1],
                "o--",
                alpha=0.45,
                color=color,
            )
            exponent = central_family_fits[name]["families"][ki][
                "fraction_exponent"
            ]
            axis.loglog(
                FRACTIONS,
                continuum[name][ki],
                "o-",
                color=color,
                label=fr"$K_D={coefficient:.2f}$, $p_\infty={exponent:.3f}$",
            )
    for name, marker in zip(accepted, ("o", "s"), strict=True):
        amplitudes = [
            row["amplitude"] for row in central_family_fits[name]["families"]
        ]
        label = "compression" if name == "compression" else "mean lapse"
        axes[1, 0].loglog(COEFFICIENTS, amplitudes, marker + "-", label=label)
        exponents = [
            row["fraction_exponent"]
            for row in central_family_fits[name]["families"]
        ]
        axes[1, 1].plot(COEFFICIENTS, exponents, marker + "-", label=label)
    axes[1, 1].axhline(
        predicted_exponent,
        color="black",
        linestyle="--",
        label=r"predicted $1+c_D^2$",
    )
    axes[0, 0].set_ylabel("compression branch split")
    axes[0, 1].set_ylabel("final mean-lapse branch split")
    axes[1, 0].set_ylabel("continuum fitted amplitude")
    axes[1, 1].set_ylabel("continuum carrier-fraction exponent")
    for axis in axes[0]:
        axis.set_xlabel(r"carrier fraction $f_D$")
    for axis in axes[1]:
        axis.set_xlabel(r"carrier coefficient $K_D$")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Theory 3.3 coefficient law after continuum extrapolation")
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    convergence_sets = (
        ("compression", compression),
        ("mean final lapse", mean_lapse),
        ("rejected point minimum", point_minimum_lapse),
    )
    coordinate = RESOLUTIONS ** -2
    for axis, (label, sequences) in zip(axes, convergence_sets, strict=True):
        for values in sequences.reshape(-1, len(RESOLUTIONS)):
            axis.plot(coordinate, values / values[-1], "o-", alpha=0.45)
        axis.set_xlabel(r"$N^{-2}$")
        axis.set_ylabel(r"split / split$_{N=32}$")
        axis.set_title(label)
        axis.grid(alpha=0.25)
    fig.suptitle("Three-level convergence: smooth averages versus a discrete extremum")
    fig.savefig(args.output_convergence_figure, dpi=180)
    plt.close(fig)

    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
