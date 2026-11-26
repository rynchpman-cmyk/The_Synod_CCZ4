"""Analyze and render the Theory 3.3 carrier-coefficient attack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    exponent, log_coefficient = np.polyfit(np.log(x), np.log(y), 1)
    actual = np.log(y)
    predicted = log_coefficient + exponent * np.log(x)
    r_squared = 1.0 - float(np.sum((actual - predicted) ** 2)) / float(
        np.sum((actual - np.mean(actual)) ** 2)
    )
    return float(exponent), float(np.exp(log_coefficient)), r_squared


def load_splits(path: Path, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))["cases"]
    indexed = {
        (
            float(row["case"]["target_fraction"]),
            float(row["case"]["relaxation_time"]),
        ): row
        for row in rows
    }
    compression = []
    lapse = []
    for fraction in fractions:
        fast = indexed[(float(fraction), 0.03)]
        slow = indexed[(float(fraction), 0.30)]
        compression.append(
            float(fast["peak_compression_ratio"])
            - float(slow["peak_compression_ratio"])
        )
        lapse.append(float(slow["minimum_lapse"]) - float(fast["minimum_lapse"]))
    return np.asarray(compression), np.asarray(lapse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kd005", type=Path)
    parser.add_argument("kd010", type=Path)
    parser.add_argument("kd020", type=Path)
    parser.add_argument("kd040", type=Path)
    parser.add_argument(
        "--output-json", type=Path, default=Path("runs/theory33_kd_attack.json")
    )
    parser.add_argument(
        "--output-figure", type=Path, default=Path("runs/theory33_kd_attack.png")
    )
    args = parser.parse_args()

    fractions = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    coefficients = np.array([0.05, 0.10, 0.20, 0.40])
    paths = [args.kd005, args.kd010, args.kd020, args.kd040]
    predicted_exponent = 1.0 + 0.7**2
    fits: dict[str, dict[str, object]] = {}
    compression_exponents = []
    lapse_exponents = []
    compression_amplitudes = []
    lapse_amplitudes = []

    for coefficient, path in zip(coefficients, paths, strict=True):
        compression, lapse = load_splits(path, fractions)
        compression_fit = power_fit(fractions, compression)
        lapse_fit = power_fit(fractions, lapse)
        key = f"{coefficient:.2f}"
        fits[key] = {
            "compression_split": compression.tolist(),
            "lapse_split": lapse.tolist(),
            "compression_exponent": compression_fit[0],
            "compression_amplitude": compression_fit[1],
            "compression_log_r_squared": compression_fit[2],
            "lapse_exponent": lapse_fit[0],
            "lapse_amplitude": lapse_fit[1],
            "lapse_log_r_squared": lapse_fit[2],
        }
        compression_exponents.append(compression_fit[0])
        lapse_exponents.append(lapse_fit[0])
        compression_amplitudes.append(compression_fit[1])
        lapse_amplitudes.append(lapse_fit[1])

    compression_exponents_array = np.asarray(compression_exponents)
    lapse_exponents_array = np.asarray(lapse_exponents)
    compression_amplitude_fit = power_fit(
        coefficients, np.asarray(compression_amplitudes)
    )
    lapse_amplitude_fit = power_fit(coefficients, np.asarray(lapse_amplitudes))
    individual_fits_pass = all(
        row["compression_log_r_squared"] > 0.995
        and row["lapse_log_r_squared"] > 0.995
        for row in fits.values()
    )
    exponent_invariance_pass = (
        float(np.max(np.abs(compression_exponents_array - predicted_exponent))) < 0.1
        and float(np.max(np.abs(lapse_exponents_array - predicted_exponent))) < 0.1
        and float(np.ptp(compression_exponents_array)) < 0.1
        and float(np.ptp(lapse_exponents_array)) < 0.1
    )
    amplitude_scaling_pass = (
        abs(compression_amplitude_fit[0] - 1.0) < 0.15
        and abs(lapse_amplitude_fit[0] - 1.0) < 0.15
        and compression_amplitude_fit[2] > 0.995
        and lapse_amplitude_fit[2] > 0.995
    )
    result = {
        "hypothesis": (
            "at fixed carrier sound speed, K_D changes the branch-split amplitude "
            "linearly without changing the carrier-fraction exponent"
        ),
        "carrier_sound_speed": 0.7,
        "predicted_fraction_exponent": predicted_exponent,
        "fractions": fractions.tolist(),
        "carrier_coefficients": coefficients.tolist(),
        "fits": fits,
        "compression_exponent_span": float(np.ptp(compression_exponents_array)),
        "lapse_exponent_span": float(np.ptp(lapse_exponents_array)),
        "compression_amplitude_K_exponent": compression_amplitude_fit[0],
        "compression_amplitude_K_coefficient": compression_amplitude_fit[1],
        "compression_amplitude_log_r_squared": compression_amplitude_fit[2],
        "lapse_amplitude_K_exponent": lapse_amplitude_fit[0],
        "lapse_amplitude_K_coefficient": lapse_amplitude_fit[1],
        "lapse_amplitude_log_r_squared": lapse_amplitude_fit[2],
        "operational_criteria": {
            "minimum_individual_log_r_squared": 0.995,
            "maximum_absolute_fraction_exponent_error": 0.1,
            "maximum_fraction_exponent_span": 0.1,
            "maximum_amplitude_K_exponent_error": 0.15,
            "minimum_amplitude_log_r_squared": 0.995,
        },
        "individual_power_laws_pass": individual_fits_pass,
        "fraction_exponent_invariance_pass": exponent_invariance_pass,
        "amplitude_scaling_pass": amplitude_scaling_pass,
        "passed": (
            individual_fits_pass and exponent_invariance_pass and amplitude_scaling_pass
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(coefficients)))
    for coefficient, color in zip(coefficients, colors, strict=True):
        row = fits[f"{coefficient:.2f}"]
        axes[0, 0].loglog(
            fractions,
            row["compression_split"],
            "o-",
            color=color,
            label=fr"$K_D={coefficient:.2f}$, $p={row['compression_exponent']:.3f}$",
        )
        axes[0, 1].loglog(
            fractions,
            row["lapse_split"],
            "o-",
            color=color,
            label=fr"$K_D={coefficient:.2f}$, $p={row['lapse_exponent']:.3f}$",
        )
    axes[1, 0].loglog(
        coefficients, compression_amplitudes, "o-", label="compression amplitude"
    )
    axes[1, 0].loglog(coefficients, lapse_amplitudes, "s-", label="lapse amplitude")
    axes[1, 1].axhline(
        predicted_exponent, color="black", linestyle="--", label=r"predicted $1+c_D^2$"
    )
    axes[1, 1].plot(
        coefficients, compression_exponents, "o-", label="compression exponent"
    )
    axes[1, 1].plot(coefficients, lapse_exponents, "s-", label="lapse exponent")
    axes[0, 0].set_ylabel("fast/slow compression split")
    axes[0, 1].set_ylabel("fast/slow lapse split")
    axes[1, 0].set_ylabel("fitted split amplitude")
    axes[1, 1].set_ylabel("fitted carrier-fraction exponent")
    for axis in axes[0]:
        axis.set_xlabel(r"carrier fraction $f_D$")
    for axis in axes[1]:
        axis.set_xlabel(r"carrier coefficient $K_D$")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(r"Theory 3.3 $K_D$ attack: amplitude changes, loading power survives")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
