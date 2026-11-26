"""Fit and render the Theory 3.3 carrier-EOS falsification campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    exponent, log_coefficient = np.polyfit(np.log(x), np.log(y), 1)
    predicted = log_coefficient + exponent * np.log(x)
    actual = np.log(y)
    r_squared = 1.0 - float(np.sum((actual - predicted) ** 2)) / float(
        np.sum((actual - np.mean(actual)) ** 2)
    )
    return float(exponent), float(np.exp(log_coefficient)), r_squared


def load_branches(path: Path, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        lapse.append(
            float(slow["minimum_lapse"]) - float(fast["minimum_lapse"])
        )
    return np.asarray(compression), np.asarray(lapse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cs050", type=Path)
    parser.add_argument("cs070", type=Path)
    parser.add_argument("cs090", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("runs/theory33_eos_falsification.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("runs/theory33_eos_falsification.png"),
    )
    args = parser.parse_args()
    fractions = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    paths = {0.5: args.cs050, 0.7: args.cs070, 0.9: args.cs090}
    fits: dict[float, dict[str, object]] = {}
    for sound_speed, path in paths.items():
        compression, lapse = load_branches(path, fractions)
        compression_fit = power_fit(fractions, compression)
        lapse_fit = power_fit(fractions, lapse)
        fits[sound_speed] = {
            "predicted_exponent": 1.0 + sound_speed**2,
            "compression_split": compression.tolist(),
            "lapse_split": lapse.tolist(),
            "compression_exponent": compression_fit[0],
            "compression_coefficient": compression_fit[1],
            "compression_log_r_squared": compression_fit[2],
            "lapse_exponent": lapse_fit[0],
            "lapse_coefficient": lapse_fit[1],
            "lapse_log_r_squared": lapse_fit[2],
        }
    predicted = np.array([fits[value]["predicted_exponent"] for value in paths])
    compression_exponents = np.array(
        [fits[value]["compression_exponent"] for value in paths]
    )
    lapse_exponents = np.array([fits[value]["lapse_exponent"] for value in paths])
    compression_tracking = np.polyfit(predicted, compression_exponents, 1)
    lapse_tracking = np.polyfit(predicted, lapse_exponents, 1)

    def tracking_r_squared(values: np.ndarray, coefficients: np.ndarray) -> float:
        estimate = np.polyval(coefficients, predicted)
        return 1.0 - float(np.sum((values - estimate) ** 2)) / float(
            np.sum((values - np.mean(values)) ** 2)
        )

    passed = all(
        float(fits[value]["compression_log_r_squared"]) > 0.995
        and float(fits[value]["lapse_log_r_squared"]) > 0.995
        and abs(
            float(fits[value]["compression_exponent"])
            - float(fits[value]["predicted_exponent"])
        )
        < 0.1
        and abs(
            float(fits[value]["lapse_exponent"])
            - float(fits[value]["predicted_exponent"])
        )
        < 0.1
        for value in paths
    )
    result = {
        "hypothesis": "loading exponent tracks 1 + carrier_sound_speed^2",
        "fractions": fractions.tolist(),
        "fits": {str(key): value for key, value in fits.items()},
        "compression_tracking_slope": float(compression_tracking[0]),
        "compression_tracking_intercept": float(compression_tracking[1]),
        "compression_tracking_r_squared": tracking_r_squared(
            compression_exponents, compression_tracking
        ),
        "lapse_tracking_slope": float(lapse_tracking[0]),
        "lapse_tracking_intercept": float(lapse_tracking[1]),
        "lapse_tracking_r_squared": tracking_r_squared(
            lapse_exponents, lapse_tracking
        ),
        "pass_criteria": {
            "minimum_individual_log_r_squared": 0.995,
            "maximum_absolute_exponent_error": 0.1,
        },
        "passed": passed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = {0.5: "tab:blue", 0.7: "tab:orange", 0.9: "tab:green"}
    for sound_speed in paths:
        row = fits[sound_speed]
        color = colors[sound_speed]
        axes[0].loglog(
            fractions,
            row["compression_split"],
            "o-",
            color=color,
            label=fr"$c_D={sound_speed:.1f}$, $p={row['compression_exponent']:.3f}$",
        )
        axes[1].loglog(
            fractions,
            row["lapse_split"],
            "o-",
            color=color,
            label=fr"$c_D={sound_speed:.1f}$, $p={row['lapse_exponent']:.3f}$",
        )
    bounds = np.array([1.15, 1.9])
    axes[2].plot(bounds, bounds, "k--", label="exact tracking")
    axes[2].plot(predicted, compression_exponents, "o-", label="compression exponent")
    axes[2].plot(predicted, lapse_exponents, "s-", label="lapse exponent")
    axes[0].set_ylabel("fast/slow compression split")
    axes[1].set_ylabel("fast/slow lapse split")
    axes[2].set_xlabel(r"predicted exponent $1+c_D^2$")
    axes[2].set_ylabel("measured loading exponent")
    for axis in axes[:2]:
        axis.set_xlabel(r"carrier fraction $f_D$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Theory 3.3 EOS falsification test: measured loading powers track $1+c_D^2$")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
