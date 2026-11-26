"""Compact high-resolution calibration of the repair-field trapping boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predict_trapping_boundary import TARGET, design, interpolate_crossing
from scan_trapping_boundary import ScanCache


CALIBRATION_MASSES = (0.35, 0.80, 1.25)
CALIBRATION_TAUS = (0.15, 0.60, 2.40)


def locate(cache: ScanCache, mass: float, tau: float, center: float,
           points: int, half_width: float, iterations: int) -> tuple[float, list[dict]]:
    lower_loading = max(0.0, center-half_width)
    upper_loading = center+half_width
    lower = cache.evaluate(mass, tau, lower_loading, points)
    upper = cache.evaluate(mass, tau, upper_loading, points)
    while upper["core_retention"] < TARGET:
        upper_loading += half_width
        upper = cache.evaluate(mass, tau, upper_loading, points)
    if lower["core_retention"] >= TARGET:
        lower = cache.evaluate(mass, tau, 0.0, points)
    if lower["core_retention"] >= TARGET:
        return 0.0, [lower, upper]
    samples = [lower, upper]
    for _ in range(iterations):
        middle = cache.evaluate(
            mass, tau, 0.5*(lower["loading"]+upper["loading"]), points
        )
        samples.append(middle)
        if middle["core_retention"] >= TARGET:
            upper = middle
        else:
            lower = middle
    crossing = interpolate_crossing([lower, upper])
    if crossing is None:
        raise RuntimeError("failed to interpolate a bracketed crossing")
    return crossing, sorted(samples, key=lambda item: item["loading"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-model", type=Path, default=Path("runs/trapping_boundary/locked_forecasts.json"))
    parser.add_argument("--validation", type=Path, default=Path("runs/trapping_boundary/convergence_256.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/trapping_boundary/calibration_256_cache.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/trapping_boundary/calibrated_boundary_256.json"))
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--half-width", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    coarse = json.loads(args.coarse_model.read_text(encoding="utf-8"))
    coefficient_map = coarse["model"]["coefficients"]
    coarse_coefficients = np.array([coefficient_map[key] for key in ("a", "b", "c", "d")])
    cache = ScanCache(args.cache)
    regimes = []
    for mass in CALIBRATION_MASSES:
        for tau in CALIBRATION_TAUS:
            center = float(design(mass, tau) @ coarse_coefficients)
            critical, samples = locate(
                cache, mass, tau, center, args.points, args.half_width, args.iterations
            )
            regime = {
                "mass": mass,
                "tau_star": tau,
                "coarse_model_center": center,
                "critical_loading_retention_0p20": critical,
                "samples": samples,
            }
            regimes.append(regime)
            print(json.dumps(regime, sort_keys=True), flush=True)

    x = np.stack([design(row["mass"], row["tau_star"]) for row in regimes])
    y = np.array([row["critical_loading_retention_0p20"] for row in regimes])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficients
    calibration_error = fitted-y

    validation_payload = json.loads(args.validation.read_text(encoding="utf-8"))
    validation = []
    for row in validation_payload["results"]:
        actual = row["critical_loading"]
        predicted = float(design(row["mass"], row["tau_star"]) @ coefficients)
        validation.append({
            "name": row["name"], "mass": row["mass"], "tau_star": row["tau_star"],
            "predicted_critical_loading": predicted,
            "measured_critical_loading": actual,
            "absolute_error": abs(predicted-actual),
        })
    errors = np.array([row["absolute_error"] for row in validation])
    # The lightest/slowest sample occupies a distinct low-node branch and is
    # reported separately. This branch split was discovered after the scan, so
    # the following regular-branch fit is explicitly post-hoc, not blind.
    regular = [
        row for row in regimes
        if not (row["mass"] == min(CALIBRATION_MASSES)
                and row["tau_star"] == max(CALIBRATION_TAUS))
    ]
    regular_x = np.stack([design(row["mass"], row["tau_star"]) for row in regular])
    regular_y = np.array([row["critical_loading_retention_0p20"] for row in regular])
    regular_coefficients = np.linalg.lstsq(regular_x, regular_y, rcond=None)[0]
    regular_residual = regular_x @ regular_coefficients-regular_y
    regular_validation = []
    for row in validation_payload["results"]:
        predicted = float(design(row["mass"], row["tau_star"]) @ regular_coefficients)
        regular_validation.append({
            "name": row["name"],
            "predicted_critical_loading": predicted,
            "measured_critical_loading": row["critical_loading"],
            "absolute_error": abs(predicted-row["critical_loading"]),
        })
    regular_errors = np.array([row["absolute_error"] for row in regular_validation])
    payload = {
        "status": "HIGH_RESOLUTION_BOUNDARY_CALIBRATED_AND_WITHHELD_CASES_RESOLVED",
        "definition": "20% repair energy retained inside abs(x)<3.5 at T=8",
        "points": args.points,
        "fit_form": "q_crit = a - b*m_A^2 + c*ln(tau_star) + d*ln(tau_star)^2",
        "coefficients": {key: float(value) for key, value in zip(("a", "b", "c", "d"), coefficients)},
        "calibration_mae": float(np.mean(np.abs(calibration_error))),
        "calibration_max_error": float(np.max(np.abs(calibration_error))),
        "withheld_mae": float(np.mean(errors)),
        "withheld_max_error": float(np.max(errors)),
        "regimes": regimes,
        "withheld_validation": validation,
        "posthoc_regular_branch": {
            "warning": "branch classification was discovered after calibration runs",
            "excluded_regime": {"mass": 0.35, "tau_star": 2.4},
            "coefficients": {
                key: float(value)
                for key, value in zip(("a", "b", "c", "d"), regular_coefficients)
            },
            "calibration_mae": float(np.mean(np.abs(regular_residual))),
            "calibration_max_error": float(np.max(np.abs(regular_residual))),
            "withheld_mae": float(np.mean(regular_errors)),
            "withheld_max_error": float(np.max(regular_errors)),
            "withheld_validation": regular_validation,
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "coefficients", "calibration_mae", "calibration_max_error",
        "withheld_mae", "withheld_max_error", "withheld_validation",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
