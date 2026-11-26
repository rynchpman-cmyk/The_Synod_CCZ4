"""Lock, resolve, and visualize out-of-sample trapping-boundary predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scan_trapping_boundary import ScanCache


BLIND_CASES = (
    {"name": "light_fast_intermediate", "mass": 0.62, "tau_star": 0.30},
    {"name": "middle_slow_intermediate", "mass": 0.93, "tau_star": 1.20},
    {"name": "heavy_fast_intermediate", "mass": 1.32, "tau_star": 0.30},
    {"name": "heavy_slow_intermediate", "mass": 1.18, "tau_star": 1.20},
)
TARGET = 0.20


def design(mass: float, tau: float) -> np.ndarray:
    log_tau = np.log(tau)
    return np.array([1.0, -mass**2, log_tau, log_tau**2])


def fit_boundary(path: Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["regimes"]
        if row["critical_loading_retention_0p20"] is not None
    ]
    x = np.stack([design(row["mass"], row["tau_star"]) for row in rows])
    y = np.array([row["critical_loading_retention_0p20"] for row in rows])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficients

    # Leave an entire mass value out at once. This is more stringent than
    # leaving one point out because all three relaxation samples disappear.
    held_out_predictions = np.empty_like(y)
    masses = np.array([row["mass"] for row in rows])
    for mass in np.unique(masses):
        train = masses != mass
        test = ~train
        local = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
        held_out_predictions[test] = x[test] @ local

    residual = fitted - y
    held_out = held_out_predictions - y
    diagnostics = {
        "form": "q_crit = a - b*m_A^2 + c*ln(tau_star) + d*ln(tau_star)^2",
        "coefficients": {
            key: float(value)
            for key, value in zip(("a", "b", "c", "d"), coefficients)
        },
        "fit_mae": float(np.mean(np.abs(residual))),
        "fit_max_error": float(np.max(np.abs(residual))),
        "fit_r_squared": float(1.0 - np.sum(residual**2) / np.sum((y-y.mean())**2)),
        "leave_one_mass_out_mae": float(np.mean(np.abs(held_out))),
        "leave_one_mass_out_max_error": float(np.max(np.abs(held_out))),
        "training_regimes": len(rows),
    }
    return coefficients, diagnostics


def lock(map_path: Path, output: Path) -> dict:
    coefficients, diagnostics = fit_boundary(map_path)
    forecasts = []
    for case in BLIND_CASES:
        predicted = float(design(case["mass"], case["tau_star"]) @ coefficients)
        forecasts.append({**case, "predicted_critical_loading": predicted})
    payload = {
        "status": "BOUNDARY_FORECASTS_LOCKED_BEFORE_BLIND_RUNS",
        "boundary_definition": (
            "loading where final repair-sector energy inside abs(x)<3.5 "
            "reaches 0.20 at T=8"
        ),
        "training_source": str(map_path),
        "model": diagnostics,
        "forecasts": forecasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def interpolate_crossing(samples: list[dict]) -> float | None:
    ordered = sorted(samples, key=lambda item: item["loading"])
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        y0, y1 = lower["core_retention"], upper["core_retention"]
        if y0 <= TARGET <= y1:
            fraction = (TARGET-y0) / max(y1-y0, 1e-15)
            return float(lower["loading"] + fraction*(upper["loading"]-lower["loading"]))
    return None


def resolve(forecast_path: Path, output: Path, cache_path: Path, points: int,
            half_width: float) -> dict:
    locked = json.loads(forecast_path.read_text(encoding="utf-8"))
    if locked.get("status") != "BOUNDARY_FORECASTS_LOCKED_BEFORE_BLIND_RUNS":
        raise ValueError("predictions must be locked before resolution")
    cache = ScanCache(cache_path)
    results = []
    for forecast in locked["forecasts"]:
        predicted = forecast["predicted_critical_loading"]
        loadings = (max(0.0, predicted-half_width), predicted, predicted+half_width)
        samples = [
            cache.evaluate(forecast["mass"], forecast["tau_star"], loading, points)
            for loading in loadings
        ]
        measured = interpolate_crossing(samples)
        result = {
            "name": forecast["name"],
            "mass": forecast["mass"],
            "tau_star": forecast["tau_star"],
            "predicted_critical_loading": predicted,
            "measured_critical_loading": measured,
            "absolute_error": None if measured is None else abs(measured-predicted),
            "retention_at_prediction": samples[1]["core_retention"],
            "target_retention_error": abs(samples[1]["core_retention"]-TARGET),
            "straddles_target": (
                samples[0]["core_retention"] <= TARGET <= samples[-1]["core_retention"]
            ),
            "samples": samples,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    errors = [result["absolute_error"] for result in results if result["absolute_error"] is not None]
    payload = {
        "status": "BOUNDARY_BLIND_RUNS_RESOLVED",
        "forecast_source": str(forecast_path),
        "points": points,
        "probe_half_width": half_width,
        "all_forecasts_straddle_target": all(r["straddles_target"] for r in results),
        "critical_loading_mae": float(np.mean(errors)) if errors else None,
        "critical_loading_max_error": float(np.max(errors)) if errors else None,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def convergence(reference_path: Path, output: Path, cache_path: Path,
                points: int, half_width: float, iterations: int) -> dict:
    """Re-locate each reference crossing on a finer grid by bisection."""
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    cache = ScanCache(cache_path)
    results = []
    for prior in reference["results"]:
        center = prior["measured_critical_loading"]
        if center is None:
            continue
        lower = cache.evaluate(
            prior["mass"], prior["tau_star"], max(0.0, center-half_width), points
        )
        upper = cache.evaluate(
            prior["mass"], prior["tau_star"], center+half_width, points
        )
        if not (lower["core_retention"] <= TARGET <= upper["core_retention"]):
            measured = None
            samples = [lower, upper]
        else:
            samples = [lower, upper]
            for _ in range(iterations):
                loading = 0.5*(lower["loading"] + upper["loading"])
                middle = cache.evaluate(
                    prior["mass"], prior["tau_star"], loading, points
                )
                samples.append(middle)
                if middle["core_retention"] >= TARGET:
                    upper = middle
                else:
                    lower = middle
            measured = interpolate_crossing([lower, upper])
        result = {
            "name": prior["name"],
            "mass": prior["mass"],
            "tau_star": prior["tau_star"],
            "reference_points": reference["points"],
            "reference_critical_loading": center,
            "points": points,
            "critical_loading": measured,
            "absolute_resolution_shift": (
                None if measured is None else abs(measured-center)
            ),
            "samples": sorted(samples, key=lambda item: item["loading"]),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    shifts = [r["absolute_resolution_shift"] for r in results if r["absolute_resolution_shift"] is not None]
    payload = {
        "status": "BOUNDARY_RESOLUTION_CONVERGENCE",
        "reference_source": str(reference_path),
        "points": points,
        "bracket_half_width": half_width,
        "bisection_iterations": iterations,
        "mean_absolute_shift": float(np.mean(shifts)) if shifts else None,
        "maximum_absolute_shift": float(np.max(shifts)) if shifts else None,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("lock", "resolve", "convergence"))
    parser.add_argument("--map", type=Path, default=Path("runs/trapping_boundary/boundary_map.json"))
    parser.add_argument("--forecasts", type=Path, default=Path("runs/trapping_boundary/locked_forecasts.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/trapping_boundary/blind_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/trapping_boundary/validation_cache.json"))
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--half-width", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "lock":
        lock(args.map, args.forecasts)
    elif args.mode == "resolve":
        resolve(args.forecasts, args.output, args.cache, args.points, args.half_width)
    else:
        convergence(
            args.forecasts, args.output, args.cache, args.points,
            args.half_width, args.iterations,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
