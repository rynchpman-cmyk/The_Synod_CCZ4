"""Blind out-of-sample forecast challenge for active-repair dynamics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scan_active_repair import Case, run_case


TARGETS = (
    "peak_mismatch_over_initial_a",
    "peak_interaction_fraction",
    "maximum_exterior_energy_fraction",
    "late_field_fraction",
    "profile_zero_crossings",
)

BLIND_CASES = {
    "balanced_intermediate": Case(0.40, 0.30, 10.0, 0.040),
    "loaded_slow_target": Case(0.85, 1.20, 15.0, 0.040),
    "heavy_fast_loaded": Case(1.00, 0.30, 20.0, 0.080),
    "light_slow_loaded": Case(0.40, 1.20, 5.0, 0.040),
    "heavy_uncoupled": Case(1.00, 1.50, 0.0, 0.020),
    "near_trap_edge": Case(0.90, 0.25, 30.0, 0.080),
    "moderate_wake": Case(0.70, 0.80, 8.0, 0.060),
    "adiabatic_dense": Case(1.10, 2.00, 10.0, 0.080),
}


def features(case: dict | Case) -> np.ndarray:
    if isinstance(case, Case):
        mass, tau, coupling, density = case.mass, case.tau_star, case.coupling, case.peak_density
    else:
        mass, tau, coupling, density = (
            case["mass"], case["tau_star"], case["coupling"], case["peak_density"]
        )
    loading = coupling * density / mass**2
    return np.array(
        [np.log(mass), np.log(mass * tau), np.log1p(loading)], dtype=float
    )


def unique_training(records: list[dict]) -> list[dict]:
    groups: dict[tuple[float, ...], list[dict]] = {}
    for record in records:
        key = tuple(np.round(features(record["case"]), 12))
        groups.setdefault(key, []).append(record)
    merged = []
    for values in groups.values():
        item = {"case": values[0]["case"]}
        for target in TARGETS:
            item[target] = float(np.mean([value[target] for value in values]))
        merged.append(item)
    return merged


class RBFSurrogate:
    def __init__(self, length_scale: float, ridge: float) -> None:
        self.length_scale = length_scale
        self.ridge = ridge

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
        distance_squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * distance_squared / length**2)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), 1e-12)
        self.x = (x - self.x_mean) / self.x_scale
        self.y_mean = y.mean(axis=0)
        self.y_scale = np.maximum(y.std(axis=0), 1e-12)
        normalized_y = (y - self.y_mean) / self.y_scale
        kernel = self._kernel(self.x, self.x, self.length_scale)
        self.weights = np.linalg.solve(
            kernel + self.ridge * np.eye(len(kernel)), normalized_y
        )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        normalized = (x - self.x_mean) / self.x_scale
        prediction = self._kernel(normalized, self.x, self.length_scale) @ self.weights
        return prediction * self.y_scale + self.y_mean


def leave_one_out(x: np.ndarray, y: np.ndarray, length: float, ridge: float) -> np.ndarray:
    predictions = np.empty_like(y)
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        predictions[index] = RBFSurrogate(length, ridge).fit(x[keep], y[keep]).predict(x[index:index+1])[0]
    return predictions


def train(records: list[dict]) -> tuple[RBFSurrogate, dict]:
    records = unique_training(records)
    x = np.stack([features(record["case"]) for record in records])
    y = np.stack([[record[target] for target in TARGETS] for record in records])
    # Scale scoring by the natural spread of each observable so one target
    # cannot dominate hyperparameter selection merely through its units.
    spread = np.maximum(y.std(axis=0), 1e-12)
    candidates = []
    for length in (0.45, 0.65, 0.9, 1.2, 1.6, 2.2):
        for ridge in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
            prediction = leave_one_out(x, y, length, ridge)
            score = float(np.sqrt(np.mean(((prediction - y) / spread) ** 2)))
            candidates.append((score, length, ridge, prediction))
    score, length, ridge, prediction = min(candidates, key=lambda value: value[0])
    errors = {}
    for index, target in enumerate(TARGETS):
        absolute = np.abs(prediction[:, index] - y[:, index])
        errors[target] = {
            "mae": float(np.mean(absolute)),
            "rmse": float(np.sqrt(np.mean(absolute**2))),
            "range": [float(y[:, index].min()), float(y[:, index].max())],
        }
    report = {
        "unique_training_regimes": len(records),
        "length_scale": length,
        "ridge": ridge,
        "normalized_loo_rmse": score,
        "leave_one_regime_out": errors,
    }
    return RBFSurrogate(length, ridge).fit(x, y), report


def make_forecasts(training_path: Path, output: Path) -> dict:
    records = json.loads(training_path.read_text(encoding="utf-8"))
    surrogate, validation = train(records)
    x_blind = np.stack([features(case) for case in BLIND_CASES.values()])
    predicted = surrogate.predict(x_blind)
    forecasts = []
    for (name, case), values in zip(BLIND_CASES.items(), predicted):
        prediction = dict(zip(TARGETS, map(float, values)))
        # Apply only exact observable-domain knowledge before locking forecasts.
        prediction["peak_mismatch_over_initial_a"] = max(
            0.0, prediction["peak_mismatch_over_initial_a"]
        )
        prediction["peak_interaction_fraction"] = max(
            0.0, prediction["peak_interaction_fraction"]
        )
        prediction["maximum_exterior_energy_fraction"] = float(
            np.clip(prediction["maximum_exterior_energy_fraction"], 0.0, 1.0)
        )
        prediction["late_field_fraction"] = max(0.0, prediction["late_field_fraction"])
        prediction["profile_zero_crossings"] = int(
            np.rint(np.clip(prediction["profile_zero_crossings"], 0.0, 8.0))
        )
        forecasts.append(
            {
                "name": name,
                "case": asdict(case),
                "dimensionless": {
                    "mass_times_tau": case.mass * case.tau_star,
                    "loading_over_mass_squared": case.coupling * case.peak_density / case.mass**2,
                },
                "prediction": prediction,
            }
        )
    payload = {
        "status": "FORECASTS_LOCKED_BEFORE_BLIND_RUNS",
        "training_source": str(training_path),
        "validation": validation,
        "forecasts": forecasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def resolve_forecasts(forecast_path: Path, output: Path, points: int) -> dict:
    locked = json.loads(forecast_path.read_text(encoding="utf-8"))
    if locked.get("status") != "FORECASTS_LOCKED_BEFORE_BLIND_RUNS":
        raise ValueError("forecast file is not a locked pre-run prediction set")
    results = []
    for forecast in locked["forecasts"]:
        case = Case(**forecast["case"])
        measured, _ = run_case(case, points, 24.0, 8.0, 81, False)
        comparison = {}
        for target in TARGETS:
            predicted = float(forecast["prediction"][target])
            actual = float(measured[target])
            comparison[target] = {
                "predicted": predicted,
                "actual": actual,
                "absolute_error": abs(predicted - actual),
            }
        results.append(
            {
                "name": forecast["name"],
                "case": forecast["case"],
                "dimensionless": forecast["dimensionless"],
                "comparison": comparison,
                "stable": measured["stable"],
                "gauss_l2": measured["final_gauss_l2"],
            }
        )
        print(json.dumps(results[-1], sort_keys=True))
    aggregate = {}
    for target in TARGETS:
        errors = np.array([result["comparison"][target]["absolute_error"] for result in results])
        actual = np.array([result["comparison"][target]["actual"] for result in results])
        aggregate[target] = {
            "mae": float(errors.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean_actual": float(np.mean(actual)),
        }
    payload = {
        "status": "BLIND_RUNS_RESOLVED",
        "forecast_source": str(forecast_path),
        "points": points,
        "aggregate": aggregate,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_high_resolution(results_path: Path, output: Path, points: int) -> dict:
    resolved = json.loads(results_path.read_text(encoding="utf-8"))
    selected = {"heavy_fast_loaded", "near_trap_edge", "adiabatic_dense"}
    verifications = []
    for prior in resolved["results"]:
        if prior["name"] not in selected:
            continue
        case = Case(**prior["case"])
        measured, _ = run_case(case, points, 24.0, 8.0, 161, False)
        comparison = {}
        for target in TARGETS:
            coarse = prior["comparison"][target]["actual"]
            fine = measured[target]
            comparison[target] = {
                "points_80": coarse,
                f"points_{points}": fine,
                "absolute_change": abs(fine - coarse),
            }
        item = {
            "name": prior["name"],
            "case": prior["case"],
            "comparison": comparison,
            "gauss_l2": measured["final_gauss_l2"],
        }
        verifications.append(item)
        print(json.dumps(item, sort_keys=True))
    payload = {
        "status": "HIGH_RESOLUTION_VERIFIED",
        "source": str(results_path),
        "points": points,
        "verifications": verifications,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("forecast", "resolve", "verify"))
    parser.add_argument("--training", type=Path, default=Path("runs/repair_exploration/sweep_summary.json"))
    parser.add_argument("--forecast", type=Path, default=Path("runs/repair_predictions/locked_forecasts.json"))
    parser.add_argument("--results", type=Path, default=Path("runs/repair_predictions/blind_results.json"))
    parser.add_argument("--verification", type=Path, default=Path("runs/repair_predictions/high_resolution.json"))
    parser.add_argument("--points", type=int, default=80)
    args = parser.parse_args()
    if args.mode == "forecast":
        payload = make_forecasts(args.training, args.forecast)
        print(json.dumps(payload, indent=2))
    elif args.mode == "resolve":
        resolve_forecasts(args.forecast, args.results, args.points)
    else:
        verify_high_resolution(args.results, args.verification, args.points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
