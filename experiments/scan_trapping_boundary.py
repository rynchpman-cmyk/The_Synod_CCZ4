"""Adaptive radiation-to-trapping boundary scan for active-repair fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scan_active_repair import Case, run_case


DENSITY = 0.08
PILOT_MASSES = (0.35, 0.70, 1.10)
PILOT_TAUS = (0.15, 0.60, 2.40)
PILOT_LOADINGS = (0.0, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
MAP_MASSES = (0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.10, 1.25, 1.40)
MAP_TAUS = (0.15, 0.60, 2.40)
BASE_LOADINGS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0)


class ScanCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @staticmethod
    def key(mass: float, tau: float, loading: float, points: int) -> str:
        return f"m={mass:.8g}|tau={tau:.8g}|q={loading:.8g}|n={points}"

    def evaluate(self, mass: float, tau: float, loading: float, points: int) -> dict:
        key = self.key(mass, tau, loading, points)
        if key not in self.records:
            case = Case(
                mass=mass,
                tau_star=tau,
                coupling=loading / DENSITY,
                peak_density=DENSITY,
            )
            metrics, _ = run_case(case, points, 24.0, 8.0, 81, False)
            self.records[key] = {
                "mass": mass,
                "tau_star": tau,
                "loading": loading,
                "points": points,
                "core_retention": metrics["final_core_retention_fraction"],
                "final_exterior": metrics["final_exterior_energy_fraction"],
                "exterior_half_time": metrics["exterior_half_time"],
                "peak_interaction_fraction": metrics["peak_interaction_fraction"],
                "profile_zero_crossings": metrics["profile_zero_crossings"],
                "gauss_l2": metrics["final_gauss_l2"],
                "stable": metrics["stable"],
            }
            self.path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        record = self.records[key]
        print(json.dumps(record, sort_keys=True), flush=True)
        return record


def run_pilot(cache: ScanCache, points: int) -> dict:
    curves = []
    for mass in PILOT_MASSES:
        for tau in PILOT_TAUS:
            records = [cache.evaluate(mass, tau, loading, points) for loading in PILOT_LOADINGS]
            retention = np.array([record["core_retention"] for record in records])
            differences = np.diff(retention)
            curves.append(
                {
                    "mass": mass,
                    "tau_star": tau,
                    "loadings": list(PILOT_LOADINGS),
                    "retention": retention.tolist(),
                    "monotone_with_tolerance_0p01": bool(np.all(differences >= -0.01)),
                    "minimum_step": float(np.min(differences)),
                    "maximum_retention": float(np.max(retention)),
                }
            )
    report = {
        "points": points,
        "density": DENSITY,
        "curves": curves,
        "monotone_curves": sum(curve["monotone_with_tolerance_0p01"] for curve in curves),
    }
    destination = cache.path.parent / "pilot_analysis.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def crossing_loading(
    cache: ScanCache,
    mass: float,
    tau: float,
    points: int,
    target: float = 0.20,
    iterations: int = 7,
) -> tuple[float | None, list[dict]]:
    records = [cache.evaluate(mass, tau, loading, points) for loading in BASE_LOADINGS]
    records.sort(key=lambda record: record["loading"])
    bracket = None
    # Select the final upward crossing, which is robust to the low-loading
    # transparency dip discovered by the pilot.
    for lower, upper in zip(records[:-1], records[1:]):
        if lower["core_retention"] < target <= upper["core_retention"]:
            bracket = (lower, upper)
    if bracket is None:
        return None, records
    lower, upper = bracket
    for _ in range(iterations):
        midpoint = 0.5 * (lower["loading"] + upper["loading"])
        middle = cache.evaluate(mass, tau, midpoint, points)
        records.append(middle)
        if middle["core_retention"] >= target:
            upper = middle
        else:
            lower = middle
    # Linear interpolation in retention makes the final estimate less tied to
    # the arbitrary last bisection side.
    fraction = (target - lower["core_retention"]) / max(
        upper["core_retention"] - lower["core_retention"], 1e-15
    )
    estimate = lower["loading"] + fraction * (
        upper["loading"] - lower["loading"]
    )
    return float(estimate), records


def run_boundary_map(cache: ScanCache, points: int) -> dict:
    regimes = []
    for mass in MAP_MASSES:
        for tau in MAP_TAUS:
            critical, records = crossing_loading(cache, mass, tau, points)
            unique = {record["loading"]: record for record in records}
            ordered = [unique[key] for key in sorted(unique)]
            low_loading = [record for record in ordered if record["loading"] <= 1.0]
            minimum = min(low_loading, key=lambda record: record["core_retention"])
            baseline = unique[0.0]["core_retention"]
            regimes.append(
                {
                    "mass": mass,
                    "tau_star": tau,
                    "critical_loading_retention_0p20": critical,
                    "transparency_loading_discrete": minimum["loading"],
                    "minimum_core_retention": minimum["core_retention"],
                    "uncoupled_core_retention": baseline,
                    "transparency_gain": baseline - minimum["core_retention"],
                    "samples": ordered,
                }
            )
    report = {
        "definition": "critical loading where final energy inside abs(x)<3.5 reaches 0.20 at T=8",
        "points": points,
        "density": DENSITY,
        "masses": list(MAP_MASSES),
        "tau_values": list(MAP_TAUS),
        "regimes": regimes,
    }
    destination = cache.path.parent / "boundary_map.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "map"))
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--cache", type=Path, default=Path("runs/trapping_boundary/cache.json"))
    args = parser.parse_args()
    cache = ScanCache(args.cache)
    if args.mode == "pilot":
        run_pilot(cache, args.points)
    else:
        report = run_boundary_map(cache, args.points)
        summary = [
            {
                key: regime[key]
                for key in (
                    "mass", "tau_star", "critical_loading_retention_0p20",
                    "transparency_loading_discrete", "transparency_gain",
                )
            }
            for regime in report["regimes"]
        ]
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
