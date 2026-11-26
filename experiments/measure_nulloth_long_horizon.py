"""Extend the small-seed full-production Nulloth trajectory to time twelve."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from experiments.map_nulloth_production_regimes import _run_configuration
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _run_configuration


def run_long_horizon(*, points: int = 4, duration: float = 12.0, dt: float = 0.02):
    result = _run_configuration(
        points,
        duration,
        dt,
        ("activated-small-long", 0.18, 0.005, "growing"),
    )
    payload = asdict(result)
    payload["transitioned_from_sustained_to_turnover"] = int(
        result.outcome == "nonlinear-turnover"
    )
    payload["coherence_lifetime_lower_bound"] = (
        result.final_time if result.final_amplitude >= 0.06 else 0.0
    )
    payload["retained_coherence_at_horizon"] = int(result.final_amplitude >= 0.06)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_long_horizon(
        points=arguments.points,
        duration=arguments.duration,
        dt=arguments.dt,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
