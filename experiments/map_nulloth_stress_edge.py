"""Confirm the low-resolution high-amplitude Nulloth dispersal edge."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from experiments.map_nulloth_production_regimes import _run_configuration
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _run_configuration


def run_stress_edge(*, points: int = 4, duration: float = 3.0, dt: float = 0.01):
    return _run_configuration(
        points,
        duration,
        dt,
        ("high-amplitude-stress", 0.18, 0.12, "rest"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_stress_edge(
        points=arguments.points,
        duration=arguments.duration,
        dt=arguments.dt,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
