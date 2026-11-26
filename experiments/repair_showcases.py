"""Reproduce the named active-repair showcase trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scan_active_repair import Case, run_case


SHOWCASES = {
    "radiative_control": Case(0.25, 0.15, 0.0, 0.002),
    "fast_quench_trap": Case(0.25, 0.15, 30.0, 0.08),
    "ringing_trap": Case(0.6, 0.15, 30.0, 0.08),
    "slow_overshoot": Case(1.2, 2.4, 0.0, 0.002),
    "interaction_dominant": Case(1.2, 2.4, 30.0, 0.08),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=(*SHOWCASES, "all"))
    parser.add_argument("--output", type=Path, default=Path("runs/repair_showcases"))
    parser.add_argument("--points", type=int, default=160)
    parser.add_argument("--length", type=float, default=24.0)
    parser.add_argument("--final-time", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=161)
    args = parser.parse_args()
    names = SHOWCASES if args.name == "all" else (args.name,)
    for name in names:
        case = SHOWCASES[name]
        metrics, trajectory = run_case(
            case, args.points, args.length, args.final_time, args.samples, True
        )
        destination = args.output / name
        destination.mkdir(parents=True, exist_ok=True)
        assert trajectory is not None
        np.savez_compressed(destination / f"{case.name}.npz", **trajectory)
        (destination / "summary.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps({"showcase": name, **metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
