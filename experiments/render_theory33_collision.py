"""Render the paired Theory 3.3 compact-collision champion trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.array(data[name]) for name in data.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fast", type=Path)
    parser.add_argument("slow", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/theory33_collision_comparison.png"),
    )
    args = parser.parse_args()
    fast = load(args.fast)
    slow = load(args.slow)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    cases = ((fast, r"Fast drag: $\tau_D=0.03$"), (slow, r"Slow drag: $\tau_D=0.30$"))
    for column, (data, title) in enumerate(cases):
        extent = [
            float(data["x"][0]),
            float(data["x"][-1]),
            float(data["time"][0]),
            float(data["time"][-1]),
        ]
        density = axes[0, column].imshow(
            data["baryon_density"],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        fig.colorbar(density, ax=axes[0, column], label=r"$n_b$")
        relative = axes[1, column].imshow(
            data["relative_lorentz_factor"] - 1.0,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
        )
        fig.colorbar(relative, ax=axes[1, column], label=r"$\gamma_D-1$")
        peak = np.max(data["baryon_density"], axis=1)
        max_gamma = np.max(data["relative_lorentz_factor"] - 1.0, axis=1)
        lapse = np.min(data["lapse"], axis=1)
        axes[2, column].plot(data["time"], peak / peak[0], label="peak density / initial")
        axes[2, column].plot(data["time"], max_gamma / max(max_gamma[0], 1.0e-14), label="counterflow / initial")
        axes[2, column].plot(data["time"], lapse, label="minimum lapse")
        axes[0, column].set_title(title)
        axes[2, column].set_xlabel("time")
        axes[2, column].set_ylim(0.0, 1.12)
        axes[2, column].grid(alpha=0.25)
        axes[2, column].legend(fontsize=8)
    axes[0, 0].set_ylabel("time")
    axes[1, 0].set_ylabel("time")
    axes[0, 1].set_ylabel("time")
    axes[1, 1].set_ylabel("time")
    axes[0, 0].set_xlabel("x")
    axes[0, 1].set_xlabel("x")
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    fig.suptitle("Theory 3.3 two-current collision: locked versus penetrating carrier flow")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
