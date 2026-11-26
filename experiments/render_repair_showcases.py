"""Render static previews of saved repair showcase trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def render(path: Path) -> Path:
    data = np.load(path)
    x, time = data["x"], data["time"]
    a, astar = data["a_z"], data["astar_z"]
    mismatch = a - astar
    extent = (float(x[0]), float(x[-1]), float(time[-1]), float(time[0]))
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.patch.set_facecolor("#090d18")
    for axis in axes.ravel():
        axis.set_facecolor("#0d1424")
        axis.tick_params(colors="#dbe7ff")
        axis.xaxis.label.set_color("#dbe7ff")
        axis.yaxis.label.set_color("#dbe7ff")
        for spine in axis.spines.values():
            spine.set_color("#34415b")
    limits = max(float(np.max(np.abs(a))), float(np.max(np.abs(astar))))
    mismatch_limit = float(np.max(np.abs(mismatch)))
    image = axes[0, 0].imshow(a, aspect="auto", extent=extent, cmap="coolwarm", vmin=-limits, vmax=limits)
    axes[0, 0].set_title(r"Dynamical repair field $A_z$", color="white")
    figure.colorbar(image, ax=axes[0, 0])
    image = axes[0, 1].imshow(mismatch, aspect="auto", extent=extent, cmap="PuOr", vmin=-mismatch_limit, vmax=mismatch_limit)
    axes[0, 1].set_title(r"Active mismatch $A_z-A_z^\star$", color="white")
    figure.colorbar(image, ax=axes[0, 1])
    axes[1, 0].plot(time, data["a_l2"] / data["a_l2"][0], label=r"$||A||$", lw=2)
    axes[1, 0].plot(time, data["astar_l2"] / data["astar_l2"][0], label=r"$||A^\star||$", lw=2)
    axes[1, 0].plot(time, data["mismatch_l2"] / data["a_l2"][0], label=r"$||A-A^\star||$", lw=2)
    axes[1, 0].set_title("Normalized amplitudes", color="white")
    axes[1, 0].legend(facecolor="#111a2d", labelcolor="white")
    ratio = data["interaction_energy"] / np.maximum(data["proca_energy"], 1e-30)
    axes[1, 1].plot(time, ratio, color="#ffb347", label=r"$E_{int}/E_A$", lw=2)
    axes[1, 1].plot(time, data["exterior_energy_fraction"], color="#62d8ff", label="exterior energy fraction", lw=2)
    axes[1, 1].plot(time, data["profile_projection"], color="#d187ff", label="initial-profile projection", lw=1.5)
    axes[1, 1].axhline(0.0, color="#77839a", lw=0.8)
    axes[1, 1].set_title("Exchange, transport, and phase", color="white")
    axes[1, 1].legend(facecolor="#111a2d", labelcolor="white")
    for axis in axes.ravel():
        axis.set_xlabel("x" if axis in axes[0] else "time")
    axes[0, 0].set_ylabel("time")
    axes[0, 1].set_ylabel("time")
    output = path.parent / "overview.png"
    figure.suptitle(path.parent.name.replace("_", " ").title(), color="white", fontsize=16)
    figure.savefig(output, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs/repair_showcases"))
    args = parser.parse_args()
    for path in sorted(args.root.glob("*/*.npz")):
        print(render(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
