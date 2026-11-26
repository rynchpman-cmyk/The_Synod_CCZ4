"""Render the trapping-boundary scan and resolution-validation summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/trapping_boundary")


def main() -> int:
    coarse = json.loads((ROOT/"blind_results.json").read_text(encoding="utf-8"))
    medium = json.loads((ROOT/"high_resolution_results.json").read_text(encoding="utf-8"))
    fine = json.loads((ROOT/"convergence_256.json").read_text(encoding="utf-8"))
    calibrated = json.loads((ROOT/"calibrated_boundary_256.json").read_text(encoding="utf-8"))
    pilot = json.loads((ROOT/"pilot_analysis.json").read_text(encoding="utf-8"))

    plt.style.use("dark_background")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {0.15: "#52d3d8", 0.60: "#f6c85f", 2.40: "#ef6f9b"}

    ax = axes[0, 0]
    for tau in (0.15, 0.60, 2.40):
        rows = sorted(
            (row for row in calibrated["regimes"] if row["tau_star"] == tau),
            key=lambda row: row["mass"],
        )
        ax.plot(
            [row["mass"] for row in rows],
            [row["critical_loading_retention_0p20"] for row in rows],
            "o-", lw=2, ms=7, color=colors[tau], label=fr"$\tau_*={tau:g}$",
        )
    ax.annotate(
        "slow-light branch", xy=(0.35, 5.8788), xytext=(0.55, 5.45),
        arrowprops={"arrowstyle": "->", "color": "white"}, fontsize=10,
    )
    ax.set(title="256-point trapping boundary", xlabel=r"bare mass $m_A$",
           ylabel=r"critical loading $q=g\rho_{peak}$", ylim=(0, 6.4))
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[0, 1]
    names = [row["name"].replace("_intermediate", "") for row in fine["results"]]
    values = {
        64: [row["measured_critical_loading"] for row in coarse["results"]],
        128: [row["measured_critical_loading"] for row in medium["results"]],
        256: [row["critical_loading"] for row in fine["results"]],
    }
    x = np.arange(len(names))
    for points, marker in ((64, "o"), (128, "s"), (256, "D")):
        ax.plot(x, values[points], marker=marker, lw=1.8, label=f"N={points}")
    ax.set(title="Boundary coordinates versus resolution", ylabel="critical loading",
           xticks=x, xticklabels=names)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1, 0]
    branch = calibrated["posthoc_regular_branch"]
    actual = np.array([r["measured_critical_loading"] for r in branch["withheld_validation"]])
    predicted = np.array([r["predicted_critical_loading"] for r in branch["withheld_validation"]])
    extent = (min(actual.min(), predicted.min())-.15, max(actual.max(), predicted.max())+.15)
    ax.plot(extent, extent, "--", color="#aaaaaa", lw=1.5, label="perfect")
    ax.scatter(actual, predicted, s=85, color="#7ee787", edgecolor="white", linewidth=.7)
    for label, xv, yv in zip(names, actual, predicted):
        ax.annotate(label, (xv, yv), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(title=f"Withheld 256-point predictions (MAE {branch['withheld_mae']:.3f})",
           xlabel="measured critical loading", ylabel="predicted critical loading",
           xlim=extent, ylim=extent)
    ax.grid(alpha=0.18)

    ax = axes[1, 1]
    for mass in (0.35, 0.70, 1.10):
        curve = next(
            row for row in pilot["curves"]
            if row["mass"] == mass and row["tau_star"] == 0.60
        )
        ax.plot(curve["loadings"], curve["retention"], "o-", lw=1.8,
                label=fr"$m_A={mass:g}$")
    ax.axhline(0.20, color="white", ls="--", lw=1, alpha=.7)
    ax.set(title=r"Low-loading transparency trough ($\tau_*=0.6$, N=64)",
           xlabel=r"loading $q=g\rho_{peak}$", ylabel="core retention at T=8")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    figure.suptitle("Active-repair radiation / matter-trapping boundary", fontsize=17)
    destination = ROOT/"trapping_boundary_summary.png"
    figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
