"""Render the blind forecast-validation summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("runs/repair_predictions")
payload = json.loads((ROOT / "blind_results.json").read_text(encoding="utf-8"))
results = payload["results"]
targets = (
    ("peak_mismatch_over_initial_a", "Peak mismatch / initial A"),
    ("peak_interaction_fraction", "Peak interaction / Proca"),
    ("maximum_exterior_energy_fraction", "Maximum exterior fraction"),
    ("late_field_fraction", "Late field fraction"),
)

plt.style.use("dark_background")
figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
figure.patch.set_facecolor("#080d17")
for axis in axes.ravel():
    axis.set_facecolor("#0e1728")
    axis.grid(color="#273850", alpha=0.7)
    for spine in axis.spines.values():
        spine.set_color("#344761")

for axis, (target, label) in zip(axes.ravel()[:4], targets):
    predicted = np.array([item["comparison"][target]["predicted"] for item in results])
    actual = np.array([item["comparison"][target]["actual"] for item in results])
    lower, upper = min(predicted.min(), actual.min()), max(predicted.max(), actual.max())
    padding = max(0.05 * (upper - lower), 0.01)
    axis.plot([lower-padding, upper+padding], [lower-padding, upper+padding], color="#61748e", ls="--")
    axis.scatter(actual, predicted, s=65, color="#64dcff", edgecolor="#d9f7ff", linewidth=0.7)
    axis.set(xlabel="Solver measurement", ylabel="Locked forecast", title=label)
    mae = np.mean(np.abs(predicted - actual))
    correlation = np.corrcoef(predicted, actual)[0, 1]
    axis.text(0.04, 0.94, f"MAE {mae:.3f}   r {correlation:.3f}", transform=axis.transAxes, va="top", color="#aebdd0")

names = [item["name"].replace("_", "\n") for item in results]
actual_crossings = np.array([item["comparison"]["profile_zero_crossings"]["actual"] for item in results])
forecast_crossings = np.array([item["comparison"]["profile_zero_crossings"]["predicted"] for item in results])
x = np.arange(len(results))
axes[1, 1].bar(x - 0.18, actual_crossings, 0.36, label="measured", color="#66e4ff")
axes[1, 1].bar(x + 0.18, forecast_crossings, 0.36, label="forecast", color="#ffad42")
axes[1, 1].set_xticks(x, names, fontsize=7)
axes[1, 1].set_title("Blind phase-reversal counts")
axes[1, 1].set_ylabel("sign changes")
axes[1, 1].legend(frameon=False)

effective_predictions = []
for item in results:
    case = item["case"]
    effective_mass = np.sqrt(case["mass"]**2 + case["coupling"] * case["peak_density"])
    effective_predictions.append(np.floor(8.0 * effective_mass / np.pi) + 1)
axes[1, 2].bar(x - 0.18, actual_crossings, 0.36, label="measured", color="#66e4ff")
axes[1, 2].bar(x + 0.18, effective_predictions, 0.36, label="effective-mass law", color="#b98cff")
axes[1, 2].set_xticks(x, names, fontsize=7)
axes[1, 2].set_title(r"$\lfloor T\sqrt{m_A^2+g\rho}/\pi\rfloor+1$")
axes[1, 2].set_ylabel("sign changes")
axes[1, 2].legend(frameon=False)

figure.suptitle("Active-Repair Blind Forecast Challenge", fontsize=18, color="white")
destination = ROOT / "forecast_validation.png"
figure.savefig(destination, dpi=160, facecolor=figure.get_facecolor())
print(destination)
