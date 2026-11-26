"""Render carrier-loading trends from the Theory 3.3 collision sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/theory33_fraction_sweep.png"),
    )
    args = parser.parse_args()
    cases = json.loads(args.summary.read_text(encoding="utf-8"))["cases"]
    indexed = {
        (
            float(row["case"]["target_fraction"]),
            float(row["case"]["relaxation_time"]),
        ): row
        for row in cases
    }
    fractions = np.array(sorted({key[0] for key in indexed}))
    fast = [indexed[(fraction, 0.03)] for fraction in fractions]
    slow = [indexed[(fraction, 0.30)] for fraction in fractions]
    compression_fast = np.array([row["peak_compression_ratio"] for row in fast])
    compression_slow = np.array([row["peak_compression_ratio"] for row in slow])
    lapse_fast = np.array([row["minimum_lapse"] for row in fast])
    lapse_slow = np.array([row["minimum_lapse"] for row in slow])
    gamma_fast = np.array([row["final_relative_lorentz_factor"] - 1.0 for row in fast])
    gamma_slow = np.array([row["final_relative_lorentz_factor"] - 1.0 for row in slow])
    delta_compression = compression_fast - compression_slow
    delta_lapse = lapse_slow - lapse_fast

    fit = fractions >= 0.1
    compression_power, compression_log_coefficient = np.polyfit(
        np.log(fractions[fit]), np.log(delta_compression[fit]), 1
    )
    lapse_power, lapse_log_coefficient = np.polyfit(
        np.log(fractions[fit]), np.log(delta_lapse[fit]), 1
    )
    fit_x = np.geomspace(fractions[fit][0], fractions[-1], 100)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(fractions, compression_fast, "o-", label=r"fast drag $\tau_D=0.03$")
    axes[0, 0].plot(fractions, compression_slow, "s-", label=r"slow drag $\tau_D=0.30$")
    axes[0, 0].set_ylabel("peak compression ratio")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(fractions, lapse_fast, "o-", label="fast drag")
    axes[0, 1].plot(fractions, lapse_slow, "s-", label="slow drag")
    axes[0, 1].set_ylabel("minimum lapse")

    axes[1, 0].semilogy(fractions, gamma_fast, "o-", label="fast drag")
    axes[1, 0].semilogy(fractions, gamma_slow, "s-", label="slow drag")
    axes[1, 0].set_ylabel(r"final max $(\gamma_D-1)$")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].loglog(fractions, delta_compression, "o", label="compression split")
    axes[1, 1].loglog(
        fit_x,
        np.exp(compression_log_coefficient) * fit_x**compression_power,
        "--",
        label=fr"$\propto f_D^{{{compression_power:.3f}}}$",
    )
    axes[1, 1].loglog(fractions, delta_lapse, "s", label="lapse split")
    axes[1, 1].loglog(
        fit_x,
        np.exp(lapse_log_coefficient) * fit_x**lapse_power,
        ":",
        label=fr"$\propto f_D^{{{lapse_power:.3f}}}$",
    )
    axes[1, 1].set_ylabel("fast/slow geometric split")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel(r"carrier fraction $f_D$")
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Theory 3.3 carrier loading: gravitational branch splitting follows the carrier EOS"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
