"""Audit the Theory 3.1 and 3.2 longitudinal target-current symbols.

The audit is deliberately independent of the evolution code.  It checks the
frozen, fluid-rest-frame principal matrices used in the formulation decision:

* Theory 3.1: conserved charge plus an advected spatial current;
* Theory 3.2: conserved charge plus a causal Maxwell--Cattaneo current.

The Euler block is the standard barotropic longitudinal block.  Lorentz boosts
change the displayed speeds but do not repair a missing eigenvector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def theory31_target_symbol(transport_speed: float = 0.0) -> np.ndarray:
    """Principal symbol for (delta q_D, delta R_D^n) in Theory 3.1."""
    return np.array(
        [[transport_speed, 1.0], [0.0, transport_speed]], dtype=float
    )


def theory32_euler_target_symbol(
    sound_speed: float,
    target_speed: float,
    *,
    baryon_density: float = 1.0,
    enthalpy_density: float = 1.0,
    target_charge: float = 1.0,
) -> np.ndarray:
    """Rest-frame symbol for (dn, dv, dp, dq_D, dR_D^n)."""
    cs = float(sound_speed)
    cD = float(target_speed)
    n = float(baryon_density)
    w = float(enthalpy_density)
    q = float(target_charge)
    return np.array(
        [
            [0.0, n, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / w, 0.0, 0.0],
            [0.0, w * cs**2, 0.0, 0.0, 0.0],
            [0.0, q, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, cD**2, 0.0],
        ],
        dtype=float,
    )


def nullity(matrix: np.ndarray, tolerance: float = 1.0e-7) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    scale = max(1.0, float(singular_values[0]))
    rank = int(np.sum(singular_values > tolerance * scale))
    return matrix.shape[1] - rank


def eigensystem_report(matrix: np.ndarray) -> dict[str, object]:
    values, vectors = np.linalg.eig(matrix)
    order = np.argsort(values.real)
    values = values[order]
    vectors = vectors[:, order]
    clusters: list[list[complex]] = []
    for value in values:
        for cluster in clusters:
            if abs(value - cluster[0]) < 1.0e-6:
                cluster.append(complex(value))
                break
        else:
            clusters.append([complex(value)])
    multiplicities = []
    for cluster in clusters:
        value = sum(cluster) / len(cluster)
        multiplicities.append(
            {
                "eigenvalue": {"real": float(value.real), "imag": float(value.imag)},
                "algebraic": len(cluster),
                "geometric": nullity(
                    matrix - value * np.eye(matrix.shape[0], dtype=complex)
                ),
            }
        )
    condition = float(np.linalg.cond(vectors))
    return {
        "eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in values
        ],
        "multiplicities": multiplicities,
        "eigenvector_condition": condition,
        "complete_real_eigensystem": bool(
            np.max(np.abs(values.imag)) < 1.0e-10
            and all(row["geometric"] == row["algebraic"] for row in multiplicities)
            and np.isfinite(condition)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("runs/theory32_closure")
    )
    parser.add_argument("--target-speed", type=float, default=0.95)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    original = eigensystem_report(theory31_target_symbol())
    representative = {
        "separated": eigensystem_report(theory32_euler_target_symbol(0.60, 0.95)),
        "coincident_nonzero_charge": eigensystem_report(
            theory32_euler_target_symbol(0.60, 0.60, target_charge=1.0)
        ),
        "coincident_zero_charge": eigensystem_report(
            theory32_euler_target_symbol(0.60, 0.60, target_charge=0.0)
        ),
    }

    sound_speeds = np.linspace(0.05, 0.99, 600)
    charge_samples = (0.0, 0.1, 1.0, 10.0)
    scan: dict[str, list[float]] = {}
    for charge in charge_samples:
        conditions = []
        for sound_speed in sound_speeds:
            report = eigensystem_report(
                theory32_euler_target_symbol(
                    sound_speed, args.target_speed, target_charge=charge
                )
            )
            conditions.append(float(report["eigenvector_condition"]))
        scan[str(charge)] = conditions

    fig, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for charge in charge_samples:
        axis.semilogy(
            sound_speeds,
            scan[str(charge)],
            label=rf"$q_D={charge:g}$",
        )
    axis.axvline(args.target_speed, color="black", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel=r"fluid sound speed $c_s$",
        ylabel="eigenvector condition number",
        title=rf"Euler/target eigenspace, $c_D={args.target_speed:g}$",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure_path = args.output / "target_eigenspace_condition.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    payload = {
        "theory31_advected_current": original,
        "theory32_representative_cases": representative,
        "scan": {
            "target_speed": args.target_speed,
            "sound_speeds": sound_speeds.tolist(),
            "target_charges": list(charge_samples),
            "eigenvector_conditions": scan,
        },
        "conclusions": {
            "theory31": (
                "Defective: the repeated material-speed eigenvalue has only one "
                "target-current eigenvector."
            ),
            "theory32": (
                "Complete for 0<c_D<1 away from c_D=c_s; nonzero background "
                "target charge makes the coincident acoustic/current eigenspace "
                "defective."
            ),
            "admissibility_rule": (
                "Choose c_D above the EOS sound-speed ceiling by a positive "
                "margin and monitor that margin at runtime."
            ),
        },
    }
    json_path = args.output / "target_eigenspace_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["conclusions"], indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
