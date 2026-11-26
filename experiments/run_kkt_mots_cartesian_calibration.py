"""Calibrate Cartesian MOTS observers against direct Brill--Lindquist surfaces.

The direct angular solve supplies one fixed physical surface at each binary
separation.  Legacy and parametric Cartesian observers then measure that same
surface without re-optimizing it.  This separates observer bias from branch-
finding error and records the physical-normal stability spectrum needed before
the diagnostic is attached to dynamical CCZ4 puncture snapshots.
"""

from __future__ import annotations

import argparse
import gc
import json
from math import pi, sqrt
from pathlib import Path

import numpy as np

from causal_repair.grid import CartesianGrid
from causal_repair.kkt_horizon import (
    DirectBrillLindquistMOTSFinder,
    NormalMOTSStabilityOperator,
    ParametricAxisymmetricMOTSFinder,
    SpectralMOTSFinder,
    SpectralMOTSParameters,
)
from run_kkt_mots_brill_lindquist import brill_lindquist_state


def comma_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 0.0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated values")
    return result


def comma_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item < 16 for item in result):
        raise argparse.ArgumentTypeError("resolutions must be comma-separated integers >= 16")
    return result


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def angular_residual(evaluator, evaluation) -> dict[str, float]:
    basis = evaluator.surface_basis.T
    mass = basis.T @ (evaluator.weights[:, None] * basis)
    projection = np.linalg.solve(
        mass, basis.T * evaluator.weights[None, :]
    )
    modal = projection @ evaluation.expansion
    spectral = sqrt(float(modal @ mass @ modal) / (4.0 * pi))
    rms = sqrt(
        float(np.sum(evaluator.weights * evaluation.expansion**2)) / (4.0 * pi)
    )
    return {
        "mean_expansion": float(
            np.sum(evaluator.weights * evaluation.expansion) / (4.0 * pi)
        ),
        "spectral_residual_norm": spectral,
        "rms_expansion": rms,
        "maximum_expansion": float(np.max(np.abs(evaluation.expansion))),
        "area": float(np.sum(evaluation.area_weights)),
    }


def observe(finder, state, surface, relative_step: float) -> dict[str, object]:
    evaluator = finder.evaluator(state, surface.center)
    evaluation = evaluator.evaluate(surface.coefficients)
    if not evaluation.valid:
        return {"valid": False, "reason": evaluation.reason}
    stability = NormalMOTSStabilityOperator(
        finder, relative_step
    ).analyze_evaluator(evaluator, surface)
    return {
        "valid": True,
        "reason": "ok",
        **angular_residual(evaluator, evaluation),
        "normal_stability": stability.diagnostics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=comma_ints, default=comma_ints("48,64,80,96"))
    parser.add_argument("--separations", type=comma_floats, default=comma_floats("0,0.75,0.765"))
    parser.add_argument("--box-length", type=float, default=6.0)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument("--maximum-degree", type=int, default=8)
    parser.add_argument("--n-theta", type=int, default=32)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--stability-relative-step", type=float, default=2.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kkt_mots_cartesian_calibration.json"),
    )
    args = parser.parse_args()
    if args.maximum_degree < 0 or args.maximum_degree % 2:
        parser.error("maximum degree must be a non-negative even integer")
    if args.box_length <= 0.0 or args.total_bare_mass <= 0.0:
        parser.error("box length and total bare mass must be positive")

    modes = ((0, 0), *((ell, 0) for ell in range(2, args.maximum_degree + 1, 2)))
    parameters = SpectralMOTSParameters(
        modes=modes,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        minimum_radius=0.05 * args.total_bare_mass,
        maximum_radius=2.0 * args.total_bare_mass,
        maximum_mode_amplitude=2.0,
        expansion_tolerance=1.0e-8,
        projected_gradient_tolerance=1.0e-11,
        jacobian_step=1.0e-5,
        gauss_newton_damping=1.0e-12,
        trust_radius=0.25,
        maximum_iterations=80,
    )
    direct_finder = DirectBrillLindquistMOTSFinder(
        parameters, args.total_bare_mass
    )
    oracle: dict[float, object] = {}
    oracle_records: list[dict[str, object]] = []
    for separation in sorted(args.separations):
        surface = direct_finder.find(
            separation, 0.65 * args.total_bare_mass
        )
        stability = direct_finder.normal_stability(
            separation, surface, args.stability_relative_step
        )
        if not surface.found:
            raise RuntimeError(
                f"direct oracle did not find the common surface at d={separation}"
            )
        oracle[separation] = surface
        oracle_records.append(
            {
                "separation": separation,
                "surface": surface.diagnostics(),
                "normal_stability": stability.diagnostics(),
            }
        )
        print(
            f"oracle d={separation:.9g} residual={surface.spectral_residual_norm:.3e} "
            f"lambda0={stability.principal_eigenvalue.real:.9e}"
        )

    cartesian_records: list[dict[str, object]] = []
    for resolution in args.resolutions:
        grid = CartesianGrid(
            (resolution,) * 3,
            (args.box_length,) * 3,
            method="fd4",
        )
        for separation in sorted(args.separations):
            state = brill_lindquist_state(
                grid, separation, args.total_bare_mass, separation
            )
            surface = oracle[separation]
            legacy = observe(
                SpectralMOTSFinder(grid, parameters),
                state,
                surface,
                args.stability_relative_step,
            )
            gc.collect()
            corrected = observe(
                ParametricAxisymmetricMOTSFinder(grid, parameters),
                state,
                surface,
                args.stability_relative_step,
            )
            record = {
                "resolution": resolution,
                "spacing": args.box_length / resolution,
                "separation": separation,
                "legacy_volume_level_set": legacy,
                "corrected_parametric": corrected,
            }
            cartesian_records.append(record)
            print(
                f"N={resolution} d={separation:.9g} "
                f"old_rms={legacy.get('rms_expansion', float('nan')):.3e} "
                f"new_rms={corrected.get('rms_expansion', float('nan')):.3e} "
                f"old_lambda={legacy.get('normal_stability', {}).get('principal_eigenvalue_real', float('nan')):.9e} "
                f"new_lambda={corrected.get('normal_stability', {}).get('principal_eigenvalue_real', float('nan')):.9e}"
            )
            del state, legacy, corrected
            gc.collect()

    payload: dict[str, object] = {
        "format": "causal_repair.kkt_mots_cartesian_calibration.v1",
        "interpretation": (
            "Direct angular surfaces are held fixed while two Cartesian observers "
            "measure expansion and physical-normal stability. No Cartesian result "
            "in this file is a re-optimized surface."
        ),
        "configuration": {
            "resolutions": list(args.resolutions),
            "separations": list(sorted(args.separations)),
            "box_length": args.box_length,
            "total_bare_mass": args.total_bare_mass,
            "maximum_degree": args.maximum_degree,
            "modes": [list(mode) for mode in modes],
            "n_theta": args.n_theta,
            "n_phi": args.n_phi,
            "stability_relative_step": args.stability_relative_step,
        },
        "oracle_records": oracle_records,
        "cartesian_records": cartesian_records,
    }
    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
