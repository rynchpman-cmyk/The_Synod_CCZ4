"""Scan common Brill--Lindquist MOTSs with direct angular surface geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_repair.kkt_horizon import (
    DirectBrillLindquistMOTSFinder,
    SpectralMOTSParameters,
)


def parse_separations(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("separations must be non-negative")
    return tuple(sorted(values))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-degree", type=int, default=8)
    parser.add_argument("--n-theta", type=int, default=40)
    parser.add_argument("--n-phi", type=int, default=8)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument(
        "--separations",
        type=parse_separations,
        default=parse_separations(
            "0.74,0.75,0.76,0.762,0.764,0.765,0.7655,0.766,0.7665,0.767"
        ),
    )
    parser.add_argument("--expansion-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--maximum-iterations", type=int, default=80)
    parser.add_argument("--initial-radius", type=float, default=0.65)
    parser.add_argument("--stability-relative-step", type=float, default=2.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kkt_mots_brill_lindquist_direct.json"),
    )
    args = parser.parse_args()
    if args.maximum_degree < 0 or args.maximum_degree % 2:
        parser.error("maximum degree must be a non-negative even integer")
    modes = ((0, 0), *((ell, 0) for ell in range(2, args.maximum_degree + 1, 2)))
    parameters = SpectralMOTSParameters(
        modes=modes,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        minimum_radius=0.05 * args.total_bare_mass,
        maximum_radius=2.0 * args.total_bare_mass,
        maximum_mode_amplitude=2.0,
        expansion_tolerance=args.expansion_tolerance,
        projected_gradient_tolerance=1.0e-11,
        jacobian_step=1.0e-5,
        gauss_newton_damping=1.0e-12,
        trust_radius=0.25,
        maximum_iterations=args.maximum_iterations,
    )
    finder = DirectBrillLindquistMOTSFinder(parameters, args.total_bare_mass)
    continuation = None
    was_found = False
    records: list[dict[str, object]] = []
    for index, separation in enumerate(args.separations):
        result = finder.find(separation, args.initial_radius, continuation)
        stability = finder.normal_stability(
            separation, result, args.stability_relative_step
        )
        if result.found:
            continuation = result.coefficients.copy()
        records.append(
            {
                "sequence_index": index,
                "separation": separation,
                "common_newly_found": result.found and not was_found,
                "common_newly_lost": was_found and not result.found,
                "branches": {
                    "common": {
                        **result.diagnostics(),
                        "normal_stability": stability.diagnostics(),
                    }
                },
            }
        )
        was_found = result.found
        print(
            f"d={separation:.9g} found={result.found} "
            f"residual={result.spectral_residual_norm:.3e} "
            f"lambda0={stability.principal_eigenvalue.real:.9e}"
        )
    found = [
        record["separation"]
        for record in records
        if record["branches"]["common"]["found"]
    ]
    payload: dict[str, object] = {
        "format": "causal_repair.kkt_mots_brill_lindquist_direct.v1",
        "interpretation": (
            "Direct axisymmetric conformal minimal-surface evaluation; no "
            "Cartesian volume differentiation or interpolation is used."
        ),
        "configuration": {
            "geometry": "direct_angular_brill_lindquist",
            "maximum_degree": args.maximum_degree,
            "modes": [list(mode) for mode in modes],
            "n_theta": args.n_theta,
            "n_phi": args.n_phi,
            "total_bare_mass": args.total_bare_mass,
            "separations": list(args.separations),
            "expansion_tolerance": args.expansion_tolerance,
            "maximum_iterations": args.maximum_iterations,
            "stability_relative_step": args.stability_relative_step,
        },
        "summary": {
            "common_found_count": len(found),
            "largest_sampled_separation_with_common_mots": max(found) if found else None,
            "transition_is_bracket_only": True,
        },
        "records": records,
    }
    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
