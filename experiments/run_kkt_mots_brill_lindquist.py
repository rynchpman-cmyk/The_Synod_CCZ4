"""Probe KKT/MOTS branch formation on Brill--Lindquist binary data.

This is a sequence of time-symmetric constraint slices, not a binary evolution.
Its purpose is to test whether the constrained surface solver locates a common
MOTS without imposing one, and to expose the active-set and shape-response
telemetry before the diagnostic is used in a dynamical production33 run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal_repair.adm import ADMState
from causal_repair.grid import CartesianGrid
from causal_repair.kkt_horizon import (
    NormalMOTSStabilityOperator,
    SpectralMOTSFinder,
    SpectralMOTSParameters,
)


def parse_separations(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("separations must be positive comma-separated values")
    return tuple(sorted(values))


def brill_lindquist_state(
    grid: CartesianGrid,
    separation: float,
    total_bare_mass: float,
    sequence_time: float,
) -> ADMState:
    """Construct equal-mass, time-symmetric Brill--Lindquist data."""

    x, y, z = grid.coordinates()
    half_separation = 0.5 * separation
    puncture_floor = 0.1 * min(grid.spacing)
    r_plus = np.maximum(
        np.sqrt(x * x + y * y + (z - half_separation) ** 2), puncture_floor
    )
    r_minus = np.maximum(
        np.sqrt(x * x + y * y + (z + half_separation) ** 2), puncture_floor
    )
    individual_bare_mass = 0.5 * total_bare_mass
    conformal_factor = (
        1.0
        + individual_bare_mass / (2.0 * r_plus)
        + individual_bare_mass / (2.0 * r_minus)
    )
    metric = grid.zeros((3, 3))
    for axis in range(3):
        metric[axis, axis] = conformal_factor**4
    return ADMState(metric, grid.zeros((3, 3)), sequence_time)


def axisymmetric_modes(maximum_degree: int, include_odd: bool) -> tuple[tuple[int, int], ...]:
    degrees = range(1 if include_odd else 2, maximum_degree + 1, 1 if include_odd else 2)
    return ((0, 0), *((ell, 0) for ell in degrees))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--box-length", type=float, default=6.0)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument(
        "--separations",
        type=parse_separations,
        default=parse_separations("0.40,0.50,0.60,0.70,0.80,0.90,1.00"),
    )
    parser.add_argument("--maximum-degree", type=int, default=4)
    parser.add_argument("--expansion-tolerance", type=float, default=2.0e-2)
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument("--common-seed-radius", type=float, default=0.65)
    parser.add_argument("--individual-seed-radius", type=float, default=0.30)
    parser.add_argument(
        "--common-only",
        action="store_true",
        help="skip the two individual branches during focused transition scans",
    )
    parser.add_argument(
        "--normal-stability",
        action="store_true",
        help="compute the induced-area, physical-normal MOTS stability operator",
    )
    parser.add_argument("--stability-relative-step", type=float, default=2.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kkt_mots_brill_lindquist.json"),
    )
    args = parser.parse_args()
    if args.resolution < 16 or args.box_length <= 0.0 or args.total_bare_mass <= 0.0:
        parser.error("resolution, box length, and mass must be positive")

    grid = CartesianGrid(
        (args.resolution,) * 3,
        (args.box_length,) * 3,
        method="fd4",
    )
    common_parameters = SpectralMOTSParameters(
        modes=axisymmetric_modes(args.maximum_degree, include_odd=False),
        n_theta=max(12, 2 * args.maximum_degree + 4),
        n_phi=max(24, 4 * args.maximum_degree + 8),
        minimum_radius=2.0 * min(grid.spacing),
        maximum_radius=0.45 * args.box_length,
        maximum_mode_amplitude=1.2,
        expansion_tolerance=args.expansion_tolerance,
        jacobian_step=8.0e-4,
        maximum_iterations=args.maximum_iterations,
    )
    individual_parameters = SpectralMOTSParameters(
        modes=axisymmetric_modes(args.maximum_degree, include_odd=True),
        include_dipole=True,
        n_theta=common_parameters.n_theta,
        n_phi=common_parameters.n_phi,
        minimum_radius=min(grid.spacing),
        maximum_radius=0.3 * args.box_length,
        maximum_mode_amplitude=1.2,
        expansion_tolerance=args.expansion_tolerance,
        jacobian_step=8.0e-4,
        maximum_iterations=args.maximum_iterations,
    )
    common_finder = SpectralMOTSFinder(grid, common_parameters)
    individual_finder = SpectralMOTSFinder(grid, individual_parameters)
    stability_operator = NormalMOTSStabilityOperator(
        common_finder, args.stability_relative_step
    )

    continuation: dict[str, np.ndarray] = {}
    was_common = False
    records: list[dict[str, object]] = []
    for index, separation in enumerate(args.separations):
        state = brill_lindquist_state(
            grid, separation, args.total_bare_mass, float(index)
        )
        centers = {"common": (0.0, 0.0, 0.0)}
        common = common_finder.find(
            state,
            args.common_seed_radius,
            centers["common"],
            continuation.get("common"),
        )
        results = {"common": common}
        if not args.common_only:
            centers.update(
                {
                    "individual_plus": (0.0, 0.0, 0.5 * separation),
                    "individual_minus": (0.0, 0.0, -0.5 * separation),
                }
            )
            results["individual_plus"] = individual_finder.find(
                state,
                args.individual_seed_radius,
                centers["individual_plus"],
                continuation.get("individual_plus"),
            )
            results["individual_minus"] = individual_finder.find(
                state,
                args.individual_seed_radius,
                centers["individual_minus"],
                continuation.get("individual_minus"),
            )
        for name, result in results.items():
            if result.found:
                continuation[name] = result.coefficients.copy()
        branch_diagnostics = {
            name: result.diagnostics() for name, result in results.items()
        }
        if args.normal_stability:
            branch_diagnostics["common"]["normal_stability"] = (
                stability_operator.analyze(state, common).diagnostics()
            )
        record = {
            "sequence_index": index,
            "separation": separation,
            "common_newly_found": common.found and not was_common,
            "common_newly_lost": was_common and not common.found,
            "branches": branch_diagnostics,
        }
        records.append(record)
        was_common = common.found
        stability_text = ""
        if args.normal_stability:
            normal = branch_diagnostics["common"]["normal_stability"]
            stability_text = (
                f" lambda0_normal={normal['principal_eigenvalue_real']:.3e}"
                if normal["valid"]
                else f" normal_stability=invalid:{normal['reason']}"
            )
        print(
            f"d={separation:.6g} common={common.found} "
            f"R=[{common.coordinate_radius_minimum:.4g},"
            f"{common.coordinate_radius_maximum:.4g}] "
            f"spectral_residual={common.spectral_residual_norm:.3e}"
            f"{stability_text}"
        )

    found_separations = [
        record["separation"]
        for record in records
        if record["branches"]["common"]["found"]
    ]
    payload: dict[str, object] = {
        "format": "causal_repair.kkt_mots_brill_lindquist.v1",
        "interpretation": (
            "A constraint-slice continuation test. Sequence index is not physical time, "
            "and KKT iterations are not spacetime evolution."
        ),
        "configuration": {
            "resolution": args.resolution,
            "box_length": args.box_length,
            "total_bare_mass": args.total_bare_mass,
            "separations": list(args.separations),
            "maximum_degree": args.maximum_degree,
            "expansion_tolerance": args.expansion_tolerance,
            "maximum_iterations": args.maximum_iterations,
            "common_only": args.common_only,
            "normal_stability": args.normal_stability,
            "stability_relative_step": args.stability_relative_step,
            "common_modes": [list(mode) for mode in common_parameters.modes],
            "individual_modes": [list(mode) for mode in individual_parameters.modes],
        },
        "summary": {
            "common_found_count": len(found_separations),
            "largest_sampled_separation_with_common_mots": (
                max(found_separations) if found_separations else None
            ),
            "transition_is_bracket_only": True,
        },
        "records": records,
    }
    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
