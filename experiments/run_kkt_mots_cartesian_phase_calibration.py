"""Attack the corrected Cartesian MOTS observer with grid-phase shifts.

The direct Brill--Lindquist surface and its physical-normal spectrum are held
fixed.  A locally refined Cartesian patch is then translated by fractions of
one cell relative to the punctures.  This exposes interpolation and puncture-
phase error separately from angular truncation and reports the lowest two
physical-normal modes needed by the dynamical CCZ4 qualification gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from causal_repair.grid import CartesianGrid
from causal_repair.kkt_horizon import (
    DirectBrillLindquistMOTSFinder,
    ParametricAxisymmetricMOTSFinder,
    SpectralMOTSParameters,
)
from run_kkt_mots_brill_lindquist import brill_lindquist_state
from run_kkt_mots_cartesian_calibration import (
    atomic_json,
    comma_floats,
    comma_ints,
    observe,
)


PHASES: dict[str, tuple[float, float, float]] = {
    "centered": (0.0, 0.0, 0.0),
    "z_quarter": (0.0, 0.0, 0.25),
    "z_half": (0.0, 0.0, 0.5),
    "xyz_half": (0.5, 0.5, 0.5),
}


def phase_names(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(result) - set(PHASES)
    if not result or unknown:
        raise argparse.ArgumentTypeError(
            f"phases must be selected from {','.join(PHASES)}"
        )
    return result


def sorted_real_modes(stability: dict[str, object]) -> list[float]:
    real = np.asarray(stability["eigenvalues_real"], dtype=float)
    imaginary = np.asarray(stability["eigenvalues_imag"], dtype=float)
    order = np.lexsort((np.abs(imaginary), real))
    return real[order].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=comma_ints, default=comma_ints("48,64,80,96"))
    parser.add_argument("--separations", type=comma_floats, default=comma_floats("0.75,0.765"))
    parser.add_argument(
        "--phases",
        type=phase_names,
        default=phase_names("centered,z_quarter,z_half,xyz_half"),
    )
    parser.add_argument("--box-length", type=float, default=2.0)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument("--maximum-degree", type=int, default=16)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--stability-relative-step", type=float, default=2.0e-4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kkt_mots_cartesian_phase_calibration.json"),
    )
    args = parser.parse_args()
    if args.maximum_degree < 2 or args.maximum_degree % 2:
        parser.error("maximum degree must be a positive even integer")
    if args.box_length <= 0.0 or args.total_bare_mass <= 0.0:
        parser.error("box length and total bare mass must be positive")

    modes = ((0, 0), *((ell, 0) for ell in range(2, args.maximum_degree + 1, 2)))
    parameters = SpectralMOTSParameters(
        modes=modes,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        minimum_radius=0.05 * args.total_bare_mass,
        maximum_radius=0.49 * args.box_length,
        maximum_mode_amplitude=2.0,
        expansion_tolerance=1.0e-8,
        projected_gradient_tolerance=1.0e-11,
        jacobian_step=1.0e-5,
        gauss_newton_damping=1.0e-12,
        trust_radius=0.25,
        maximum_iterations=80,
    )
    direct = DirectBrillLindquistMOTSFinder(parameters, args.total_bare_mass)
    oracle: dict[float, object] = {}
    oracle_records: list[dict[str, object]] = []
    for separation in sorted(args.separations):
        surface = direct.find(separation, 0.65 * args.total_bare_mass)
        stability = direct.normal_stability(
            separation, surface, args.stability_relative_step
        )
        if not surface.found or not stability.valid:
            raise RuntimeError(f"direct oracle failed at d={separation}")
        oracle[separation] = surface
        oracle_modes = sorted_real_modes(stability.diagnostics())
        oracle_records.append(
            {
                "separation": separation,
                "surface": surface.diagnostics(),
                "normal_stability": stability.diagnostics(),
                "lowest_real_modes": oracle_modes[:2],
            }
        )
        print(
            f"oracle d={separation:.9g} "
            f"lambda0={oracle_modes[0]:.9e} lambda1={oracle_modes[1]:.9e}"
        )

    records: list[dict[str, object]] = []
    for resolution in args.resolutions:
        spacing = args.box_length / resolution
        for phase_name in args.phases:
            phase = PHASES[phase_name]
            origin = tuple(
                -0.5 * args.box_length + phase[axis] * spacing
                for axis in range(3)
            )
            grid = CartesianGrid(
                (resolution,) * 3,
                (args.box_length,) * 3,
                origin=origin,
                method="fd4",
            )
            finder = ParametricAxisymmetricMOTSFinder(grid, parameters)
            for separation in sorted(args.separations):
                surface = oracle[separation]
                state = brill_lindquist_state(
                    grid, separation, args.total_bare_mass, separation
                )
                measurement = observe(
                    finder, state, surface, args.stability_relative_step
                )
                if not measurement.get("valid", False):
                    raise RuntimeError(
                        f"Cartesian observer failed at N={resolution}, "
                        f"phase={phase_name}, d={separation}: "
                        f"{measurement.get('reason')}"
                    )
                stability = measurement["normal_stability"]
                measured_modes = sorted_real_modes(stability)
                record = {
                    "resolution": resolution,
                    "spacing": spacing,
                    "phase": phase_name,
                    "phase_fraction": list(phase),
                    "origin": list(origin),
                    "separation": separation,
                    "cells_per_mean_radius": (
                        surface.coordinate_radius_mean / spacing
                    ),
                    "measurement": measurement,
                    "lowest_real_modes": measured_modes[:2],
                }
                records.append(record)
                print(
                    f"N={resolution} phase={phase_name} d={separation:.9g} "
                    f"cells/r={record['cells_per_mean_radius']:.2f} "
                    f"rms={measurement['rms_expansion']:.3e} "
                    f"lambda0={measured_modes[0]:.9e} "
                    f"lambda1={measured_modes[1]:.9e}"
                )
                del state, measurement
                gc.collect()

    summaries: list[dict[str, object]] = []
    oracle_by_separation = {
        item["separation"]: item for item in oracle_records
    }
    for resolution in args.resolutions:
        for separation in sorted(args.separations):
            selected = [
                record
                for record in records
                if record["resolution"] == resolution
                and record["separation"] == separation
            ]
            measured = np.asarray(
                [record["lowest_real_modes"] for record in selected], dtype=float
            )
            oracle_modes = np.asarray(
                oracle_by_separation[separation]["lowest_real_modes"], dtype=float
            )
            rms = np.asarray(
                [record["measurement"]["rms_expansion"] for record in selected],
                dtype=float,
            )
            summaries.append(
                {
                    "resolution": resolution,
                    "separation": separation,
                    "cells_per_mean_radius": selected[0]["cells_per_mean_radius"],
                    "phase_count": len(selected),
                    "rms_expansion_maximum": float(np.max(rms)),
                    "rms_expansion_mean": float(np.mean(rms)),
                    "mode_mean": np.mean(measured, axis=0).tolist(),
                    "mode_phase_span": np.ptp(measured, axis=0).tolist(),
                    "mode_maximum_absolute_error": np.max(
                        np.abs(measured - oracle_modes[None, :]), axis=0
                    ).tolist(),
                }
            )

    payload: dict[str, object] = {
        "format": "causal_repair.kkt_mots_cartesian_phase_calibration.v1",
        "interpretation": (
            "A fixed lmax-resolved direct surface is measured on translated "
            "local Cartesian patches. Phase summaries quantify observer bias "
            "for the lowest two physical-normal modes."
        ),
        "configuration": {
            "resolutions": list(args.resolutions),
            "separations": list(sorted(args.separations)),
            "phases": list(args.phases),
            "phase_fractions": {
                name: list(PHASES[name]) for name in args.phases
            },
            "box_length": args.box_length,
            "total_bare_mass": args.total_bare_mass,
            "maximum_degree": args.maximum_degree,
            "modes": [list(mode) for mode in modes],
            "n_theta": args.n_theta,
            "n_phi": args.n_phi,
            "stability_relative_step": args.stability_relative_step,
        },
        "oracle_records": oracle_records,
        "records": records,
        "phase_summaries": summaries,
    }
    atomic_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
