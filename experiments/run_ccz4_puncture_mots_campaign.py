"""Evolve head-on punctures and track dynamical MOTS branches on CCZ4 snapshots.

This is a vacuum, equal-mass Brill--Lindquist campaign.  The surface observers
are read-only: CCZ4 supplies physical ADM snapshots, while independent common
and individual branch trackers evolve only their own KKT pseudo-time.  Output
and observer continuation are restartable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters, CCZ4Solver, CCZ4State
from causal_repair.grid import CartesianGrid
from causal_repair.io import load_checkpoint, save_checkpoint
from causal_repair.kkt_horizon import (
    CCZ4PhysicalNormalMOTSTracker,
    MOTSBranchDefinition,
    ParametricAxisymmetricMOTSFinder,
    SpectralMOTSParameters,
)
from run_kkt_mots_brill_lindquist import brill_lindquist_state


STATE_FIELDS = (
    "conformal_metric",
    "conformal_A",
    "conformal_factor",
    "trace_K",
    "theta",
    "gamma_hat",
    "lapse",
    "shift",
    "shift_driver",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(path)


def save_campaign_checkpoint(
    path: Path,
    state: CCZ4State,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + ".pending.npz")
    arrays = {name: getattr(state, name) for name in STATE_FIELDS}
    save_checkpoint(pending, metadata=metadata, time=np.asarray(state.time), **arrays)
    pending.replace(path)


def load_campaign_checkpoint(path: Path) -> tuple[CCZ4State, dict[str, object]]:
    arrays, metadata = load_checkpoint(path)
    missing = set(STATE_FIELDS) - set(arrays)
    if missing or "time" not in arrays:
        raise ValueError(f"incomplete CCZ4 puncture checkpoint: {sorted(missing)}")
    state = CCZ4State(
        *(arrays[name] for name in STATE_FIELDS),
        float(np.asarray(arrays["time"])),
    )
    return state, metadata


def quadratic_minimum(coordinates: np.ndarray, values: np.ndarray) -> float:
    index = int(np.argmin(values))
    if index == 0 or index == len(values) - 1:
        return float(coordinates[index])
    left, center, right = values[index - 1 : index + 2]
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float(coordinates[index])
    offset = 0.5 * (left - right) / denominator
    return float(coordinates[index] + offset * (coordinates[index + 1] - coordinates[index]))


def puncture_centers(grid: CartesianGrid, state: CCZ4State) -> tuple[float, float]:
    """Locate the two punctures from axial minima of the conformal factor."""

    z = grid.coordinates()[2][0, 0, :]
    profile = np.min(state.conformal_factor, axis=(0, 1))
    negative = z < 0.0
    positive = z > 0.0
    if not np.any(negative) or not np.any(positive):
        raise FloatingPointError("grid does not straddle both punctures")
    lower = quadratic_minimum(z[negative], profile[negative])
    upper = quadratic_minimum(z[positive], profile[positive])
    if not lower < upper:
        raise FloatingPointError("puncture locator lost branch ordering")
    return lower, upper


def modes(maximum_degree: int, even_only: bool) -> tuple[tuple[int, int], ...]:
    start = 2 if even_only else 1
    stride = 2 if even_only else 1
    return ((0, 0), *((ell, 0) for ell in range(start, maximum_degree + 1, stride)))


def parameters(
    grid: CartesianGrid,
    maximum_degree: int,
    even_only: bool,
    expansion_tolerance: float,
    maximum_iterations: int,
) -> SpectralMOTSParameters:
    return SpectralMOTSParameters(
        modes=modes(maximum_degree, even_only),
        include_dipole=not even_only,
        n_theta=max(20, 4 * maximum_degree),
        n_phi=max(12, 2 * maximum_degree),
        minimum_radius=max(0.55 * min(grid.spacing), 0.015),
        maximum_radius=0.46 * min(grid.lengths),
        maximum_mode_amplitude=2.0,
        expansion_tolerance=expansion_tolerance,
        projected_gradient_tolerance=1.0e-9,
        jacobian_step=2.0e-4,
        gauss_newton_damping=1.0e-10,
        trust_radius=0.25,
        maximum_iterations=maximum_iterations,
    )


def tracker_diagnostics(observations) -> dict[str, object]:
    return {
        name: observation.diagnostics()
        for name, observation in observations.items()
    }


def classify_common(observations) -> dict[str, object]:
    """Classify a common pair by stability sign rather than seed name."""

    candidates: dict[str, list[tuple[str, object]]] = {
        "common_outer": [],
        "common_inner": [],
    }
    ambiguous: list[str] = []
    for seed_name, observation in observations.items():
        stability = observation.normal_stability
        if not observation.branch.result.found or stability is None or not stability.valid:
            continue
        real = float(stability.principal_eigenvalue.real)
        if real > 0.0:
            candidates["common_outer"].append((seed_name, observation))
        elif real < 0.0:
            candidates["common_inner"].append((seed_name, observation))
        else:
            ambiguous.append(seed_name)
    result: dict[str, object] = {"ambiguous_seed_names": ambiguous}
    for physical_name, values in candidates.items():
        if not values:
            result[physical_name] = None
            continue
        values.sort(key=lambda item: item[1].branch.result.spectral_residual_norm)
        seed_name, observation = values[0]
        diagnostic = observation.diagnostics()
        diagnostic["tracker_seed_name"] = seed_name
        result[physical_name] = diagnostic
        ambiguous.extend(name for name, _ in values[1:])
    return result


def masked_constraint_diagnostics(
    solver: CCZ4Solver,
    state: CCZ4State,
    punctures: tuple[float, float],
    mask_cells: float,
    boundary_mask_cells: float,
) -> dict[str, float]:
    hamiltonian, momentum = solver.constraints(state)
    x, y, z = solver.grid.coordinates()
    mask_radius = mask_cells * max(solver.grid.spacing)
    mask = np.ones(solver.grid.shape, dtype=bool)
    for puncture in punctures:
        mask &= x * x + y * y + (z - puncture) ** 2 >= mask_radius**2
    if not np.any(mask):
        raise FloatingPointError("puncture mask removed the entire grid")
    momentum_squared = np.sum(momentum * momentum, axis=0)
    boundary_distance = np.minimum.reduce(
        [
            x - solver.grid.origin[0],
            solver.grid.origin[0] + solver.grid.lengths[0] - x,
            y - solver.grid.origin[1],
            solver.grid.origin[1] + solver.grid.lengths[1] - y,
            z - solver.grid.origin[2],
            solver.grid.origin[2] + solver.grid.lengths[2] - z,
        ]
    )
    boundary_mask_radius = boundary_mask_cells * max(solver.grid.spacing)
    interior = mask & (boundary_distance >= boundary_mask_radius)
    if not np.any(interior):
        raise FloatingPointError("combined puncture/boundary mask removed the grid")
    return {
        "puncture_mask_radius": mask_radius,
        "boundary_mask_radius": boundary_mask_radius,
        "unmasked_fraction": float(np.mean(mask)),
        "interior_fraction": float(np.mean(interior)),
        "hamiltonian_l2_full": float(np.sqrt(np.mean(hamiltonian**2))),
        "hamiltonian_l2_masked": float(np.sqrt(np.mean(hamiltonian[mask] ** 2))),
        "hamiltonian_l2_interior": float(
            np.sqrt(np.mean(hamiltonian[interior] ** 2))
        ),
        "momentum_l2_full": float(np.sqrt(np.mean(momentum_squared))),
        "momentum_l2_masked": float(np.sqrt(np.mean(momentum_squared[mask]))),
        "momentum_l2_interior": float(
            np.sqrt(np.mean(momentum_squared[interior]))
        ),
    }


def state_health(state: CCZ4State) -> dict[str, float | bool]:
    finite = all(np.all(np.isfinite(getattr(state, name))) for name in STATE_FIELDS)
    matrices = np.moveaxis(state.conformal_metric, (0, 1), (-2, -1))
    minimum_metric = float(np.min(np.linalg.eigvalsh(matrices)))
    return {
        "finite": finite,
        "minimum_lapse": float(np.min(state.lapse)),
        "maximum_lapse": float(np.max(state.lapse)),
        "minimum_conformal_factor": float(np.min(state.conformal_factor)),
        "maximum_conformal_factor": float(np.max(state.conformal_factor)),
        "minimum_conformal_metric_eigenvalue": minimum_metric,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--box-length", type=float, default=2.0)
    parser.add_argument("--separation", type=float, default=0.78)
    parser.add_argument("--total-bare-mass", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--cfl", type=float, default=0.02)
    parser.add_argument("--observation-cadence", type=int, default=10)
    parser.add_argument("--checkpoint-cadence", type=int, default=40)
    parser.add_argument("--maximum-degree", type=int, default=6)
    parser.add_argument("--individual-maximum-degree", type=int, default=4)
    parser.add_argument("--expansion-tolerance", type=float, default=2.0e-3)
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument("--mask-cells", type=float, default=5.0)
    parser.add_argument("--boundary-mask-cells", type=float, default=5.0)
    parser.add_argument("--branch-split-log-radius", type=float, default=0.2)
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs/ccz4_puncture_mots_pilot"),
    )
    args = parser.parse_args()
    if args.resolution < 16 or args.box_length <= 0.0 or args.separation <= 0.0:
        parser.error("resolution, box length, and separation must be positive")
    if args.steps < 0 or args.cfl <= 0.0 or args.observation_cadence <= 0:
        parser.error("steps, CFL, and observation cadence are invalid")
    if args.maximum_degree < 2 or args.maximum_degree % 2:
        parser.error("common maximum degree must be a positive even integer")

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    grid = CartesianGrid(
        (args.resolution,) * 3,
        (args.box_length,) * 3,
        method="fd4",
    )
    dt = args.cfl * min(grid.spacing)
    ccz4_parameters = CCZ4Parameters(
        slicing="1+log",
        evolve_shift=True,
        radiative_boundaries=True,
    )
    solver = CCZ4Solver(grid, ccz4_parameters)
    common_finder = ParametricAxisymmetricMOTSFinder(
        grid,
        parameters(
            grid,
            args.maximum_degree,
            True,
            args.expansion_tolerance,
            args.maximum_iterations,
        ),
    )
    common_tracker = CCZ4PhysicalNormalMOTSTracker(
        solver,
        common_finder,
        (
            MOTSBranchDefinition("common_outer", (0.0, 0.0, 0.0), 0.58),
            MOTSBranchDefinition("common_inner", (0.0, 0.0, 0.0), 0.32),
        ),
    )
    individual_tracker = None
    if not args.skip_individual:
        individual_finder = ParametricAxisymmetricMOTSFinder(
            grid,
            parameters(
                grid,
                args.individual_maximum_degree,
                False,
                args.expansion_tolerance,
                args.maximum_iterations,
            ),
        )
        individual_tracker = CCZ4PhysicalNormalMOTSTracker(
            solver,
            individual_finder,
            (
                MOTSBranchDefinition("individual_minus", (0.0, 0.0, -0.5 * args.separation), 0.18),
                MOTSBranchDefinition("individual_plus", (0.0, 0.0, 0.5 * args.separation), 0.18),
            ),
        )

    configuration = {
        "resolution": args.resolution,
        "box_length": args.box_length,
        "separation": args.separation,
        "total_bare_mass": args.total_bare_mass,
        "dt": dt,
        "cfl": args.cfl,
        "observation_cadence": args.observation_cadence,
        "checkpoint_cadence": args.checkpoint_cadence,
        "maximum_degree": args.maximum_degree,
        "individual_maximum_degree": args.individual_maximum_degree,
        "expansion_tolerance": args.expansion_tolerance,
        "maximum_iterations": args.maximum_iterations,
        "mask_cells": args.mask_cells,
        "boundary_mask_cells": args.boundary_mask_cells,
        "branch_split_log_radius": args.branch_split_log_radius,
        "skip_individual": args.skip_individual,
        "ccz4_parameters": asdict(ccz4_parameters),
    }
    trajectory_path = output / "trajectory.json"
    records: list[dict[str, object]] = []
    starting_step = 0
    if args.restart_from is None:
        adm = brill_lindquist_state(
            grid, args.separation, args.total_bare_mass, 0.0
        )
        state = solver.from_adm(adm)
        # CCZ4 stores W=det(gamma)^(-1/6)=psi^(-2) for conformally flat
        # puncture data.  The standard pre-collapsed lapse is alpha=psi^(-2),
        # hence alpha=W rather than sqrt(W).
        state.lapse = state.conformal_factor.copy()
    else:
        state, metadata = load_campaign_checkpoint(args.restart_from)
        stored_configuration = metadata.get("configuration")
        if isinstance(stored_configuration, dict):
            stored_configuration.setdefault(
                "boundary_mask_cells", args.boundary_mask_cells
            )
            stored_configuration.setdefault(
                "branch_split_log_radius", args.branch_split_log_radius
            )
        if stored_configuration != configuration:
            raise ValueError("restart configuration does not match requested campaign")
        starting_step = int(metadata["step"])
        common_tracker.restore(metadata["common_tracker"])
        if individual_tracker is not None:
            individual_payload = metadata.get("individual_tracker")
            if not isinstance(individual_payload, dict):
                raise ValueError("restart is missing individual tracker state")
            individual_tracker.restore(individual_payload)
        if trajectory_path.exists():
            records = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if records and int(records[-1]["step"]) != starting_step:
            raise ValueError("trajectory and restart checkpoint steps disagree")

    def checkpoint_metadata(step: int) -> dict[str, object]:
        return {
            "format": "causal_repair.ccz4_puncture_mots_campaign.v1",
            "step": step,
            "configuration": configuration,
            "common_tracker": common_tracker.checkpoint(),
            "individual_tracker": (
                None if individual_tracker is None else individual_tracker.checkpoint()
            ),
        }

    def observe(step: int) -> None:
        lower, upper = puncture_centers(grid, state)
        midpoint = 0.5 * (lower + upper)
        common_tracker.update_center("common_outer", (0.0, 0.0, midpoint))
        common_tracker.update_center("common_inner", (0.0, 0.0, midpoint))
        common = common_tracker.observe(state)
        found_common = [
            value
            for value in common.values()
            if value.branch.result.found
            and value.normal_stability is not None
            and value.normal_stability.valid
        ]
        # At a saddle-node birth the two surfaces are close.  If only one seed
        # converges, use its stability sign to plant the missing partner on the
        # appropriate side in log radius for the next snapshot.
        if len(found_common) == 1:
            source = found_common[0]
            seed = source.branch.result.coefficients.copy()
            if source.normal_stability.principal_eigenvalue.real < 0.0:
                common_tracker.seed_branch("common_inner", seed)
                seed[0] += args.branch_split_log_radius
                common_tracker.seed_branch("common_outer", seed)
            else:
                common_tracker.seed_branch("common_outer", seed)
                seed[0] -= args.branch_split_log_radius
                common_tracker.seed_branch("common_inner", seed)
        individual = None
        if individual_tracker is not None:
            individual_tracker.update_center("individual_minus", (0.0, 0.0, lower))
            individual_tracker.update_center("individual_plus", (0.0, 0.0, upper))
            individual = individual_tracker.observe(state)
        health = state_health(state)
        constraints = masked_constraint_diagnostics(
            solver,
            state,
            (lower, upper),
            args.mask_cells,
            args.boundary_mask_cells,
        )
        record: dict[str, object] = {
            "step": step,
            "time": float(state.time),
            "puncture_z": [lower, upper],
            "coordinate_separation": upper - lower,
            "health": health,
            "constraints": constraints,
            "common": tracker_diagnostics(common),
            "common_classified": classify_common(common),
            "individual": (
                None if individual is None else tracker_diagnostics(individual)
            ),
        }
        records.append(record)
        atomic_json(trajectory_path, records)
        classified = record["common_classified"]
        found = [
            name
            for name in ("common_outer", "common_inner")
            if classified[name] is not None
        ]
        print(
            f"step={step} t={state.time:.8f} puncture_d={upper-lower:.8f} "
            f"common={found or '-'} Hint={constraints['hamiltonian_l2_interior']:.3e} "
            f"alpha_min={health['minimum_lapse']:.3e}"
        )
        for name in found:
            value = classified[name]
            qualification = value["cartesian_qualification"]
            stability = value["normal_stability"]
            if stability is not None:
                print(
                    f"  {name}: r={value['coordinate_radius_mean']:.6f} "
                    f"lambda0={stability['principal_eigenvalue_real']:.6e} "
                    f"cells/r={qualification['cells_per_mean_radius']:.2f} "
                    f"sign_gate={qualification['principal_mode_sign_gate_passed']}"
                )
        if args.save_snapshots:
            save_campaign_checkpoint(
                output / "snapshots" / f"step_{step:07d}.npz",
                state,
                checkpoint_metadata(step),
            )

    if not records or int(records[-1]["step"]) < starting_step:
        observe(starting_step)
    for step in range(starting_step + 1, args.steps + 1):
        state = solver.step(state, dt)
        health = state_health(state)
        if (
            not health["finite"]
            or health["minimum_lapse"] <= 0.0
            or health["minimum_conformal_factor"] <= 0.0
            or health["minimum_conformal_metric_eigenvalue"] <= 0.0
        ):
            raise FloatingPointError(f"CCZ4 health gate failed at step {step}: {health}")
        if step % args.observation_cadence == 0 or step == args.steps:
            observe(step)
        if step % args.checkpoint_cadence == 0 or step == args.steps:
            save_campaign_checkpoint(
                output / "final_checkpoint.npz",
                state,
                checkpoint_metadata(step),
            )

    summary = {
        "format": "causal_repair.ccz4_puncture_mots_summary.v1",
        "configuration": configuration,
        "completed_steps": args.steps,
        "final_time": float(state.time),
        "records": len(records),
        "common_outer_first_found_time": next(
            (
                record["time"]
                for record in records
                if record.get("common_classified", {}).get("common_outer") is not None
            ),
            None,
        ),
        "common_inner_first_found_time": next(
            (
                record["time"]
                for record in records
                if record.get("common_classified", {}).get("common_inner") is not None
            ),
            None,
        ),
        "final_health": state_health(state),
    }
    atomic_json(output / "summary.json", summary)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
