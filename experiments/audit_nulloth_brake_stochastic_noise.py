"""Run a seeded multimode shape-noise ensemble for the periodic brake."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.adm import inverse_metric
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import CTTParameters, PeriodicCMCInitialDataSolver
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.theory33 import CarrierPrimitive

from search_nulloth_brake_controls import _solver


SEEDS = (1103, 2207, 3313, 4421, 5527, 6637, 7753, 8861)


@dataclass(frozen=True)
class NoiseRun:
    seed: int
    noise_rms_fraction: float
    profile_peak: float
    coefficient_l2: float
    seed_coordinate_energy: float
    final_scalar_amplitude: float
    carrier_kinetic_proxy: float
    relative_suppression: float
    meets_objective: bool
    baryon_relative_drift: float
    carrier_relative_drift: float
    entropy_change: float
    hamiltonian_l2: float
    momentum_l2: float
    gauss_max_l2: float
    scalar_constraint_l2: float
    recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    admissible: bool


def stochastic_profile(
    coordinate: np.ndarray,
    wavenumber: float,
    seed: int,
    noise_fraction: float = 0.10,
) -> tuple[np.ndarray, float, float]:
    base = np.cos(wavenumber * coordinate)
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(4, 2))
    noise = np.zeros_like(base)
    for mode, (cosine, sine) in enumerate(coefficients, start=2):
        noise += cosine * np.cos(mode * wavenumber * coordinate)
        noise += sine * np.sin(mode * wavenumber * coordinate)
    noise -= np.mean(noise)
    noise *= (
        noise_fraction
        * np.sqrt(np.mean(base**2))
        / max(np.sqrt(np.mean(noise**2)), 1.0e-300)
    )
    raw = base + noise
    profile = raw / np.max(np.abs(raw))
    realized_fraction = float(
        np.sqrt(np.mean((noise / np.max(np.abs(raw))) ** 2))
        / np.sqrt(np.mean((base / np.max(np.abs(raw))) ** 2))
    )
    return profile, realized_fraction, float(np.linalg.norm(coefficients))


def _source_values(record_path: Path, assessment_path: Path) -> tuple[float, int]:
    source = json.loads(record_path.read_text(encoding="utf-8"))
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    match = next(
        result
        for result in source["parsed_stdout"]["resolution_results"]
        if result["active"]["points"] == 32
    )
    return (
        float(match["control"]["carrier_kinetic_proxy"]),
        int(assessment["scientific_acceptance"]["passed"]),
    )


def _run(seed: int, control_kinetic: float) -> NoiseRun:
    points = 32
    duration = 2.0
    dt = 0.01
    amplitude = 0.02
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid, NULLOTH_DISCOVERY_PARAMETERS)
    (coordinate,) = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    carrier_profile = np.cos(wavenumber * coordinate)
    profile, realized_noise, coefficient_l2 = stochastic_profile(
        coordinate, wavenumber, seed
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(0.18 + 0.04 * carrier_profile, grid.zeros((3,)))
    varphi = amplitude * profile
    peak_effective_mass = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass**2
        - NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling * 0.22**2
    )
    growing_rate = float(np.sqrt(-peak_effective_mass - wavenumber**2))
    scalar_momentum = -growing_rate * varphi
    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )
    provisional_recovery = solver.recover(provisional)
    constraints = PeriodicCMCInitialDataSolver(
        grid,
        CTTParameters(
            kappa=solver.ccz4.parameters.kappa_gravity,
            maximum_iterations=4000,
            tolerance=1.0e-8,
            relaxation=0.3,
            report_interval=20,
        ),
        expanding=False,
    ).solve(
        provisional_recovery.total_stress.rho,
        provisional_recovery.total_stress.momentum,
    )
    if not constraints.report.converged:
        raise FloatingPointError(f"CMC solve failed for stochastic seed {seed}")
    state = solver.initialize(
        solver.ccz4.from_adm(constraints.geometry),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )
    initial = solver.diagnostics(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    _, _, sqrt_h = inverse_metric(h)
    scalar_stress = solver.scalar_system.stress(
        h, state.scalar, solver.master.scalar_potential(state.scalar.varphi)
    )
    seed_energy = float(grid.integrate(sqrt_h * scalar_stress.rho))
    for _ in range(int(round(duration / dt))):
        state = solver.step(state, dt)

    diagnostics = solver.diagnostics(state)
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    speed_squared = np.maximum(
        np.einsum(
            "ij...,i...,j...->...",
            h,
            recovery.carrier.velocity,
            recovery.carrier.velocity,
        ),
        0.0,
    )
    carrier_W = 1.0 / np.sqrt(np.maximum(1.0 - speed_squared, 1.0e-300))
    kinetic = float(
        grid.integrate(recovery.carrier.number_density * (carrier_W - 1.0))
    )
    suppression = (control_kinetic - kinetic) / abs(control_kinetic)
    admissible = bool(
        diagnostics["recovery_failures"] == 0
        and diagnostics["minimum_legendre_eigenvalue"] > 0.0
        and diagnostics["minimum_thermodynamic_eigenvalue"] > 0.0
        and diagnostics["maximum_scalar_amplitude"]
        < NULLOTH_DISCOVERY_PARAMETERS.scalar_amplitude_limit
    )
    return NoiseRun(
        seed,
        realized_noise,
        float(np.max(np.abs(profile))),
        coefficient_l2,
        seed_energy,
        diagnostics["maximum_scalar_amplitude"],
        kinetic,
        suppression,
        suppression >= 0.02,
        abs(diagnostics["baryon_mass"] - initial["baryon_mass"])
        / abs(initial["baryon_mass"]),
        abs(diagnostics["carrier_number"] - initial["carrier_number"])
        / abs(initial["carrier_number"]),
        diagnostics["entropy_integral"] - initial["entropy_integral"],
        diagnostics["hamiltonian_l2"],
        diagnostics["momentum_l2"],
        max(diagnostics["gauss_A_l2"], diagnostics["gauss_B_l2"]),
        diagnostics["scalar_constraint_l2"],
        diagnostics["recovery_maximum_residual"],
        diagnostics["minimum_legendre_eigenvalue"],
        diagnostics["minimum_thermodynamic_eigenvalue"],
        admissible,
    )


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    fraction = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (fraction + z**2 / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(fraction * (1.0 - fraction) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return float(center - half), float(center + half)


def audit(source_record: Path, assessment_record: Path) -> dict[str, object]:
    control_kinetic, source_accepted = _source_values(source_record, assessment_record)
    runs = [_run(seed, control_kinetic) for seed in SEEDS]
    suppressions = np.asarray([run.relative_suppression for run in runs])
    passes = sum(run.meets_objective for run in runs)
    lower, upper = _wilson_interval(passes, len(runs))
    return {
        "source_assessment_accepted": source_accepted,
        "points": 32,
        "ensemble_size": len(runs),
        "distinct_seeds": len(set(run.seed for run in runs)),
        "target_relative_suppression": 0.02,
        "realizations_meeting_objective": passes,
        "realizations_missing_objective": len(runs) - passes,
        "empirical_pass_fraction": passes / len(runs),
        "wilson_95_lower_pass_fraction": lower,
        "wilson_95_upper_pass_fraction": upper,
        "minimum_relative_suppression": float(np.min(suppressions)),
        "mean_relative_suppression": float(np.mean(suppressions)),
        "maximum_relative_suppression": float(np.max(suppressions)),
        "suppression_standard_deviation": float(np.std(suppressions, ddof=1)),
        "maximum_noise_fraction_error": max(
            abs(run.noise_rms_fraction - 0.10) for run in runs
        ),
        "maximum_profile_peak_error": max(abs(run.profile_peak - 1.0) for run in runs),
        "all_runs_admissible": int(all(run.admissible for run in runs)),
        "maximum_number_relative_drift": max(
            max(run.baryon_relative_drift, run.carrier_relative_drift)
            for run in runs
        ),
        "minimum_entropy_change": min(run.entropy_change for run in runs),
        "maximum_gauss_l2": max(run.gauss_max_l2 for run in runs),
        "maximum_scalar_constraint_l2": max(run.scalar_constraint_l2 for run in runs),
        "maximum_gravitational_constraint_l2": max(
            max(run.hamiltonian_l2, run.momentum_l2) for run in runs
        ),
        "maximum_recovery_residual": max(run.recovery_residual for run in runs),
        "minimum_legendre_eigenvalue": min(
            run.minimum_legendre_eigenvalue for run in runs
        ),
        "minimum_thermodynamic_eigenvalue": min(
            run.minimum_thermodynamic_eigenvalue for run in runs
        ),
        "runs": [asdict(run) for run in runs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_record", type=Path)
    parser.add_argument("assessment_record", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            audit(arguments.source_record, arguments.assessment_record),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
