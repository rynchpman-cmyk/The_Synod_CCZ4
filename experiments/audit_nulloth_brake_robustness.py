"""Audit targeting robustness of the periodic Carrier Coherence Brake."""

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


@dataclass(frozen=True)
class RobustnessVariant:
    name: str
    commanded_amplitude: float
    realized_amplitude: float
    phase_offset_radians: float
    noise_fraction: float
    momentum_scale: float
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


VARIANTS = (
    ("amplitude_minus_10_percent", 0.018, 0.0, 0.0, 1.0),
    ("amplitude_plus_10_percent", 0.022, 0.0, 0.0, 1.0),
    ("phase_minus_pi_over_8", 0.020, -np.pi / 8.0, 0.0, 1.0),
    ("phase_plus_pi_over_8", 0.020, np.pi / 8.0, 0.0, 1.0),
    ("second_harmonic_10_percent", 0.020, 0.0, 0.10, 1.0),
    ("momentum_underdrive_20_percent", 0.020, 0.0, 0.0, 0.8),
)


def normalized_seed_profile(
    coordinate: np.ndarray,
    wavenumber: float,
    phase: float,
    noise_fraction: float,
) -> np.ndarray:
    profile = np.cos(wavenumber * coordinate + phase)
    if noise_fraction:
        profile = profile + noise_fraction * np.cos(
            2.0 * wavenumber * coordinate + 0.37
        )
    return profile / np.max(np.abs(profile))


def _source_values(record_path: Path, assessment_path: Path) -> tuple[float, float, int]:
    source = json.loads(record_path.read_text(encoding="utf-8"))
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    matches = [
        result
        for result in source["parsed_stdout"]["resolution_results"]
        if result["active"]["points"] == 32
    ]
    if len(matches) != 1:
        raise ValueError("expected one sealed 32-point source result")
    result = matches[0]
    return (
        float(result["control"]["carrier_kinetic_proxy"]),
        float(result["active"]["carrier_kinetic_proxy"]),
        int(assessment["scientific_acceptance"]["passed"]),
    )


def _run_variant(
    *,
    name: str,
    amplitude: float,
    phase: float,
    noise_fraction: float,
    momentum_scale: float,
    control_kinetic: float,
    points: int,
    duration: float,
    dt: float,
    target: float,
) -> RobustnessVariant:
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid, NULLOTH_DISCOVERY_PARAMETERS)
    (coordinate,) = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    carrier_profile = np.cos(wavenumber * coordinate)
    scalar_profile = normalized_seed_profile(
        coordinate, wavenumber, phase, noise_fraction
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(0.18 + 0.04 * carrier_profile, grid.zeros((3,)))
    varphi = amplitude * scalar_profile
    peak_effective_mass = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass**2
        - NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling * 0.22**2
    )
    growing_rate = float(np.sqrt(-peak_effective_mass - wavenumber**2))
    scalar_momentum = -momentum_scale * growing_rate * varphi
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
        raise FloatingPointError(f"CMC solve failed for {name}")
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
    steps = int(round(duration / dt))
    for _ in range(steps):
        state = solver.step(state, dt)

    diagnostics = solver.diagnostics(state)
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    carrier_speed_squared = np.maximum(
        np.einsum(
            "ij...,i...,j...->...",
            h,
            recovery.carrier.velocity,
            recovery.carrier.velocity,
        ),
        0.0,
    )
    carrier_W = 1.0 / np.sqrt(np.maximum(1.0 - carrier_speed_squared, 1.0e-300))
    kinetic = float(
        grid.integrate(recovery.carrier.number_density * (carrier_W - 1.0))
    )
    suppression = (control_kinetic - kinetic) / max(
        abs(control_kinetic), 1.0e-300
    )
    admissible = bool(
        diagnostics["recovery_failures"] == 0
        and diagnostics["minimum_legendre_eigenvalue"] > 0.0
        and diagnostics["minimum_thermodynamic_eigenvalue"] > 0.0
        and diagnostics["maximum_scalar_amplitude"]
        < NULLOTH_DISCOVERY_PARAMETERS.scalar_amplitude_limit
    )
    return RobustnessVariant(
        name,
        amplitude,
        float(np.max(np.abs(varphi))),
        phase,
        noise_fraction,
        momentum_scale,
        seed_energy,
        diagnostics["maximum_scalar_amplitude"],
        kinetic,
        suppression,
        suppression >= target,
        abs(diagnostics["baryon_mass"] - initial["baryon_mass"])
        / max(abs(initial["baryon_mass"]), 1.0e-300),
        abs(diagnostics["carrier_number"] - initial["carrier_number"])
        / max(abs(initial["carrier_number"]), 1.0e-300),
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


def audit(
    source_record: Path,
    assessment_record: Path,
    *,
    points: int = 32,
    duration: float = 2.0,
    dt: float = 0.01,
    target: float = 0.02,
) -> dict[str, object]:
    control_kinetic, nominal_kinetic, source_accepted = _source_values(
        source_record, assessment_record
    )
    nominal_suppression = (control_kinetic - nominal_kinetic) / abs(control_kinetic)
    variants = [
        _run_variant(
            name=name,
            amplitude=amplitude,
            phase=phase,
            noise_fraction=noise,
            momentum_scale=momentum,
            control_kinetic=control_kinetic,
            points=points,
            duration=duration,
            dt=dt,
            target=target,
        )
        for name, amplitude, phase, noise, momentum in VARIANTS
    ]
    by_name = {variant.name: variant for variant in variants}
    phase_minus = by_name["phase_minus_pi_over_8"].relative_suppression
    phase_plus = by_name["phase_plus_pi_over_8"].relative_suppression
    phase_asymmetry = abs(phase_plus - phase_minus) / max(
        abs(phase_plus), abs(phase_minus), 1.0e-300
    )
    return {
        "source_assessment_accepted": source_accepted,
        "points": points,
        "variant_count": len(variants),
        "target_relative_suppression": target,
        "sealed_nominal_relative_suppression": nominal_suppression,
        "variants_meeting_objective": sum(v.meets_objective for v in variants),
        "minimum_variant_relative_suppression": min(
            v.relative_suppression for v in variants
        ),
        "amplitude_response_ordered": int(
            by_name["amplitude_minus_10_percent"].relative_suppression
            < nominal_suppression
            < by_name["amplitude_plus_10_percent"].relative_suppression
        ),
        "phase_pair_relative_asymmetry": phase_asymmetry,
        "shape_noise_meets_objective": int(
            by_name["second_harmonic_10_percent"].meets_objective
        ),
        "momentum_underdrive_meets_objective": int(
            by_name["momentum_underdrive_20_percent"].meets_objective
        ),
        "all_variants_admissible": int(all(v.admissible for v in variants)),
        "maximum_number_relative_drift": max(
            max(v.baryon_relative_drift, v.carrier_relative_drift)
            for v in variants
        ),
        "minimum_entropy_change": min(v.entropy_change for v in variants),
        "maximum_gauss_l2": max(v.gauss_max_l2 for v in variants),
        "maximum_scalar_constraint_l2": max(v.scalar_constraint_l2 for v in variants),
        "maximum_gravitational_constraint_l2": max(
            max(v.hamiltonian_l2, v.momentum_l2) for v in variants
        ),
        "maximum_recovery_residual": max(v.recovery_residual for v in variants),
        "minimum_legendre_eigenvalue": min(
            v.minimum_legendre_eigenvalue for v in variants
        ),
        "minimum_thermodynamic_eigenvalue": min(
            v.minimum_thermodynamic_eigenvalue for v in variants
        ),
        "variants": [asdict(variant) for variant in variants],
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
