"""Promote Nulloth coherence candidates through the full production solver."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import (
    CTTParameters,
    PeriodicCMCInitialDataSolver,
)
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.production_nulloth import (
    NullothProductionSolver,
    load_nulloth_production_state,
    save_nulloth_production_state,
)
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class FullCandidateRun:
    active: bool
    points: int
    duration: float
    steps: int
    initial_scalar_amplitude: float
    final_scalar_amplitude: float
    peak_scalar_amplitude: float
    peak_to_final_drop: float
    nonlinear_turnover: bool
    scalar_amplification: float
    fitted_growth_rate: float
    late_fitted_growth_rate: float
    equilibrium_amplitude: float
    maximum_fluid_speed: float
    maximum_carrier_speed: float
    carrier_kinetic_proxy: float
    carrier_contrast: float
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
    restart_maximum_error: float
    admissible: bool


@dataclass(frozen=True)
class SpellPromotionResult:
    candidate: str
    active: FullCandidateRun
    control: FullCandidateRun
    carrier_density_effect_l2: float
    carrier_velocity_effect_l2: float
    carrier_kinetic_effect: float
    relative_carrier_kinetic_effect: float
    effect_kind: str
    promoted: bool
    rejection_reasons: tuple[str, ...]


def _solver(grid: PeriodicGrid) -> NullothProductionSolver:
    p = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=p.relaxation_time,
        gamma_ad=p.gamma_ad,
    )
    geometry = CCZ4Parameters(
        kappa_gravity=1.0e-3,
        slicing="harmonic",
        evolve_shift=False,
        radiative_boundaries=False,
    )
    return NullothProductionSolver(grid, p, vector, geometry)


def _initial_state(
    solver: NullothProductionSolver, *, active: bool
):
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    profile = np.cos(wavenumber * coordinate)
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        0.18 + 0.04 * profile,
        grid.zeros((3,)),
    )
    amplitude = 0.02 if active else 0.0
    varphi = amplitude * profile
    peak_effective_mass = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass**2
        - NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling * 0.22**2
    )
    growth_rate = np.sqrt(-peak_effective_mass - wavenumber**2)
    scalar_momentum = -growth_rate * varphi

    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )
    recovery = solver.recover(provisional)
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
    ).solve(recovery.total_stress.rho, recovery.total_stress.momentum)
    if not constraints.report.converged:
        raise FloatingPointError("periodic CMC candidate data did not converge")
    geometry = solver.ccz4.from_adm(constraints.geometry)
    return solver.initialize(
        geometry,
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )


def _run(
    *, points: int, duration: float, dt: float, active: bool
):
    steps = int(round(duration / dt))
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must be a positive integer multiple of dt")
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid)
    state = _initial_state(solver, active=active)
    initial_diagnostics = solver.diagnostics(state)
    initial_baryons = initial_diagnostics["baryon_mass"]
    initial_carriers = initial_diagnostics["carrier_number"]
    initial_entropy = initial_diagnostics["entropy_integral"]
    initial_amplitude = max(
        initial_diagnostics["maximum_scalar_amplitude"], 1.0e-300
    )
    sample_times = [state.time]
    sample_amplitudes = [initial_amplitude]

    for step in range(steps):
        state = solver.step(state, dt)
        if step == steps - 1 or (step + 1) % max(1, steps // 20) == 0:
            sample_times.append(state.time)
            sample_amplitudes.append(
                max(float(np.max(np.abs(state.scalar.varphi))), 1.0e-300)
            )

    diagnostics = solver.diagnostics(state)
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    fluid_speed = np.sqrt(
        np.maximum(
            np.einsum(
                "ij...,i...,j...->...",
                h,
                recovery.fluid.velocity,
                recovery.fluid.velocity,
            ),
            0.0,
        )
    )
    carrier_speed = np.sqrt(
        np.maximum(
            np.einsum(
                "ij...,i...,j...->...",
                h,
                recovery.carrier.velocity,
                recovery.carrier.velocity,
            ),
            0.0,
        )
    )
    carrier_W = 1.0 / np.sqrt(np.maximum(1.0 - carrier_speed**2, 1.0e-300))
    carrier_kinetic = float(
        grid.integrate(recovery.carrier.number_density * (carrier_W - 1.0))
    )
    carrier_contrast = float(
        (np.max(recovery.carrier.number_density) - np.min(recovery.carrier.number_density))
        / np.mean(recovery.carrier.number_density)
    )
    final_amplitude = diagnostics["maximum_scalar_amplitude"]
    fitted_growth = (
        float(np.polyfit(sample_times, np.log(sample_amplitudes), 1)[0])
        if active
        else 0.0
    )
    late_start = len(sample_times) // 2
    late_growth = (
        float(
            np.polyfit(
                sample_times[late_start:],
                np.log(sample_amplitudes[late_start:]),
                1,
            )[0]
        )
        if active and len(sample_times) - late_start >= 2
        else 0.0
    )
    peak_effective_mass = min(
        diagnostics["minimum_effective_mass_squared"], -1.0e-300
    )
    equilibrium_amplitude = float(
        np.sqrt(
            -peak_effective_mass
            / NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
        )
    )

    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "candidate.npz"
        save_nulloth_production_state(
            checkpoint,
            state,
            parameters=NULLOTH_DISCOVERY_PARAMETERS,
        )
        restarted, _ = load_nulloth_production_state(checkpoint)
        reference_next = solver.step(state, dt)
        restarted_next = solver.step(restarted, dt)
        restart_error = max(
            float(np.max(np.abs(first - second)))
            for first, second in zip(
                solver._pack(reference_next), solver._pack(restarted_next)
            )
        )

    admissible = bool(
        diagnostics["recovery_failures"] == 0
        and diagnostics["minimum_legendre_eigenvalue"] > 0.0
        and diagnostics["minimum_thermodynamic_eigenvalue"] > 0.0
        and final_amplitude
        < NULLOTH_DISCOVERY_PARAMETERS.scalar_amplitude_limit
    )
    run = FullCandidateRun(
        active,
        points,
        duration,
        steps,
        0.0 if not active else initial_amplitude,
        final_amplitude,
        float(max(sample_amplitudes)),
        float(
            (max(sample_amplitudes) - final_amplitude)
            / max(max(sample_amplitudes), 1.0e-300)
        ),
        bool(active and final_amplitude < 0.95 * max(sample_amplitudes)),
        1.0 if not active else final_amplitude / initial_amplitude,
        fitted_growth,
        late_growth,
        equilibrium_amplitude,
        float(np.max(fluid_speed)),
        float(np.max(carrier_speed)),
        carrier_kinetic,
        carrier_contrast,
        abs(diagnostics["baryon_mass"] - initial_baryons)
        / max(abs(initial_baryons), 1.0e-300),
        abs(diagnostics["carrier_number"] - initial_carriers)
        / max(abs(initial_carriers), 1.0e-300),
        diagnostics["entropy_integral"] - initial_entropy,
        diagnostics["hamiltonian_l2"],
        diagnostics["momentum_l2"],
        max(diagnostics["gauss_A_l2"], diagnostics["gauss_B_l2"]),
        diagnostics["scalar_constraint_l2"],
        diagnostics["recovery_maximum_residual"],
        diagnostics["minimum_legendre_eigenvalue"],
        diagnostics["minimum_thermodynamic_eigenvalue"],
        restart_error,
        admissible,
    )
    return run, recovery


def promote_candidate(
    *, points: int = 4, duration: float = 2.0, dt: float = 1.0e-2
) -> SpellPromotionResult:
    """Run a localized active candidate and its scalar-free control."""
    active, active_recovery = _run(
        points=points, duration=duration, dt=dt, active=True
    )
    control, control_recovery = _run(
        points=points, duration=duration, dt=dt, active=False
    )
    density_effect = float(
        np.sqrt(
            np.mean(
                (
                    active_recovery.carrier.number_density
                    - control_recovery.carrier.number_density
                )
                ** 2
            )
        )
    )
    velocity_effect = float(
        np.sqrt(
            np.mean(
                (
                    active_recovery.carrier.velocity
                    - control_recovery.carrier.velocity
                )
                ** 2
            )
        )
    )
    kinetic_effect = active.carrier_kinetic_proxy - control.carrier_kinetic_proxy
    relative_kinetic_effect = kinetic_effect / max(
        abs(control.carrier_kinetic_proxy), 1.0e-300
    )
    effect_kind = (
        "carrier_kinetic_suppression"
        if relative_kinetic_effect < 0.0
        else "carrier_kinetic_enhancement"
    )
    reasons: list[str] = []
    if not active.admissible:
        reasons.append("active evolution left the admissible domain")
    if active.scalar_amplification < 2.0:
        reasons.append("coherence did not amplify by at least two")
    if active.peak_scalar_amplitude < 0.9 * active.equilibrium_amplitude:
        reasons.append("coherence did not reach the nonlinear condensed branch")
    if not active.nonlinear_turnover:
        reasons.append("coherence did not demonstrate nonlinear turnover")
    if max(active.baryon_relative_drift, active.carrier_relative_drift) > 1.0e-10:
        reasons.append("number conservation exceeded tolerance")
    if active.gauss_max_l2 > 1.0e-10 or active.scalar_constraint_l2 > 1.0e-12:
        reasons.append("field constraints exceeded tolerance")
    if max(active.hamiltonian_l2, active.momentum_l2) > 1.0e-3:
        reasons.append("gravitational constraints exceeded tolerance")
    if active.restart_maximum_error > 1.0e-11:
        reasons.append("checkpoint restart was not reproducible")
    if density_effect < 1.0e-6 and velocity_effect < 1.0e-6:
        reasons.append("no resolved carrier effect above the scalar-free control")
    if abs(relative_kinetic_effect) < 1.0e-2:
        reasons.append("carrier kinetic effect was below one percent of control")
    return SpellPromotionResult(
        "localized_carrier_coherence_mode_1",
        active,
        control,
        density_effect,
        velocity_effect,
        kinetic_effect,
        relative_kinetic_effect,
        effect_kind,
        not reasons,
        tuple(reasons),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    args = parser.parse_args()
    result = promote_candidate(
        points=args.points, duration=args.duration, dt=args.dt
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
