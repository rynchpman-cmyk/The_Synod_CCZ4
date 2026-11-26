"""Map long-time low-resolution Nulloth production regimes and budgets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import CTTParameters, PeriodicCMCInitialDataSolver
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class ProductionRegime:
    name: str
    carrier_density: float
    initial_amplitude: float
    completed: int
    outcome: str
    final_time: float
    peak_amplitude: float
    final_amplitude: float
    peak_to_final_drop_fraction: float
    maximum_high_wavenumber_fraction: float
    combined_energy_relative_change: float
    baryon_relative_drift: float
    carrier_relative_drift: float
    entropy_change: float
    cumulative_step_entropy_change: float
    entropy_telescope_error: float
    cumulative_drag_entropy: float
    cumulative_drag_heat: float
    cumulative_damping_heat: float
    cumulative_projection_entropy: float
    minimum_step_entropy_change: float
    maximum_hamiltonian_l2: float
    maximum_momentum_l2: float
    maximum_gauss_l2: float
    maximum_scalar_constraint_l2: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float
    failure: str | None


@dataclass(frozen=True)
class ProductionRegimeAtlas:
    points: int
    duration: float
    dt: float
    regimes: tuple[ProductionRegime, ...]
    attempted_runs: int
    completed_runs: int
    denied_runs: int
    distinct_outcomes: int
    scalar_free_runs: int
    nonlinear_turnover_runs: int
    sustained_coherence_runs: int
    oscillatory_runs: int
    maximum_peak_amplitude: float
    maximum_high_wavenumber_fraction: float
    maximum_number_relative_drift: float
    maximum_combined_energy_relative_change: float
    minimum_entropy_change: float
    minimum_step_entropy_change: float
    maximum_entropy_telescope_error: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float
    maximum_projection_stage_sum_to_net_entropy_ratio: float
    metastable_lifetime_lower_bound: float


_CONFIGURATIONS = (
    ("activated-zero", 0.18, 0.0, "zero"),
    ("activated-small", 0.18, 0.005, "growing"),
    ("activated-standard", 0.18, 0.02, "growing"),
    ("near-condensed", 0.18, 0.06, "rest"),
)


def _solver(grid: PeriodicGrid) -> NullothProductionSolver:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
    )
    geometry = CCZ4Parameters(
        kappa_gravity=1.0e-3,
        slicing="harmonic",
        evolve_shift=False,
        radiative_boundaries=False,
    )
    return NullothProductionSolver(grid, parameters, vector, geometry)


def _initial_state(
    solver: NullothProductionSolver,
    carrier_density: float,
    amplitude: float,
    momentum_kind: str,
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
        carrier_density * (1.0 + 0.2 * profile), grid.zeros((3,))
    )
    varphi = amplitude * profile
    effective_mass = 0.4**2 - 12.0 * carrier_density**2
    growth_squared = -effective_mass - wavenumber**2
    scalar_momentum = grid.zeros()
    if momentum_kind == "growing" and growth_squared > 0.0:
        scalar_momentum = -np.sqrt(growth_squared) * varphi
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
        raise FloatingPointError("periodic CMC production-atlas data did not converge")
    return solver.initialize(
        solver.ccz4.from_adm(constraints.geometry),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )


def _high_wavenumber_fraction(field: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fft(field) / len(field)) ** 2
    indices = np.abs(np.fft.fftfreq(len(field)) * len(field))
    return float(np.sum(spectrum[indices > len(field) / 3.0]) / max(np.sum(spectrum), 1.0e-300))


def _classify(initial: float, peak: float, final: float, completed: bool) -> str:
    if not completed:
        return "denied"
    if initial == 0.0 and peak < 1.0e-14:
        return "scalar-free"
    drop = (peak - final) / max(peak, 1.0e-300)
    if peak >= 1.2 * max(initial, 1.0e-14) and drop >= 0.05:
        return "nonlinear-turnover"
    if peak >= 0.06 and final >= 0.5 * peak:
        return "sustained-coherence"
    if final < 0.5 * max(initial, 1.0e-14) and peak < 1.2 * max(initial, 1.0e-14):
        return "dispersed"
    return "bounded-oscillatory"


def _run_configuration(
    points: int,
    duration: float,
    dt: float,
    configuration: tuple[str, float, float, str],
) -> ProductionRegime:
    name, carrier_density, amplitude, momentum_kind = configuration
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid)
    state = _initial_state(solver, carrier_density, amplitude, momentum_kind)
    initial = solver.diagnostics(state)
    peak = float(np.max(np.abs(state.scalar.varphi)))
    maximum_high = _high_wavenumber_fraction(state.scalar.varphi)
    maximum_hamiltonian = float(initial["hamiltonian_l2"])
    maximum_momentum = float(initial["momentum_l2"])
    maximum_gauss = max(float(initial["gauss_A_l2"]), float(initial["gauss_B_l2"]))
    maximum_scalar_constraint = float(initial["scalar_constraint_l2"])
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermo = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    cumulative_step_entropy = 0.0
    cumulative_drag_entropy = 0.0
    cumulative_drag_heat = 0.0
    cumulative_damping_heat = 0.0
    cumulative_projection_entropy = 0.0
    minimum_step_entropy = np.inf
    completed = True
    failure = None
    steps = int(round(duration / dt))
    try:
        for _ in range(steps):
            state = solver.step(state, dt)
            amplitude_now = float(np.max(np.abs(state.scalar.varphi)))
            peak = max(peak, amplitude_now)
            maximum_high = max(
                maximum_high, _high_wavenumber_fraction(state.scalar.varphi)
            )
            cumulative_step_entropy += solver.last_step_entropy_change
            cumulative_drag_entropy += solver.last_drag_entropy_change
            cumulative_drag_heat += solver.last_drag_heat
            cumulative_damping_heat += solver.last_damping_report.irreversible_heat
            cumulative_projection_entropy += solver.last_projection_entropy_change
            minimum_step_entropy = min(minimum_step_entropy, solver.last_step_entropy_change)
            if int(round(state.time / dt)) % 25 == 0:
                diagnostic = solver.diagnostics(state)
                maximum_hamiltonian = max(maximum_hamiltonian, float(diagnostic["hamiltonian_l2"]))
                maximum_momentum = max(maximum_momentum, float(diagnostic["momentum_l2"]))
                maximum_gauss = max(
                    maximum_gauss,
                    float(diagnostic["gauss_A_l2"]),
                    float(diagnostic["gauss_B_l2"]),
                )
                maximum_scalar_constraint = max(
                    maximum_scalar_constraint, float(diagnostic["scalar_constraint_l2"])
                )
                minimum_legendre = min(
                    minimum_legendre, float(diagnostic["minimum_legendre_eigenvalue"])
                )
                minimum_thermo = min(
                    minimum_thermo, float(diagnostic["minimum_thermodynamic_eigenvalue"])
                )
                maximum_recovery = max(
                    maximum_recovery, float(diagnostic["recovery_maximum_residual"])
                )
    except (FloatingPointError, ValueError) as error:
        completed = False
        failure = f"{type(error).__name__}: {error}"
    final = solver.diagnostics(state)
    final_amplitude = float(np.max(np.abs(state.scalar.varphi)))
    entropy_change = float(final["entropy_integral"] - initial["entropy_integral"])
    energy_change = abs(float(final["combined_energy"] - initial["combined_energy"])) / max(
        abs(float(initial["combined_energy"])), 1.0e-300
    )
    baryon_drift = abs(float(final["baryon_mass"] - initial["baryon_mass"])) / max(
        abs(float(initial["baryon_mass"])), 1.0e-300
    )
    carrier_drift = abs(float(final["carrier_number"] - initial["carrier_number"])) / max(
        abs(float(initial["carrier_number"])), 1.0e-300
    )
    return ProductionRegime(
        name,
        carrier_density,
        amplitude,
        int(completed),
        _classify(amplitude, peak, final_amplitude, completed),
        float(state.time),
        peak,
        final_amplitude,
        (peak - final_amplitude) / max(peak, 1.0e-300),
        maximum_high,
        energy_change,
        baryon_drift,
        carrier_drift,
        entropy_change,
        cumulative_step_entropy,
        abs(entropy_change - cumulative_step_entropy),
        cumulative_drag_entropy,
        cumulative_drag_heat,
        cumulative_damping_heat,
        cumulative_projection_entropy,
        0.0 if not np.isfinite(minimum_step_entropy) else minimum_step_entropy,
        maximum_hamiltonian,
        maximum_momentum,
        maximum_gauss,
        maximum_scalar_constraint,
        minimum_legendre,
        minimum_thermo,
        maximum_recovery,
        failure,
    )


def run_atlas(
    *, points: int = 4, duration: float = 6.0, dt: float = 0.01
) -> ProductionRegimeAtlas:
    regimes = tuple(
        _run_configuration(points, duration, dt, configuration)
        for configuration in _CONFIGURATIONS
    )
    completed = tuple(regime for regime in regimes if regime.completed)
    coherent_lifetimes = [
        regime.final_time
        for regime in completed
        if regime.outcome in {"nonlinear-turnover", "sustained-coherence"}
    ]
    return ProductionRegimeAtlas(
        points,
        duration,
        dt,
        regimes,
        len(regimes),
        len(completed),
        len(regimes) - len(completed),
        len({regime.outcome for regime in regimes}),
        sum(regime.outcome == "scalar-free" for regime in regimes),
        sum(regime.outcome == "nonlinear-turnover" for regime in regimes),
        sum(regime.outcome == "sustained-coherence" for regime in regimes),
        sum(regime.outcome == "bounded-oscillatory" for regime in regimes),
        max(regime.peak_amplitude for regime in completed) if completed else np.inf,
        max(regime.maximum_high_wavenumber_fraction for regime in completed)
        if completed
        else np.inf,
        max(regime.baryon_relative_drift for regime in completed)
        if completed
        else np.inf,
        max(regime.combined_energy_relative_change for regime in completed)
        if completed
        else np.inf,
        min(regime.entropy_change for regime in completed) if completed else -np.inf,
        min(regime.minimum_step_entropy_change for regime in completed)
        if completed
        else -np.inf,
        max(regime.entropy_telescope_error for regime in completed)
        if completed
        else np.inf,
        max(max(regime.maximum_hamiltonian_l2, regime.maximum_momentum_l2) for regime in completed)
        if completed
        else np.inf,
        max(max(regime.maximum_gauss_l2, regime.maximum_scalar_constraint_l2) for regime in completed)
        if completed
        else np.inf,
        min(regime.minimum_legendre for regime in completed) if completed else -np.inf,
        min(regime.minimum_thermodynamic for regime in completed) if completed else -np.inf,
        max(regime.maximum_recovery_residual for regime in completed)
        if completed
        else np.inf,
        max(
            abs(regime.cumulative_projection_entropy)
            / max(abs(regime.entropy_change), 1.0e-300)
            for regime in completed
        )
        if completed
        else np.inf,
        min(coherent_lifetimes) if coherent_lifetimes else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(points=arguments.points, duration=arguments.duration, dt=arguments.dt)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
