"""Find the least resolved scalar seed that meets a brake objective."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from causal_repair.adm import inverse_metric
from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import CTTParameters, PeriodicCMCInitialDataSolver
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS, NullothParameters
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class BranchResult:
    branch: str
    seed_amplitude: float
    seed_coordinate_energy: float
    final_scalar_amplitude: float
    carrier_kinetic_proxy: float
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


@dataclass(frozen=True)
class SearchSample:
    amplitude: float
    relative_suppression: float
    passes_target: bool


def _solver(grid: PeriodicGrid, parameters: NullothParameters) -> NullothProductionSolver:
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
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
    amplitude: float,
    growing_rate: float,
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
    carrier = CarrierPrimitive(0.18 + 0.04 * profile, grid.zeros((3,)))
    varphi = amplitude * profile
    scalar_momentum = -growing_rate * varphi

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
        raise FloatingPointError("periodic CMC search data did not converge")
    return solver.initialize(
        solver.ccz4.from_adm(constraints.geometry),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=scalar_momentum,
    )


def _run_branch(
    *,
    branch: str,
    parameters: NullothParameters,
    points: int,
    duration: float,
    dt: float,
    amplitude: float,
    growing_rate: float,
) -> BranchResult:
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid, parameters)
    state = _initial_state(solver, amplitude, growing_rate)
    initial = solver.diagnostics(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    _, _, sqrt_h = inverse_metric(h)
    scalar_stress = solver.scalar_system.stress(
        h, state.scalar, solver.master.scalar_potential(state.scalar.varphi)
    )
    seed_energy = float(grid.integrate(sqrt_h * scalar_stress.rho))
    steps = int(round(duration / dt))
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must be a positive integer multiple of dt")
    for _ in range(steps):
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
    carrier_kinetic = float(
        grid.integrate(recovery.carrier.number_density * (carrier_W - 1.0))
    )
    admissible = bool(
        diagnostics["recovery_failures"] == 0
        and diagnostics["minimum_legendre_eigenvalue"] > 0.0
        and diagnostics["minimum_thermodynamic_eigenvalue"] > 0.0
        and diagnostics["maximum_scalar_amplitude"]
        < parameters.scalar_amplitude_limit
    )
    return BranchResult(
        branch,
        amplitude,
        seed_energy,
        diagnostics["maximum_scalar_amplitude"],
        carrier_kinetic,
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


def _relative_suppression(active: BranchResult, control: BranchResult) -> float:
    return (control.carrier_kinetic_proxy - active.carrier_kinetic_proxy) / max(
        abs(control.carrier_kinetic_proxy), 1.0e-300
    )


def search(
    *,
    points: int = 8,
    duration: float = 2.0,
    dt: float = 1.0e-2,
    target_suppression: float = 0.02,
    refinement_steps: int = 6,
) -> dict[str, object]:
    canonical = NULLOTH_DISCOVERY_PARAMETERS
    uncoupled = replace(
        canonical,
        scalar_baryon_coupling=0.0,
        scalar_carrier_coupling=0.0,
        scalar_counterflow_coupling=0.0,
    )
    wavenumber = 2.0 * np.pi / 32.0
    peak_effective_mass = (
        canonical.scalar_mass**2 - canonical.scalar_carrier_coupling * 0.22**2
    )
    growing_rate = float(np.sqrt(-peak_effective_mass - wavenumber**2))
    no_command = _run_branch(
        branch="no_command",
        parameters=canonical,
        points=points,
        duration=duration,
        dt=dt,
        amplitude=0.0,
        growing_rate=growing_rate,
    )

    cache: dict[float, BranchResult] = {}

    def evaluate(amplitude: float) -> SearchSample:
        active = _run_branch(
            branch="active",
            parameters=canonical,
            points=points,
            duration=duration,
            dt=dt,
            amplitude=amplitude,
            growing_rate=growing_rate,
        )
        cache[amplitude] = active
        suppression = _relative_suppression(active, no_command)
        return SearchSample(amplitude, suppression, suppression >= target_suppression)

    samples: list[SearchSample] = []
    lower = 0.0
    upper = 0.0
    for amplitude in (0.001, 0.002, 0.004, 0.008, 0.016, 0.032):
        sample = evaluate(amplitude)
        samples.append(sample)
        if sample.passes_target:
            upper = amplitude
            break
        lower = amplitude
    if upper == 0.0:
        raise RuntimeError("coarse search did not bracket the suppression target")

    for _ in range(refinement_steps):
        midpoint = 0.5 * (lower + upper)
        sample = evaluate(midpoint)
        samples.append(sample)
        if sample.passes_target:
            upper = midpoint
        else:
            lower = midpoint

    selected = cache[upper]
    scalar_coupling_free = _run_branch(
        branch="scalar_coupling_free",
        parameters=uncoupled,
        points=points,
        duration=duration,
        dt=dt,
        amplitude=upper,
        growing_rate=growing_rate,
    )
    selected_suppression = _relative_suppression(selected, no_command)
    uncoupled_suppression = _relative_suppression(
        scalar_coupling_free, no_command
    )
    branches = (no_command, selected, scalar_coupling_free)
    return {
        "search_method": "coarse_bracket_plus_bisection",
        "points": points,
        "duration": duration,
        "time_step": dt,
        "target_relative_suppression": target_suppression,
        "search_evaluations": len(samples),
        "bracket_lower_amplitude": lower,
        "bracket_upper_amplitude": upper,
        "bracket_width": upper - lower,
        "lower_bracket_passes": int(
            _relative_suppression(cache[lower], no_command) >= target_suppression
        ),
        "upper_bracket_passes": int(selected_suppression >= target_suppression),
        "selected_seed_amplitude": upper,
        "selected_seed_coordinate_energy": selected.seed_coordinate_energy,
        "selected_relative_suppression": selected_suppression,
        "uncoupled_relative_suppression": uncoupled_suppression,
        "coupling_attributed_suppression": (
            selected_suppression - uncoupled_suppression
        ),
        "active_beats_scalar_coupling_free": int(
            selected.carrier_kinetic_proxy
            < scalar_coupling_free.carrier_kinetic_proxy
        ),
        "no_command_scalar_amplitude": no_command.final_scalar_amplitude,
        "all_branches_admissible": int(all(branch.admissible for branch in branches)),
        "maximum_number_relative_drift": max(
            max(branch.baryon_relative_drift, branch.carrier_relative_drift)
            for branch in branches
        ),
        "minimum_entropy_change": min(branch.entropy_change for branch in branches),
        "maximum_gauss_l2": max(branch.gauss_max_l2 for branch in branches),
        "maximum_scalar_constraint_l2": max(
            branch.scalar_constraint_l2 for branch in branches
        ),
        "maximum_gravitational_constraint_l2": max(
            max(branch.hamiltonian_l2, branch.momentum_l2) for branch in branches
        ),
        "maximum_recovery_residual": max(
            branch.recovery_residual for branch in branches
        ),
        "minimum_legendre_eigenvalue": min(
            branch.minimum_legendre_eigenvalue for branch in branches
        ),
        "minimum_thermodynamic_eigenvalue": min(
            branch.minimum_thermodynamic_eigenvalue for branch in branches
        ),
        "samples": [asdict(sample) for sample in samples],
        "branches": [asdict(branch) for branch in branches],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0e-2)
    parser.add_argument("--target-suppression", type=float, default=0.02)
    parser.add_argument("--refinement-steps", type=int, default=6)
    arguments = parser.parse_args()
    print(
        json.dumps(
            search(
                points=arguments.points,
                duration=arguments.duration,
                dt=arguments.dt,
                target_suppression=arguments.target_suppression,
                refinement_steps=arguments.refinement_steps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
