"""Close the complete-step Nulloth entropy budget under active counterflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import CTTParameters, PeriodicCMCInitialDataSolver
from causal_repair.matter import FluidPrimitive
from causal_repair.theory33 import CarrierPrimitive

try:
    from experiments.map_nulloth_production_regimes import _solver
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _solver


@dataclass(frozen=True)
class EntropyCase:
    points: int
    dt: float
    net_entropy_change: float
    drag_entropy_change: float
    numerical_entropy_change: float
    numerical_entropy_fraction: float
    raw_projection_stage_sum: float
    raw_projection_to_numerical_ratio: float
    minimum_step_entropy_change: float
    entropy_telescope_error: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class EntropyAtlas:
    duration: float
    cases: tuple[EntropyCase, ...]
    time_net_relative_spread: float
    time_drag_relative_spread: float
    time_numerical_relative_spread: float
    numerical_entropy_monotone_with_resolution: int
    coarse_to_medium_numerical_reduction: float
    medium_to_fine_numerical_reduction: float
    coarse_numerical_entropy_fraction: float
    fine_numerical_entropy_fraction: float
    minimum_net_entropy_change: float
    minimum_drag_entropy_change: float
    minimum_numerical_entropy_change: float
    minimum_raw_projection_to_numerical_ratio: float
    maximum_raw_projection_to_numerical_ratio: float
    maximum_entropy_telescope_error: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _initial_state(points: int):
    grid = PeriodicGrid((points,), (32.0,), method="spectral")
    solver = _solver(grid)
    (coordinate,) = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier_velocity = grid.zeros((3,))
    carrier_velocity[0] = 0.15 * np.sin(wavenumber * coordinate)
    carrier = CarrierPrimitive(
        0.18 * (1.0 + 0.1 * np.cos(wavenumber * coordinate)),
        carrier_velocity,
    )
    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
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
        raise FloatingPointError("counterflow entropy CMC data did not converge")
    return solver, solver.initialize(
        solver.ccz4.from_adm(constraints.geometry),
        fluid,
        carrier=carrier,
    )


def _run_case(points: int, duration: float, dt: float) -> EntropyCase:
    solver, state = _initial_state(points)
    initial = solver.diagnostics(state)
    net = 0.0
    drag = 0.0
    raw_projection = 0.0
    minimum_step = np.inf
    maximum_gravity = 0.0
    maximum_field = 0.0
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermodynamic = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    for _ in range(int(round(duration / dt))):
        state = solver.step(state, dt)
        net += solver.last_step_entropy_change
        drag += solver.last_drag_entropy_change
        raw_projection += solver.last_projection_entropy_change
        minimum_step = min(minimum_step, solver.last_step_entropy_change)
        diagnostic = solver.diagnostics(state)
        maximum_gravity = max(
            maximum_gravity,
            float(diagnostic["hamiltonian_l2"]),
            float(diagnostic["momentum_l2"]),
        )
        maximum_field = max(
            maximum_field,
            float(diagnostic["gauss_A_l2"]),
            float(diagnostic["gauss_B_l2"]),
            float(diagnostic["scalar_constraint_l2"]),
        )
        minimum_legendre = min(
            minimum_legendre,
            float(diagnostic["minimum_legendre_eigenvalue"]),
        )
        minimum_thermodynamic = min(
            minimum_thermodynamic,
            float(diagnostic["minimum_thermodynamic_eigenvalue"]),
        )
        maximum_recovery = max(
            maximum_recovery,
            float(diagnostic["recovery_maximum_residual"]),
        )
    final = solver.diagnostics(state)
    numerical = net - drag
    actual_net = float(final["entropy_integral"] - initial["entropy_integral"])
    number_drift = max(
        abs(float(final[key] - initial[key]))
        / max(abs(float(initial[key])), 1.0e-300)
        for key in ("baryon_mass", "carrier_number")
    )
    return EntropyCase(
        points,
        dt,
        net,
        drag,
        numerical,
        numerical / net,
        raw_projection,
        raw_projection / numerical,
        minimum_step,
        abs(net - actual_net),
        maximum_gravity,
        maximum_field,
        number_drift,
        minimum_legendre,
        minimum_thermodynamic,
        maximum_recovery,
    )


def _relative_spread(values: list[float]) -> float:
    return (max(values) - min(values)) / max(abs(values[-1]), 1.0e-300)


def run_atlas(*, duration: float = 0.1) -> EntropyAtlas:
    configurations = ((16, 0.02), (16, 0.01), (16, 0.005), (8, 0.01), (32, 0.01))
    cases_by_key = {
        configuration: _run_case(*configuration[:1], duration, configuration[1])
        for configuration in configurations
    }
    cases = tuple(cases_by_key[configuration] for configuration in configurations)
    time_cases = [cases_by_key[(16, dt)] for dt in (0.02, 0.01, 0.005)]
    spatial_cases = [cases_by_key[(points, 0.01)] for points in (8, 16, 32)]
    coarse, medium, fine = spatial_cases
    return EntropyAtlas(
        duration,
        cases,
        _relative_spread([case.net_entropy_change for case in time_cases]),
        _relative_spread([case.drag_entropy_change for case in time_cases]),
        _relative_spread([case.numerical_entropy_change for case in time_cases]),
        int(
            coarse.numerical_entropy_change
            > medium.numerical_entropy_change
            > fine.numerical_entropy_change
        ),
        coarse.numerical_entropy_change / medium.numerical_entropy_change,
        medium.numerical_entropy_change / fine.numerical_entropy_change,
        coarse.numerical_entropy_fraction,
        fine.numerical_entropy_fraction,
        min(case.net_entropy_change for case in cases),
        min(case.drag_entropy_change for case in cases),
        min(case.numerical_entropy_change for case in cases),
        min(case.raw_projection_to_numerical_ratio for case in cases),
        max(case.raw_projection_to_numerical_ratio for case in cases),
        max(case.entropy_telescope_error for case in cases),
        max(case.maximum_gravitational_constraint_l2 for case in cases),
        max(case.maximum_field_constraint_l2 for case in cases),
        max(case.maximum_number_relative_drift for case in cases),
        min(case.minimum_legendre for case in cases),
        min(case.minimum_thermodynamic for case in cases),
        max(case.maximum_recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(duration=arguments.duration)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
