"""Measure independent 3D vector and material pulse responses in Nulloth."""

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
    from experiments.measure_nulloth_periodic_tensor_response import _tt_l2
except ModuleNotFoundError:  # Direct execution places experiments/ on sys.path.
    from map_nulloth_production_regimes import _solver
    from measure_nulloth_periodic_tensor_response import _tt_l2


@dataclass(frozen=True)
class SourcePulseCase:
    source: str
    initial_pulse_amplitude: float
    final_pulse_amplitude: float
    pulse_retention: float
    initial_physical_metric_contrast_l2: float
    final_physical_metric_contrast_l2: float
    dynamical_physical_metric_contrast_l2: float
    initial_extrinsic_curvature_tt_l2: float
    final_extrinsic_curvature_tt_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


@dataclass(frozen=True)
class SourcePulseAtlas:
    points_per_axis: int
    duration: float
    dt: float
    cases: tuple[SourcePulseCase, ...]
    vector_pulse_retention: float
    material_pulse_retention: float
    vector_final_metric_contrast_l2: float
    material_final_metric_contrast_l2: float
    minimum_dynamical_metric_contrast_l2: float
    minimum_final_extrinsic_curvature_tt_l2: float
    maximum_initial_extrinsic_curvature_tt_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_field_constraint_l2: float
    maximum_number_relative_drift: float
    minimum_entropy_change: float
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_recovery_residual: float


def _l2(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(field) ** 2)))


def _mode_amplitude(field: np.ndarray, coordinate: np.ndarray, wavenumber: float) -> float:
    coefficient = np.mean(field * np.exp(-1j * wavenumber * coordinate))
    return float(2.0 * abs(coefficient))


def _initial_state(grid: PeriodicGrid, source: str):
    solver = _solver(grid)
    x, _, z = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    density = np.full(grid.shape, 0.6)
    pi_A = grid.zeros((3,))
    if source == "material":
        density = 0.6 * (1.0 + 0.02 * np.cos(wavenumber * x))
    elif source == "vector":
        pi_A[1] = 0.001 * np.cos(wavenumber * x)
    elif source != "control":
        raise ValueError(f"unknown source {source}")
    fluid = FluidPrimitive(
        density,
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        0.18 * (1.0 + 0.1 * np.cos(wavenumber * z)),
        grid.zeros((3,)),
    )
    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        pi_A=pi_A,
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
        raise FloatingPointError(f"3D periodic CMC {source} data did not converge")
    state = solver.initialize(
        solver.ccz4.from_adm(constraints.geometry),
        fluid,
        carrier=carrier,
        pi_A=pi_A,
    )
    return solver, state


def _evolve(solver, state, duration: float, dt: float):
    initial = solver.diagnostics(state)
    maximum_gravity = max(
        float(initial["hamiltonian_l2"]),
        float(initial["momentum_l2"]),
    )
    maximum_field = max(
        float(initial["gauss_A_l2"]),
        float(initial["gauss_B_l2"]),
        float(initial["scalar_constraint_l2"]),
    )
    minimum_legendre = float(initial["minimum_legendre_eigenvalue"])
    minimum_thermodynamic = float(initial["minimum_thermodynamic_eigenvalue"])
    maximum_recovery = float(initial["recovery_maximum_residual"])
    for _ in range(int(round(duration / dt))):
        state = solver.step(state, dt)
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
    number_drift = max(
        abs(float(final[key] - initial[key]))
        / max(abs(float(initial[key])), 1.0e-300)
        for key in ("baryon_mass", "carrier_number")
    )
    return (
        state,
        initial,
        final,
        maximum_gravity,
        maximum_field,
        number_drift,
        minimum_legendre,
        minimum_thermodynamic,
        maximum_recovery,
    )


def run_atlas(
    *, points: int = 4, duration: float = 0.01, dt: float = 0.005
) -> SourcePulseAtlas:
    grid = PeriodicGrid((points,) * 3, (32.0,) * 3, method="spectral")
    x, _, _ = grid.coordinates()
    wavenumber = 2.0 * np.pi / grid.lengths[0]
    control_solver, control_initial_state = _initial_state(grid, "control")
    control_h_initial, control_K_initial = control_solver.ccz4.physical_geometry(
        control_initial_state.geometry
    )
    control_result = _evolve(control_solver, control_initial_state, duration, dt)
    control_final_state = control_result[0]
    control_h_final, control_K_final = control_solver.ccz4.physical_geometry(
        control_final_state.geometry
    )
    cases = []
    for source in ("vector", "material"):
        solver, initial_state = _initial_state(grid, source)
        initial_recovery = solver.recover(initial_state)
        active_h_initial, active_K_initial = solver.ccz4.physical_geometry(
            initial_state.geometry
        )
        if source == "vector":
            initial_amplitude = _mode_amplitude(
                initial_state.pi_A[1], x, wavenumber
            )
        else:
            initial_amplitude = _mode_amplitude(
                initial_recovery.fluid.baryon_density, x, wavenumber
            )
        result = _evolve(solver, initial_state, duration, dt)
        final_state, initial, final = result[:3]
        final_recovery = solver.recover(final_state)
        active_h_final, active_K_final = solver.ccz4.physical_geometry(
            final_state.geometry
        )
        if source == "vector":
            final_amplitude = _mode_amplitude(final_state.pi_A[1], x, wavenumber)
        else:
            final_amplitude = _mode_amplitude(
                final_recovery.fluid.baryon_density, x, wavenumber
            )
        cases.append(
            SourcePulseCase(
                source,
                initial_amplitude,
                final_amplitude,
                final_amplitude / initial_amplitude,
                _l2(active_h_initial - control_h_initial),
                _l2(active_h_final - control_h_final),
                _l2(
                    (active_h_final - active_h_initial)
                    - (control_h_final - control_h_initial)
                ),
                _tt_l2(active_K_initial - control_K_initial, grid),
                _tt_l2(active_K_final - control_K_final, grid),
                max(result[3], control_result[3]),
                max(result[4], control_result[4]),
                max(result[5], control_result[5]),
                min(
                    float(final["entropy_integral"] - initial["entropy_integral"]),
                    float(
                        control_result[2]["entropy_integral"]
                        - control_result[1]["entropy_integral"]
                    ),
                ),
                min(result[6], control_result[6]),
                min(result[7], control_result[7]),
                max(result[8], control_result[8]),
            )
        )
    vector, material = cases
    return SourcePulseAtlas(
        points,
        duration,
        dt,
        tuple(cases),
        vector.pulse_retention,
        material.pulse_retention,
        vector.final_physical_metric_contrast_l2,
        material.final_physical_metric_contrast_l2,
        min(case.dynamical_physical_metric_contrast_l2 for case in cases),
        min(case.final_extrinsic_curvature_tt_l2 for case in cases),
        max(case.initial_extrinsic_curvature_tt_l2 for case in cases),
        max(case.maximum_gravitational_constraint_l2 for case in cases),
        max(case.maximum_field_constraint_l2 for case in cases),
        max(case.maximum_number_relative_drift for case in cases),
        min(case.entropy_change for case in cases),
        min(case.minimum_legendre for case in cases),
        min(case.minimum_thermodynamic for case in cases),
        max(case.maximum_recovery_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--duration", type=float, default=0.01)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(
        points=arguments.points,
        duration=arguments.duration,
        dt=arguments.dt,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
