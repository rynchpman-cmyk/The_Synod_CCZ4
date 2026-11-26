"""Converge Nulloth's finite scalar passage in resolution and patch size."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import CartesianGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.theory33 import CarrierPrimitive

try:
    from experiments.audit_nulloth_finite_scalar_outflow import _scalar_energy, _solver
except ModuleNotFoundError:
    from audit_nulloth_finite_scalar_outflow import _scalar_energy, _solver


@dataclass(frozen=True)
class FiniteConvergenceCase:
    name: str
    points: int
    length: float
    origin: float
    spacing: float
    scalar_energy_loss_fraction: float
    final_peak_amplitude: float
    incoming_to_outgoing_characteristic_ratio: float
    maximum_opposite_edge_amplitude: float
    final_scalar_constraint_l2: float
    final_gravitational_constraint_l2: float
    final_proca_constraint_l2: float
    recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_step_entropy_balance_residual: float
    cumulative_entropy_balance_residual: float
    maximum_carrier_number_balance_residual: float


@dataclass(frozen=True)
class FiniteConvergenceAtlas:
    duration: float
    dt: float
    cases: tuple[FiniteConvergenceCase, ...]
    finest_scalar_energy_loss_fraction: float
    energy_loss_convergence_ratio: float
    medium_to_fine_energy_loss_difference: float
    patch_size_energy_loss_relative_difference: float
    maximum_incoming_to_outgoing_characteristic_ratio: float
    maximum_opposite_edge_amplitude: float
    maximum_scalar_constraint_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_proca_constraint_l2: float
    maximum_recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_step_entropy_balance_residual: float
    maximum_absolute_cumulative_entropy_balance_residual: float
    maximum_carrier_number_balance_residual: float


def _initial_state(solver):
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.1),
        np.full(grid.shape, 0.5),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        np.full(grid.shape, 0.03),
        grid.zeros((3,)),
    )
    varphi = 1.0e-5 * np.exp(-0.5 * ((coordinate - 0.2) / 0.1) ** 2)
    gradient = grid.gradient(varphi)
    return solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=gradient[0],
        scalar_gradient=gradient,
    )


def _run_case(
    name: str,
    points: int,
    length: float,
    origin: float,
    duration: float,
    dt: float,
) -> FiniteConvergenceCase:
    grid = CartesianGrid(
        (points,), (length,), origin=(origin,), method="fd2"
    )
    solver = _solver(grid)
    state = _initial_state(solver)
    initial_energy = _scalar_energy(solver, state)
    maximum_incoming = 0.0
    maximum_outgoing = 0.0
    maximum_opposite = abs(float(state.scalar.varphi[0]))
    maximum_entropy_residual = 0.0
    cumulative_entropy_residual = 0.0
    maximum_carrier_residual = 0.0
    for _ in range(int(round(duration / dt))):
        state = solver.step(state, dt)
        maximum_incoming = max(
            maximum_incoming,
            abs(
                float(
                    state.scalar.momentum[-1]
                    - state.scalar.gradient[0, -1]
                )
            ),
        )
        maximum_outgoing = max(
            maximum_outgoing,
            abs(
                float(
                    state.scalar.momentum[-1]
                    + state.scalar.gradient[0, -1]
                )
            ),
        )
        maximum_opposite = max(
            maximum_opposite, abs(float(state.scalar.varphi[0]))
        )
        maximum_entropy_residual = max(
            maximum_entropy_residual,
            abs(solver.last_entropy_balance_residual),
        )
        cumulative_entropy_residual += solver.last_entropy_balance_residual
        maximum_carrier_residual = max(
            maximum_carrier_residual,
            abs(solver.last_carrier_number_balance_residual),
        )
    diagnostics = solver.diagnostics(state)
    final_energy = _scalar_energy(solver, state)
    return FiniteConvergenceCase(
        name,
        points,
        length,
        origin,
        grid.spacing[0],
        1.0 - final_energy / max(initial_energy, 1.0e-300),
        float(np.max(np.abs(state.scalar.varphi))),
        maximum_incoming / max(maximum_outgoing, 1.0e-300),
        maximum_opposite,
        float(diagnostics["scalar_constraint_l2"]),
        max(
            float(diagnostics["hamiltonian_l2"]),
            float(diagnostics["momentum_l2"]),
        ),
        max(
            float(diagnostics["gauss_A_l2"]),
            float(diagnostics["gauss_B_l2"]),
        ),
        float(diagnostics["recovery_maximum_residual"]),
        float(diagnostics["minimum_legendre_eigenvalue"]),
        float(diagnostics["minimum_thermodynamic_eigenvalue"]),
        maximum_entropy_residual,
        cumulative_entropy_residual,
        maximum_carrier_residual,
    )


def run_atlas(
    *, duration: float = 0.32, dt: float = 0.002
) -> FiniteConvergenceAtlas:
    steps = int(round(duration / dt))
    if steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must be a positive integer multiple of dt")
    cases = (
        _run_case("coarse", 12, 1.0, -0.5, duration, dt),
        _run_case("medium", 18, 1.0, -0.5, duration, dt),
        _run_case("fine", 24, 1.0, -0.5, duration, dt),
        _run_case("expanded-equal-spacing", 36, 1.5, -1.0, duration, dt),
    )
    coarse, medium, fine, expanded = cases
    coarse_error = abs(
        coarse.scalar_energy_loss_fraction - fine.scalar_energy_loss_fraction
    )
    medium_error = abs(
        medium.scalar_energy_loss_fraction - fine.scalar_energy_loss_fraction
    )
    patch_difference = abs(
        fine.scalar_energy_loss_fraction - expanded.scalar_energy_loss_fraction
    ) / max(abs(fine.scalar_energy_loss_fraction), 1.0e-300)
    return FiniteConvergenceAtlas(
        duration,
        dt,
        cases,
        fine.scalar_energy_loss_fraction,
        coarse_error / max(medium_error, 1.0e-300),
        medium_error,
        patch_difference,
        max(case.incoming_to_outgoing_characteristic_ratio for case in cases),
        max(case.maximum_opposite_edge_amplitude for case in cases),
        max(case.final_scalar_constraint_l2 for case in cases),
        max(case.final_gravitational_constraint_l2 for case in cases),
        max(case.final_proca_constraint_l2 for case in cases),
        max(case.recovery_residual for case in cases),
        min(case.minimum_legendre_eigenvalue for case in cases),
        min(case.minimum_thermodynamic_eigenvalue for case in cases),
        max(case.maximum_step_entropy_balance_residual for case in cases),
        max(abs(case.cumulative_entropy_balance_residual) for case in cases),
        max(case.maximum_carrier_number_balance_residual for case in cases),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.32)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(duration=arguments.duration, dt=arguments.dt)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
