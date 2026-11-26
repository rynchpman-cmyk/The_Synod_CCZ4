"""Audit a compact Nulloth scalar pulse on a finite Cartesian patch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.finite_nulloth import FiniteNullothProductionSolver
from causal_repair.grid import CartesianGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class FiniteScalarOutflow:
    points: int
    duration: float
    dt: float
    steps: int
    initial_scalar_energy: float
    final_scalar_energy: float
    retained_scalar_energy_fraction: float
    scalar_energy_loss_fraction: float
    initial_peak_amplitude: float
    final_peak_amplitude: float
    maximum_right_incoming_characteristic: float
    maximum_right_outgoing_characteristic: float
    incoming_to_outgoing_characteristic_ratio: float
    maximum_opposite_edge_amplitude: float
    maximum_scalar_constraint_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_proca_constraint_l2: float
    maximum_recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_step_entropy_balance_residual: float
    cumulative_entropy_balance_residual: float
    maximum_carrier_number_balance_residual: float


def _solver(grid: CartesianGrid) -> FiniteNullothProductionSolver:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        gamma_ad=parameters.gamma_ad,
    )
    geometry = CCZ4Parameters(
        kappa_gravity=1.0e-12,
        slicing="harmonic",
        evolve_shift=False,
        radiative_boundaries=True,
    )
    return FiniteNullothProductionSolver(
        grid,
        parameters,
        vector,
        geometry,
        scalar_boundary_damping=1.0,
    )


def _initial_state(solver: FiniteNullothProductionSolver):
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
    varphi = 1.0e-5 * np.exp(-0.5 * ((coordinate - 0.4) / 0.035) ** 2)
    gradient = grid.gradient(varphi)
    # Pi=Phi_x is the right-moving flat-space characteristic convention.
    return solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        varphi=varphi,
        scalar_momentum=gradient[0],
        scalar_gradient=gradient,
    )


def _scalar_energy(solver, state) -> float:
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    return solver.grid.integrate(solver._scalar_stress(h, state).rho)


def run_finite_scalar_outflow(
    *, points: int = 24, duration: float = 0.12, dt: float = 0.001
) -> FiniteScalarOutflow:
    steps = int(round(duration / dt))
    if points < 6 or steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("invalid finite-pulse grid or time interval")
    grid = CartesianGrid(
        (points,), (1.0,), origin=(-0.5,), method="fd2"
    )
    solver = _solver(grid)
    state = _initial_state(solver)
    initial_energy = _scalar_energy(solver, state)
    initial_peak = float(np.max(np.abs(state.scalar.varphi)))
    maximum_incoming = 0.0
    maximum_outgoing = 0.0
    maximum_left = abs(float(state.scalar.varphi[0]))
    maximum_scalar_constraint = 0.0
    maximum_gravity = 0.0
    maximum_proca = 0.0
    maximum_recovery = 0.0
    minimum_legendre = float("inf")
    minimum_thermodynamic = float("inf")
    maximum_entropy_residual = 0.0
    cumulative_entropy_residual = 0.0
    maximum_carrier_residual = 0.0

    for _ in range(steps):
        state = solver.step(state, dt)
        incoming = abs(
            float(state.scalar.momentum[-1] - state.scalar.gradient[0, -1])
        )
        outgoing = abs(
            float(state.scalar.momentum[-1] + state.scalar.gradient[0, -1])
        )
        maximum_incoming = max(maximum_incoming, incoming)
        maximum_outgoing = max(maximum_outgoing, outgoing)
        maximum_left = max(maximum_left, abs(float(state.scalar.varphi[0])))
        diagnostics = solver.diagnostics(state)
        maximum_scalar_constraint = max(
            maximum_scalar_constraint,
            float(diagnostics["scalar_constraint_l2"]),
        )
        maximum_gravity = max(
            maximum_gravity,
            float(diagnostics["hamiltonian_l2"]),
            float(diagnostics["momentum_l2"]),
        )
        maximum_proca = max(
            maximum_proca,
            float(diagnostics["gauss_A_l2"]),
            float(diagnostics["gauss_B_l2"]),
        )
        maximum_recovery = max(
            maximum_recovery,
            float(diagnostics["recovery_maximum_residual"]),
        )
        minimum_legendre = min(
            minimum_legendre,
            float(diagnostics["minimum_legendre_eigenvalue"]),
        )
        minimum_thermodynamic = min(
            minimum_thermodynamic,
            float(diagnostics["minimum_thermodynamic_eigenvalue"]),
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

    final_energy = _scalar_energy(solver, state)
    retained = final_energy / max(initial_energy, 1.0e-300)
    return FiniteScalarOutflow(
        points,
        duration,
        dt,
        steps,
        initial_energy,
        final_energy,
        retained,
        1.0 - retained,
        initial_peak,
        float(np.max(np.abs(state.scalar.varphi))),
        maximum_incoming,
        maximum_outgoing,
        maximum_incoming / max(maximum_outgoing, 1.0e-300),
        maximum_left,
        maximum_scalar_constraint,
        maximum_gravity,
        maximum_proca,
        maximum_recovery,
        minimum_legendre,
        minimum_thermodynamic,
        maximum_entropy_residual,
        cumulative_entropy_residual,
        maximum_carrier_residual,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_finite_scalar_outflow(
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
