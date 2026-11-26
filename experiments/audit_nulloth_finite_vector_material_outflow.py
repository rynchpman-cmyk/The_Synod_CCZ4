"""Audit independent vector and material outflow on a finite Nulloth patch."""

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
    from experiments.audit_nulloth_finite_scalar_outflow import _solver
except ModuleNotFoundError:
    from audit_nulloth_finite_scalar_outflow import _solver


@dataclass(frozen=True)
class FiniteSourceOutflow:
    duration: float
    dt: float
    vector_norm_loss_fraction: float
    material_excess_baryon_loss_fraction: float
    material_excess_carrier_loss_fraction: float
    vector_opposite_edge_amplitude: float
    material_opposite_edge_density_contrast: float
    maximum_gravitational_constraint_l2: float
    maximum_proca_constraint_l2: float
    maximum_scalar_constraint_l2: float
    maximum_recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_step_entropy_balance_residual: float
    maximum_absolute_cumulative_entropy_balance_residual: float
    maximum_carrier_number_balance_residual: float


def _initial_state(solver, source: str):
    grid = solver.grid
    (coordinate,) = grid.coordinates()
    profile = np.exp(-0.5 * ((coordinate - 0.4) / 0.05) ** 2)
    density = np.full(grid.shape, 0.1)
    carrier_density = np.full(grid.shape, 0.03)
    fluid_velocity = grid.zeros((3,))
    carrier_velocity = grid.zeros((3,))
    a = grid.zeros((3,))
    pi_A = grid.zeros((3,))
    if source == "vector":
        a[1] = 1.0e-5 * profile
        pi_A[1] = -grid.derivative(a[1], 0)
    elif source == "material":
        density += 0.02 * profile
        carrier_density += 0.006 * profile
        fluid_velocity[0] = 0.25 * profile
        carrier_velocity[0] = fluid_velocity[0]
    elif source != "control":
        raise ValueError(f"unknown finite source {source}")
    fluid = FluidPrimitive(
        density,
        np.full(grid.shape, 0.5),
        fluid_velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(carrier_density, carrier_velocity)
    return solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        a=a,
        pi_A=pi_A,
    )


def _field_norm(grid, active, control) -> float:
    difference_a = active.a[1] - control.a[1]
    difference_pi = active.pi_A[1] - control.pi_A[1]
    return grid.integrate(difference_a**2 + difference_pi**2)


def _evolve(solver, state, steps: int, dt: float):
    maximum_entropy_residual = 0.0
    cumulative_entropy_residual = 0.0
    maximum_carrier_residual = 0.0
    for _ in range(steps):
        state = solver.step(state, dt)
        maximum_entropy_residual = max(
            maximum_entropy_residual,
            abs(solver.last_entropy_balance_residual),
        )
        cumulative_entropy_residual += solver.last_entropy_balance_residual
        maximum_carrier_residual = max(
            maximum_carrier_residual,
            abs(solver.last_carrier_number_balance_residual),
        )
    return (
        state,
        maximum_entropy_residual,
        cumulative_entropy_residual,
        maximum_carrier_residual,
    )


def run_outflow(
    *, points: int = 24, duration: float = 0.12, dt: float = 0.001
) -> FiniteSourceOutflow:
    steps = int(round(duration / dt))
    if points < 6 or steps < 1 or not np.isclose(steps * dt, duration):
        raise ValueError("invalid finite source trial")
    grid = CartesianGrid((points,), (1.0,), origin=(-0.5,), method="fd2")
    solvers = {name: _solver(grid) for name in ("control", "vector", "material")}
    initial = {
        name: _initial_state(solvers[name], name)
        for name in ("control", "vector", "material")
    }
    initial_vector_norm = _field_norm(grid, initial["vector"], initial["control"])
    initial_diagnostics = {
        name: solvers[name].diagnostics(initial[name]) for name in initial
    }
    initial_baryon_excess = (
        float(initial_diagnostics["material"]["baryon_mass"])
        - float(initial_diagnostics["control"]["baryon_mass"])
    )
    initial_carrier_excess = (
        float(initial_diagnostics["material"]["carrier_number"])
        - float(initial_diagnostics["control"]["carrier_number"])
    )
    evolved = {
        name: _evolve(solvers[name], initial[name], steps, dt)
        for name in ("control", "vector", "material")
    }
    final = {name: result[0] for name, result in evolved.items()}
    final_diagnostics = {
        name: solvers[name].diagnostics(final[name]) for name in final
    }
    final_vector_norm = _field_norm(grid, final["vector"], final["control"])
    final_baryon_excess = (
        float(final_diagnostics["material"]["baryon_mass"])
        - float(final_diagnostics["control"]["baryon_mass"])
    )
    final_carrier_excess = (
        float(final_diagnostics["material"]["carrier_number"])
        - float(final_diagnostics["control"]["carrier_number"])
    )
    material_recovery = solvers["material"].recover(final["material"])
    control_recovery = solvers["control"].recover(final["control"])
    diagnostics = tuple(final_diagnostics.values())
    return FiniteSourceOutflow(
        duration,
        dt,
        1.0 - final_vector_norm / max(initial_vector_norm, 1.0e-300),
        1.0 - final_baryon_excess / max(initial_baryon_excess, 1.0e-300),
        1.0 - final_carrier_excess / max(initial_carrier_excess, 1.0e-300),
        max(
            abs(float(final["vector"].a[1, 0] - final["control"].a[1, 0])),
            abs(
                float(
                    final["vector"].pi_A[1, 0]
                    - final["control"].pi_A[1, 0]
                )
            ),
        ),
        abs(
            float(
                material_recovery.fluid.baryon_density[0]
                - control_recovery.fluid.baryon_density[0]
            )
        ),
        max(
            max(float(item["hamiltonian_l2"]), float(item["momentum_l2"]))
            for item in diagnostics
        ),
        max(
            max(float(item["gauss_A_l2"]), float(item["gauss_B_l2"]))
            for item in diagnostics
        ),
        max(float(item["scalar_constraint_l2"]) for item in diagnostics),
        max(float(item["recovery_maximum_residual"]) for item in diagnostics),
        min(float(item["minimum_legendre_eigenvalue"]) for item in diagnostics),
        min(float(item["minimum_thermodynamic_eigenvalue"]) for item in diagnostics),
        max(result[1] for result in evolved.values()),
        max(abs(result[2]) for result in evolved.values()),
        max(result[3] for result in evolved.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=24)
    parser.add_argument("--duration", type=float, default=0.12)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_outflow(
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
