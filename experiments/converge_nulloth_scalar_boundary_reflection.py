"""Converge scalar boundary reflection against an extended-domain control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.adm import flat_metric
from causal_repair.finite_nulloth import NullothScalarCharacteristicBoundary
from causal_repair.grid import CartesianGrid
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem


@dataclass(frozen=True)
class ReflectionCase:
    points: int
    spacing: float
    dt: float
    steps: int
    normalized_reflection_energy: float
    maximum_reflected_amplitude: float
    finite_scalar_constraint_l2: float
    extended_scalar_constraint_l2: float


@dataclass(frozen=True)
class ReflectionConvergence:
    duration: float
    cfl_ratio: float
    cases: tuple[ReflectionCase, ...]
    finest_normalized_reflection_energy: float
    coarse_to_medium_reflection_reduction: float
    medium_to_fine_reflection_reduction: float
    finest_maximum_reflected_amplitude: float
    maximum_scalar_constraint_l2: float
    extended_boundary_causally_disconnected: int


def _initial_state(system: NullothScalarSystem) -> NullothScalarState:
    coordinate = system.grid.coordinates()[0]
    varphi = 1.0e-6 * np.exp(-0.5 * ((coordinate - 0.2) / 0.06) ** 2)
    gradient = system.grid.gradient(varphi)
    return system.initialize(varphi, gradient[0], gradient)


def _advance(grid: CartesianGrid, duration: float, cfl_ratio: float):
    system = NullothScalarSystem(grid)
    boundary = NullothScalarCharacteristicBoundary(grid)
    state = _initial_state(system)
    h = flat_metric(grid)
    lapse = np.ones(grid.shape)
    shift = grid.zeros((3,))
    effective_mass_squared = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass**2
        - NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling * 0.03**2
    )
    dt = cfl_ratio * grid.spacing[0]
    steps = int(round(duration / dt))
    if steps < 1 or abs(steps * dt - duration) > 1.0e-14:
        raise ValueError("duration must close on the reflection ladder")

    def rhs(current: NullothScalarState):
        source = (
            effective_mass_squared * current.varphi
            + NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
            * current.varphi**3
        )
        raw = system.rhs(
            h,
            grid.zeros(),
            lapse,
            shift,
            current,
            source,
        )
        return boundary.scalar_rhs(h, current, raw, lapse, shift)

    def stage(current, slope, factor):
        return NullothScalarState(
            current.varphi + factor * dt * slope[0],
            current.momentum + factor * dt * slope[1],
            current.gradient + factor * dt * slope[2],
        )

    for _ in range(steps):
        k1 = rhs(state)
        k2 = rhs(stage(state, k1, 0.5))
        k3 = rhs(stage(state, k2, 0.5))
        k4 = rhs(stage(state, k3, 1.0))
        state = NullothScalarState(
            state.varphi
            + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state.momentum
            + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state.gradient
            + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )
    constraint_l2 = float(np.sqrt(np.mean(system.constraint(state) ** 2)))
    return state, effective_mass_squared, dt, steps, constraint_l2


def _run_case(
    points: int, duration: float, cfl_ratio: float
) -> ReflectionCase:
    finite = CartesianGrid((points,), (1.0,), origin=(-0.5,), method="fd2")
    extended = CartesianGrid(
        (2 * points,), (2.0,), origin=(-0.5,), method="fd2"
    )
    finite_state, mass_squared, dt, steps, finite_constraint = _advance(
        finite, duration, cfl_ratio
    )
    extended_state, _, _, _, extended_constraint = _advance(
        extended, duration, cfl_ratio
    )
    coordinate = finite.coordinates()[0]
    comparison = coordinate < 0.4
    dvarphi = finite_state.varphi - extended_state.varphi[:points]
    dmomentum = finite_state.momentum - extended_state.momentum[:points]
    dgradient = (
        finite_state.gradient[0] - extended_state.gradient[0, :points]
    )
    reflected_energy = 0.5 * finite.integrate(
        comparison
        * (
            dmomentum**2
            + dgradient**2
            + mass_squared * dvarphi**2
        )
    )
    initial = _initial_state(NullothScalarSystem(finite))
    initial_energy = 0.5 * finite.integrate(
        initial.momentum**2
        + initial.gradient[0] ** 2
        + mass_squared * initial.varphi**2
    )
    return ReflectionCase(
        points,
        finite.spacing[0],
        dt,
        steps,
        reflected_energy / max(initial_energy, 1.0e-300),
        float(np.max(np.abs(dvarphi[comparison]))),
        finite_constraint,
        extended_constraint,
    )


def run_convergence(
    *, duration: float = 0.6, cfl_ratio: float = 0.1
) -> ReflectionConvergence:
    cases = tuple(
        _run_case(points, duration, cfl_ratio) for points in (32, 64, 128)
    )
    coarse, medium, fine = cases
    return ReflectionConvergence(
        duration,
        cfl_ratio,
        cases,
        fine.normalized_reflection_energy,
        coarse.normalized_reflection_energy
        / max(medium.normalized_reflection_energy, 1.0e-300),
        medium.normalized_reflection_energy
        / max(fine.normalized_reflection_energy, 1.0e-300),
        fine.maximum_reflected_amplitude,
        max(
            max(case.finite_scalar_constraint_l2, case.extended_scalar_constraint_l2)
            for case in cases
        ),
        int(0.2 + duration < 1.5),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.6)
    parser.add_argument("--cfl-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_convergence(
        duration=arguments.duration,
        cfl_ratio=arguments.cfl_ratio,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
