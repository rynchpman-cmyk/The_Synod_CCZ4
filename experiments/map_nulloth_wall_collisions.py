"""Evolve localized Nulloth domain walls and controlled collisions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.integrators import rk4_arrays
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem


@dataclass(frozen=True)
class WallRun:
    dimensions: int
    x_points: int
    initial_zero_crossings: int
    final_zero_crossings: int
    minimum_wall_separation: float
    measured_incoming_speed: float
    incoming_speed_error: float
    maximum_energy_relative_drift: float
    final_constraint_l2: float
    final_high_wavenumber_energy_fraction: float
    maximum_amplitude: float
    bounded: int


@dataclass(frozen=True)
class WallCollisionAtlas:
    one_dimensional_runs: tuple[WallRun, ...]
    embedded_runs: tuple[WallRun, ...]
    wall_thickness: float
    vacuum_amplitude: float
    controlled_collisions: int
    maximum_speed_error: float
    maximum_energy_relative_drift: float
    maximum_constraint_l2: float
    high_wavenumber_fraction_change_96_to_128: float
    maximum_dimensional_embedding_error: float
    bounded_runs: int
    annihilation_runs: int


def _zero_positions(profile: np.ndarray, length: float) -> np.ndarray:
    signs = np.signbit(profile)
    indices = np.flatnonzero(signs != np.roll(signs, -1))
    positions = []
    spacing = length / len(profile)
    for index in indices:
        left = float(profile[index])
        right = float(profile[(index + 1) % len(profile)])
        fraction = abs(left) / max(abs(left) + abs(right), 1.0e-300)
        positions.append((index + fraction) * spacing % length)
    return np.sort(np.asarray(positions))


def _energy(
    grid: PeriodicGrid, state: NullothScalarState, mass_squared: float
) -> float:
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    gradient_squared = np.sum(state.gradient**2, axis=0)
    vacuum_squared = -mass_squared / quartic
    vacuum_potential = (
        0.5 * mass_squared * vacuum_squared
        + 0.25 * quartic * vacuum_squared**2
    )
    density = (
        0.5 * (state.momentum**2 + gradient_squared)
        + 0.5 * mass_squared * state.varphi**2
        + 0.25 * quartic * state.varphi**4
        - vacuum_potential
    )
    return grid.integrate(density)


def _high_wavenumber_fraction(
    profile: np.ndarray, momentum: np.ndarray, length: float, mass_squared: float
) -> float:
    count = len(profile)
    phi_hat = np.fft.fft(profile) / count
    pi_hat = np.fft.fft(momentum) / count
    wavenumbers = 2.0 * np.pi * np.fft.fftfreq(count, d=length / count)
    quadratic = 0.5 * (
        np.abs(pi_hat) ** 2
        + (wavenumbers**2 + abs(mass_squared)) * np.abs(phi_hat) ** 2
    )
    high = np.abs(np.fft.fftfreq(count) * count) > count / 3.0
    return float(np.sum(quadratic[high]) / max(np.sum(quadratic), 1.0e-300))


def _run(
    x_points: int,
    dimensions: int,
    duration: float,
    dt: float,
    incoming_speed: float,
) -> tuple[WallRun, np.ndarray]:
    shape = (x_points,) + (4,) * (dimensions - 1)
    lengths = (32.0,) + (4.0,) * (dimensions - 1)
    grid = PeriodicGrid(shape, lengths, method="spectral")
    coordinate = grid.coordinates()[0]
    mass_squared = 0.4**2 - 12.0 * 0.18**2
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    vacuum = np.sqrt(-mass_squared / quartic)
    thickness = np.sqrt(2.0 / -mass_squared)
    gamma = 1.0 / np.sqrt(1.0 - incoming_speed**2)
    left_argument = gamma * (coordinate - 8.0) / thickness
    right_argument = gamma * (24.0 - coordinate) / thickness
    left = np.tanh(left_argument)
    right = np.tanh(right_argument)
    phi = vacuum * left * right
    phi_time = -vacuum * gamma * incoming_speed / thickness * (
        (1.0 / np.cosh(left_argument) ** 2) * right
        + left * (1.0 / np.cosh(right_argument) ** 2)
    )
    momentum = -phi_time
    system = NullothScalarSystem(grid)
    state = NullothScalarState(phi, momentum, grid.gradient(phi))
    initial_energy = _energy(grid, state, mass_squared)
    maximum_energy_drift = 0.0
    maximum_amplitude = float(np.max(np.abs(phi)))
    minimum_separation = 16.0
    tracking_times = []
    tracking_positions = []
    initial_positions = _zero_positions(phi[(slice(None),) + (0,) * (dimensions - 1)], 32.0)

    def rhs(_time: float, values):
        current = NullothScalarState(values[0], values[1], values[2])
        source = mass_squared * current.varphi + quartic * current.varphi**3
        return system.rhs(
            np.broadcast_to(np.eye(3)[(...,) + (None,) * dimensions], (3, 3) + shape),
            grid.zeros(),
            np.ones(shape),
            grid.zeros((3,)),
            current,
            source,
        )

    steps = int(round(duration / dt))
    sample_interval = max(1, int(round(0.25 / dt)))
    for step in range(steps):
        values = rk4_arrays(
            (state.varphi, state.momentum, state.gradient), step * dt, dt, rhs
        )
        state = NullothScalarState(*values)
        if (step + 1) % sample_interval == 0 or step == steps - 1:
            time = (step + 1) * dt
            profile = state.varphi[(slice(None),) + (0,) * (dimensions - 1)]
            positions = _zero_positions(profile, 32.0)
            if len(positions) >= 2:
                separation = float(positions[1] - positions[0])
                minimum_separation = min(minimum_separation, separation)
                if time <= 8.0:
                    tracking_times.append(time)
                    tracking_positions.append(float(positions[0]))
            maximum_amplitude = max(maximum_amplitude, float(np.max(np.abs(state.varphi))))
            maximum_energy_drift = max(
                maximum_energy_drift,
                abs(_energy(grid, state, mass_squared) - initial_energy)
                / max(abs(initial_energy), 1.0e-300),
            )

    final_profile = state.varphi[(slice(None),) + (0,) * (dimensions - 1)]
    final_momentum = state.momentum[(slice(None),) + (0,) * (dimensions - 1)]
    final_positions = _zero_positions(final_profile, 32.0)
    measured_speed = float(np.polyfit(tracking_times, tracking_positions, 1)[0])
    constraint = system.constraint(state)
    run = WallRun(
        dimensions,
        x_points,
        len(initial_positions),
        len(final_positions),
        minimum_separation,
        measured_speed,
        abs(measured_speed - incoming_speed),
        maximum_energy_drift,
        float(np.sqrt(np.mean(constraint**2))),
        _high_wavenumber_fraction(final_profile, final_momentum, 32.0, mass_squared),
        maximum_amplitude,
        int(np.isfinite(maximum_amplitude) and maximum_amplitude < 0.17),
    )
    return run, final_profile


def run_atlas(
    *, duration: float = 28.0, dt: float = 0.01, incoming_speed: float = 0.4
) -> WallCollisionAtlas:
    one_dimensional = []
    profiles: dict[tuple[int, int], np.ndarray] = {}
    for points in (64, 96, 128):
        run, profile = _run(points, 1, duration, dt, incoming_speed)
        one_dimensional.append(run)
        profiles[(1, points)] = profile
    embedded = []
    for dimensions in (2, 3):
        run, profile = _run(64, dimensions, duration, dt, incoming_speed)
        embedded.append(run)
        profiles[(dimensions, 64)] = profile
    embedding_error = max(
        float(np.max(np.abs(profiles[(dimensions, 64)] - profiles[(1, 64)])))
        for dimensions in (2, 3)
    )
    high_change = abs(
        one_dimensional[-1].final_high_wavenumber_energy_fraction
        - one_dimensional[-2].final_high_wavenumber_energy_fraction
    )
    all_runs = one_dimensional + embedded
    return WallCollisionAtlas(
        tuple(one_dimensional),
        tuple(embedded),
        float(np.sqrt(2.0 / -(0.4**2 - 12.0 * 0.18**2))),
        float(np.sqrt(-(0.4**2 - 12.0 * 0.18**2) / 50.0)),
        sum(run.minimum_wall_separation < 2.0 for run in all_runs),
        max(run.incoming_speed_error for run in all_runs),
        max(run.maximum_energy_relative_drift for run in all_runs),
        max(run.final_constraint_l2 for run in all_runs),
        high_change,
        embedding_error,
        sum(run.bounded for run in all_runs),
        sum(run.final_zero_crossings == 0 for run in all_runs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=28.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--incoming-speed", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(
        duration=arguments.duration,
        dt=arguments.dt,
        incoming_speed=arguments.incoming_speed,
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
