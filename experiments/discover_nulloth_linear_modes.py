"""Discover and rank linearly growing Nulloth coherence modes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class DiscoveredMode:
    carrier_density: float
    mode_number: int
    wavenumber: float
    effective_mass_squared: float
    predicted_growth_rate: float
    measured_growth_rate: float
    relative_growth_error: float
    amplification: float
    equilibrium_amplitude: float
    scalar_constraint_l2: float
    hyperbolic_at_equilibrium: bool
    accepted: bool


@dataclass(frozen=True)
class LinearModeDiscoveryResult:
    tested_states: int
    unstable_modes: int
    accepted_candidates: int
    candidates: tuple[DiscoveredMode, ...]


def discover_linear_modes(
    *,
    points: int = 64,
    length: float = 32.0,
    duration: float = 1.0,
    dt: float = 5.0e-3,
    initial_amplitude: float = 1.0e-7,
    carrier_densities: tuple[float, ...] = (0.10, 0.12, 0.16, 0.20, 0.24),
    mode_numbers: tuple[int, ...] = (0, 1, 2, 3),
) -> LinearModeDiscoveryResult:
    """Evolve all unstable modes and return candidates passing validation."""
    if duration <= 0.0 or dt <= 0.0 or initial_amplitude <= 0.0:
        raise ValueError("duration, dt, and initial_amplitude must be positive")
    steps = int(round(duration / dt))
    if steps < 2 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must be an integer multiple of dt")
    grid = PeriodicGrid((points,), (length,), method="spectral")
    (coordinate,) = grid.coordinates()
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    scalar_system = NullothScalarSystem(grid)
    h = grid.zeros((3, 3))
    for component in range(3):
        h[component, component] = 1.0
    trace_K = grid.zeros()
    lapse = np.ones(grid.shape)
    shift = grid.zeros((3,))
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )

    candidates: list[DiscoveredMode] = []
    tested = 0
    for carrier_density in carrier_densities:
        carrier = CarrierPrimitive(
            np.full(grid.shape, carrier_density), grid.zeros((3,))
        )
        symmetric = master.evaluate(h, fluid, carrier, grid.zeros())
        effective_mass_squared = float(symmetric.effective_mass_squared[0])
        for mode_number in mode_numbers:
            tested += 1
            wavenumber = 2.0 * np.pi * mode_number / length
            growth_squared = -effective_mass_squared - wavenumber**2
            if growth_squared <= 0.0:
                continue
            predicted = float(np.sqrt(growth_squared))
            profile = (
                np.ones(grid.shape)
                if mode_number == 0
                else np.cos(wavenumber * coordinate)
            )
            phi = initial_amplitude * profile
            pi = -predicted * phi
            amplitudes = np.empty(steps + 1)

            def mode_amplitude(value: np.ndarray) -> float:
                factor = 1.0 if mode_number == 0 else 2.0
                return abs(float(factor * np.mean(value * profile)))

            def rhs(value_phi: np.ndarray, value_pi: np.ndarray):
                state = NullothScalarState(
                    value_phi,
                    value_pi,
                    grid.gradient(value_phi),
                )
                constitutive = master.evaluate(h, fluid, carrier, value_phi)
                dphi, dpi, _ = scalar_system.rhs(
                    h,
                    trace_K,
                    lapse,
                    shift,
                    state,
                    constitutive.scalar_equation_source,
                )
                return dphi, dpi

            amplitudes[0] = mode_amplitude(phi)
            for step in range(steps):
                k1_phi, k1_pi = rhs(phi, pi)
                k2_phi, k2_pi = rhs(
                    phi + 0.5 * dt * k1_phi,
                    pi + 0.5 * dt * k1_pi,
                )
                k3_phi, k3_pi = rhs(
                    phi + 0.5 * dt * k2_phi,
                    pi + 0.5 * dt * k2_pi,
                )
                k4_phi, k4_pi = rhs(
                    phi + dt * k3_phi, pi + dt * k3_pi
                )
                phi += (dt / 6.0) * (
                    k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
                )
                pi += (dt / 6.0) * (
                    k1_pi + 2.0 * k2_pi + 2.0 * k3_pi + k4_pi
                )
                amplitudes[step + 1] = mode_amplitude(phi)

            times = np.linspace(0.0, duration, steps + 1)
            measured = float(np.polyfit(times, np.log(amplitudes), 1)[0])
            relative_error = abs(measured - predicted) / predicted
            final_state = NullothScalarState(phi, pi, grid.gradient(phi))
            constraint_l2 = float(
                np.sqrt(np.mean(scalar_system.constraint(final_state) ** 2))
            )
            equilibrium_amplitude = float(
                np.sqrt(
                    -effective_mass_squared
                    / NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
                )
            )
            equilibrium_audit = master.coupled_characteristic_audit(
                0.6,
                0.2,
                np.zeros(3),
                carrier_density,
                np.zeros(3),
                equilibrium_amplitude,
            )
            hyperbolic = bool(
                equilibrium_audit.strongly_hyperbolic
                and equilibrium_audit.causal
            )
            accepted = bool(
                relative_error < 1.0e-4
                and constraint_l2 < 1.0e-12
                and hyperbolic
            )
            candidates.append(
                DiscoveredMode(
                    carrier_density,
                    mode_number,
                    wavenumber,
                    effective_mass_squared,
                    predicted,
                    measured,
                    relative_error,
                    float(amplitudes[-1] / amplitudes[0]),
                    equilibrium_amplitude,
                    constraint_l2,
                    hyperbolic,
                    accepted,
                )
            )

    candidates.sort(key=lambda candidate: candidate.measured_growth_rate, reverse=True)
    return LinearModeDiscoveryResult(
        tested,
        len(candidates),
        sum(candidate.accepted for candidate in candidates),
        tuple(candidates),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=5.0e-3)
    args = parser.parse_args()
    result = discover_linear_modes(
        points=args.points, duration=args.duration, dt=args.dt
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
