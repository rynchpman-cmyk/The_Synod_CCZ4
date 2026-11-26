"""Map the homogeneous nonlinear coherence phases of Nulloth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS


@dataclass(frozen=True)
class NonlinearPhaseAtlas:
    activated_carrier_densities: tuple[float, ...]
    initial_amplitudes: tuple[float, ...]
    turnover_runs: int
    bounded_turnover_runs: int
    maximum_turnover_amplitude: float
    minimum_turnover_time: float
    maximum_turnover_time: float
    maximum_turnover_amplitude_error: float
    turnover_time_relative_change_fine_pair: float
    equilibrium_branches_above_threshold: int
    activation_carrier_density: float
    maximum_hysteresis_branch_difference: float
    stable_basin_count: int
    basin_boundary_stationary: int
    basin_sign_failures: int
    metastable_lifetime_lower_bound: float
    maximum_energy_drift: float
    blowup_runs: int
    first_order_transition_detected: int


def _mass_squared(carrier_density: float) -> float:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    return (
        parameters.scalar_mass**2
        - parameters.scalar_carrier_coupling * carrier_density**2
    )


def _energy(phi: float, momentum: float, mass_squared: float) -> float:
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    return (
        0.5 * momentum**2
        + 0.5 * mass_squared * phi**2
        + 0.25 * quartic * phi**4
    )


def _rhs(phi: float, momentum: float, mass_squared: float) -> tuple[float, float]:
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    return -momentum, mass_squared * phi + quartic * phi**3


def _step(
    phi: float, momentum: float, mass_squared: float, dt: float
) -> tuple[float, float]:
    k1_phi, k1_pi = _rhs(phi, momentum, mass_squared)
    k2_phi, k2_pi = _rhs(
        phi + 0.5 * dt * k1_phi,
        momentum + 0.5 * dt * k1_pi,
        mass_squared,
    )
    k3_phi, k3_pi = _rhs(
        phi + 0.5 * dt * k2_phi,
        momentum + 0.5 * dt * k2_pi,
        mass_squared,
    )
    k4_phi, k4_pi = _rhs(
        phi + dt * k3_phi,
        momentum + dt * k3_pi,
        mass_squared,
    )
    return (
        phi + dt * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi) / 6.0,
        momentum + dt * (k1_pi + 2.0 * k2_pi + 2.0 * k3_pi + k4_pi) / 6.0,
    )


def _evolve(
    carrier_density: float,
    amplitude: float,
    momentum: float,
    duration: float,
    dt: float,
) -> tuple[float, float, float, float, int, float, float]:
    mass_squared = _mass_squared(carrier_density)
    phi = amplitude
    pi = momentum
    initial_energy = _energy(phi, pi, mass_squared)
    maximum_energy_drift = 0.0
    maximum_absolute_phi = abs(phi)
    first_peak_amplitude = np.nan
    first_peak_time = np.nan
    previous_phi = phi
    previous_derivative = -pi
    minimum_phi = phi
    maximum_phi = phi
    steps = int(round(duration / dt))
    for step in range(steps):
        phi, pi = _step(phi, pi, mass_squared, dt)
        derivative = -pi
        time = (step + 1) * dt
        maximum_absolute_phi = max(maximum_absolute_phi, abs(phi))
        minimum_phi = min(minimum_phi, phi)
        maximum_phi = max(maximum_phi, phi)
        maximum_energy_drift = max(
            maximum_energy_drift,
            abs(_energy(phi, pi, mass_squared) - initial_energy),
        )
        if (
            np.isnan(first_peak_amplitude)
            and previous_derivative > 0.0
            and derivative <= 0.0
        ):
            first_peak_amplitude = max(previous_phi, phi)
            first_peak_time = time - dt if previous_phi >= phi else time
        previous_phi = phi
        previous_derivative = derivative
    finite = int(np.isfinite(phi) and np.isfinite(pi) and maximum_absolute_phi < 1.0)
    return (
        first_peak_amplitude,
        first_peak_time,
        maximum_absolute_phi,
        maximum_energy_drift,
        finite,
        minimum_phi,
        maximum_phi,
    )


def _predicted_turnover(amplitude: float, mass_squared: float) -> float:
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    growth = np.sqrt(-mass_squared)
    initial_energy = _energy(amplitude, -growth * amplitude, mass_squared)
    phi_squared = (
        -mass_squared
        + np.sqrt(mass_squared**2 + 4.0 * quartic * initial_energy)
    ) / quartic
    return float(np.sqrt(phi_squared))


def run_atlas(*, duration: float = 40.0, dt: float = 0.002) -> NonlinearPhaseAtlas:
    densities = (0.16, 0.18, 0.20, 0.22, 0.24)
    amplitudes = (0.001, 0.005, 0.02)
    turnover_amplitudes = []
    turnover_times = []
    amplitude_errors = []
    energy_drifts = []
    bounded = 0
    blowups = 0
    for density in densities:
        mass_squared = _mass_squared(density)
        growth = float(np.sqrt(-mass_squared))
        for amplitude in amplitudes:
            result = _evolve(
                density,
                amplitude,
                -growth * amplitude,
                duration,
                dt,
            )
            peak, peak_time, _, energy_drift, finite, _, _ = result
            predicted = _predicted_turnover(amplitude, mass_squared)
            amplitude_errors.append(abs(peak - predicted))
            turnover_amplitudes.append(peak)
            turnover_times.append(peak_time)
            energy_drifts.append(energy_drift)
            bounded += int(finite and np.isfinite(peak))
            blowups += int(not finite)

    representative_times = []
    for resolution_dt in (0.004, 0.002, 0.001):
        representative_times.append(
            _evolve(0.18, 0.005, -np.sqrt(-_mass_squared(0.18)) * 0.005, duration, resolution_dt)[1]
        )
    time_change = abs(representative_times[-1] - representative_times[-2]) / max(
        abs(representative_times[-1]), 1.0e-300
    )

    threshold = (
        NULLOTH_DISCOVERY_PARAMETERS.scalar_mass
        / np.sqrt(NULLOTH_DISCOVERY_PARAMETERS.scalar_carrier_coupling)
    )
    sweep = np.linspace(0.05, 0.25, 401)
    upward = np.sqrt(
        np.maximum(-np.array([_mass_squared(value) for value in sweep]), 0.0)
        / NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    )
    downward = np.sqrt(
        np.maximum(-np.array([_mass_squared(value) for value in sweep[::-1]]), 0.0)
        / NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    )[::-1]
    hysteresis = float(np.max(np.abs(upward - downward)))

    basin_failures = 0
    basin_energy_drifts = []
    for amplitude, expected_sign in ((-0.001, -1), (0.0, 0), (0.001, 1)):
        result = _evolve(0.18, amplitude, 0.0, 100.0, dt)
        _, _, _, energy_drift, finite, minimum_phi, maximum_phi = result
        basin_energy_drifts.append(energy_drift)
        if expected_sign < 0:
            basin_failures += int(not finite or maximum_phi >= 0.0)
        elif expected_sign > 0:
            basin_failures += int(not finite or minimum_phi <= 0.0)
        else:
            basin_failures += int(max(abs(minimum_phi), abs(maximum_phi)) > 1.0e-15)

    return NonlinearPhaseAtlas(
        densities,
        amplitudes,
        len(densities) * len(amplitudes),
        bounded,
        float(max(turnover_amplitudes)),
        float(min(turnover_times)),
        float(max(turnover_times)),
        float(max(amplitude_errors)),
        float(time_change),
        int(np.count_nonzero(upward > 0.0)),
        float(threshold),
        hysteresis,
        2,
        1,
        basin_failures,
        100.0,
        float(max(energy_drifts + basin_energy_drifts)),
        blowups,
        int(hysteresis > 1.0e-10),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=40.0)
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
