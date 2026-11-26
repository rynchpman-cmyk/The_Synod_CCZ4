"""Measure the linear growth rate of the homogeneous Nulloth scalar branch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction, NullothParameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class HomogeneousGrowthResult:
    predicted_rate: float
    measured_rate: float
    relative_error: float
    effective_mass_squared: float
    initial_amplitude: float
    final_amplitude: float
    duration: float
    steps: int


def run_homogeneous_growth(
    *,
    duration: float = 0.5,
    dt: float = 5.0e-4,
    initial_amplitude: float = 1.0e-8,
) -> HomogeneousGrowthResult:
    """Evolve the frozen-material homogeneous mode and fit ``log|varphi|``.

    At linear order the material state is fixed because its Nulloth correction
    is quadratic in ``varphi``.  Initializing ``Pi=-rate*varphi`` selects the
    growing eigenmode for the convention ``varphi_t=-Pi``.
    """
    if duration <= 0.0 or dt <= 0.0 or initial_amplitude <= 0.0:
        raise ValueError("duration, dt, and initial_amplitude must be positive")
    steps = int(round(duration / dt))
    if steps < 2 or not np.isclose(steps * dt, duration):
        raise ValueError("duration must be an integer multiple of dt")

    parameters = NullothParameters(
        scalar_mass=0.4,
        scalar_quartic=2.0,
        scalar_baryon_coupling=0.0,
        scalar_carrier_coupling=12.0,
        scalar_counterflow_coupling=0.0,
    )
    master = NullothMasterFunction(parameters)
    h = np.eye(3).reshape(3, 3, 1)
    fluid = FluidPrimitive(
        np.array([0.8]),
        np.array([0.2]),
        np.zeros((3, 1)),
        np.zeros(1),
    )
    carrier = CarrierPrimitive(np.array([0.2]), np.zeros((3, 1)))
    symmetric = master.evaluate(h, fluid, carrier, np.zeros(1))
    effective_mass_squared = float(symmetric.effective_mass_squared[0])
    if effective_mass_squared >= 0.0:
        raise ValueError("the selected homogeneous state is not scalar-unstable")
    predicted = float(np.sqrt(-effective_mass_squared))

    phi = float(initial_amplitude)
    pi = -predicted * phi
    times = np.linspace(0.0, duration, steps + 1)
    amplitudes = np.empty(steps + 1)
    amplitudes[0] = abs(phi)

    def rhs(value: np.ndarray) -> np.ndarray:
        local_phi, local_pi = value
        state = master.evaluate(
            h, fluid, carrier, np.array([local_phi], dtype=float)
        )
        return np.array(
            [-local_pi, float(state.scalar_equation_source[0])], dtype=float
        )

    value = np.array([phi, pi], dtype=float)
    for index in range(steps):
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * dt * k1)
        k3 = rhs(value + 0.5 * dt * k2)
        k4 = rhs(value + dt * k3)
        value += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        amplitudes[index + 1] = abs(value[0])

    measured = float(np.polyfit(times, np.log(amplitudes), 1)[0])
    return HomogeneousGrowthResult(
        predicted,
        measured,
        abs(measured - predicted) / predicted,
        effective_mass_squared,
        initial_amplitude,
        float(amplitudes[-1]),
        duration,
        steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=5.0e-4)
    parser.add_argument("--amplitude", type=float, default=1.0e-8)
    args = parser.parse_args()
    result = run_homogeneous_growth(
        duration=args.duration,
        dt=args.dt,
        initial_amplitude=args.amplitude,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
