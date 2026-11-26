"""Qualify driven scalar latency and cutoff on a finite Nulloth patch."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.driven_finite_nulloth import DrivenFiniteNullothProductionSolver
from causal_repair.grid import CartesianGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class LatencyRun:
    points: int
    dt: float
    measured_speed: float
    maximum_arrival_error: float
    prearrival_leakage_ratio: float
    final_characteristic_ratio: float
    right_reflection_ratio: float
    maximum_scalar_constraint_l2: float
    maximum_gravitational_constraint_l2: float
    maximum_recovery_residual: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float


def _run(points: int, dt: float) -> LatencyRun:
    length, duration, center, width, drive_amplitude = 0.3, 0.42, 0.04, 0.01, 1e-4
    grid = CartesianGrid((points,), (length,), origin=(0.0,), method="fd2")

    def drive(side: int, time: float) -> tuple[float, float]:
        if side == 1:
            return 0.0, 0.0
        value = drive_amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)
        derivative = -(time - center) * value / width**2
        return float(value), float(derivative)

    p = NULLOTH_DISCOVERY_PARAMETERS
    solver = DrivenFiniteNullothProductionSolver(
        grid,
        p,
        Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=p.relaxation_time,
            gamma_ad=p.gamma_ad,
        ),
        CCZ4Parameters(
            kappa_gravity=1e-12,
            slicing="harmonic",
            evolve_shift=False,
            radiative_boundaries=True,
        ),
        scalar_drive=drive,
        drive_damping=100.0,
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.1),
        np.full(grid.shape, 0.5),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(np.full(grid.shape, 0.03), grid.zeros((3,)))
    state = solver.initialize(
        solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=0.0
    )
    (coordinate,) = grid.coordinates()
    sensor_targets = (0.075, 0.15, 0.225)
    sensors = [int(np.argmin(abs(coordinate - target))) for target in sensor_targets]
    histories = [[] for _ in sensors]
    times = []
    right_incoming, right_outgoing = 0.0, 0.0
    scalar_constraint = gravity_constraint = recovery_residual = 0.0
    min_legendre = min_thermo = float("inf")
    steps = int(round(duration / dt))
    for _ in range(steps):
        state = solver.step(state, dt)
        characteristic = state.scalar.momentum + state.scalar.gradient[0]
        times.append(state.time)
        for history, index in zip(histories, sensors):
            history.append(abs(float(characteristic[index])))
        right_incoming = max(
            right_incoming,
            abs(float(state.scalar.momentum[-1] - state.scalar.gradient[0, -1])),
        )
        right_outgoing = max(
            right_outgoing,
            abs(float(state.scalar.momentum[-1] + state.scalar.gradient[0, -1])),
        )
        diagnostics = solver.diagnostics(state)
        scalar_constraint = max(scalar_constraint, diagnostics["scalar_constraint_l2"])
        gravity_constraint = max(
            gravity_constraint,
            diagnostics["hamiltonian_l2"],
            diagnostics["momentum_l2"],
        )
        recovery_residual = max(
            recovery_residual, diagnostics["recovery_maximum_residual"]
        )
        min_legendre = min(min_legendre, diagnostics["minimum_legendre_eigenvalue"])
        min_thermo = min(min_thermo, diagnostics["minimum_thermodynamic_eigenvalue"])

    times_array = np.asarray(times)
    peaks = np.asarray([times_array[int(np.argmax(history))] for history in histories])
    positions = coordinate[sensors]
    slope, intercept = np.polyfit(positions, peaks, 1)
    measured_speed = 1.0 / slope
    boundary_position = coordinate[0]
    expected = center + positions - boundary_position
    global_peak = max(max(history) for history in histories)
    leakage = 0.0
    for history, arrival in zip(histories, expected):
        mask = times_array < arrival - 4.0 * width
        if np.any(mask):
            leakage = max(leakage, float(np.max(np.asarray(history)[mask])))
    final_characteristic = max(history[-1] for history in histories)
    return LatencyRun(
        points,
        dt,
        float(measured_speed),
        float(np.max(np.abs(peaks - expected))),
        leakage / max(global_peak, 1e-300),
        final_characteristic / max(global_peak, 1e-300),
        right_incoming / max(right_outgoing, 1e-300),
        float(scalar_constraint),
        float(gravity_constraint),
        float(recovery_residual),
        float(min_legendre),
        float(min_thermo),
    )


def main() -> None:
    runs = (_run(24, 0.001), _run(48, 0.0005))
    result = {
        "resolution_count": 2,
        "coarse_speed": runs[0].measured_speed,
        "fine_speed": runs[1].measured_speed,
        "speed_relative_change": abs(runs[1].measured_speed - runs[0].measured_speed)
        / abs(runs[1].measured_speed),
        "maximum_arrival_error": max(run.maximum_arrival_error for run in runs),
        "maximum_prearrival_leakage_ratio": max(
            run.prearrival_leakage_ratio for run in runs
        ),
        "maximum_final_characteristic_ratio": max(
            run.final_characteristic_ratio for run in runs
        ),
        "maximum_right_reflection_ratio": max(run.right_reflection_ratio for run in runs),
        "maximum_scalar_constraint_l2": max(
            run.maximum_scalar_constraint_l2 for run in runs
        ),
        "maximum_gravitational_constraint_l2": max(
            run.maximum_gravitational_constraint_l2 for run in runs
        ),
        "maximum_recovery_residual": max(run.maximum_recovery_residual for run in runs),
        "minimum_legendre_eigenvalue": min(run.minimum_legendre_eigenvalue for run in runs),
        "minimum_thermodynamic_eigenvalue": min(
            run.minimum_thermodynamic_eigenvalue for run in runs
        ),
        "runs": [asdict(run) for run in runs],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
