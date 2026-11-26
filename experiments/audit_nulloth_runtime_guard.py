"""Audit SAFE, WATCH, DENY, frame, and production paths of the runtime guard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS, NullothMasterFunction
from causal_repair.nulloth_guard import (
    NullothAdmissibilityError,
    NullothRuntimeGuard,
)
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class RuntimeGuardAudit:
    direction_count: int
    safe_status_correct: int
    watch_status_correct: int
    amplitude_deny_status_correct: int
    superluminal_deny_status_correct: int
    enforce_raised_on_deny: int
    production_recovery_status_correct: int
    frame_status_agreement: int
    frame_maximum_metric_speed_error: float
    safe_minimum_legendre: float
    safe_minimum_thermodynamic: float
    watch_maximum_condition: float
    production_symbol_evaluations: int


def _primitive(
    shape: tuple[int, ...],
    density: float,
    internal: float,
    velocity: np.ndarray,
    carrier_density: float,
    carrier_velocity: np.ndarray,
) -> tuple[FluidPrimitive, CarrierPrimitive]:
    fluid = FluidPrimitive(
        np.full(shape, density),
        np.full(shape, internal),
        np.broadcast_to(velocity.reshape((3,) + (1,) * len(shape)), (3,) + shape).copy(),
        np.zeros(shape),
    )
    carrier = CarrierPrimitive(
        np.full(shape, carrier_density),
        np.broadcast_to(
            carrier_velocity.reshape((3,) + (1,) * len(shape)),
            (3,) + shape,
        ).copy(),
    )
    return fluid, carrier


def run_audit() -> RuntimeGuardAudit:
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    guard = NullothRuntimeGuard(master)
    shape = (4,)
    identity = np.broadcast_to(np.eye(3)[:, :, None], (3, 3) + shape).copy()
    safe_fluid, safe_carrier = _primitive(
        shape, 0.65, 0.225, np.array([0.04, 0.01, 0.0]), 0.18, np.array([-0.03, 0.02, 0.01])
    )
    safe = guard.audit(identity, safe_fluid, safe_carrier, np.full(shape, 0.05))
    watch = guard.audit(identity, safe_fluid, safe_carrier, np.full(shape, 0.16))
    amplitude_deny = guard.audit(
        identity, safe_fluid, safe_carrier, np.full(shape, 0.18)
    )
    low_carrier_fluid, low_carrier = _primitive(
        shape, 0.65, 0.225, np.zeros(3), 0.02, np.zeros(3)
    )
    superluminal = guard.audit(
        identity, low_carrier_fluid, low_carrier, np.zeros(shape)
    )
    raised = 0
    try:
        guard.enforce(identity, safe_fluid, safe_carrier, np.full(shape, 0.18))
    except NullothAdmissibilityError:
        raised = 1

    metric_cell = np.array(
        [[1.2, 0.04, 0.0], [0.04, 0.9, 0.03], [0.0, 0.03, 1.1]]
    )
    metric = np.broadcast_to(metric_cell[:, :, None], (3, 3) + shape).copy()
    frame = np.linalg.cholesky(metric_cell).T
    coordinate_velocity = np.linalg.solve(frame, np.array([0.04, 0.01, 0.0]))
    coordinate_carrier = np.linalg.solve(frame, np.array([-0.03, 0.02, 0.01]))
    curved_fluid, curved_carrier = _primitive(
        shape,
        0.65,
        0.225,
        coordinate_velocity,
        0.18,
        coordinate_carrier,
    )
    curved = guard.audit(
        metric, curved_fluid, curved_carrier, np.full(shape, 0.05)
    )
    speed_error = max(
        abs(
            float(coordinate_velocity @ metric_cell @ coordinate_velocity)
            - float(np.array([0.04, 0.01, 0.0]) @ np.array([0.04, 0.01, 0.0]))
        ),
        abs(
            float(coordinate_carrier @ metric_cell @ coordinate_carrier)
            - float(np.array([-0.03, 0.02, 0.01]) @ np.array([-0.03, 0.02, 0.01]))
        ),
    )

    grid = PeriodicGrid((4,), (2.0,), method="fd2")
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=NULLOTH_DISCOVERY_PARAMETERS.relaxation_time,
        target_strength=0.0,
        gamma_ad=NULLOTH_DISCOVERY_PARAMETERS.gamma_ad,
    )
    solver = NullothProductionSolver(
        grid, NULLOTH_DISCOVERY_PARAMETERS, vector
    )
    production_fluid, production_carrier = _primitive(
        grid.shape, 0.65, 0.225, np.zeros(3), 0.18, np.zeros(3)
    )
    state = solver.initialize(
        solver.ccz4.flat_state(),
        production_fluid,
        carrier=production_carrier,
        varphi=0.05,
    )
    recovery = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    production = guard.audit(
        h, recovery.fluid, recovery.carrier, state.scalar.varphi
    )
    return RuntimeGuardAudit(
        guard.thresholds.direction_count,
        int(safe.status == "SAFE"),
        int(watch.status == "WATCH"),
        int(amplitude_deny.status == "DENY"),
        int(superluminal.status == "DENY"),
        raised,
        int(production.status == "SAFE"),
        int(curved.status == safe.status),
        speed_error,
        safe.minimum_legendre_eigenvalue,
        safe.minimum_thermodynamic_eigenvalue,
        watch.maximum_eigenvector_condition,
        production.symbol_evaluations,
    )


def main() -> None:
    print(json.dumps(asdict(run_audit()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
