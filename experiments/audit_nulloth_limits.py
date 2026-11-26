"""Audit the constitutive and production limits connecting Nulloth to Theory 3.3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction, NullothParameters
from causal_repair.production33 import Theory33ProductionSolver
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import (
    CarrierPrimitive,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


@dataclass(frozen=True)
class NullothLimitAudit:
    constitutive_fields_compared: int
    constitutive_max_abs_error: float
    smallest_grid_failed_cells: int
    nulloth_state_groups: int
    theory33_rhs_groups: int
    nulloth_rhs_groups: int
    rhs_groups_compared: int
    rhs_max_abs_error: float
    scalar_amplitudes: tuple[float, ...]
    one_step_errors: tuple[float, ...]
    one_step_error_ratios: tuple[float, ...]
    one_step_scaling_orders: tuple[float, ...]
    minimum_one_step_scaling_order: float


def _parameters() -> tuple[Theory33MasterParameters, NullothParameters]:
    shared = dict(
        entrainment_linear=0.01,
        entrainment_quadratic=0.005,
        relaxation_time=1.0,
    )
    return (
        Theory33MasterParameters(**shared),
        NullothParameters(
            **shared,
            scalar_baryon_coupling=0.0,
            scalar_carrier_coupling=0.0,
            scalar_counterflow_coupling=0.0,
        ),
    )


def _initial_data(
    grid: PeriodicGrid,
) -> tuple[FluidPrimitive, CarrierPrimitive]:
    (coordinate,) = grid.coordinates()
    fluid = FluidPrimitive(
        0.8 * (1.0 + 0.01 * np.sin(np.pi * coordinate)),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        0.2 * (1.0 + 0.01 * np.cos(np.pi * coordinate)),
        grid.zeros((3,)),
    )
    return fluid, carrier


def _solvers(
    grid: PeriodicGrid,
) -> tuple[Theory33ProductionSolver, NullothProductionSolver]:
    base_parameters, nulloth_parameters = _parameters()
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=base_parameters.relaxation_time,
        gamma_ad=base_parameters.gamma_ad,
    )
    return (
        Theory33ProductionSolver(grid, base_parameters, vector),
        NullothProductionSolver(grid, nulloth_parameters, vector),
    )


def _maximum_difference(left: object, right: object) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def run_limit_audit() -> NullothLimitAudit:
    h = np.eye(3).reshape(3, 3, 1)
    fluid = FluidPrimitive(
        np.array([0.8]),
        np.array([0.2]),
        np.array([[0.08], [0.0], [0.0]]),
        np.array([0.0]),
    )
    carrier = CarrierPrimitive(
        np.array([0.2]),
        np.array([[-0.04], [0.0], [0.0]]),
    )
    base_parameters = Theory33MasterParameters(
        entrainment_linear=0.03,
        entrainment_quadratic=0.02,
    )
    nulloth_parameters = NullothParameters(
        entrainment_linear=0.03,
        entrainment_quadratic=0.02,
        scalar_baryon_coupling=0.7,
        scalar_carrier_coupling=1.3,
        scalar_counterflow_coupling=0.9,
    )
    base_constitutive = Theory33MasterFunction(base_parameters).evaluate(
        h, fluid, carrier
    )
    nulloth_constitutive = NullothMasterFunction(nulloth_parameters).evaluate(
        h, fluid, carrier, np.array([0.0])
    )
    constitutive_names = (
        "lambda_mf",
        "generalized_pressure",
        "B_N",
        "B_D",
        "entrainment",
        "chemical_N",
        "chemical_D",
        "thermodynamic_hessian_nn",
        "thermodynamic_hessian_nd",
        "thermodynamic_hessian_dd",
        "carrier_momentum_down",
        "carrier_momentum_normal",
        "source_charge_eulerian",
        "source_current_up",
    )
    constitutive_errors = [
        _maximum_difference(
            getattr(nulloth_constitutive, name),
            getattr(base_constitutive, name),
        )
        for name in constitutive_names
    ]
    stress_names = ("rho", "momentum", "stress", "trace")
    constitutive_errors.extend(
        _maximum_difference(
            getattr(nulloth_constitutive.stress, name),
            getattr(base_constitutive.stress, name),
        )
        for name in stress_names
    )

    grid = PeriodicGrid((4,), (2.0,), method="fd2")
    base_solver, nulloth_solver = _solvers(grid)
    fluid_grid, carrier_grid = _initial_data(grid)
    base_state = base_solver.initialize(
        base_solver.ccz4.flat_state(), fluid_grid, carrier=carrier_grid
    )
    nulloth_state = nulloth_solver.initialize(
        nulloth_solver.ccz4.flat_state(),
        fluid_grid,
        carrier=carrier_grid,
        varphi=0.0,
    )
    recovery = nulloth_solver.recover(nulloth_state)
    base_rhs = base_solver.rhs(base_state.time, base_solver._pack(base_state))
    nulloth_rhs = nulloth_solver.rhs(
        nulloth_state.time, nulloth_solver._pack(nulloth_state)
    )
    rhs_errors = [
        _maximum_difference(expected, actual)
        for expected, actual in zip(base_rhs, nulloth_rhs[:24])
    ]

    dt = 1.0e-6
    base_after = base_solver.step(base_state, dt)
    base_values = base_solver._pack(base_after)
    amplitudes = (2.0e-3, 1.0e-3, 5.0e-4)
    one_step_errors = []
    for amplitude in amplitudes:
        _, amplitude_solver = _solvers(grid)
        state = amplitude_solver.initialize(
            amplitude_solver.ccz4.flat_state(),
            fluid_grid,
            carrier=carrier_grid,
            varphi=amplitude,
        )
        after = amplitude_solver.step(state, dt)
        values = amplitude_solver._pack(after)
        one_step_errors.append(
            max(
                _maximum_difference(expected, actual)
                for expected, actual in zip(base_values, values[:24])
            )
        )
    ratios = tuple(
        right / left for left, right in zip(one_step_errors, one_step_errors[1:])
    )
    orders = tuple(-float(np.log2(ratio)) for ratio in ratios)
    return NullothLimitAudit(
        len(constitutive_errors),
        max(constitutive_errors),
        recovery.report.failed_cells,
        len(nulloth_solver._pack(nulloth_state)),
        len(base_rhs),
        len(nulloth_rhs),
        min(len(base_rhs), 24),
        max(rhs_errors),
        amplitudes,
        tuple(one_step_errors),
        ratios,
        orders,
        min(orders),
    )


def main() -> None:
    print(json.dumps(asdict(run_limit_audit()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

