"""Audit Nulloth mode mixing, avoided crossings, damping, and drag."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive, lorentz_factor
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters, Theory3System
from causal_repair.theory33 import CarrierPrimitive

try:
    from experiments.audit_nulloth_linear_spectrum import (
        _ordered,
        _vector_expected,
        _vector_generator,
    )
except ImportError:  # Direct experiment execution.
    from audit_nulloth_linear_spectrum import _ordered, _vector_expected, _vector_generator


@dataclass(frozen=True)
class MixingRelaxationAudit:
    mass_samples: int
    exact_crossings_detected: int
    avoided_crossings_detected: int
    uncoupled_minimum_mass_squared_gap: float
    coupled_minimum_mass_squared_gap: float
    maximum_mode_mixing_fraction: float
    cleaning_damping_samples: int
    maximum_cleaning_dispersion_error: float
    electric_damping_parameter_sets: int
    electric_damping_modes: int
    maximum_electric_damping_error: float
    minimum_irreversible_heat: float
    carrier_relaxation_times: tuple[float, ...]
    maximum_carrier_drag_decay_error: float
    carrier_drag_monotonic_with_inverse_relaxation_time: int
    minimum_drag_entropy_change: float
    maximum_drag_energy_conservation_error: float
    maximum_drag_momentum_conservation_error: float


def _mass_crossing_audit(samples: int = 81) -> tuple[float, float, float]:
    masses_B = np.linspace(0.3, 0.7, samples)
    uncoupled_gaps = []
    coupled_gaps = []
    mixing_fractions = []
    for mass_B in masses_B:
        uncoupled = Theory3Parameters(
            mass_A=0.5,
            mass_B=float(mass_B),
            mixing_mass_squared=0.0,
            target_speed=0.9,
        )
        coupled = replace(uncoupled, mixing_mass_squared=0.05)
        uncoupled_values = np.sort(np.linalg.eigvals(uncoupled.gauss_matrix).real)
        coupled_values, coupled_vectors = np.linalg.eigh(coupled.gauss_matrix)
        uncoupled_gaps.append(float(np.diff(uncoupled_values)[0]))
        coupled_gaps.append(float(np.diff(coupled_values)[0]))
        for column in range(2):
            weights = np.abs(coupled_vectors[:, column]) ** 2
            mixing_fractions.append(float(np.min(weights)))
    return min(uncoupled_gaps), min(coupled_gaps), max(mixing_fractions)


def _cleaning_audit() -> float:
    maximum_error = 0.0
    for damping in (0.0, 0.5, 1.0, 3.0):
        generator, parameters = _vector_generator(24, 1, damping)
        numerical = _ordered(np.linalg.eigvals(generator))
        expected = _vector_expected(1, damping, parameters)
        maximum_error = max(
            maximum_error, float(np.max(np.abs(numerical - expected)))
        )
    return maximum_error


def _electric_damping_audit() -> tuple[int, float, float]:
    grid = PeriodicGrid((4,), (2.0,), method="spectral")
    metric = grid.zeros((3, 3))
    for component in range(3):
        metric[component, component] = 1.0
    fluid = FluidPrimitive(
        np.ones(grid.shape), np.ones(grid.shape), grid.zeros((3,)), grid.zeros()
    )
    zero = grid.zeros()
    zero_vector = grid.zeros((3,))
    lapse = np.ones(grid.shape)
    maximum_error = 0.0
    minimum_heat = np.inf
    tested_modes = 0
    dt = 0.1
    for gamma_W, gamma_B in ((0.2, 0.0), (0.2, 0.3), (1.0, 1.0)):
        parameters = Theory3Parameters(
            target_speed=0.9,
            gamma_W=gamma_W,
            gamma_B=gamma_B,
        )
        system = Theory3System(grid, parameters)
        damping_matrix = np.array(
            [
                [-gamma_W / parameters.Z_A, gamma_W / parameters.Z_A],
                [gamma_W / parameters.Z_B, -(gamma_W + gamma_B) / parameters.Z_B],
            ]
        )
        rates, vectors = np.linalg.eig(damping_matrix)
        for rate, vector in zip(rates, vectors.T):
            electric_A = zero_vector.copy()
            electric_B = zero_vector.copy()
            electric_A[1] = vector[0]
            electric_B[1] = vector[1]
            result = system.damping_step(
                metric,
                zero_vector,
                -parameters.Z_A * electric_A,
                zero_vector,
                -parameters.Z_B * electric_B,
                zero,
                zero,
                fluid,
                lapse,
                dt,
            )
            new_A = -result.pi_A[1] / parameters.Z_A
            new_B = -result.pi_B[1] / parameters.Z_B
            observed = np.array([float(np.mean(new_A)), float(np.mean(new_B))])
            expected = np.exp(float(rate) * dt) * vector
            maximum_error = max(
                maximum_error, float(np.max(np.abs(observed - expected)))
            )
            minimum_heat = min(minimum_heat, float(np.min(result.irreversible_heat)))
            tested_modes += 1
    return tested_modes, maximum_error, minimum_heat


def _carrier_drag_audit() -> tuple[tuple[float, ...], float, int, float, float, float]:
    times = (0.25, 0.5, 1.0, 2.0)
    ratios = []
    maximum_error = 0.0
    minimum_entropy = np.inf
    maximum_energy_error = 0.0
    maximum_momentum_error = 0.0
    dt = 0.05
    for relaxation_time in times:
        grid = PeriodicGrid((4,), (2.0,), method="fd2")
        nulloth = replace(
            NULLOTH_DISCOVERY_PARAMETERS, relaxation_time=relaxation_time
        )
        vector = Theory3Parameters(
            target_speed=0.9,
            target_relaxation_time=relaxation_time,
            target_strength=0.0,
            gamma_W=0.0,
            gamma_B=0.0,
        )
        solver = NullothProductionSolver(grid, nulloth, vector)
        fluid = FluidPrimitive(
            np.full(grid.shape, 0.6),
            np.full(grid.shape, 0.2),
            grid.zeros((3,)),
            grid.zeros(),
        )
        carrier_velocity = grid.zeros((3,))
        carrier_velocity[0] = 0.2
        carrier = CarrierPrimitive(np.full(grid.shape, 0.18), carrier_velocity)
        state = solver.initialize(
            solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=0.0
        )
        before = solver.recover(state)
        h, _ = solver.ccz4.physical_geometry(state.geometry)
        q_N_before = lorentz_factor(h, before.fluid.velocity)[None, ...] * before.fluid.velocity
        q_D_before = lorentz_factor(h, before.carrier.velocity)[None, ...] * before.carrier.velocity
        difference_before = q_D_before - q_N_before
        energy_before = state.matter.energy.copy()
        momentum_before = state.matter.momentum.copy()
        updated = solver._apply_carrier_drag(state, dt)
        after = solver.recover(updated)
        q_N_after = lorentz_factor(h, after.fluid.velocity)[None, ...] * after.fluid.velocity
        q_D_after = lorentz_factor(h, after.carrier.velocity)[None, ...] * after.carrier.velocity
        observed = float(np.linalg.norm(q_D_after - q_N_after) / np.linalg.norm(difference_before))
        expected = float(np.exp(-dt / relaxation_time))
        ratios.append(observed)
        maximum_error = max(maximum_error, abs(observed - expected))
        minimum_entropy = min(minimum_entropy, solver.last_drag_entropy_change)
        maximum_energy_error = max(
            maximum_energy_error, float(np.max(np.abs(updated.matter.energy - energy_before)))
        )
        maximum_momentum_error = max(
            maximum_momentum_error,
            float(np.max(np.abs(updated.matter.momentum - momentum_before))),
        )
    monotonic = int(all(left < right for left, right in zip(ratios, ratios[1:])))
    return (
        times,
        maximum_error,
        monotonic,
        minimum_entropy,
        maximum_energy_error,
        maximum_momentum_error,
    )


def run_audit() -> MixingRelaxationAudit:
    uncoupled_gap, coupled_gap, mixing = _mass_crossing_audit()
    cleaning_error = _cleaning_audit()
    electric_modes, electric_error, minimum_heat = _electric_damping_audit()
    drag = _carrier_drag_audit()
    return MixingRelaxationAudit(
        81,
        int(uncoupled_gap < 1.0e-12),
        int(coupled_gap > 0.05),
        uncoupled_gap,
        coupled_gap,
        mixing,
        4,
        cleaning_error,
        3,
        electric_modes,
        electric_error,
        minimum_heat,
        *drag,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_audit()
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
