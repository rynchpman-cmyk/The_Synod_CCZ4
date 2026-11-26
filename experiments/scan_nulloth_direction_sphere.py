"""Scan a near-uniform sphere of Nulloth propagation directions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class DirectionSphereAudit:
    states: int
    directions_per_state: int
    samples: int
    failures: int
    activated_samples: int
    approximate_covering_radius_radians: float
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    maximum_eigenvector_condition: float
    maximum_constraint_reduced_condition: float
    maximum_eigenspace_residual: float
    maximum_reduction_spectrum_error: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_characteristic_multiplicity: int
    minimum_distinct_characteristic_families: int
    worst_state_index: int
    worst_direction: tuple[float, float, float]
    first_failure: dict[str, object] | None


def _fibonacci_sphere(count: int, phase: float = 0.5) -> np.ndarray:
    indices = np.arange(count, dtype=float) + phase
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = golden_angle * indices
    return np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )


def _covering_radius(directions: np.ndarray) -> float:
    probes = _fibonacci_sphere(4096, phase=0.25)
    nearest_cosine = np.max(probes @ directions.T, axis=1)
    return float(np.max(np.arccos(np.clip(nearest_cosine, -1.0, 1.0))))


def _random_velocity(
    rng: np.random.Generator, maximum: float
) -> np.ndarray:
    direction = rng.normal(size=3)
    return direction * rng.uniform(0.0, maximum) / np.linalg.norm(direction)


def _states(seed: int) -> list[tuple[float, float, np.ndarray, float, np.ndarray, float]]:
    states = [
        (0.4, 0.1, np.zeros(3), 0.12, np.array([-0.3, 0.0, 0.0]), -0.16),
        (0.4, 0.35, np.array([0.0, 0.2, 0.0]), 0.24, np.zeros(3), 0.16),
        (0.9, 0.1, np.zeros(3), 0.24, np.array([0.0, 0.0, 0.3]), 0.16),
        (
            0.9,
            0.35,
            np.array([0.2, 0.0, 0.0]),
            0.12,
            np.array([0.0, -0.3, 0.0]),
            -0.16,
        ),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(8):
        states.append(
            (
                rng.uniform(0.4, 0.9),
                rng.uniform(0.1, 0.35),
                _random_velocity(rng, 0.2),
                rng.uniform(0.12, 0.24),
                _random_velocity(rng, 0.3),
                rng.uniform(-0.16, 0.16),
            )
        )
    return states


def _multiplicities(speeds: np.ndarray, tolerance: float = 1.0e-7) -> list[int]:
    ordered = np.sort(np.asarray(speeds, dtype=float))
    multiplicities = [1]
    for left, right in zip(ordered[:-1], ordered[1:]):
        if abs(right - left) <= tolerance * max(1.0, abs(left), abs(right)):
            multiplicities[-1] += 1
        else:
            multiplicities.append(1)
    return multiplicities


def run_audit(
    *, seed: int = 20260725, directions_per_state: int = 256
) -> DirectionSphereAudit:
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    states = _states(seed)
    directions = _fibonacci_sphere(directions_per_state)
    failures = 0
    activated = 0
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    maximum_reduced_condition = 0.0
    maximum_residual = 0.0
    maximum_reduction_error = 0.0
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_multiplicity = 0
    minimum_families = 25
    worst_state_index = 0
    worst_direction = directions[0]
    first_failure = None
    for state_index, state in enumerate(states):
        for direction in directions:
            audit = master.coupled_characteristic_audit(*state, direction)
            if audit.effective_mass_squared < 0.0:
                activated += 1
            if audit.eigenvector_condition > maximum_condition:
                maximum_condition = audit.eigenvector_condition
                worst_state_index = state_index
                worst_direction = direction.copy()
            maximum_speed = max(maximum_speed, audit.maximum_absolute_speed)
            maximum_imaginary = max(
                maximum_imaginary, audit.maximum_imaginary_speed
            )
            maximum_reduced_condition = max(
                maximum_reduced_condition,
                audit.constraint_reduced_eigenvector_condition,
            )
            maximum_residual = max(
                maximum_residual, audit.maximum_eigenspace_residual
            )
            maximum_reduction_error = max(
                maximum_reduction_error,
                audit.constraint_reduction_spectrum_error,
            )
            minimum_legendre = min(
                minimum_legendre, audit.legendre_minimum_eigenvalue
            )
            minimum_thermo = min(
                minimum_thermo, audit.thermodynamic_minimum_eigenvalue
            )
            multiplicities = _multiplicities(audit.speeds)
            maximum_multiplicity = max(maximum_multiplicity, max(multiplicities))
            minimum_families = min(minimum_families, len(multiplicities))
            if not (audit.strongly_hyperbolic and audit.causal):
                failures += 1
                if first_failure is None:
                    first_failure = {
                        "state_index": state_index,
                        "state": [
                            state[0],
                            state[1],
                            state[2].tolist(),
                            state[3],
                            state[4].tolist(),
                            state[5],
                        ],
                        "direction": direction.tolist(),
                        "rank": audit.eigenvector_rank,
                        "condition": audit.eigenvector_condition,
                        "maximum_speed": audit.maximum_absolute_speed,
                    }
    return DirectionSphereAudit(
        len(states),
        directions_per_state,
        len(states) * directions_per_state,
        failures,
        activated,
        _covering_radius(directions),
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
        maximum_reduced_condition,
        maximum_residual,
        maximum_reduction_error,
        float(minimum_legendre),
        float(minimum_thermo),
        maximum_multiplicity,
        minimum_families,
        worst_state_index,
        tuple(float(value) for value in worst_direction),
        first_failure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--directions-per-state", type=int, default=256)
    arguments = parser.parse_args()
    result = run_audit(
        seed=arguments.seed,
        directions_per_state=arguments.directions_per_state,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
