"""Adaptively map Nulloth thermodynamic, counterflow, and coherence margins."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class StateResult:
    point: tuple[float, ...]
    failed: bool
    activated: bool
    maximum_speed: float
    maximum_imaginary_speed: float
    maximum_condition: float
    maximum_reduced_condition: float
    maximum_eigenspace_residual: float
    maximum_reduction_error: float
    minimum_legendre: float
    minimum_thermodynamic: float
    minimum_distinct_mode_gap: float
    maximum_characteristic_multiplicity: int
    effective_mass_squared: float
    hazard: float


@dataclass(frozen=True)
class AdaptiveStateAtlas:
    dimensions: int
    boundary_states: int
    initial_cells: int
    refinements: int
    adaptive_states: int
    sampled_states: int
    directions_per_state: int
    symbol_evaluations: int
    maximum_refinement_depth: int
    failures: int
    activated_states: int
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    maximum_eigenvector_condition: float
    maximum_constraint_reduced_condition: float
    maximum_eigenspace_residual: float
    maximum_reduction_spectrum_error: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    minimum_distinct_mode_gap: float
    maximum_characteristic_multiplicity: int
    minimum_effective_mass_squared: float
    maximum_effective_mass_squared: float
    worst_condition_point: tuple[float, ...]
    narrowest_convexity_point: tuple[float, ...]
    nearest_mode_collision_point: tuple[float, ...]
    first_failure: dict[str, object] | None


@dataclass(frozen=True)
class Cell:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    depth: int
    result: StateResult


_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
)


def _mode_structure(speeds: np.ndarray) -> tuple[float, int]:
    ordered = np.sort(np.asarray(speeds, dtype=float))
    clusters = [[float(ordered[0])]]
    for speed in ordered[1:]:
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(speed - center) <= 1.0e-7 * max(1.0, abs(center), abs(speed)):
            clusters[-1].append(float(speed))
        else:
            clusters.append([float(speed)])
    centers = np.array([sum(cluster) / len(cluster) for cluster in clusters])
    gap = float(np.min(np.diff(centers))) if len(centers) > 1 else np.inf
    return gap, max(len(cluster) for cluster in clusters)


def _evaluate(
    master: NullothMasterFunction, point: tuple[float, ...]
) -> StateResult:
    density, internal, carrier_density, counterflow, varphi = point
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    maximum_reduced_condition = 0.0
    maximum_residual = 0.0
    maximum_reduction_error = 0.0
    minimum_gap = np.inf
    maximum_multiplicity = 0
    failed = False
    effective_mass = None
    for direction in _DIRECTIONS:
        audit = master.coupled_characteristic_audit(
            density,
            internal,
            np.zeros(3),
            carrier_density,
            np.array([counterflow, 0.0, 0.0]),
            varphi,
            direction,
        )
        gap, multiplicity = _mode_structure(audit.speeds)
        minimum_gap = min(minimum_gap, gap)
        maximum_multiplicity = max(maximum_multiplicity, multiplicity)
        minimum_legendre = min(
            minimum_legendre, audit.legendre_minimum_eigenvalue
        )
        minimum_thermo = min(
            minimum_thermo, audit.thermodynamic_minimum_eigenvalue
        )
        maximum_speed = max(maximum_speed, audit.maximum_absolute_speed)
        maximum_imaginary = max(
            maximum_imaginary, audit.maximum_imaginary_speed
        )
        maximum_condition = max(
            maximum_condition, audit.eigenvector_condition
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
        effective_mass = audit.effective_mass_squared
        failed |= not (audit.strongly_hyperbolic and audit.causal)
    convexity = min(minimum_legendre, minimum_thermo)
    hazard = max(
        1.0 / max(convexity, 1.0e-12),
        maximum_condition,
        0.02 / max(minimum_gap, 1.0e-12),
    )
    return StateResult(
        point,
        failed,
        effective_mass < 0.0,
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
        maximum_reduced_condition,
        maximum_residual,
        maximum_reduction_error,
        minimum_legendre,
        minimum_thermo,
        minimum_gap,
        maximum_multiplicity,
        float(effective_mass),
        hazard,
    )


def _split(cell: Cell, master: NullothMasterFunction) -> list[Cell]:
    lower = np.asarray(cell.lower)
    upper = np.asarray(cell.upper)
    midpoint = 0.5 * (lower + upper)
    children = []
    for choices in itertools.product((0, 1), repeat=len(lower)):
        child_lower = np.where(choices, midpoint, lower)
        child_upper = np.where(choices, upper, midpoint)
        center = tuple(float(value) for value in 0.5 * (child_lower + child_upper))
        children.append(
            Cell(
                tuple(child_lower),
                tuple(child_upper),
                cell.depth + 1,
                _evaluate(master, center),
            )
        )
    return children


def run_atlas(*, refinements: int = 24) -> AdaptiveStateAtlas:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    master = NullothMasterFunction(parameters)
    lower = np.array([0.4, 0.1, 0.12, -0.3, -0.16])
    upper = np.array([0.9, 0.35, 0.24, 0.3, 0.16])
    root = Cell(
        tuple(lower),
        tuple(upper),
        0,
        _evaluate(master, tuple(0.5 * (lower + upper))),
    )
    initial_cells = _split(root, master)
    evaluated = [cell.result for cell in initial_cells]
    queue = []
    serial = 0
    for cell in initial_cells:
        heapq.heappush(queue, (-cell.result.hazard, serial, cell))
        serial += 1
    maximum_depth = 1
    for _ in range(refinements):
        _, _, selected = heapq.heappop(queue)
        children = _split(selected, master)
        maximum_depth = max(maximum_depth, children[0].depth)
        for child in children:
            evaluated.append(child.result)
            heapq.heappush(queue, (-child.result.hazard, serial, child))
            serial += 1

    boundary = []
    for choices in itertools.product((0, 1), repeat=5):
        point = tuple(float(value) for value in np.where(choices, upper, lower))
        boundary.append(_evaluate(master, point))
    results = boundary + evaluated
    worst_condition = max(results, key=lambda result: result.maximum_condition)
    narrowest_convexity = min(
        results,
        key=lambda result: min(
            result.minimum_legendre, result.minimum_thermodynamic
        ),
    )
    nearest_collision = min(
        results, key=lambda result: result.minimum_distinct_mode_gap
    )
    first_failed = next((result for result in results if result.failed), None)
    return AdaptiveStateAtlas(
        5,
        len(boundary),
        len(initial_cells),
        refinements,
        len(evaluated),
        len(results),
        len(_DIRECTIONS),
        len(results) * len(_DIRECTIONS),
        maximum_depth,
        sum(result.failed for result in results),
        sum(result.activated for result in results),
        max(result.maximum_speed for result in results),
        max(result.maximum_imaginary_speed for result in results),
        max(result.maximum_condition for result in results),
        max(result.maximum_reduced_condition for result in results),
        max(result.maximum_eigenspace_residual for result in results),
        max(result.maximum_reduction_error for result in results),
        min(result.minimum_legendre for result in results),
        min(result.minimum_thermodynamic for result in results),
        min(result.minimum_distinct_mode_gap for result in results),
        max(result.maximum_characteristic_multiplicity for result in results),
        min(result.effective_mass_squared for result in results),
        max(result.effective_mass_squared for result in results),
        worst_condition.point,
        narrowest_convexity.point,
        nearest_collision.point,
        None
        if first_failed is None
        else {
            "point": first_failed.point,
            "condition": first_failed.maximum_condition,
            "minimum_legendre": first_failed.minimum_legendre,
            "minimum_thermodynamic": first_failed.minimum_thermodynamic,
            "maximum_speed": first_failed.maximum_speed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, default=24)
    arguments = parser.parse_args()
    print(
        json.dumps(
            asdict(run_atlas(refinements=arguments.refinements)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
