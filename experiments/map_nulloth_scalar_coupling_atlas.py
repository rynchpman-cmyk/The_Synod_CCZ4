"""Adaptively map Nulloth's three scalar couplings and coherence amplitude."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class CouplingResult:
    point: tuple[float, ...]
    qualified: bool
    activated_probes: int
    failure_cause: str | None
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_speed: float
    maximum_imaginary: float
    maximum_condition: float
    maximum_reduced_condition: float
    maximum_eigenspace_residual: float
    hazard: float


@dataclass(frozen=True)
class ScalarCouplingAtlas:
    coupling_dimensions: int
    state_probes: int
    directions_per_probe: int
    initial_cells: int
    refinements: int
    adaptive_points: int
    corner_points: int
    coupling_points: int
    symbol_evaluations: int
    maximum_refinement_depth: int
    qualified_points: int
    proscribed_points: int
    activated_points: int
    canonical_point_failures: int
    zero_coupling_point_failures: int
    legendre_failure_points: int
    thermodynamic_failure_points: int
    superluminal_failure_points: int
    complex_mode_failure_points: int
    incomplete_or_condition_failure_points: int
    unclassified_failures: int
    qualified_maximum_speed: float
    qualified_maximum_condition: float
    qualified_maximum_reduced_condition: float
    qualified_maximum_eigenspace_residual: float
    qualified_minimum_legendre: float
    qualified_minimum_thermodynamic: float
    nearest_qualified_point_to_failure: tuple[float, ...]
    least_severe_proscribed_point: tuple[float, ...]


@dataclass(frozen=True)
class Cell:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    depth: int
    result: CouplingResult


_STATE_PROBES = (
    (0.65, 0.225, 0.18, 0.0),
    (0.9, 0.1, 0.24, -0.3),
    (0.4, 0.1, 0.12, -0.3),
    (0.9, 0.35, 0.24, 0.3),
    (0.4, 0.35, 0.24, 0.3),
)

_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
)


def _evaluate(point: tuple[float, ...]) -> CouplingResult:
    baryon, carrier, counterflow_coupling, amplitude = point
    parameters = replace(
        NULLOTH_DISCOVERY_PARAMETERS,
        scalar_baryon_coupling=baryon,
        scalar_carrier_coupling=carrier,
        scalar_counterflow_coupling=counterflow_coupling,
        scalar_amplitude_limit=0.18,
    )
    master = NullothMasterFunction(parameters)
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    maximum_reduced_condition = 0.0
    maximum_residual = 0.0
    activated = 0
    failure_cause = None
    for density, internal, carrier_density, counterflow in _STATE_PROBES:
        probe_activated = False
        for direction in _DIRECTIONS:
            try:
                audit = master.coupled_characteristic_audit(
                    density,
                    internal,
                    np.zeros(3),
                    carrier_density,
                    np.array([counterflow, 0.0, 0.0]),
                    amplitude,
                    direction,
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                failure_cause = failure_cause or "incomplete_or_condition"
                continue
            probe_activated |= audit.effective_mass_squared < 0.0
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
            if failure_cause is None:
                if audit.legendre_minimum_eigenvalue <= parameters.convexity_floor:
                    failure_cause = "legendre"
                elif (
                    audit.thermodynamic_minimum_eigenvalue
                    <= parameters.convexity_floor
                ):
                    failure_cause = "thermodynamic"
                elif audit.maximum_absolute_speed > 1.0 + 1.0e-8:
                    failure_cause = "superluminal"
                elif audit.maximum_imaginary_speed >= 1.0e-8:
                    failure_cause = "complex_mode"
                elif (
                    not audit.complete_eigenbasis
                    or audit.eigenvector_condition
                    >= parameters.characteristic_condition_limit
                    or audit.constraint_reduced_eigenvector_condition
                    >= parameters.characteristic_condition_limit
                ):
                    failure_cause = "incomplete_or_condition"
        activated += int(probe_activated)
    qualified = failure_cause is None
    convexity = min(minimum_legendre, minimum_thermo)
    if qualified:
        hazard = max(
            1.0 / max(convexity, 1.0e-12),
            maximum_condition,
        )
    else:
        # Failed points remain competitive for refinement, but safe points
        # near a boundary receive comparable priority.
        severity = max(-convexity, maximum_imaginary, maximum_speed - 1.0, 0.0)
        hazard = 50.0 / (1.0 + 100.0 * severity)
    return CouplingResult(
        point,
        qualified,
        activated,
        failure_cause,
        float(minimum_legendre),
        float(minimum_thermo),
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
        maximum_reduced_condition,
        maximum_residual,
        hazard,
    )


def _split(cell: Cell) -> list[Cell]:
    lower = np.asarray(cell.lower)
    upper = np.asarray(cell.upper)
    midpoint = 0.5 * (lower + upper)
    children = []
    for choices in itertools.product((0, 1), repeat=4):
        child_lower = np.where(choices, midpoint, lower)
        child_upper = np.where(choices, upper, midpoint)
        point = tuple(float(value) for value in 0.5 * (child_lower + child_upper))
        children.append(
            Cell(
                tuple(child_lower),
                tuple(child_upper),
                cell.depth + 1,
                _evaluate(point),
            )
        )
    return children


def run_atlas(*, refinements: int = 32) -> ScalarCouplingAtlas:
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([6.0, 20.0, 16.0, 0.17])
    root = Cell(
        tuple(lower),
        tuple(upper),
        0,
        _evaluate(tuple(0.5 * (lower + upper))),
    )
    initial = _split(root)
    adaptive = [cell.result for cell in initial]
    queue = []
    serial = 0
    for cell in initial:
        heapq.heappush(queue, (-cell.result.hazard, serial, cell))
        serial += 1
    maximum_depth = 1
    for _ in range(refinements):
        _, _, selected = heapq.heappop(queue)
        children = _split(selected)
        maximum_depth = max(maximum_depth, children[0].depth)
        for child in children:
            adaptive.append(child.result)
            heapq.heappush(queue, (-child.result.hazard, serial, child))
            serial += 1

    corners = [
        _evaluate(tuple(float(value) for value in np.where(bits, upper, lower)))
        for bits in itertools.product((0, 1), repeat=4)
    ]
    canonical = _evaluate((0.0, 12.0, 0.2, 0.16))
    zero = _evaluate((0.0, 0.0, 0.0, 0.17))
    unique = {
        result.point: result
        for result in adaptive + corners + [canonical, zero]
    }
    results = list(unique.values())
    qualified = [result for result in results if result.qualified]
    failed = [result for result in results if not result.qualified]
    causes = [result.failure_cause for result in failed]
    nearest_safe = max(qualified, key=lambda result: result.hazard)
    least_failed = max(failed, key=lambda result: result.hazard)
    return ScalarCouplingAtlas(
        4,
        len(_STATE_PROBES),
        len(_DIRECTIONS),
        len(initial),
        refinements,
        len(adaptive),
        len(corners),
        len(results),
        len(results) * len(_STATE_PROBES) * len(_DIRECTIONS),
        maximum_depth,
        len(qualified),
        len(failed),
        sum(result.activated_probes > 0 for result in results),
        int(not canonical.qualified),
        int(not zero.qualified),
        causes.count("legendre"),
        causes.count("thermodynamic"),
        causes.count("superluminal"),
        causes.count("complex_mode"),
        causes.count("incomplete_or_condition"),
        sum(
            cause
            not in {
                "legendre",
                "thermodynamic",
                "superluminal",
                "complex_mode",
                "incomplete_or_condition",
            }
            for cause in causes
        ),
        max(result.maximum_speed for result in qualified),
        max(result.maximum_condition for result in qualified),
        max(result.maximum_reduced_condition for result in qualified),
        max(result.maximum_eigenspace_residual for result in qualified),
        min(result.minimum_legendre for result in qualified),
        min(result.minimum_thermodynamic for result in qualified),
        nearest_safe.point,
        least_failed.point,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, default=32)
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
