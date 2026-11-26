"""Locate production and physical denial surfaces outside the qualified box."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class PointClassification:
    physical_pass: bool
    production_pass: bool
    physical_cause: str | None
    production_cause: str | None
    minimum_legendre: float
    minimum_thermodynamic: float
    maximum_speed: float
    maximum_imaginary: float
    maximum_condition: float


@dataclass(frozen=True)
class RayBoundary:
    name: str
    start: tuple[float, ...]
    end: tuple[float, ...]
    first_physical_failure_fraction: float | None
    physical_bracket_width: float | None
    physical_cause: str | None
    physical_boundary_point: tuple[float, ...] | None
    first_production_failure_fraction: float | None
    production_bracket_width: float | None
    production_cause: str | None
    production_boundary_point: tuple[float, ...] | None


@dataclass(frozen=True)
class BoundaryAtlas:
    rays: int
    coarse_steps_per_ray: int
    bisection_steps: int
    safe_starting_points: int
    physically_proscribed_rays: int
    production_proscribed_rays: int
    legendre_failure_rays: int
    thermodynamic_failure_rays: int
    complex_mode_rays: int
    incomplete_basis_rays: int
    ill_conditioned_rays: int
    superluminal_rays: int
    amplitude_cap_rays: int
    relative_lorentz_cap_rays: int
    unclassified_errors: int
    minimum_physical_boundary_fraction: float
    minimum_production_boundary_fraction: float
    maximum_boundary_bracket_width: float
    boundaries: tuple[RayBoundary, ...]


_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
)


def _rays() -> tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]:
    center = (0.65, 0.225, 0.18, 0.0, 0.0)
    rays = []
    endpoints = (
        ("density_low", (0.02, 0.225, 0.18, 0.0, 0.0)),
        ("density_high", (2.0, 0.225, 0.18, 0.0, 0.0)),
        ("internal_low", (0.65, 0.005, 0.18, 0.0, 0.0)),
        ("internal_high", (0.65, 1.5, 0.18, 0.0, 0.0)),
        ("carrier_low", (0.65, 0.225, 0.005, 0.0, 0.0)),
        ("carrier_high", (0.65, 0.225, 0.8, 0.0, 0.0)),
        ("counterflow_negative", (0.65, 0.225, 0.18, -0.9, 0.0)),
        ("counterflow_positive", (0.65, 0.225, 0.18, 0.9, 0.0)),
        ("coherence_negative", (0.65, 0.225, 0.18, 0.0, -0.8)),
        ("coherence_positive", (0.65, 0.225, 0.18, 0.0, 0.8)),
    )
    rays.extend((name, center, endpoint) for name, endpoint in endpoints)
    rays.extend(
        (
            name,
            start,
            end,
        )
        for name, start, end in (
            (
                "convexity_corner_outward",
                (0.9, 0.1, 0.24, -0.3, -0.16),
                (2.0, 0.005, 0.8, -0.9, -0.8),
            ),
            (
                "collision_corner_outward",
                (0.4, 0.1, 0.12, -0.3, -0.16),
                (0.02, 0.005, 0.005, -0.9, -0.8),
            ),
            (
                "condition_corner_outward",
                (0.9, 0.35, 0.24, 0.3, -0.16),
                (2.0, 1.5, 0.8, 0.9, -0.8),
            ),
        )
    )
    return tuple(rays)


def _classify(
    master: NullothMasterFunction, point: tuple[float, ...]
) -> PointClassification:
    density, internal, carrier_density, counterflow, varphi = point
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    physical_cause = None
    incomplete_basis = False
    try:
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
                maximum_condition,
                audit.eigenvector_condition,
                audit.constraint_reduced_eigenvector_condition,
            )
            incomplete_basis |= (
                audit.eigenvector_rank < 25
                or audit.constraint_reduced_eigenvector_rank < 25
            )
        if minimum_legendre <= master.parameters.convexity_floor:
            physical_cause = "legendre"
        elif minimum_thermo <= master.parameters.convexity_floor:
            physical_cause = "thermodynamic"
        elif maximum_speed > 1.0 + 1.0e-8:
            physical_cause = "superluminal"
        elif maximum_imaginary >= 1.0e-8:
            physical_cause = "complex_mode"
        elif incomplete_basis:
            physical_cause = "incomplete_basis"
        elif (
            not np.isfinite(maximum_condition)
            or maximum_condition
            >= master.parameters.characteristic_condition_limit
        ):
            physical_cause = "ill_conditioned"
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        physical_cause = "incomplete_basis"
    relative_lorentz = 1.0 / np.sqrt(1.0 - counterflow**2)
    production_cause = None
    if abs(varphi) > NULLOTH_DISCOVERY_PARAMETERS.scalar_amplitude_limit:
        production_cause = "amplitude_cap"
    elif (
        relative_lorentz
        > NULLOTH_DISCOVERY_PARAMETERS.maximum_relative_lorentz_factor
    ):
        production_cause = "relative_lorentz_cap"
    elif physical_cause is not None:
        production_cause = physical_cause
    return PointClassification(
        physical_cause is None,
        production_cause is None,
        physical_cause,
        production_cause,
        float(minimum_legendre),
        float(minimum_thermo),
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
    )


def _interpolate(
    start: tuple[float, ...], end: tuple[float, ...], fraction: float
) -> tuple[float, ...]:
    left = np.asarray(start)
    right = np.asarray(end)
    return tuple(float(value) for value in left + fraction * (right - left))


def _first_crossing(
    master: NullothMasterFunction,
    start: tuple[float, ...],
    end: tuple[float, ...],
    coarse_steps: int,
    bisection_steps: int,
    attribute: str,
) -> tuple[float | None, float | None, str | None, tuple[float, ...] | None]:
    fractions = np.linspace(0.0, 1.0, coarse_steps)
    previous_fraction = 0.0
    previous_pass = bool(getattr(_classify(master, start), attribute))
    if not previous_pass:
        return 0.0, 0.0, "unsafe_start", start
    failed_fraction = None
    failed_classification = None
    for fraction in fractions[1:]:
        classification = _classify(master, _interpolate(start, end, fraction))
        if not bool(getattr(classification, attribute)):
            failed_fraction = float(fraction)
            failed_classification = classification
            break
        previous_fraction = float(fraction)
    if failed_fraction is None:
        return None, None, None, None
    lower = previous_fraction
    upper = failed_fraction
    for _ in range(bisection_steps):
        midpoint = 0.5 * (lower + upper)
        classification = _classify(master, _interpolate(start, end, midpoint))
        if bool(getattr(classification, attribute)):
            lower = midpoint
        else:
            upper = midpoint
            failed_classification = classification
    cause_attribute = (
        "physical_cause" if attribute == "physical_pass" else "production_cause"
    )
    return (
        upper,
        upper - lower,
        getattr(failed_classification, cause_attribute),
        _interpolate(start, end, upper),
    )


def run_atlas(
    *, coarse_steps: int = 65, bisection_steps: int = 24
) -> BoundaryAtlas:
    parameters = replace(
        NULLOTH_DISCOVERY_PARAMETERS,
        scalar_amplitude_limit=1.0,
        maximum_relative_lorentz_factor=10.0,
    )
    master = NullothMasterFunction(parameters)
    boundaries = []
    safe_starts = 0
    for name, start, end in _rays():
        starting = _classify(master, start)
        safe_starts += int(starting.physical_pass and starting.production_pass)
        physical = _first_crossing(
            master,
            start,
            end,
            coarse_steps,
            bisection_steps,
            "physical_pass",
        )
        production = _first_crossing(
            master,
            start,
            end,
            coarse_steps,
            bisection_steps,
            "production_pass",
        )
        boundaries.append(
            RayBoundary(
                name,
                start,
                end,
                physical[0],
                physical[1],
                physical[2],
                physical[3],
                production[0],
                production[1],
                production[2],
                production[3],
            )
        )
    physical = [
        boundary
        for boundary in boundaries
        if boundary.first_physical_failure_fraction is not None
    ]
    production = [
        boundary
        for boundary in boundaries
        if boundary.first_production_failure_fraction is not None
    ]
    physical_causes = [boundary.physical_cause for boundary in physical]
    production_causes = [boundary.production_cause for boundary in production]
    widths = [
        width
        for boundary in boundaries
        for width in (
            boundary.physical_bracket_width,
            boundary.production_bracket_width,
        )
        if width is not None
    ]
    recognized = {
        "legendre",
        "thermodynamic",
        "complex_mode",
        "incomplete_basis",
        "ill_conditioned",
        "superluminal",
        "amplitude_cap",
        "relative_lorentz_cap",
    }
    causes = physical_causes + production_causes
    return BoundaryAtlas(
        len(boundaries),
        coarse_steps,
        bisection_steps,
        safe_starts,
        len(physical),
        len(production),
        physical_causes.count("legendre"),
        physical_causes.count("thermodynamic"),
        physical_causes.count("complex_mode"),
        physical_causes.count("incomplete_basis"),
        physical_causes.count("ill_conditioned"),
        physical_causes.count("superluminal"),
        production_causes.count("amplitude_cap"),
        production_causes.count("relative_lorentz_cap"),
        sum(cause not in recognized for cause in causes),
        min(boundary.first_physical_failure_fraction for boundary in physical),
        min(boundary.first_production_failure_fraction for boundary in production),
        max(widths),
        tuple(boundaries),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-steps", type=int, default=65)
    parser.add_argument("--bisection-steps", type=int, default=24)
    arguments = parser.parse_args()
    result = run_atlas(
        coarse_steps=arguments.coarse_steps,
        bisection_steps=arguments.bisection_steps,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
