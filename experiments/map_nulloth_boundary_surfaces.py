"""Map high-precision convexity, causal, and nonhyperbolic Nulloth surfaces."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace

import mpmath as mp
import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class CoherenceSurfacePoint:
    carrier_density: float
    condition_denial_amplitude: float
    thermodynamic_zero_amplitude: float
    legendre_zero_amplitude: float
    condition_bracket_width: float


@dataclass(frozen=True)
class CarrierCausalSurfacePoint:
    density: float
    internal: float
    minimum_qualified_carrier_density: float
    bracket_width: float


@dataclass(frozen=True)
class BoundarySurfaceAtlas:
    precision_decimal_digits: int
    coherence_surface_points: int
    carrier_causal_surface_points: int
    carrier_coherence_grid_cells: int
    counterflow_coherence_grid_cells: int
    safe_grid_cells: int
    superluminal_grid_cells: int
    complex_or_ill_conditioned_grid_cells: int
    thermodynamic_grid_cells: int
    legendre_grid_cells: int
    unclassified_grid_cells: int
    failed_region_components: int
    production_to_condition_order_violations: int
    condition_to_thermodynamic_order_violations: int
    thermodynamic_to_legendre_order_violations: int
    missing_carrier_causal_crossings: int
    maximum_characteristic_bracket_width: float
    minimum_condition_margin_above_production_cap: float
    minimum_thermodynamic_to_legendre_gap: float
    minimum_qualified_carrier_density: float
    maximum_qualified_carrier_density: float
    coherence_surface: tuple[CoherenceSurfacePoint, ...]
    carrier_causal_surface: tuple[CarrierCausalSurfacePoint, ...]


_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 0.0]),
    np.array([1.0, 1.0, 1.0]),
)


def _positive_root(function, upper: mp.mpf = mp.mpf("1")) -> mp.mpf:
    lower = mp.mpf("0")
    for index in range(1, 2001):
        candidate = upper * index / 2000
        if function(candidate) <= 0:
            upper = candidate
            break
        lower = candidate
    else:
        raise ValueError("no determinant zero in root interval")
    for _ in range(256):
        midpoint = (lower + upper) / 2
        if function(midpoint) > 0:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _physical_pass(
    master: NullothMasterFunction,
    density: float,
    internal: float,
    carrier_density: float,
    varphi: float,
) -> bool:
    try:
        for direction in _DIRECTIONS:
            audit = master.coupled_characteristic_audit(
                density,
                internal,
                np.zeros(3),
                carrier_density,
                np.zeros(3),
                varphi,
                direction,
            )
            if not (audit.strongly_hyperbolic and audit.causal):
                return False
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return False
    return True


def _bisect_transition(
    function, lower: float, upper: float, iterations: int = 32
) -> tuple[float, float]:
    if not function(lower) or function(upper):
        raise ValueError("transition bracket does not contain pass-to-fail")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if function(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return upper, upper - lower


def _bisect_fail_to_pass(
    function, lower: float, upper: float, iterations: int = 32
) -> tuple[float, float]:
    if function(lower) or not function(upper):
        raise ValueError("transition bracket does not contain fail-to-pass")
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if function(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return upper, upper - lower


def _coherence_surface(
    master: NullothMasterFunction, points: int
) -> tuple[CoherenceSurfacePoint, ...]:
    p = NULLOTH_DISCOVERY_PARAMETERS
    gamma = mp.mpf(str(p.gamma_ad))
    target = mp.mpf(str(p.target_fraction))
    susceptibility = mp.mpf(str(p.chemical_susceptibility))
    carrier_k = mp.mpf(str(p.carrier_K))
    cs = mp.mpf(str(p.carrier_sound_speed))
    cs2 = cs**2
    scalar_n = mp.mpf(str(p.scalar_baryon_coupling))
    scalar_d = mp.mpf(str(p.scalar_carrier_coupling))
    scalar_w = mp.mpf(str(p.scalar_counterflow_coupling))
    n = mp.mpf("0.65")
    internal = mp.mpf("0.225")
    thermo_nn_base = (
        gamma * (gamma - 1) * internal / n
        + target**2 / susceptibility
    )
    result = []
    for carrier_density in np.linspace(0.12, 0.24, points):
        d = mp.mpf(str(float(carrier_density)))
        thermo_dd_base = (
            carrier_k * (1 + cs2) * cs2 * d ** (cs2 - 1)
            + 1 / susceptibility
        )

        def thermo_determinant(phi: mp.mpf) -> mp.mpf:
            nn = thermo_nn_base - scalar_n * phi**2
            dd = thermo_dd_base - scalar_d * phi**2
            return nn * dd - (target / susceptibility) ** 2

        anchor = d - target * n

        def legendre_determinant(phi: mp.mpf) -> mp.mpf:
            q = -mp.mpf("0.5") * scalar_w * phi**2
            rho_n = (
                1
                + gamma * internal
                - target * anchor / susceptibility
                - scalar_n * phi**2 * n
            )
            rho_d = (
                carrier_k * (1 + cs2) * d**cs2
                + anchor / susceptibility
                - scalar_d * phi**2 * d
            )
            b_n = (rho_n - q * d) / n
            b_d = (rho_d - q * n) / d
            return b_n * b_d - q**2

        thermo_root = _positive_root(thermo_determinant)
        legendre_root = _positive_root(legendre_determinant)
        upper = float(thermo_root * mp.mpf("1.001"))
        condition, width = _bisect_transition(
            lambda phi: _physical_pass(
                master, 0.65, 0.225, float(carrier_density), phi
            ),
            0.17,
            upper,
        )
        result.append(
            CoherenceSurfacePoint(
                float(carrier_density),
                condition,
                float(thermo_root),
                float(legendre_root),
                width,
            )
        )
    return tuple(result)


def _carrier_surface(
    master: NullothMasterFunction, thermodynamic_points: int
) -> tuple[CarrierCausalSurfacePoint, ...]:
    result = []
    for density in np.linspace(0.4, 0.9, thermodynamic_points):
        for internal in np.linspace(0.1, 0.35, thermodynamic_points):
            boundary, width = _bisect_fail_to_pass(
                lambda carrier: _physical_pass(
                    master, float(density), float(internal), carrier, 0.0
                ),
                0.005,
                0.12,
            )
            result.append(
                CarrierCausalSurfacePoint(
                    float(density), float(internal), boundary, width
                )
            )
    return tuple(result)


def _grid_cause(
    master: NullothMasterFunction,
    carrier_density: float,
    varphi: float,
    *,
    density: float = 0.65,
    internal: float = 0.225,
    counterflow: float = 0.0,
) -> str:
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    complete = True
    try:
        for direction in _DIRECTIONS[:2]:
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
            complete &= audit.complete_eigenbasis
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return "complex_or_ill_conditioned"
    if minimum_legendre <= master.parameters.convexity_floor:
        return "legendre"
    if minimum_thermo <= master.parameters.convexity_floor:
        return "thermodynamic"
    if maximum_speed > 1.0 + 1.0e-8:
        return "superluminal"
    if (
        maximum_imaginary >= 1.0e-8
        or not complete
        or not np.isfinite(maximum_condition)
        or maximum_condition >= master.parameters.characteristic_condition_limit
    ):
        return "complex_or_ill_conditioned"
    return "safe"


def _components(failed: np.ndarray) -> int:
    visited = np.zeros_like(failed, dtype=bool)
    components = 0
    for row in range(failed.shape[0]):
        for column in range(failed.shape[1]):
            if not failed[row, column] or visited[row, column]:
                continue
            components += 1
            stack = [(row, column)]
            visited[row, column] = True
            while stack:
                current_row, current_column = stack.pop()
                for offset_row, offset_column in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    neighbor_row = current_row + offset_row
                    neighbor_column = current_column + offset_column
                    if (
                        0 <= neighbor_row < failed.shape[0]
                        and 0 <= neighbor_column < failed.shape[1]
                        and failed[neighbor_row, neighbor_column]
                        and not visited[neighbor_row, neighbor_column]
                    ):
                        visited[neighbor_row, neighbor_column] = True
                        stack.append((neighbor_row, neighbor_column))
    return components


def run_atlas(
    *, coherence_points: int = 17, thermodynamic_points: int = 5, grid_points: int = 31
) -> BoundarySurfaceAtlas:
    mp.mp.dps = 80
    parameters = replace(
        NULLOTH_DISCOVERY_PARAMETERS,
        scalar_amplitude_limit=0.4,
        maximum_relative_lorentz_factor=10.0,
    )
    master = NullothMasterFunction(parameters)
    coherence = _coherence_surface(master, coherence_points)
    carrier = _carrier_surface(master, thermodynamic_points)
    causes = np.empty((grid_points, grid_points), dtype=object)
    carrier_values = np.linspace(0.01, 0.3, grid_points)
    amplitude_values = np.linspace(0.0, 0.26, grid_points)
    for row, carrier_density in enumerate(carrier_values):
        for column, amplitude in enumerate(amplitude_values):
            causes[row, column] = _grid_cause(
                master, float(carrier_density), float(amplitude)
            )
    counterflow_causes = np.empty((grid_points, grid_points), dtype=object)
    counterflow_values = np.linspace(-0.6, 0.6, grid_points)
    for row, counterflow in enumerate(counterflow_values):
        for column, amplitude in enumerate(amplitude_values):
            counterflow_causes[row, column] = _grid_cause(
                master,
                0.24,
                float(amplitude),
                density=0.9,
                internal=0.1,
                counterflow=float(counterflow),
            )
    failed = causes != "safe"
    counterflow_failed = counterflow_causes != "safe"
    combined = np.concatenate((causes.ravel(), counterflow_causes.ravel()))
    production_cap = NULLOTH_DISCOVERY_PARAMETERS.scalar_amplitude_limit
    widths = [point.condition_bracket_width for point in coherence]
    widths.extend(point.bracket_width for point in carrier)
    return BoundarySurfaceAtlas(
        80,
        len(coherence),
        len(carrier),
        causes.size,
        counterflow_causes.size,
        int(np.count_nonzero(combined == "safe")),
        int(np.count_nonzero(combined == "superluminal")),
        int(np.count_nonzero(combined == "complex_or_ill_conditioned")),
        int(np.count_nonzero(combined == "thermodynamic")),
        int(np.count_nonzero(combined == "legendre")),
        int(
            np.count_nonzero(
                ~np.isin(
                    combined,
                    (
                        "safe",
                        "superluminal",
                        "complex_or_ill_conditioned",
                        "thermodynamic",
                        "legendre",
                    ),
                )
            )
        ),
        _components(failed) + _components(counterflow_failed),
        sum(
            point.condition_denial_amplitude <= production_cap
            for point in coherence
        ),
        sum(
            point.thermodynamic_zero_amplitude
            <= point.condition_denial_amplitude
            for point in coherence
        ),
        sum(
            point.legendre_zero_amplitude
            <= point.thermodynamic_zero_amplitude
            for point in coherence
        ),
        len(carrier) - thermodynamic_points**2,
        max(widths),
        min(
            point.condition_denial_amplitude - production_cap
            for point in coherence
        ),
        min(
            point.legendre_zero_amplitude
            - point.thermodynamic_zero_amplitude
            for point in coherence
        ),
        min(point.minimum_qualified_carrier_density for point in carrier),
        max(point.minimum_qualified_carrier_density for point in carrier),
        coherence,
        carrier,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coherence-points", type=int, default=17)
    parser.add_argument("--thermodynamic-points", type=int, default=5)
    parser.add_argument("--grid-points", type=int, default=31)
    arguments = parser.parse_args()
    result = run_atlas(
        coherence_points=arguments.coherence_points,
        thermodynamic_points=arguments.thermodynamic_points,
        grid_points=arguments.grid_points,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
