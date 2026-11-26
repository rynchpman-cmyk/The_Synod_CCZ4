"""Map Nulloth linear growth over the qualified state atlas and in 3D."""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS, NullothMasterFunction
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from causal_repair.theory33 import CarrierPrimitive

try:
    from experiments.map_nulloth_adaptive_state_atlas import Cell, _evaluate, _split
except ImportError:  # Direct experiment execution.
    from map_nulloth_adaptive_state_atlas import Cell, _evaluate, _split


@dataclass(frozen=True)
class GrowthAtlas:
    sampled_states: int
    characteristic_failures: int
    locally_growing_states: int
    locally_oscillatory_states: int
    maximum_growth_rate: float
    minimum_scalar_curvature: float
    maximum_scalar_curvature: float
    maximum_condensed_frequency: float
    box_lengths: tuple[float, ...]
    states_with_nonhomogeneous_instability: tuple[int, ...]
    maximum_unstable_mode_number: tuple[int, ...]
    multidimensional_resolutions: tuple[int, ...]
    tested_wavevectors: tuple[tuple[int, ...], ...]
    maximum_multidimensional_dispersion_error: float
    maximum_directional_growth_difference: float
    candidate_mechanisms_registered: int


def _atlas_results(refinements: int):
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    lower = np.array([0.4, 0.1, 0.12, -0.3, -0.16])
    upper = np.array([0.9, 0.35, 0.24, 0.3, 0.16])
    root = Cell(
        tuple(lower),
        tuple(upper),
        0,
        _evaluate(master, tuple(0.5 * (lower + upper))),
    )
    initial = _split(root, master)
    evaluated = [cell.result for cell in initial]
    queue = []
    serial = 0
    for cell in initial:
        heapq.heappush(queue, (-cell.result.hazard, serial, cell))
        serial += 1
    for _ in range(refinements):
        _, _, selected = heapq.heappop(queue)
        children = _split(selected, master)
        for child in children:
            evaluated.append(child.result)
            heapq.heappush(queue, (-child.result.hazard, serial, child))
            serial += 1
    boundary = []
    for choices in itertools.product((0, 1), repeat=5):
        point = tuple(float(value) for value in np.where(choices, upper, lower))
        boundary.append(_evaluate(master, point))
    return boundary + evaluated


def _multidimensional_scalar_spectrum(
    points: int, wavevector: tuple[int, ...]
) -> np.ndarray:
    dimensions = len(wavevector)
    grid = PeriodicGrid((points,) * dimensions, (32.0,) * dimensions, method="spectral")
    coordinates = grid.coordinates()
    phase = sum(
        2.0 * np.pi * mode * coordinate / 32.0
        for mode, coordinate in zip(wavevector, coordinates)
    )
    cosine = np.cos(phase)
    sine = np.sin(phase)
    metric = grid.zeros((3, 3))
    for component in range(3):
        metric[component, component] = 1.0
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(np.full(grid.shape, 0.18), grid.zeros((3,)))
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    system = NullothScalarSystem(grid)
    epsilon = 1.0e-7
    matrix = np.zeros((5, 5), dtype=complex)

    def rhs(column: int, sign: float):
        phi = grid.zeros()
        momentum = grid.zeros()
        gradient = grid.zeros((3,))
        perturbation = sign * epsilon * cosine
        if column == 0:
            phi += perturbation
        elif column == 1:
            momentum += perturbation
        else:
            gradient[column - 2] += perturbation
        state = NullothScalarState(phi, momentum, gradient)
        source = master.evaluate(metric, fluid, carrier, phi).scalar_equation_source
        return system.rhs(
            metric,
            grid.zeros(),
            np.ones(grid.shape),
            grid.zeros((3,)),
            state,
            source,
        )

    for column in range(5):
        plus = rhs(column, 1.0)
        minus = rhs(column, -1.0)
        rows = (
            (plus[0] - minus[0]) / (2.0 * epsilon),
            (plus[1] - minus[1]) / (2.0 * epsilon),
            *((plus[2][axis] - minus[2][axis]) / (2.0 * epsilon) for axis in range(3)),
        )
        matrix[:, column] = [
            complex(2.0 * np.mean(row * cosine), -2.0 * np.mean(row * sine))
            for row in rows
        ]
    return np.linalg.eigvals(matrix)


def _ordered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    return values[
        np.lexsort((np.round(values.real, 10), np.round(values.imag, 10)))
    ]


def _expected_scalar(wavevector: tuple[int, ...]) -> np.ndarray:
    curvature = 0.4**2 - 12.0 * 0.18**2
    wavenumber_squared = sum((2.0 * np.pi * mode / 32.0) ** 2 for mode in wavevector)
    root = np.sqrt(complex(-(wavenumber_squared + curvature)))
    return _ordered(np.array((-root, 0.0, 0.0, 0.0, root), dtype=complex))


def run_atlas(
    *, refinements: int = 24, resolutions: tuple[int, ...] = (8, 12, 16)
) -> GrowthAtlas:
    if len(resolutions) < 3:
        raise ValueError("the multidimensional audit requires three resolutions")
    results = _atlas_results(refinements)
    quartic = NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
    curvatures = np.array(
        [result.effective_mass_squared + 3.0 * quartic * result.point[4] ** 2 for result in results]
    )
    growing = curvatures < 0.0
    growth_rates = np.sqrt(np.maximum(-curvatures, 0.0))
    condensed_frequencies = np.sqrt(
        np.maximum(-2.0 * np.array([result.effective_mass_squared for result in results]), 0.0)
    )
    lengths = (8.0, 16.0, 32.0, 64.0)
    nonhomogeneous_counts = []
    maximum_modes = []
    for length in lengths:
        thresholds = length * growth_rates / (2.0 * np.pi)
        maximum = np.maximum(np.ceil(thresholds).astype(int) - 1, 0)
        nonhomogeneous_counts.append(int(np.count_nonzero(maximum >= 1)))
        maximum_modes.append(int(np.max(maximum)))

    wavevectors = ((1,), (1, 1), (1, 1, 1))
    multidimensional_error = 0.0
    directional_rates = []
    for resolution in resolutions:
        for wavevector in wavevectors:
            numerical = _ordered(
                _multidimensional_scalar_spectrum(resolution, wavevector)
            )
            expected = _expected_scalar(wavevector)
            multidimensional_error = max(
                multidimensional_error,
                float(np.max(np.abs(numerical - expected))),
            )
            if resolution == resolutions[-1]:
                directional_rates.append(float(np.max(numerical.real)))

    normalized_rates = []
    curvature = 0.4**2 - 12.0 * 0.18**2
    for wavevector, rate in zip(wavevectors, directional_rates):
        k_squared = sum((2.0 * np.pi * mode / 32.0) ** 2 for mode in wavevector)
        normalized_rates.append(rate**2 + k_squared + curvature)
    directional_difference = float(np.max(np.abs(normalized_rates)))
    return GrowthAtlas(
        len(results),
        sum(result.failed for result in results),
        int(np.count_nonzero(growing)),
        int(np.count_nonzero(~growing)),
        float(np.max(growth_rates)),
        float(np.min(curvatures)),
        float(np.max(curvatures)),
        float(np.max(condensed_frequencies)),
        lengths,
        tuple(nonhomogeneous_counts),
        tuple(maximum_modes),
        resolutions,
        wavevectors,
        multidimensional_error,
        directional_difference,
        3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, default=24)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(8, 12, 16))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_atlas(
        refinements=arguments.refinements,
        resolutions=tuple(arguments.resolutions),
    )
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
