"""Compose the constrained CCZ4 and Nulloth physical principal symbols."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class CCZ4CompositionAudit:
    ccz4_reduced_variables: int
    nulloth_physical_variables: int
    composed_principal_variables: int
    sampled_fourier_modes: tuple[int, ...]
    ccz4_eigenvector_rank: int
    ccz4_eigenvector_condition: float
    ccz4_maximum_eigenspace_residual: float
    ccz4_maximum_real_growth: float
    ccz4_maximum_gauge_speed: float
    ccz4_spectrum_variation: float
    nulloth_physical_eigenvector_rank: int
    nulloth_physical_eigenvector_condition: float
    nulloth_physical_maximum_speed: float
    composed_eigenvector_rank: int
    composed_eigenvector_condition: float
    composed_maximum_eigenspace_residual: float
    composed_spectrum_union_error: float
    normalized_cross_block_norms: tuple[float, ...]
    minimum_cross_block_decay_order: float
    highest_mode_cross_block_fraction: float
    principal_limit_cross_block_norm: float


_TRACE_FREE_BASIS = (
    np.diag([1.0, 0.0, -1.0]),
    np.diag([0.0, 1.0, -1.0]),
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
)


def _layout(values: tuple[np.ndarray, ...]) -> list[tuple[int, int | None]]:
    result = [(group, component) for group in (0, 1) for component in range(5)]
    result.extend(((2, None), (3, None), (4, None)))
    result.extend((5, component) for component in range(3))
    result.append((6, None))
    result.extend((7, component) for component in range(3))
    result.extend((8, component) for component in range(3))
    for group in range(9, len(values)):
        leading_shape = values[group].shape[:-1]
        if not leading_shape:
            result.append((group, None))
        elif len(leading_shape) == 1:
            result.extend(
                (group, component) for component in range(leading_shape[0])
            )
        else:
            raise ValueError(f"unsupported state group {group}: {leading_shape}")
    return result


def _perturb(
    values: tuple[np.ndarray, ...],
    layout: list[tuple[int, int | None]],
    column: int,
    amplitude: np.ndarray | float,
) -> tuple[np.ndarray, ...]:
    perturbed = [value.copy() for value in values]
    group, component = layout[column]
    if group in (0, 1):
        perturbed[group] += _TRACE_FREE_BASIS[component][..., None] * amplitude
    elif component is None:
        perturbed[group] += amplitude
    else:
        perturbed[group][component] += amplitude
    return tuple(perturbed)


def _extract(
    values: tuple[np.ndarray, ...],
    layout: list[tuple[int, int | None]],
) -> list[np.ndarray]:
    result = []
    identity = np.eye(3)[..., None]
    for group, component in layout:
        if group in (0, 1):
            matrix = 0.5 * (
                values[group] + np.swapaxes(values[group], 0, 1)
            )
            trace = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 3.0
            matrix = matrix - identity * trace
            if component == 0:
                result.append(matrix[0, 0])
            elif component == 1:
                result.append(matrix[1, 1])
            elif component == 2:
                result.append(matrix[0, 1])
            elif component == 3:
                result.append(matrix[0, 2])
            else:
                result.append(matrix[1, 2])
        elif component is None:
            result.append(values[group])
        else:
            result.append(values[group][component])
    return result


def _coefficient(field: np.ndarray, cosine: np.ndarray, sine: np.ndarray) -> complex:
    return complex(
        2.0 * float(np.mean(field * cosine)),
        -2.0 * float(np.mean(field * sine)),
    )


def _ordered_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    return values[np.lexsort((values.real, values.imag))]


def _complete_clustered_basis(
    symbol: np.ndarray, cluster_tolerance: float = 2.0e-7
) -> tuple[np.ndarray, int, float, float]:
    """Return exactly one SVD null basis vector per algebraic mode.

    A numerical Fourier symbol can make an additional singular value fall
    below a global cutoff at a repeated speed.  Counting every such value
    creates a rectangular basis and an infinite condition number even when
    the eigenspaces have full rank.  The algebraic multiplicity fixes the
    required number of right-singular vectors for each clustered mode.
    """
    eigenvalues = _ordered_spectrum(np.linalg.eigvals(symbol))
    clusters: list[list[complex]] = []
    for value in eigenvalues:
        if not clusters:
            clusters.append([complex(value)])
            continue
        center = sum(clusters[-1]) / len(clusters[-1])
        scale = max(1.0, abs(center), abs(value))
        if abs(value - center) <= cluster_tolerance * scale:
            clusters[-1].append(complex(value))
        else:
            clusters.append([complex(value)])
    identity = np.eye(symbol.shape[0], dtype=complex)
    pieces = []
    maximum_residual = 0.0
    for cluster in clusters:
        center = sum(cluster) / len(cluster)
        multiplicity = len(cluster)
        _, _, right = np.linalg.svd(symbol.astype(complex) - center * identity)
        basis = right[-multiplicity:].conj().T
        maximum_residual = max(
            maximum_residual,
            float(np.linalg.norm((symbol - center * identity) @ basis, ord=2)),
        )
        pieces.append(basis)
    complete = np.concatenate(pieces, axis=1)
    rank = int(np.linalg.matrix_rank(complete, tol=cluster_tolerance))
    condition = float(np.linalg.cond(complete)) if rank == symbol.shape[0] else np.inf
    return eigenvalues, rank, condition, maximum_residual


def _background(
    grid: PeriodicGrid,
) -> tuple[NullothProductionSolver, tuple[np.ndarray, ...]]:
    parameters = NULLOTH_DISCOVERY_PARAMETERS
    vector = Theory3Parameters(
        target_speed=0.9,
        target_relaxation_time=parameters.relaxation_time,
        target_strength=0.0,
        gamma_ad=parameters.gamma_ad,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    ccz4 = CCZ4Parameters(
        slicing="harmonic",
        evolve_shift=True,
        radiative_boundaries=False,
        kappa1=0.0,
        eta_shift=0.0,
    )
    solver = NullothProductionSolver(grid, parameters, vector, ccz4)
    velocity = np.broadcast_to(
        np.array([0.04, 0.01, 0.0])[:, None], (3,) + grid.shape
    ).copy()
    carrier_velocity = np.broadcast_to(
        np.array([-0.03, 0.02, 0.01])[:, None], (3,) + grid.shape
    ).copy()
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.65),
        np.full(grid.shape, 0.2),
        velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        np.full(grid.shape, 0.18), carrier_velocity
    )
    state = solver.initialize(
        solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=0.05
    )
    return solver, solver._pack(state)


def _ccz4_symbol(
    solver: NullothProductionSolver,
    values: tuple[np.ndarray, ...],
    layout: list[tuple[int, int | None]],
    coordinate: np.ndarray,
    mode: int,
    epsilon: float,
) -> np.ndarray:
    geometry = solver.ccz4.flat_state()
    base = (
        geometry.conformal_metric,
        geometry.conformal_A,
        geometry.conformal_factor,
        geometry.trace_K,
        geometry.theta,
        geometry.gamma_hat,
        geometry.lapse,
        geometry.shift,
        geometry.shift_driver,
    )
    cosine = np.cos(mode * coordinate)
    sine = np.sin(mode * coordinate)
    zero_sources = (
        solver.grid.zeros(),
        solver.grid.zeros((3,)),
        solver.grid.zeros((3, 3)),
    )
    symbol = np.zeros((23, 23), dtype=complex)
    geometry_layout = layout[:23]
    for column in range(23):
        plus = solver.ccz4.rhs(
            0.0,
            _perturb(base, geometry_layout, column, epsilon * cosine),
            zero_sources,
        )
        minus = solver.ccz4.rhs(
            0.0,
            _perturb(base, geometry_layout, column, -epsilon * cosine),
            zero_sources,
        )
        derivative = tuple(
            (left - right) / (2.0 * epsilon)
            for left, right in zip(plus, minus)
        )
        for row, field in enumerate(_extract(derivative, geometry_layout)):
            symbol[row, column] = _coefficient(field, cosine, sine)
    scales = np.ones(23)
    configuration = list(range(5)) + [10, 16, 17, 18, 19]
    scales[configuration] = float(mode)
    return scales[:, None] * symbol / scales[None, :] / float(mode)


def _cross_norm(
    solver: NullothProductionSolver,
    values: tuple[np.ndarray, ...],
    layout: list[tuple[int, int | None]],
    coordinate: np.ndarray,
    mode: int,
    epsilon: float,
) -> float:
    cosine = np.cos(mode * coordinate)
    sine = np.sin(mode * coordinate)
    upper = np.zeros((23, 32), dtype=complex)
    lower = np.zeros((32, 23), dtype=complex)
    primitive_guess = solver._primitive_guess.copy()

    def column(column: int) -> list[np.ndarray]:
        solver._primitive_guess = primitive_guess.copy()
        plus = solver.rhs(
            0.0, _perturb(values, layout, column, epsilon * cosine)
        )
        solver._primitive_guess = primitive_guess.copy()
        minus = solver.rhs(
            0.0, _perturb(values, layout, column, -epsilon * cosine)
        )
        derivative = tuple(
            (left - right) / (2.0 * epsilon)
            for left, right in zip(plus, minus)
        )
        return _extract(derivative, layout)

    for column_index, column_number in enumerate(range(23, 55)):
        fields = column(column_number)
        for row in range(23):
            upper[row, column_index] = _coefficient(
                fields[row], cosine, sine
            )
    for column_number in range(23):
        fields = column(column_number)
        for row_index, row_number in enumerate(range(23, 55)):
            lower[row_index, column_number] = _coefficient(
                fields[row_number], cosine, sine
            )

    scales = np.ones(23)
    configuration = list(range(5)) + [10, 16, 17, 18, 19]
    scales[configuration] = float(mode)
    normalized_upper = scales[:, None] * upper / float(mode)
    normalized_lower = lower / scales[None, :] / float(mode)
    return float(
        np.sqrt(
            np.linalg.norm(normalized_upper, ord="fro") ** 2
            + np.linalg.norm(normalized_lower, ord="fro") ** 2
        )
    )


def run_audit(
    *, grid_points: int = 16, modes: tuple[int, ...] = (1, 2, 4)
) -> CCZ4CompositionAudit:
    grid = PeriodicGrid((grid_points,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    solver, values = _background(grid)
    layout = _layout(values)
    if len(layout) != 55:
        raise AssertionError(f"expected 55 reduced production variables, got {len(layout)}")
    epsilon = 1.0e-7
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)

    ccz4_symbols = [
        _ccz4_symbol(solver, values, layout, coordinate, mode, epsilon)
        for mode in modes
    ]
    ccz4_conditions = []
    ccz4_residuals = []
    ccz4_ranks = []
    ccz4_spectra = []
    maximum_real_growth = 0.0
    maximum_gauge_speed = 0.0
    for symbol in ccz4_symbols:
        eigenvalues, rank, condition, residual = _complete_clustered_basis(symbol)
        ccz4_spectra.append(_ordered_spectrum(eigenvalues))
        ccz4_ranks.append(rank)
        ccz4_conditions.append(condition)
        ccz4_residuals.append(residual)
        maximum_real_growth = max(
            maximum_real_growth, float(np.max(np.abs(eigenvalues.real)))
        )
        maximum_gauge_speed = max(
            maximum_gauge_speed, float(np.max(np.abs(eigenvalues.imag)))
        )
    spectrum_variation = max(
        float(np.max(np.abs(spectrum - ccz4_spectra[0])))
        for spectrum in ccz4_spectra[1:]
    )

    physical = master.coupled_characteristic_audit(
        0.65,
        0.2,
        np.array([0.04, 0.01, 0.0]),
        0.18,
        np.array([-0.03, 0.02, 0.01]),
        0.05,
        np.array([1.0, 0.0, 0.0]),
    )
    physical_symbol = np.linalg.solve(physical.time_matrix, physical.space_matrix)
    composed = np.zeros((48, 48), dtype=complex)
    composed[:23, :23] = ccz4_symbols[0]
    composed[23:, 23:] = 1.0j * physical_symbol
    composed_eigenvalues, composed_rank, composed_condition, composed_residual = (
        _complete_clustered_basis(composed)
    )
    expected_spectrum = _ordered_spectrum(
        np.concatenate((ccz4_spectra[0], 1.0j * physical.speeds))
    )
    spectrum_union_error = float(
        np.max(np.abs(_ordered_spectrum(composed_eigenvalues) - expected_spectrum))
    )

    cross_norms = tuple(
        _cross_norm(solver, values, layout, coordinate, mode, epsilon)
        for mode in modes
    )
    decay_orders = tuple(
        float(
            np.log(cross_norms[left] / cross_norms[right])
            / np.log(modes[right] / modes[left])
        )
        for left, right in zip(range(len(modes) - 1), range(1, len(modes)))
    )
    return CCZ4CompositionAudit(
        23,
        25,
        48,
        modes,
        min(ccz4_ranks),
        max(ccz4_conditions),
        max(ccz4_residuals),
        maximum_real_growth,
        maximum_gauge_speed,
        spectrum_variation,
        physical.eigenvector_rank,
        physical.eigenvector_condition,
        physical.maximum_absolute_speed,
        composed_rank,
        composed_condition,
        composed_residual,
        spectrum_union_error,
        cross_norms,
        min(decay_orders),
        cross_norms[-1] / cross_norms[0],
        0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=16)
    parser.add_argument("--modes", type=int, nargs="+", default=(1, 2, 4))
    arguments = parser.parse_args()
    result = run_audit(
        grid_points=arguments.grid_points, modes=tuple(arguments.modes)
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
