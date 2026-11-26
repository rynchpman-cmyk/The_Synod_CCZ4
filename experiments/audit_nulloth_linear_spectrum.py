"""Audit the branch-resolved linear dispersion relations of Nulloth."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS, NullothMasterFunction
from causal_repair.nulloth_scalar import NullothScalarState, NullothScalarSystem
from causal_repair.theory3 import Theory3Parameters, Theory3System
from causal_repair.theory33 import CarrierPrimitive
try:
    from experiments.audit_nulloth_ccz4_composition import (
        _background,
        _ccz4_symbol,
        _layout,
    )
except ImportError:  # Direct experiment execution.
    from audit_nulloth_ccz4_composition import (
        _background,
        _ccz4_symbol,
        _layout,
    )


@dataclass(frozen=True)
class EquilibriumSpectrum:
    name: str
    carrier_density: float
    scalar_amplitude: float
    effective_mass_squared: float
    scalar_curvature: float
    unstable_fourier_modes: int
    maximum_scalar_dispersion_error: float
    acoustic_modes: int
    contact_or_vorticity_modes: int
    minimum_acoustic_speed: float
    maximum_acoustic_speed: float


@dataclass(frozen=True)
class LinearSpectrumAudit:
    equilibria: tuple[EquilibriumSpectrum, ...]
    resolutions: tuple[int, ...]
    scalar_fourier_modes: tuple[int, ...]
    vector_fourier_modes: tuple[int, ...]
    material_modes_per_equilibrium: int
    scalar_physical_modes: int
    scalar_reduction_constraint_modes: int
    proca_physical_modes: int
    proca_cleaning_modes: int
    ccz4_gravitational_tensor_modes: int
    ccz4_gamma_driver_modes: int
    ccz4_gauge_constraint_modes: int
    maximum_vector_dispersion_error: float
    maximum_ccz4_principal_frequency_error: float
    maximum_resolution_spectrum_change: float
    maximum_spurious_real_growth: float
    finite_volume_critical_length: float
    finite_volume_unstable_mode_count: int
    branch_families_identified: int


def _ordered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    rounded_real = np.round(values.real, 10)
    rounded_imag = np.round(values.imag, 10)
    return values[np.lexsort((rounded_real, rounded_imag))]


def _coefficient(
    field: np.ndarray,
    cosine: np.ndarray,
    sine: np.ndarray,
    mode: int,
) -> complex:
    factor = 1.0 if mode == 0 else 2.0
    return complex(
        factor * float(np.mean(field * cosine)),
        -factor * float(np.mean(field * sine)),
    )


def _flat_metric(grid: PeriodicGrid) -> np.ndarray:
    metric = grid.zeros((3, 3))
    for component in range(3):
        metric[component, component] = 1.0
    return metric


def _scalar_generator(
    points: int,
    length: float,
    mode: int,
    carrier_density: float,
    scalar_amplitude: float,
) -> np.ndarray:
    grid = PeriodicGrid((points,), (length,), method="spectral")
    (coordinate,) = grid.coordinates()
    wavenumber = 2.0 * np.pi * mode / length
    cosine = np.ones(grid.shape) if mode == 0 else np.cos(wavenumber * coordinate)
    sine = np.zeros(grid.shape) if mode == 0 else np.sin(wavenumber * coordinate)
    system = NullothScalarSystem(grid)
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    metric = _flat_metric(grid)
    fluid = FluidPrimitive(
        np.full(grid.shape, 0.6),
        np.full(grid.shape, 0.2),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        np.full(grid.shape, carrier_density), grid.zeros((3,))
    )
    lapse = np.ones(grid.shape)
    shift = grid.zeros((3,))
    trace_K = grid.zeros()
    epsilon = 1.0e-7
    matrix = np.zeros((5, 5), dtype=complex)

    def rhs(column: int, sign: float) -> tuple[np.ndarray, ...]:
        phi = np.full(grid.shape, scalar_amplitude)
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
        constitutive = master.evaluate(metric, fluid, carrier, phi)
        return system.rhs(
            metric,
            trace_K,
            lapse,
            shift,
            state,
            constitutive.scalar_equation_source,
        )

    for column in range(5):
        plus = rhs(column, 1.0)
        minus = rhs(column, -1.0)
        fields = (
            (plus[0] - minus[0]) / (2.0 * epsilon),
            (plus[1] - minus[1]) / (2.0 * epsilon),
            *((plus[2][axis] - minus[2][axis]) / (2.0 * epsilon) for axis in range(3)),
        )
        matrix[:, column] = [
            _coefficient(field, cosine, sine, mode) for field in fields
        ]
    return matrix


def _scalar_expected(wavenumber: float, curvature: float) -> np.ndarray:
    frequency_squared = wavenumber**2 + curvature
    root = np.sqrt(complex(-frequency_squared))
    return _ordered(np.array((-root, 0.0, 0.0, 0.0, root), dtype=complex))


_VECTOR_COMPONENTS = (3, 3, 3, 3, 1, 1, 1, 1)


def _vector_generator(points: int, mode: int, damping: float) -> tuple[np.ndarray, Theory3Parameters]:
    grid = PeriodicGrid((points,), (2.0 * np.pi,), method="spectral")
    (coordinate,) = grid.coordinates()
    cosine = np.cos(mode * coordinate)
    sine = np.sin(mode * coordinate)
    parameters = Theory3Parameters(
        target_speed=0.9,
        target_strength=0.0,
        gamma_W=0.0,
        gamma_B=0.0,
    )
    system = Theory3System(grid, parameters)
    metric = _flat_metric(grid)
    zero = grid.zeros()
    zero_vector = grid.zeros((3,))
    lapse = np.ones(grid.shape)
    epsilon = 1.0e-7
    matrix = np.zeros((16, 16), dtype=complex)

    def fields(column: int, sign: float) -> list[np.ndarray]:
        values: list[np.ndarray] = []
        offset = 0
        for components in _VECTOR_COMPONENTS:
            value = zero.copy() if components == 1 else zero_vector.copy()
            if offset <= column < offset + components:
                if components == 1:
                    value += sign * epsilon * cosine
                else:
                    value[column - offset] += sign * epsilon * cosine
            offset += components
            values.append(value)
        return values

    for column in range(16):
        plus = system.rhs_fields(
            metric,
            *fields(column, 1.0),
            zero,
            zero_vector,
            lapse,
            zero_vector,
            damping,
        )
        minus = system.rhs_fields(
            metric,
            *fields(column, -1.0),
            zero,
            zero_vector,
            lapse,
            zero_vector,
            damping,
        )
        rows: list[np.ndarray] = []
        for left, right, components in zip(plus, minus, _VECTOR_COMPONENTS):
            derivative = (left - right) / (2.0 * epsilon)
            rows.extend([derivative] if components == 1 else list(derivative))
        matrix[:, column] = [
            _coefficient(field, cosine, sine, mode) for field in rows
        ]
    return matrix, parameters


def _vector_expected(
    mode: int, damping: float, parameters: Theory3Parameters
) -> np.ndarray:
    values: list[complex] = []
    for mass_squared in np.linalg.eigvals(parameters.gauss_matrix):
        frequency = np.sqrt(mode**2 + float(mass_squared.real))
        values.extend((-1.0j * frequency, 1.0j * frequency) * 3)
    cleaning_root = np.sqrt(complex(damping**2 - 4.0 * mode**2))
    cleaning_pair = (
        (-damping - cleaning_root) / 2.0,
        (-damping + cleaning_root) / 2.0,
    )
    values.extend(cleaning_pair * 2)
    return _ordered(np.asarray(values, dtype=complex))


def _material_speeds(
    master: NullothMasterFunction,
    carrier_density: float,
    scalar_amplitude: float,
) -> np.ndarray:
    audit = master.coupled_characteristic_audit(
        0.6,
        0.2,
        np.zeros(3),
        carrier_density,
        np.zeros(3),
        scalar_amplitude,
        np.array([1.0, 0.0, 0.0]),
    )
    material = np.linalg.solve(
        audit.time_matrix[:8, :8], audit.space_matrix[:8, :8]
    )
    eigenvalues = np.linalg.eigvals(material)
    if np.max(np.abs(eigenvalues.imag)) > 1.0e-8:
        raise AssertionError("material equilibrium has complex characteristic speeds")
    return np.sort(eigenvalues.real)


def run_audit(
    *, resolutions: tuple[int, ...] = (16, 24, 32)
) -> LinearSpectrumAudit:
    if len(resolutions) < 3:
        raise ValueError("the spectrum audit requires at least three resolutions")
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    scalar_modes = (0, 1, 2, 3)
    vector_modes = (1, 2, 3)
    scalar_length = 32.0
    equilibrium_inputs = (
        ("symmetric-stable", 0.10, 0.0),
        ("symmetric-activated", 0.18, 0.0),
        (
            "condensed",
            0.18,
            float(np.sqrt((12.0 * 0.18**2 - 0.4**2) / 50.0)),
        ),
    )
    equilibrium_results: list[EquilibriumSpectrum] = []
    all_resolution_spectra: dict[tuple[str, int, int], np.ndarray] = {}
    maximum_scalar_error = 0.0
    activated_curvature = 0.0

    metric = np.eye(3).reshape(3, 3, 1)
    fluid = FluidPrimitive(
        np.array([0.6]), np.array([0.2]), np.zeros((3, 1)), np.zeros(1)
    )
    for name, carrier_density, amplitude in equilibrium_inputs:
        carrier = CarrierPrimitive(np.array([carrier_density]), np.zeros((3, 1)))
        state = master.evaluate(metric, fluid, carrier, np.array([amplitude]))
        effective_mass_squared = float(state.effective_mass_squared[0])
        curvature = effective_mass_squared + (
            3.0 * NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic * amplitude**2
        )
        if name == "symmetric-activated":
            activated_curvature = curvature
        unstable = 0
        local_error = 0.0
        for points in resolutions:
            for mode in scalar_modes:
                wavenumber = 2.0 * np.pi * mode / scalar_length
                numerical = _ordered(
                    np.linalg.eigvals(
                        _scalar_generator(
                            points,
                            scalar_length,
                            mode,
                            carrier_density,
                            amplitude,
                        )
                    )
                )
                expected = _scalar_expected(wavenumber, curvature)
                error = float(np.max(np.abs(numerical - expected)))
                local_error = max(local_error, error)
                all_resolution_spectra[(name, points, mode)] = numerical
                if points == resolutions[0] and wavenumber**2 + curvature < 0.0:
                    unstable += 1
        maximum_scalar_error = max(maximum_scalar_error, local_error)
        speeds = _material_speeds(master, carrier_density, amplitude)
        acoustic = np.abs(speeds) > 1.0e-8
        equilibrium_results.append(
            EquilibriumSpectrum(
                name,
                carrier_density,
                amplitude,
                effective_mass_squared,
                curvature,
                unstable,
                local_error,
                int(np.count_nonzero(acoustic)),
                int(np.count_nonzero(~acoustic)),
                float(np.min(np.abs(speeds[acoustic]))),
                float(np.max(np.abs(speeds[acoustic]))),
            )
        )

    vector_error = 0.0
    ccz4_error = 0.0
    spurious_growth = 0.0
    for points in resolutions:
        for mode in vector_modes:
            generator, parameters = _vector_generator(points, mode, 1.0)
            numerical = _ordered(np.linalg.eigvals(generator))
            expected = _vector_expected(mode, 1.0, parameters)
            vector_error = max(
                vector_error, float(np.max(np.abs(numerical - expected)))
            )
        grid = PeriodicGrid((points,), (2.0 * np.pi,), method="spectral")
        (coordinate,) = grid.coordinates()
        solver, values = _background(grid)
        layout = _layout(values)
        for mode in vector_modes:
            symbol = _ccz4_symbol(
                solver, values, layout, coordinate, mode, 1.0e-7
            )
            spectrum = np.linalg.eigvals(symbol)
            expected_speeds = np.array(
                [-np.sqrt(4.0 / 3.0)]
                + [-1.0] * 9
                + [0.0] * 3
                + [1.0] * 9
                + [np.sqrt(4.0 / 3.0)]
            )
            numerical_speeds = np.sort(spectrum.imag)
            ccz4_error = max(
                ccz4_error,
                float(np.max(np.abs(numerical_speeds - expected_speeds))),
            )
            spurious_growth = max(
                spurious_growth, float(np.max(np.abs(spectrum.real)))
            )

    resolution_change = 0.0
    for name, _, _ in equilibrium_inputs:
        for mode in scalar_modes:
            reference = all_resolution_spectra[(name, resolutions[-1], mode)]
            for points in resolutions[:-1]:
                resolution_change = max(
                    resolution_change,
                    float(
                        np.max(
                            np.abs(
                                all_resolution_spectra[(name, points, mode)]
                                - reference
                            )
                        )
                    ),
                )
    critical_length = 2.0 * np.pi / np.sqrt(-activated_curvature)
    activated_unstable = next(
        result.unstable_fourier_modes
        for result in equilibrium_results
        if result.name == "symmetric-activated"
    )
    return LinearSpectrumAudit(
        tuple(equilibrium_results),
        resolutions,
        scalar_modes,
        vector_modes,
        8,
        2,
        3,
        12,
        4,
        4,
        2,
        17,
        vector_error,
        ccz4_error,
        resolution_change,
        spurious_growth,
        float(critical_length),
        activated_unstable,
        7,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=(16, 24, 32))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_audit(resolutions=tuple(arguments.resolutions))
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
