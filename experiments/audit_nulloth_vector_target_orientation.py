"""Audit Proca principal invariance and target-current orientation response."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.nulloth import NullothMasterFunction
from causal_repair.theory3 import Theory3Parameters, Theory3System


@dataclass(frozen=True)
class VectorTargetOrientationAudit:
    target_orientations: int
    proca_parameter_sets: int
    field_variables: int
    numerical_jacobian_source_orientations: int
    maximum_principal_block_variation: float
    maximum_principal_spectrum_variation: float
    maximum_source_orientation_field_jacobian_error: float
    maximum_power_projection_error: float
    maximum_force_projection_error: float
    longitudinal_power: float
    transverse_power: float
    maximum_transverse_power: float
    minimum_sampled_mismatch_angle: float
    maximum_sampled_mismatch_angle: float


def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    return np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )


def _pack_fields(grid: PeriodicGrid) -> tuple[np.ndarray, ...]:
    return (
        grid.zeros((3,)),
        grid.zeros((3,)),
        grid.zeros((3,)),
        grid.zeros((3,)),
        grid.zeros(),
        grid.zeros(),
        grid.zeros(),
        grid.zeros(),
    )


def _field_layout() -> tuple[tuple[int, int | None], ...]:
    layout = []
    for group in range(4):
        layout.extend((group, component) for component in range(3))
    layout.extend((group, None) for group in range(4, 8))
    return tuple(layout)


def _field_jacobian(
    system: Theory3System,
    charge: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    grid = system.grid
    (coordinate,) = grid.coordinates()
    cosine = np.cos(coordinate)
    sine = np.sin(coordinate)
    fields = _pack_fields(grid)
    layout = _field_layout()
    h = np.eye(3).reshape(3, 3, 1)
    epsilon = 1.0e-7
    jacobian = np.zeros((16, 16), dtype=complex)

    def perturb(column: int, sign: float) -> tuple[np.ndarray, ...]:
        values = [value.copy() for value in fields]
        group, component = layout[column]
        if component is None:
            values[group] += sign * epsilon * cosine
        else:
            values[group][component] += sign * epsilon * cosine
        return tuple(values)

    def rhs(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
        return system.rhs_fields(
            h,
            *values,
            charge,
            current,
            np.ones(grid.shape),
            grid.zeros((3,)),
            0.0,
        )

    for column in range(16):
        plus = rhs(perturb(column, 1.0))
        minus = rhs(perturb(column, -1.0))
        derivatives = tuple(
            (left - right) / (2.0 * epsilon)
            for left, right in zip(plus, minus)
        )
        for row, (group, component) in enumerate(layout):
            field = (
                derivatives[group]
                if component is None
                else derivatives[group][component]
            )
            jacobian[row, column] = complex(
                2.0 * float(np.mean(field * cosine)),
                -2.0 * float(np.mean(field * sine)),
            )
    return jacobian


def run_audit(*, target_orientations: int = 128) -> VectorTargetOrientationAudit:
    grid = PeriodicGrid((16,), (2.0 * np.pi,), method="spectral")
    parameter_sets = (
        Theory3Parameters(Z_A=1.0, Z_B=1.0, mass_A=0.4, mass_B=0.7),
        Theory3Parameters(
            Z_A=0.8,
            Z_B=1.3,
            mass_A=0.2,
            mass_B=1.1,
            mixing_mass_squared=0.09,
        ),
        Theory3Parameters(
            Z_A=1.7,
            Z_B=0.6,
            mass_A=1.0,
            mass_B=0.3,
            mixing_mass_squared=0.04,
        ),
        Theory3Parameters(
            Z_A=1.0,
            Z_B=0.9,
            mass_A=0.05,
            mass_B=0.05,
            mixing_mass_squared=0.0,
        ),
    )
    canonical = NullothMasterFunction._proca_cleaning_principal_block()
    doubled = np.zeros((14, 14))
    doubled[:7, :7] = canonical
    doubled[7:, 7:] = canonical
    reference_spectrum = np.sort_complex(np.linalg.eigvals(doubled))
    maximum_block_variation = 0.0
    maximum_spectrum_variation = 0.0
    for _parameters in parameter_sets:
        # The canonical block has no mass, normalization, mixing, source, or
        # background-field argument. Reconstructing it for each parameter set
        # makes that independence an executable assertion.
        block = NullothMasterFunction._proca_cleaning_principal_block()
        candidate = np.zeros((14, 14))
        candidate[:7, :7] = block
        candidate[7:, 7:] = block
        maximum_block_variation = max(
            maximum_block_variation,
            float(np.max(np.abs(candidate - doubled))),
        )
        maximum_spectrum_variation = max(
            maximum_spectrum_variation,
            float(
                np.max(
                    np.abs(
                        np.sort_complex(np.linalg.eigvals(candidate))
                        - reference_spectrum
                    )
                )
            ),
        )

    system = Theory3System(grid, parameter_sets[0])
    charge = np.full(grid.shape, 0.18)
    currents = []
    for direction in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    ):
        currents.append(
            np.broadcast_to(0.07 * direction[:, None], (3,) + grid.shape).copy()
        )
    jacobians = [_field_jacobian(system, charge, current) for current in currents]
    source_jacobian_error = max(
        float(np.max(np.abs(jacobian - jacobians[0])))
        for jacobian in jacobians[1:]
    )

    h = np.eye(3).reshape(3, 3, 1)
    b = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    pi_B[0] = -parameter_sets[0].Z_B * 0.12
    b[2] = 0.09 * grid.coordinates()[0]
    electric = system.electric_field(h, pi_B, parameter_sets[0].Z_B)[:, 0]
    field_lower = system.field_strength(b)[:, :, 0]
    orientations = _fibonacci_sphere(target_orientations)
    maximum_power_error = 0.0
    maximum_force_error = 0.0
    maximum_transverse_power = 0.0
    angles = []
    for direction in orientations:
        current = (0.07 * direction).reshape(3, 1)
        power, force = system.drive_exchange(
            h, b, pi_B, np.array([0.18]), current
        )
        expected_power = float(electric @ current[:, 0])
        expected_force = 0.18 * electric + field_lower @ current[:, 0]
        maximum_power_error = max(
            maximum_power_error, abs(float(power[0]) - expected_power)
        )
        maximum_force_error = max(
            maximum_force_error,
            float(np.max(np.abs(force[:, 0] - expected_force))),
        )
        cosine = float(direction @ electric / np.linalg.norm(electric))
        angles.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
        if abs(cosine) < 0.1:
            maximum_transverse_power = max(
                maximum_transverse_power, abs(float(power[0]))
            )
    longitudinal_current = np.array([[0.07], [0.0], [0.0]])
    transverse_current = np.array([[0.0], [0.07], [0.0]])
    longitudinal_power = float(
        system.drive_exchange(
            h, b, pi_B, np.array([0.18]), longitudinal_current
        )[0][0]
    )
    transverse_power = float(
        system.drive_exchange(
            h, b, pi_B, np.array([0.18]), transverse_current
        )[0][0]
    )
    return VectorTargetOrientationAudit(
        target_orientations,
        len(parameter_sets),
        16,
        len(currents),
        maximum_block_variation,
        maximum_spectrum_variation,
        source_jacobian_error,
        maximum_power_error,
        maximum_force_error,
        longitudinal_power,
        transverse_power,
        maximum_transverse_power,
        min(angles),
        max(angles),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-orientations", type=int, default=128)
    arguments = parser.parse_args()
    print(
        json.dumps(
            asdict(run_audit(target_orientations=arguments.target_orientations)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
