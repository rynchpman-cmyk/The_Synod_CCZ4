"""Runtime characteristic and constitutive guard for Nulloth states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matter import FluidPrimitive
from .nulloth import NullothMasterFunction
from .theory33 import CarrierPrimitive


class NullothAdmissibilityError(FloatingPointError):
    """Raised when a runtime state enters a denied Nulloth domain."""


@dataclass(frozen=True)
class NullothGuardThresholds:
    warning_scalar_amplitude_fraction: float = 0.9
    warning_relative_lorentz_fraction: float = 0.9
    warning_legendre_eigenvalue: float = 0.15
    warning_thermodynamic_eigenvalue: float = 0.08
    warning_eigenvector_condition: float = 1.0e5
    warning_reduced_condition: float = 1.0e4
    maximum_cells: int = 16
    direction_count: int = 18

    def __post_init__(self) -> None:
        if not 0.0 < self.warning_scalar_amplitude_fraction < 1.0:
            raise ValueError("scalar warning fraction must lie in (0, 1)")
        if not 0.0 < self.warning_relative_lorentz_fraction < 1.0:
            raise ValueError("relative-Lorentz warning fraction must lie in (0, 1)")
        if min(
            self.warning_legendre_eigenvalue,
            self.warning_thermodynamic_eigenvalue,
            self.warning_eigenvector_condition,
            self.warning_reduced_condition,
        ) <= 0.0:
            raise ValueError("guard warning thresholds must be positive")
        if self.maximum_cells <= 0 or self.direction_count < 6:
            raise ValueError("guard sampling controls are invalid")


@dataclass(frozen=True)
class NullothGuardReport:
    status: str
    sampled_cells: int
    directions_per_cell: int
    symbol_evaluations: int
    reasons: tuple[str, ...]
    maximum_scalar_amplitude: float
    maximum_relative_lorentz_factor: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_eigenvector_condition: float
    maximum_reduced_condition: float
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    maximum_eigenspace_residual: float
    worst_flat_cell_index: int


def _fibonacci_sphere(count: int) -> np.ndarray:
    indices = np.arange(count, dtype=float) + 0.5
    z = 1.0 - 2.0 * indices / count
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    return np.column_stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z)
    )


class NullothRuntimeGuard:
    """Sample local states and classify them as ``SAFE``, ``WATCH``, or ``DENY``.

    The guard is diagnostic and does not modify evolved variables. Coordinate
    velocities are transformed into a local orthonormal spatial frame before
    the flat frozen-coefficient characteristic audit is evaluated.
    """

    def __init__(
        self,
        master: NullothMasterFunction,
        thresholds: NullothGuardThresholds | None = None,
    ) -> None:
        self.master = master
        self.thresholds = thresholds or NullothGuardThresholds()
        self.directions = _fibonacci_sphere(self.thresholds.direction_count)

    def audit(
        self,
        h: np.ndarray,
        fluid: FluidPrimitive,
        carrier: CarrierPrimitive,
        varphi: np.ndarray,
    ) -> NullothGuardReport:
        shape = np.asarray(varphi).shape
        count = int(np.prod(shape))
        if h.shape != (3, 3) + shape:
            raise ValueError("guard metric and scalar shapes disagree")
        if fluid.baryon_density.shape != shape or carrier.number_density.shape != shape:
            raise ValueError("guard primitive shapes disagree")
        sample_count = min(count, self.thresholds.maximum_cells)
        indices = np.unique(
            np.linspace(0, count - 1, sample_count, dtype=int)
        )
        h_flat = h.reshape(3, 3, count)
        n_flat = fluid.baryon_density.reshape(count)
        e_flat = fluid.specific_internal_energy.reshape(count)
        u_flat = fluid.velocity.reshape(3, count)
        d_flat = carrier.number_density.reshape(count)
        v_flat = carrier.velocity.reshape(3, count)
        phi_flat = np.asarray(varphi, dtype=float).reshape(count)
        p = self.master.parameters
        warning_phi = (
            self.thresholds.warning_scalar_amplitude_fraction
            * p.scalar_amplitude_limit
        )
        warning_gamma = (
            1.0
            + self.thresholds.warning_relative_lorentz_fraction
            * (p.maximum_relative_lorentz_factor - 1.0)
        )
        deny_reasons: set[str] = set()
        watch_reasons: set[str] = set()
        minimum_legendre = np.inf
        minimum_thermo = np.inf
        maximum_condition = 0.0
        maximum_reduced = 0.0
        maximum_speed = 0.0
        maximum_imaginary = 0.0
        maximum_residual = 0.0
        maximum_phi = 0.0
        maximum_gamma = 1.0
        worst_index = int(indices[0])
        worst_score = -np.inf
        evaluations = 0
        for flat_index in indices:
            metric = h_flat[:, :, flat_index]
            frame = np.linalg.cholesky(metric).T
            velocity = frame @ u_flat[:, flat_index]
            carrier_velocity = frame @ v_flat[:, flat_index]
            phi = float(phi_flat[flat_index])
            maximum_phi = max(maximum_phi, abs(phi))
            speed_u2 = float(velocity @ velocity)
            speed_v2 = float(carrier_velocity @ carrier_velocity)
            if speed_u2 >= 1.0 or speed_v2 >= 1.0:
                deny_reasons.add("subluminal_primitive")
                continue
            gamma_u = 1.0 / np.sqrt(1.0 - speed_u2)
            gamma_v = 1.0 / np.sqrt(1.0 - speed_v2)
            relative_gamma = gamma_u * gamma_v * (
                1.0 - float(velocity @ carrier_velocity)
            )
            maximum_gamma = max(maximum_gamma, relative_gamma)
            if abs(phi) > p.scalar_amplitude_limit:
                deny_reasons.add("scalar_amplitude_cap")
                continue
            if relative_gamma > p.maximum_relative_lorentz_factor:
                deny_reasons.add("relative_lorentz_cap")
                continue
            if abs(phi) >= warning_phi:
                watch_reasons.add("scalar_amplitude_margin")
            if relative_gamma >= warning_gamma:
                watch_reasons.add("relative_lorentz_margin")
            for direction in self.directions:
                evaluations += 1
                audit = self.master.coupled_characteristic_audit(
                    float(n_flat[flat_index]),
                    float(e_flat[flat_index]),
                    velocity,
                    float(d_flat[flat_index]),
                    carrier_velocity,
                    phi,
                    direction,
                )
                minimum_legendre = min(
                    minimum_legendre, audit.legendre_minimum_eigenvalue
                )
                minimum_thermo = min(
                    minimum_thermo, audit.thermodynamic_minimum_eigenvalue
                )
                maximum_condition = max(
                    maximum_condition, audit.eigenvector_condition
                )
                maximum_reduced = max(
                    maximum_reduced,
                    audit.constraint_reduced_eigenvector_condition,
                )
                maximum_speed = max(
                    maximum_speed, audit.maximum_absolute_speed
                )
                maximum_imaginary = max(
                    maximum_imaginary, audit.maximum_imaginary_speed
                )
                maximum_residual = max(
                    maximum_residual, audit.maximum_eigenspace_residual
                )
                if not audit.strongly_hyperbolic:
                    deny_reasons.add("strong_hyperbolicity")
                if not audit.causal:
                    deny_reasons.add("physical_causality")
                if (
                    audit.legendre_minimum_eigenvalue
                    <= self.thresholds.warning_legendre_eigenvalue
                ):
                    watch_reasons.add("legendre_margin")
                if (
                    audit.thermodynamic_minimum_eigenvalue
                    <= self.thresholds.warning_thermodynamic_eigenvalue
                ):
                    watch_reasons.add("thermodynamic_margin")
                if (
                    audit.eigenvector_condition
                    >= self.thresholds.warning_eigenvector_condition
                ):
                    watch_reasons.add("eigenvector_condition")
                if (
                    audit.constraint_reduced_eigenvector_condition
                    >= self.thresholds.warning_reduced_condition
                ):
                    watch_reasons.add("reduced_condition")
                score = max(
                    audit.eigenvector_condition
                    / self.thresholds.warning_eigenvector_condition,
                    self.thresholds.warning_legendre_eigenvalue
                    / max(audit.legendre_minimum_eigenvalue, 1.0e-300),
                    self.thresholds.warning_thermodynamic_eigenvalue
                    / max(audit.thermodynamic_minimum_eigenvalue, 1.0e-300),
                )
                if score > worst_score:
                    worst_score = score
                    worst_index = int(flat_index)
        if not np.isfinite(minimum_legendre):
            minimum_legendre = np.nan
            minimum_thermo = np.nan
        if deny_reasons:
            status = "DENY"
            reasons = tuple(sorted(deny_reasons | watch_reasons))
        elif watch_reasons:
            status = "WATCH"
            reasons = tuple(sorted(watch_reasons))
        else:
            status = "SAFE"
            reasons = ()
        return NullothGuardReport(
            status,
            len(indices),
            len(self.directions),
            evaluations,
            reasons,
            maximum_phi,
            maximum_gamma,
            float(minimum_legendre),
            float(minimum_thermo),
            maximum_condition,
            maximum_reduced,
            maximum_speed,
            maximum_imaginary,
            maximum_residual,
            worst_index,
        )

    def enforce(self, *args, **kwargs) -> NullothGuardReport:
        report = self.audit(*args, **kwargs)
        if report.status == "DENY":
            raise NullothAdmissibilityError(
                "Nulloth runtime guard denied state: " + ", ".join(report.reasons)
            )
        return report
