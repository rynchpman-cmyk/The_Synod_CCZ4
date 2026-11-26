"""Driven scalar-characteristic extension of the staged finite Nulloth solver."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .adm import inverse_metric
from .finite_nulloth import (
    FiniteNullothProductionSolver,
    NullothScalarCharacteristicBoundary,
)
from .grid import Array, CartesianGrid


ScalarDrive = Callable[[int, float], tuple[float, float]]


class DrivenNullothScalarBoundary(NullothScalarCharacteristicBoundary):
    """Penalty-track prescribed incoming scalar characteristics.

    The drive returns the target incoming characteristic and its time
    derivative for ``side=-1`` (left) or ``side=1`` (right). Homogeneous
    outflow is recovered by returning ``(0, 0)``.
    """

    def __init__(
        self,
        grid: CartesianGrid,
        drive: ScalarDrive,
        damping: float = 100.0,
    ) -> None:
        super().__init__(grid, damping)
        self.drive = drive
        self.time = 0.0

    def scalar_rhs(
        self,
        h: Array,
        state,
        rhs: tuple[Array, Array, Array],
        lapse: Array,
        shift: Array,
    ) -> tuple[Array, Array, Array]:
        dvarphi, dmomentum, dgradient = (
            np.array(value, copy=True) for value in rhs
        )
        h_inv = inverse_metric(h)[0]
        normal_scale = np.sqrt(np.maximum(h_inv[0, 0], 1.0e-300))
        for side, index in ((-1, 0), (1, -1)):
            beta_normal = side * shift[0, index] / normal_scale[index]
            if lapse[index] <= abs(beta_normal):
                raise FloatingPointError(
                    "driven finite boundary ceased to be outward subluminal"
                )
            normal_up = side * h_inv[:, 0, index] / normal_scale[index]
            phi_normal = float(
                np.einsum("i,i->", normal_up, state.gradient[:, index])
            )
            dphi_normal = float(
                np.einsum("i,i->", normal_up, dgradient[:, index])
            )
            outgoing_rhs = float(dmomentum[index]) + dphi_normal
            target, target_derivative = self.drive(side, self.time)
            incoming = float(state.momentum[index]) - phi_normal
            incoming_rhs = target_derivative - self.damping * (incoming - target)
            dmomentum[index] = 0.5 * (outgoing_rhs + incoming_rhs)
            target_phi_rhs = 0.5 * (outgoing_rhs - incoming_rhs)
            delta = target_phi_rhs - dphi_normal
            dgradient[0, index] += side * delta / normal_scale[index]
        return dvarphi, dmomentum, dgradient


class DrivenFiniteNullothProductionSolver(FiniteNullothProductionSolver):
    """Finite production solver with prescribed scalar incoming data."""

    def __init__(self, *args, scalar_drive: ScalarDrive, drive_damping=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar_boundary = DrivenNullothScalarBoundary(
            self.grid, scalar_drive, drive_damping
        )

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        self.scalar_boundary.time = float(time)
        return super().rhs(time, values)
