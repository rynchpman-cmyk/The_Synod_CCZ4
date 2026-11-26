"""First-order canonical scalar subsystem used by the Nulloth backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .grid import Array, PeriodicGrid
from .theory2 import StressEnergy3p1


@dataclass
class NullothScalarState:
    varphi: Array
    momentum: Array
    gradient: Array

    def copy(self) -> "NullothScalarState":
        return NullothScalarState(
            self.varphi.copy(), self.momentum.copy(), self.gradient.copy()
        )


@dataclass(frozen=True)
class NullothScalarDiagnostics:
    constraint_l2: float
    constraint_linf: float
    minimum_energy_density: float
    maximum_amplitude: float


class NullothScalarSystem:
    """Canonical scalar field in a supplied 3+1 geometry.

    ``momentum`` is ``Pi = -n^mu grad_mu varphi`` and ``gradient`` is the
    independently evolved spatial covector ``Phi_i``.
    """

    def __init__(self, grid: PeriodicGrid, constraint_damping: float = 0.0) -> None:
        if constraint_damping < 0.0:
            raise ValueError("constraint_damping must be nonnegative")
        self.grid = grid
        self.constraint_damping = float(constraint_damping)

    def initialize(
        self,
        varphi: Array | float,
        momentum: Array | float = 0.0,
        gradient: Array | None = None,
    ) -> NullothScalarState:
        phi = np.broadcast_to(np.asarray(varphi, dtype=float), self.grid.shape).copy()
        pi = np.broadcast_to(np.asarray(momentum, dtype=float), self.grid.shape).copy()
        grad = self.grid.gradient(phi) if gradient is None else np.asarray(gradient, dtype=float).copy()
        if grad.shape != (3,) + self.grid.shape:
            raise ValueError("scalar gradient must have shape (3,) + grid.shape")
        return NullothScalarState(phi, pi, grad)

    def constraint(self, state: NullothScalarState) -> Array:
        return state.gradient - self.grid.gradient(state.varphi)

    def stress(
        self,
        h: Array,
        state: NullothScalarState,
        potential: Array,
    ) -> StressEnergy3p1:
        h_inv = inverse_metric(h)[0]
        grad_sq = np.einsum(
            "ij...,i...,j...->...", h_inv, state.gradient, state.gradient
        )
        rho = 0.5 * (state.momentum**2 + grad_sq) + potential
        momentum = state.momentum[None, ...] * state.gradient
        stress = (
            state.gradient[:, None, ...] * state.gradient[None, :, ...]
            + h
            * (0.5 * (state.momentum**2 - grad_sq) - potential)[None, None, ...]
        )
        trace = np.einsum("ij...,ij...->...", h_inv, stress)
        return StressEnergy3p1(rho, momentum, stress, trace)

    def rhs(
        self,
        h: Array,
        trace_K: Array,
        lapse: Array,
        shift: Array,
        state: NullothScalarState,
        equation_source: Array,
    ) -> tuple[Array, Array, Array]:
        h_inv, _, sqrt_h = inverse_metric(h)
        phi_up = np.einsum("ij...,j...->i...", h_inv, state.gradient)
        div_phi = self.grid.divergence(sqrt_h[None, ...] * phi_up) / sqrt_h
        grad_lapse = self.grid.gradient(lapse)
        grad_pi = self.grid.gradient(state.momentum)
        grad_phi_direct = self.grid.gradient(state.varphi)
        constraint = state.gradient - grad_phi_direct

        dvarphi = -lapse * state.momentum
        for axis in range(self.grid.ndim):
            dvarphi += shift[axis] * state.gradient[axis]

        dmomentum = lapse * (
            -div_phi + trace_K * state.momentum + equation_source
        )
        dmomentum += np.einsum("i...,i...->...", phi_up, grad_lapse)
        for axis in range(self.grid.ndim):
            dmomentum += shift[axis] * grad_pi[axis]

        dgradient = -lapse[None, ...] * grad_pi
        dgradient -= state.momentum[None, ...] * grad_lapse
        dgradient -= (
            self.constraint_damping * lapse[None, ...] * constraint
        )
        for i in range(3):
            for j in range(self.grid.ndim):
                dgradient[i] += shift[j] * self.grid.derivative(
                    state.gradient[i], j
                )
                if i < self.grid.ndim:
                    dgradient[i] += state.gradient[j] * self.grid.derivative(
                        shift[j], i
                    )
        return dvarphi, dmomentum, dgradient

    def diagnostics(
        self,
        h: Array,
        state: NullothScalarState,
        potential: Array,
    ) -> NullothScalarDiagnostics:
        constraint = self.constraint(state)
        stress = self.stress(h, state, potential)
        return NullothScalarDiagnostics(
            float(np.sqrt(np.mean(constraint**2))),
            float(np.max(np.abs(constraint))),
            float(np.min(stress.rho)),
            float(np.max(np.abs(state.varphi))),
        )

