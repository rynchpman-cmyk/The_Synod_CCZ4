"""Deterministic and stochastic Maxwell--Cattaneo rapidity relaxation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .grid import Array, PeriodicGrid
from .integrators import rk4_arrays

Target = Callable[[float], Array]
Velocity = Callable[[float], Array]


@dataclass(frozen=True)
class RelaxationParameters:
    mobility: float = 1.0
    susceptibility: float = 1.0
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.mobility <= 0.0 or self.susceptibility <= 0.0:
            raise ValueError("mobility and susceptibility must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature cannot be negative")

    @property
    def tau(self) -> float:
        return 1.0 / (self.mobility * self.susceptibility)


@dataclass
class RelaxationState:
    xi: Array
    time: float = 0.0

    def copy(self) -> "RelaxationState":
        return RelaxationState(self.xi.copy(), self.time)


@dataclass(frozen=True)
class RelaxationDiagnostics:
    time: float
    free_energy: float
    entropy_production: float
    mismatch_l2: float
    max_rapidity: float


class RelaxationSolver:
    """Evolve a spatial rapidity field in a prescribed matter frame.

    In Cartesian coordinates the implemented deterministic equation is

    ``partial_t xi + velocity^j partial_j xi = (target - xi) / tau``.

    Noise is normalized as a cell-averaged discretization of
    ``<eta eta> = 2 T Gamma delta(x-x')`` and integrated with the exact local
    Ornstein--Uhlenbeck variance.
    """

    def __init__(
        self,
        grid: PeriodicGrid,
        target: Target,
        parameters: RelaxationParameters | None = None,
        velocity: Velocity | None = None,
        seed: int | None = None,
    ) -> None:
        self.grid = grid
        self.target = target
        self.parameters = parameters or RelaxationParameters()
        self.velocity = velocity
        self.rng = np.random.default_rng(seed)

    def zero_state(self) -> RelaxationState:
        return RelaxationState(self.grid.zeros((3,)))

    def _target(self, time: float) -> Array:
        target = np.asarray(self.target(time), dtype=float)
        if target.shape != (3,) + self.grid.shape:
            raise ValueError("target returned an incompatible shape")
        return target

    def _velocity(self, time: float) -> Array:
        if self.velocity is None:
            return self.grid.zeros((3,))
        velocity = np.asarray(self.velocity(time), dtype=float)
        if velocity.shape != (3,) + self.grid.shape:
            raise ValueError("velocity returned an incompatible shape")
        speed_sq = np.sum(velocity * velocity, axis=0)
        if np.any(speed_sq >= 1.0):
            raise ValueError("prescribed transport velocity must be subluminal")
        return velocity

    def _advection_rhs(self, time: float, values: tuple[Array]) -> tuple[Array]:
        (xi,) = values
        velocity = self._velocity(time)
        advective = np.zeros_like(xi)
        for axis in range(self.grid.ndim):
            advective += velocity[axis][None, ...] * self.grid.derivative(xi, axis)
        return (-advective,)

    def _relax(self, xi: Array, target: Array, duration: float) -> Array:
        decay = np.exp(-duration / self.parameters.tau)
        return target + decay * (xi - target)

    def step(self, state: RelaxationState, dt: float, *, stochastic: bool = True) -> RelaxationState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        expected = (3,) + self.grid.shape
        if state.xi.shape != expected:
            raise ValueError(f"xi must have shape {expected}")

        # Strang split: local relaxation, transport, local relaxation.
        xi = self._relax(state.xi, self._target(state.time), 0.5 * dt)
        (xi,) = rk4_arrays((xi,), state.time, dt, self._advection_rhs)
        xi = self._relax(xi, self._target(state.time + dt), 0.5 * dt)

        p = self.parameters
        if stochastic and p.temperature > 0.0:
            variance = (p.temperature / (p.susceptibility * self.grid.cell_volume)) * (
                1.0 - np.exp(-2.0 * dt / p.tau)
            )
            xi = xi + np.sqrt(variance) * self.rng.standard_normal(xi.shape)

        if not np.all(np.isfinite(xi)):
            raise FloatingPointError("relaxation state contains a non-finite value")
        return RelaxationState(xi, state.time + dt)

    def diagnostics(self, state: RelaxationState) -> RelaxationDiagnostics:
        p = self.parameters
        mismatch = state.xi - self._target(state.time)
        mismatch_sq = np.sum(mismatch * mismatch, axis=0)
        force_sq = p.susceptibility**2 * mismatch_sq
        free_energy = self.grid.integrate(0.5 * p.susceptibility * mismatch_sq)
        entropy = (
            self.grid.integrate(p.mobility * force_sq / p.temperature)
            if p.temperature > 0.0
            else float("nan")
        )
        return RelaxationDiagnostics(
            time=state.time,
            free_energy=free_energy,
            entropy_production=entropy,
            mismatch_l2=float(np.sqrt(np.mean(mismatch_sq))),
            max_rapidity=float(np.max(np.sqrt(np.sum(state.xi * state.xi, axis=0)))),
        )

