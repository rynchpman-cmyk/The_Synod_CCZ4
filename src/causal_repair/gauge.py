"""Moving-puncture gauge interface selected by Theory 2.0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Array, PeriodicGrid


@dataclass
class MovingPunctureGaugeState:
    lapse: Array
    shift: Array
    driver: Array


@dataclass(frozen=True)
class GaugeRHS:
    lapse: Array
    shift: Array
    driver: Array


def moving_puncture_rhs(
    grid: PeriodicGrid,
    state: MovingPunctureGaugeState,
    trace_K: Array,
    gamma_tilde_rhs: Array,
    *,
    theta: Array | None = None,
    eta_shift: float = 1.0,
) -> GaugeRHS:
    """Advective 1+log and Gamma-driver equations.

    ``gamma_tilde_rhs`` is the CCZ4/BSSN evolution RHS for the conformal
    connection, which must be supplied by that formulation.
    """
    theta = np.zeros_like(trace_K) if theta is None else theta
    adv_lapse = np.zeros_like(state.lapse)
    adv_shift = np.zeros_like(state.shift)
    adv_driver = np.zeros_like(state.driver)
    for axis in range(grid.ndim):
        adv_lapse += state.shift[axis] * grid.derivative(state.lapse, axis)
        adv_shift += state.shift[axis][None, ...] * grid.derivative(state.shift, axis)
        adv_driver += state.shift[axis][None, ...] * grid.derivative(state.driver, axis)
    # The equation requires (partial_t - beta.d) Gamma_tilde, conventionally
    # provided directly as gamma_tilde_rhs_minus_advection.
    return GaugeRHS(
        lapse=adv_lapse - 2.0 * state.lapse * (trace_K - 2.0 * theta),
        shift=adv_shift + 0.75 * state.driver,
        driver=adv_driver + gamma_tilde_rhs - eta_shift * state.driver,
    )
