"""Experimental free evolution of the coupled raw-ADM--Proca system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import ADMState, ADMSolver, inverse_metric
from .curved_proca import CurvedProcaSystem
from .grid import Array
from .integrators import rk4_arrays
from .proca import ProcaState


@dataclass
class CoupledState:
    h: Array
    K: Array
    a: Array
    pi: Array
    time: float = 0.0


class CoupledADMProcaSolver:
    def __init__(self, adm: ADMSolver, proca: CurvedProcaSystem) -> None:
        if adm.grid != proca.grid:
            raise ValueError("ADM and Proca systems must share the same grid")
        self.adm = adm
        self.proca = proca
        self.grid = adm.grid

    def flat_state(self) -> CoupledState:
        geometry = self.adm.flat_state()
        return CoupledState(
            geometry.h,
            geometry.K,
            self.grid.zeros((3,)),
            self.grid.zeros((3,)),
        )

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        h, K, a, pi = values
        lapse, shift = self.adm.gauge(time, h, K)
        matter = self.proca.stress_energy(h, a, pi)
        dh, dK = self.adm.rhs(time, h, K, matter)
        da, dpi = self.proca.rhs(h, a, pi, lapse, shift)
        return dh, dK, da, dpi

    def step(self, state: CoupledState, dt: float) -> CoupledState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        values = rk4_arrays((state.h, state.K, state.a, state.pi), state.time, dt, self.rhs)
        h, K, a, pi = values
        h = 0.5 * (h + np.swapaxes(h, 0, 1))
        K = 0.5 * (K + np.swapaxes(K, 0, 1))
        inverse_metric(h)  # positive-definiteness check
        return CoupledState(h, K, a, pi, state.time + dt)

    def constraints(self, state: CoupledState) -> tuple[Array, Array]:
        matter = self.proca.stress_energy(state.h, state.a, state.pi)
        geometry = ADMState(state.h, state.K, state.time)
        return self.adm.constraints(geometry, matter)

    def constraint_norms(self, state: CoupledState) -> dict[str, float]:
        hamiltonian, momentum = self.constraints(state)
        return {
            "hamiltonian_l2": float(np.sqrt(np.mean(hamiltonian**2))),
            "momentum_l2": float(np.sqrt(np.mean(np.sum(momentum**2, axis=0)))),
            "hamiltonian_linf": float(np.max(np.abs(hamiltonian))),
            "momentum_linf": float(np.max(np.abs(momentum))),
        }

