"""Experimental raw-ADM bridge for Theory 2.0 with prescribed fluid fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import ADMState, ADMSolver, inverse_metric
from .curved_proca import Theory2ProcaSystem
from .grid import Array
from .integrators import rk4_arrays
from .matter import FluidPrimitive, validate_fluid
from .theory2 import TargetEvolution, Theory2Parameters, total_stress_energy


@dataclass
class Theory2ADMState:
    h: Array
    K: Array
    a: Array
    pi: Array
    astar: Array
    time: float = 0.0


class Theory2PrescribedMatterADM:
    """Back-reacting ADM--repair evolution with non-evolved fluid primitives.

    This bridge is useful for metric-response tests. It intentionally does not
    advertise a conservative GRHD evolution: Theory 2.0 has not fixed the
    target-relaxation exchange four-force needed to evolve the fluid sector.
    """

    def __init__(
        self,
        adm: ADMSolver,
        fluid: FluidPrimitive,
        parameters: Theory2Parameters | None = None,
    ) -> None:
        self.adm = adm
        self.grid = adm.grid
        self.fluid = fluid
        self.parameters = parameters or Theory2Parameters()
        self.proca = Theory2ProcaSystem(self.grid, self.parameters)
        self.target = TargetEvolution(self.grid, self.parameters)

    def matter_sources(
        self, h: Array, a: Array, pi: Array, astar: Array
    ) -> tuple[Array, Array, Array]:
        validate_fluid(self.grid, h, self.fluid)
        phi = self.proca.constrained_phi(h, pi, self.fluid, astar)
        proca_part = self.proca.stress_energy_theory2(
            h, a, pi, self.fluid, astar
        )
        total = total_stress_energy(
            self.grid,
            h,
            self.fluid,
            proca_part,
            a,
            phi,
            astar,
            self.parameters,
        )
        return total.rho, total.momentum, total.stress

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        h, K, a, pi, astar = values
        lapse, shift = self.adm.gauge(time, h, K)
        sources = self.matter_sources(h, a, pi, astar)
        dh, dK = self.adm.rhs(time, h, K, sources)
        da, dpi = self.proca.rhs_theory2(
            h, a, pi, self.fluid, astar, lapse, shift
        )
        (dastar,) = self.target.rhs(time, (astar,), h, self.fluid)
        return dh, dK, da, dpi, dastar

    def step(self, state: Theory2ADMState, dt: float) -> Theory2ADMState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        h, K, a, pi, astar = rk4_arrays(
            (state.h, state.K, state.a, state.pi, state.astar),
            state.time,
            dt,
            self.rhs,
        )
        h = 0.5 * (h + np.swapaxes(h, 0, 1))
        K = 0.5 * (K + np.swapaxes(K, 0, 1))
        inverse_metric(h)
        return Theory2ADMState(h, K, a, pi, astar, state.time + dt)

    def constraints(self, state: Theory2ADMState) -> dict[str, float]:
        matter = self.matter_sources(state.h, state.a, state.pi, state.astar)
        hamiltonian, momentum = self.adm.constraints(
            ADMState(state.h, state.K, state.time), matter
        )
        phi = self.proca.constrained_phi(
            state.h, state.pi, self.fluid, state.astar
        )
        gauss = self.proca.gauss_constraint(
            state.h,
            state.a,
            state.pi,
            self.fluid,
            state.astar,
            phi,
        )
        return {
            "hamiltonian_l2": float(np.sqrt(np.mean(hamiltonian**2))),
            "momentum_l2": float(np.sqrt(np.mean(np.sum(momentum**2, axis=0)))),
            "gauss_l2": float(np.sqrt(np.mean(gauss**2))),
        }

