"""One-dimensional finite-patch Nulloth backend with scalar characteristics.

This module stages finite-domain qualification without changing the sealed
periodic production backend.  It reuses the canonical Nulloth continuum RHS
and adds boundary projections for the mixed-Proca and scalar subsystems.
"""

from __future__ import annotations

import numpy as np

from .adm import inverse_metric
from .boundary import Theory3CharacteristicBoundary
from .ccz4 import CCZ4Parameters
from .grhd import GRHDParameters
from .grid import Array, CartesianGrid
from .nulloth import NullothMasterFunction, NullothParameters
from .nulloth_scalar import NullothScalarSystem
from .production3 import Theory3ProductionParameters
from .production33 import Theory33ProductionSolver
from .production_nulloth import NullothProductionSolver, NullothProductionState
from .theory3 import Theory3Parameters


class NullothScalarCharacteristicBoundary:
    """Constraint-compatible incoming scalar projection on a 1D patch.

    With ``Pi=-n^mu grad_mu varphi`` and outward unit derivative ``Phi_s``,
    the outgoing and incoming physical characteristics are ``Pi+Phi_s`` and
    ``Pi-Phi_s``.  The outgoing RHS is preserved and the incoming field is
    damped homogeneously.  The production RK stage projection reconstructs
    ``Phi_i=partial_i varphi`` after every stage.
    """

    def __init__(self, grid: CartesianGrid, damping: float = 1.0) -> None:
        if grid.ndim != 1:
            raise NotImplementedError(
                "finite Nulloth scalar characteristics are qualified only in 1D"
            )
        if damping < 0.0:
            raise ValueError("scalar boundary damping must be nonnegative")
        self.grid = grid
        self.damping = float(damping)

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
                    "finite Nulloth boundary ceased to be outward subluminal"
                )
            normal_up = (
                side * h_inv[:, 0, index] / normal_scale[index]
            )
            phi_normal = float(
                np.einsum("i,i->", normal_up, state.gradient[:, index])
            )
            dphi_normal = float(
                np.einsum("i,i->", normal_up, dgradient[:, index])
            )
            outgoing_rhs = float(dmomentum[index]) + dphi_normal
            incoming_rhs = -self.damping * (
                float(state.momentum[index]) - phi_normal
            )
            target_momentum_rhs = 0.5 * (outgoing_rhs + incoming_rhs)
            target_phi_normal_rhs = 0.5 * (outgoing_rhs - incoming_rhs)
            delta = target_phi_normal_rhs - dphi_normal
            dmomentum[index] = target_momentum_rhs
            # The unit normal covector is side*dx_0/sqrt(h^00).
            dgradient[0, index] += side * delta / normal_scale[index]
        return dvarphi, dmomentum, dgradient


class FiniteNullothProductionSolver(NullothProductionSolver):
    """Canonical Nulloth production on a 1D nonperiodic Cartesian patch."""

    def __init__(
        self,
        grid: CartesianGrid,
        nulloth: NullothParameters | None = None,
        vector: Theory3Parameters | None = None,
        ccz4: CCZ4Parameters | None = None,
        grhd: GRHDParameters | None = None,
        production: Theory3ProductionParameters | None = None,
        *,
        scalar_boundary_damping: float = 1.0,
    ) -> None:
        if grid.ndim != 1:
            raise NotImplementedError(
                "FiniteNullothProductionSolver currently supports only 1D patches"
            )
        parameters = nulloth or NullothParameters()
        if vector is None:
            vector = Theory3Parameters(
                target_speed=max(parameters.carrier_sound_speed, 0.9),
                target_relaxation_time=parameters.relaxation_time,
                target_strength=0.0,
                gamma_ad=parameters.gamma_ad,
            )
        Theory33ProductionSolver.__init__(
            self,
            grid,
            parameters,
            vector,
            ccz4,
            grhd,
            production,
        )
        self.master_parameters = parameters
        self.master = NullothMasterFunction(parameters)
        self.scalar_system = NullothScalarSystem(
            grid, parameters.scalar_constraint_damping
        )
        self.scalar_boundary = NullothScalarCharacteristicBoundary(
            grid, scalar_boundary_damping
        )
        self.last_entropy_boundary_outflow = 0.0
        self.last_entropy_balance_residual = 0.0

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        result = list(super().rhs(time, values))
        state = self._unpack(values, time)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        proca = Theory3CharacteristicBoundary(self.grid).mixed_proca_rhs(
            self.system,
            h,
            state,
            tuple(result[14:22]),
        )
        result[14:22] = proca
        scalar = self.scalar_boundary.scalar_rhs(
            h,
            state.scalar,
            tuple(result[24:27]),
            state.geometry.lapse,
            state.geometry.shift,
        )
        result[24:27] = scalar
        return tuple(result)

    def step(
        self, state: NullothProductionState, dt: float
    ) -> NullothProductionState:
        """Advance one step with entropy loss charged to the open boundary."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        entropy_before_step = float(self.grid.integrate(state.matter.entropy))
        self.last_projection_entropy_change = 0.0
        self.last_projection_minimum_change = 0.0
        values = self._pack(state)
        time = state.time

        def stage(base, slope, factor, stage_time):
            candidate = self._unpack(
                tuple(y + factor * dt * k for y, k in zip(base, slope)),
                stage_time,
            )
            candidate.geometry = self.ccz4.project_algebraic(candidate.geometry)
            candidate.geometry.time = stage_time
            candidate = self._project_to_master_manifold(candidate)
            return self._pack(candidate)

        k1 = tuple(self.rhs(time, values))
        y2 = stage(values, k1, 0.5, time + 0.5 * dt)
        k2 = tuple(self.rhs(time + 0.5 * dt, y2))
        y3 = stage(values, k2, 0.5, time + 0.5 * dt)
        k3 = tuple(self.rhs(time + 0.5 * dt, y3))
        y4 = stage(values, k3, 1.0, time + dt)
        k4 = tuple(self.rhs(time + dt, y4))
        stage_entropy_change = (
            dt
            * self.grid.integrate(k1[12] + 2.0 * k2[12] + 2.0 * k3[12] + k4[12])
            / 6.0
        )
        charge_to_number = 1.0 / self.master_parameters.carrier_charge
        stage_number_change = (
            dt
            * charge_to_number
            * self.grid.integrate(k1[22] + 2.0 * k2[22] + 2.0 * k3[22] + k4[22])
            / 6.0
        )
        evolved = tuple(
            y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        result = self._unpack(evolved, time + dt)
        result.geometry = self.ccz4.project_algebraic(result.geometry)
        result.geometry.time = result.time
        result = self._project_to_master_manifold(result)
        result = self._apply_carrier_drag(result, dt)
        result = self._apply_damping(result, dt)
        self.recover(result)
        actual_number_change = charge_to_number * self.grid.integrate(
            result.target_charge - state.target_charge
        )
        self.last_carrier_boundary_number_outflow = -stage_number_change
        self.last_carrier_number_change = actual_number_change
        self.last_carrier_number_balance_residual = (
            actual_number_change + self.last_carrier_boundary_number_outflow
        )
        self.last_step_entropy_change = (
            float(self.grid.integrate(result.matter.entropy)) - entropy_before_step
        )
        self.last_entropy_boundary_outflow = -stage_entropy_change
        self.last_entropy_balance_residual = (
            self.last_step_entropy_change + self.last_entropy_boundary_outflow
        )
        return result
