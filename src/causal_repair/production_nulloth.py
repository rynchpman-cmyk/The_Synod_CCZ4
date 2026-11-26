"""Nulloth production-reference backend.

The backend extends ``production33`` with a canonical repair-coherence scalar
whose algebraic coupling is included in the two-current master action.  The
CCZ4 principal part is unchanged; it receives the complete Einstein-frame
Eulerian stress of material, mixed Proca, and scalar sectors.

Qualification boundary
----------------------
This first backend is restricted to periodic Cartesian grids.  It is intended
for homogeneous phase scans, localized periodic-box experiments, convergence
studies, and coupled conservation tests.  Finite boundaries require the full
coupled Nulloth characteristic treatment and are intentionally rejected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .adm import inverse_metric
from .ccz4 import CCZ4Parameters, CCZ4State
from .grhd import GRHDParameters
from .grid import Array, PeriodicGrid
from .io import load_checkpoint, save_checkpoint
from .matter import FluidPrimitive, lorentz_factor
from .nulloth import NullothMasterFunction, NullothMasterState, NullothParameters
from .nulloth_scalar import NullothScalarState, NullothScalarSystem
from .production3 import (
    FiniteDampingReport,
    Theory3MatterState,
    Theory3ProductionParameters,
    _add_stress,
)
from .production33 import Theory33ProductionSolver
from .theory3 import Theory3Parameters
from .theory33 import CarrierPrimitive


@dataclass
class NullothProductionState:
    geometry: CCZ4State
    matter: Theory3MatterState
    a: Array
    pi_A: Array
    b: Array
    pi_B: Array
    longitudinal_A: Array
    longitudinal_B: Array
    cleaning_A: Array
    cleaning_B: Array
    target_charge: Array
    target_current: Array
    scalar: NullothScalarState
    time: float = 0.0


@dataclass(frozen=True)
class NullothRecoveryReport:
    failed_cells: int
    atmosphere_cells: int
    maximum_residual: float
    minimum_carrier_density: float
    maximum_relative_lorentz_factor: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    minimum_effective_mass_squared: float
    maximum_coupling_coefficient: float
    maximum_scalar_amplitude: float
    scalar_constraint_l2: float


@dataclass(frozen=True)
class NullothRecovery:
    fluid: FluidPrimitive
    carrier: CarrierPrimitive
    master: NullothMasterState
    entropy_per_baryon: Array
    sound_speed: Array
    material_stress: Any
    vector_stress: Any
    scalar_stress: Any
    total_stress: Any
    target_charge_eulerian: Array
    target_current_up: Array
    report: NullothRecoveryReport


class NullothProductionSolver(Theory33ProductionSolver):
    """CCZ4 + mixed Proca + two-current Nulloth scalarization."""

    def __init__(
        self,
        grid: PeriodicGrid,
        nulloth: NullothParameters | None = None,
        vector: Theory3Parameters | None = None,
        ccz4: CCZ4Parameters | None = None,
        grhd: GRHDParameters | None = None,
        production: Theory3ProductionParameters | None = None,
    ) -> None:
        if not getattr(grid, "is_periodic", True):
            raise NotImplementedError(
                "Nulloth supports periodic grids only; finite coupled "
                "characteristic boundaries are not yet qualified"
            )
        parameters = nulloth or NullothParameters()
        if vector is None:
            vector = Theory3Parameters(
                target_speed=max(parameters.carrier_sound_speed, 0.9),
                target_relaxation_time=parameters.relaxation_time,
                target_strength=0.0,
                gamma_ad=parameters.gamma_ad,
            )
        super().__init__(grid, parameters, vector, ccz4, grhd, production)
        self.master_parameters = parameters
        self.master = NullothMasterFunction(parameters)
        self.scalar_system = NullothScalarSystem(
            grid, parameters.scalar_constraint_damping
        )

    def _cell_predictions(
        self,
        x: np.ndarray,
        h: np.ndarray,
        sqrt_h: float,
        entropy_per_baryon: float,
        varphi: float,
    ) -> np.ndarray:
        return self.master.cell_predictions(
            x, h, sqrt_h, entropy_per_baryon, varphi
        )

    def _recover_cells(
        self,
        h: Array,
        D_N: Array,
        D_D: Array,
        momentum: Array,
        energy: Array,
        P_D: Array,
        entropy_per_baryon: Array,
        varphi: Array,
    ) -> tuple[FluidPrimitive, CarrierPrimitive, Array, int]:
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = inverse_metric(h)[2].reshape(count)
        D_N_flat = D_N.reshape(count)
        D_D_flat = D_D.reshape(count)
        S_flat = momentum.reshape(3, count)
        E_flat = energy.reshape(count)
        P_flat = P_D.reshape(3, count)
        entropy_flat = entropy_per_baryon.reshape(count)
        phi_flat = varphi.reshape(count)
        guess_flat = None
        if self._primitive_guess is not None and self._primitive_guess.shape == (8,) + shape:
            guess_flat = self._primitive_guess.reshape(8, count)
        solved = np.empty((8, count), dtype=float)
        residuals = np.empty(count, dtype=float)
        failed = 0
        for cell in range(count):
            target = np.concatenate(
                ([D_N_flat[cell], D_D_flat[cell]], S_flat[:, cell], [E_flat[cell]], P_flat[:, cell])
            )
            energy_scale = max(abs(E_flat[cell]), np.linalg.norm(S_flat[:, cell]), 1.0e-10)
            carrier_scale = max(abs(D_D_flat[cell]), 1.0e-10)
            momentum_scale = max(np.linalg.norm(P_flat[:, cell]), carrier_scale, 1.0e-10)
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    carrier_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    momentum_scale,
                    momentum_scale,
                    momentum_scale,
                ]
            )
            if guess_flat is not None:
                x = guess_flat[:, cell].copy()
            else:
                n0 = max(
                    D_N_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.grhd.parameters.density_floor,
                )
                d0 = max(
                    D_D_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.master_parameters.carrier_floor,
                )
                denominator = max(E_flat[cell] + 1.0e-8, 1.0e-8)
                velocity = np.linalg.solve(h_flat[:, :, cell], S_flat[:, cell]) / denominator
                speed_sq = float(velocity @ h_flat[:, :, cell] @ velocity)
                if speed_sq >= 0.5:
                    velocity *= np.sqrt(0.5 / max(speed_sq, 1.0e-300))
                W0 = 1.0 / np.sqrt(1.0 - float(velocity @ h_flat[:, :, cell] @ velocity))
                q0 = W0 * velocity
                x = np.concatenate(([np.log(n0)], q0, [np.log(d0)], q0))

            function = lambda value: self._cell_predictions(
                value,
                h_flat[:, :, cell],
                float(sqrt_flat[cell]),
                float(entropy_flat[cell]),
                float(phi_flat[cell]),
            )
            for _ in range(self.production_parameters.recovery_iterations):
                prediction = function(x)
                residual = (prediction - target) / scale
                norm = float(np.max(np.abs(residual)))
                if norm < max(self.production_parameters.recovery_tolerance, 2.0e-8):
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.lstsq(jac, -residual, rcond=1.0e-12)[0]
                accepted = False
                for power in range(10):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[4] = np.clip(trial[4], -700.0, 700.0)
                    trial_residual = (function(trial) - target) / scale
                    if np.max(np.abs(trial_residual)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            residuals[cell] = final
            if not np.isfinite(final) or final > 2.0e-6:
                failed += 1
            solved[:, cell] = x

        self._primitive_guess = solved.reshape((8,) + shape)
        n = np.exp(solved[0]).reshape(shape)
        d = np.exp(solved[4]).reshape(shape)
        q_N = solved[1:4].reshape((3,) + shape)
        q_D = solved[5:8].reshape((3,) + shape)
        W_N = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N))
        W_D = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D))
        internal = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            internal[index] = self._entropy_internal(
                float(n[index]), float(entropy_per_baryon[index])
            )
        fluid = FluidPrimitive(n, internal, q_N / W_N[None, ...], np.zeros(shape))
        carrier = CarrierPrimitive(d, q_D / W_D[None, ...])
        return fluid, carrier, residuals.reshape(shape), failed

    def _recover_cells_from_energy(
        self,
        h: Array,
        D_N: Array,
        D_D: Array,
        momentum: Array,
        energy: Array,
        P_D: Array,
        entropy_guess: Array,
        varphi: Array,
    ) -> tuple[FluidPrimitive, CarrierPrimitive, Array, Array, int]:
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = inverse_metric(h)[2].reshape(count)
        D_N_flat = D_N.reshape(count)
        D_D_flat = D_D.reshape(count)
        S_flat = momentum.reshape(3, count)
        E_flat = energy.reshape(count)
        P_flat = P_D.reshape(3, count)
        entropy_flat = entropy_guess.reshape(count)
        phi_flat = varphi.reshape(count)
        guess_flat = None
        if self._primitive_guess is not None and self._primitive_guess.shape == (8,) + shape:
            guess_flat = self._primitive_guess.reshape(8, count)
        solved = np.empty((9, count), dtype=float)
        residuals = np.empty(count, dtype=float)
        failed = 0
        for cell in range(count):
            target = np.concatenate(
                ([D_N_flat[cell], D_D_flat[cell]], S_flat[:, cell], [E_flat[cell]], P_flat[:, cell])
            )
            energy_scale = max(abs(E_flat[cell]), np.linalg.norm(S_flat[:, cell]), 1.0e-10)
            carrier_scale = max(abs(D_D_flat[cell]), 1.0e-10)
            momentum_scale = max(np.linalg.norm(P_flat[:, cell]), carrier_scale, 1.0e-10)
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    carrier_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    momentum_scale,
                    momentum_scale,
                    momentum_scale,
                ]
            )
            if guess_flat is not None:
                primitive = guess_flat[:, cell]
                x = np.concatenate(
                    ([primitive[0], entropy_flat[cell]], primitive[1:4], [primitive[4]], primitive[5:8])
                )
            else:
                n0 = max(
                    D_N_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.grhd.parameters.density_floor,
                )
                d0 = max(
                    D_D_flat[cell] / max(sqrt_flat[cell], 1.0e-300),
                    self.master_parameters.carrier_floor,
                )
                denominator = max(E_flat[cell] + 1.0e-8, 1.0e-8)
                velocity = np.linalg.solve(h_flat[:, :, cell], S_flat[:, cell]) / denominator
                speed_sq = float(velocity @ h_flat[:, :, cell] @ velocity)
                if speed_sq >= 0.5:
                    velocity *= np.sqrt(0.5 / max(speed_sq, 1.0e-300))
                W0 = 1.0 / np.sqrt(1.0 - float(velocity @ h_flat[:, :, cell] @ velocity))
                q0 = W0 * velocity
                x = np.concatenate(([np.log(n0), entropy_flat[cell]], q0, [np.log(d0)], q0))

            def function(value: np.ndarray) -> np.ndarray:
                primitive = np.concatenate(
                    ([value[0]], value[2:5], [value[5]], value[6:9])
                )
                return self._cell_predictions(
                    primitive,
                    h_flat[:, :, cell],
                    float(sqrt_flat[cell]),
                    float(value[1]),
                    float(phi_flat[cell]),
                )

            for _ in range(self.production_parameters.recovery_iterations):
                residual = (function(x) - target) / scale
                norm = float(np.max(np.abs(residual)))
                if norm < max(self.production_parameters.recovery_tolerance, 2.0e-9):
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.solve(jac, -residual)
                accepted = False
                for power in range(12):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[5] = np.clip(trial[5], -700.0, 700.0)
                    if np.max(np.abs((function(trial) - target) / scale)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            residuals[cell] = final
            if not np.isfinite(final) or final > 2.0e-7:
                failed += 1
            solved[:, cell] = x

        density = np.exp(solved[0]).reshape(shape)
        entropy = solved[1].reshape(shape)
        q_N = solved[2:5].reshape((3,) + shape)
        carrier_density = np.exp(solved[5]).reshape(shape)
        q_D = solved[6:9].reshape((3,) + shape)
        W_N = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N))
        W_D = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D))
        internal = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            internal[index] = self._entropy_internal(
                float(density[index]), float(entropy[index])
            )
        fluid = FluidPrimitive(
            density, internal, q_N / W_N[None, ...], np.zeros(shape)
        )
        carrier = CarrierPrimitive(
            carrier_density, q_D / W_D[None, ...]
        )
        self._primitive_guess = np.concatenate(
            (solved[0:1], solved[2:5], solved[5:6], solved[6:9]), axis=0
        ).reshape((8,) + shape)
        return fluid, carrier, entropy, residuals.reshape(shape), failed

    def _scalar_stress(self, h: Array, state: NullothProductionState):
        potential = self.master.scalar_potential(state.scalar.varphi)
        return self.scalar_system.stress(h, state.scalar, potential)

    def _project_to_master_manifold(
        self, state: NullothProductionState
    ) -> NullothProductionState:
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        source_charge = state.target_charge / sqrt_h
        vector_stress = self.system.vector_stress(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            source_charge,
        )
        scalar_stress = self._scalar_stress(h, state)
        entropy_guess = state.matter.entropy / np.maximum(state.matter.D, 1.0e-300)
        fluid, carrier, entropy, residual, failed = self._recover_cells_from_energy(
            h,
            state.matter.D,
            state.target_charge / self.master_parameters.carrier_charge,
            state.matter.momentum / sqrt_h[None, ...]
            - vector_stress.momentum
            - scalar_stress.momentum,
            state.matter.energy / sqrt_h - vector_stress.rho - scalar_stress.rho,
            state.target_current,
            entropy_guess,
            state.scalar.varphi,
        )
        if failed:
            raise FloatingPointError(
                "Nulloth energy-manifold projection failed: "
                f"cells={failed}, residual={float(np.max(residual)):.3e}"
            )
        master = self.master.evaluate(h, fluid, carrier, state.scalar.varphi)
        if np.any(
            master.relative_lorentz_factor
            > self.master_parameters.maximum_relative_lorentz_factor
        ):
            raise FloatingPointError(
                "Nulloth energy-manifold projection left the counterflow domain"
            )
        projected_entropy = state.matter.D * entropy
        correction = projected_entropy - state.matter.entropy
        self.last_projection_entropy_change += float(self.grid.integrate(correction))
        self.last_projection_minimum_change = min(
            self.last_projection_minimum_change, float(np.min(correction))
        )
        state.matter.entropy = projected_entropy
        return state

    def initialize(
        self,
        geometry: CCZ4State,
        fluid: FluidPrimitive,
        *,
        carrier: CarrierPrimitive | None = None,
        a: Array | None = None,
        pi_A: Array | None = None,
        b: Array | None = None,
        pi_B: Array | None = None,
        varphi: Array | float = 0.0,
        scalar_momentum: Array | float = 0.0,
        scalar_gradient: Array | None = None,
    ) -> NullothProductionState:
        h, _ = self.ccz4.physical_geometry(geometry)
        _, _, sqrt_h = inverse_metric(h)
        a = self.grid.zeros((3,)) if a is None else np.asarray(a, dtype=float)
        b = self.grid.zeros((3,)) if b is None else np.asarray(b, dtype=float)
        pi_A = self.grid.zeros((3,)) if pi_A is None else np.asarray(pi_A, dtype=float)
        pi_B = self.grid.zeros((3,)) if pi_B is None else np.asarray(pi_B, dtype=float)
        scalar = self.scalar_system.initialize(varphi, scalar_momentum, scalar_gradient)
        if carrier is None:
            carrier = CarrierPrimitive(
                np.maximum(
                    self.master_parameters.target_fraction * fluid.baryon_density,
                    self.master_parameters.carrier_floor,
                ),
                np.array(fluid.velocity, copy=True),
            )
        master = self.master.evaluate(h, fluid, carrier, scalar.varphi)
        D_D, P_D = self.master.conserved_carrier(h, master)
        target_charge = self.master_parameters.carrier_charge * D_D
        longitudinal_A, longitudinal_B = self.system.initial_longitudinal_charges(
            h, pi_A, pi_B
        )
        vector_stress = self.system.vector_stress(
            h,
            a,
            pi_A,
            b,
            pi_B,
            longitudinal_A,
            longitudinal_B,
            master.source_charge_eulerian,
        )
        scalar_stress = self.scalar_system.stress(
            h, scalar, self.master.scalar_potential(scalar.varphi)
        )
        total = _add_stress(_add_stress(master.stress, vector_stress), scalar_stress)
        W_N = lorentz_factor(h, fluid.velocity)
        D_N = sqrt_h * fluid.baryon_density * W_N
        matter = Theory3MatterState(
            D_N,
            sqrt_h[None, ...] * total.momentum,
            sqrt_h * total.rho,
            D_N * master.entropy_per_baryon,
            D_N * fluid.sigma,
        )
        state = NullothProductionState(
            geometry,
            matter,
            a.copy(),
            pi_A.copy(),
            b.copy(),
            pi_B.copy(),
            longitudinal_A,
            longitudinal_B,
            self.grid.zeros(),
            self.grid.zeros(),
            target_charge,
            P_D,
            scalar,
            geometry.time,
        )
        q_N = W_N[None, ...] * fluid.velocity
        W_D = lorentz_factor(h, carrier.velocity)
        q_D = W_D[None, ...] * carrier.velocity
        self._primitive_guess = np.concatenate(
            (
                np.log(fluid.baryon_density)[None, ...],
                q_N,
                np.log(carrier.number_density)[None, ...],
                q_D,
            ),
            axis=0,
        )
        self.recover(state)
        return state

    def vacuum_state(self) -> NullothProductionState:
        fluid = FluidPrimitive(
            np.full(self.grid.shape, self.grhd.parameters.density_floor),
            np.full(self.grid.shape, self.grhd.parameters.internal_energy_floor),
            self.grid.zeros((3,)),
            self.grid.zeros(),
        )
        carrier = CarrierPrimitive(
            np.full(self.grid.shape, self.master_parameters.carrier_floor),
            self.grid.zeros((3,)),
        )
        return self.initialize(self.ccz4.flat_state(), fluid, carrier=carrier)

    def recover(self, state: NullothProductionState) -> NullothRecovery:
        h, _ = self.ccz4.physical_geometry(state.geometry)
        _, _, sqrt_h = inverse_metric(h)
        source_charge = state.target_charge / sqrt_h
        vector_stress = self.system.vector_stress(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            source_charge,
        )
        scalar_stress = self._scalar_stress(h, state)
        material_energy = state.matter.energy / sqrt_h - vector_stress.rho - scalar_stress.rho
        material_momentum = (
            state.matter.momentum / sqrt_h[None, ...]
            - vector_stress.momentum
            - scalar_stress.momentum
        )
        D_D = state.target_charge / self.master_parameters.carrier_charge
        entropy_per_baryon = state.matter.entropy / np.maximum(state.matter.D, 1.0e-300)
        fluid, carrier, residual, failed = self._recover_cells(
            h,
            state.matter.D,
            D_D,
            material_momentum,
            material_energy,
            state.target_current,
            entropy_per_baryon,
            state.scalar.varphi,
        )
        fluid = FluidPrimitive(
            fluid.baryon_density,
            fluid.specific_internal_energy,
            fluid.velocity,
            state.matter.tracer / np.maximum(state.matter.D, 1.0e-300),
        )
        master = self.master.evaluate(h, fluid, carrier, state.scalar.varphi)
        legendre_trace = master.B_N + master.B_D
        legendre_disc = np.sqrt(
            np.maximum((master.B_N - master.B_D) ** 2 + 4.0 * master.entrainment**2, 0.0)
        )
        legendre_min = 0.5 * (legendre_trace - legendre_disc)
        thermo_trace = master.thermodynamic_hessian_nn + master.thermodynamic_hessian_dd
        thermo_disc = np.sqrt(
            np.maximum(
                (master.thermodynamic_hessian_nn - master.thermodynamic_hessian_dd) ** 2
                + 4.0 * master.thermodynamic_hessian_nd**2,
                0.0,
            )
        )
        thermo_min = 0.5 * (thermo_trace - thermo_disc)
        scalar_constraint = self.scalar_system.constraint(state.scalar)
        invalid = (
            (legendre_min <= self.master_parameters.convexity_floor)
            | (thermo_min <= self.master_parameters.convexity_floor)
            | (master.relative_lorentz_factor < 1.0 - 1.0e-10)
            | (
                master.relative_lorentz_factor
                > self.master_parameters.maximum_relative_lorentz_factor
            )
            | (master.effective_mass_squared < self.master_parameters.effective_mass_floor)
            | (np.abs(state.scalar.varphi) > self.master_parameters.scalar_amplitude_limit)
        )
        failed += int(np.count_nonzero(invalid))
        report = NullothRecoveryReport(
            failed,
            int(
                np.count_nonzero(
                    state.matter.D
                    <= sqrt_h
                    * self.grhd.parameters.density_floor
                    * (1.0 + 1.0e-10)
                )
            ),
            float(np.max(residual)),
            float(np.min(carrier.number_density)),
            float(np.max(master.relative_lorentz_factor)),
            float(np.min(legendre_min)),
            float(np.min(thermo_min)),
            float(np.min(master.effective_mass_squared)),
            float(np.max(master.coupling_coefficient)),
            float(np.max(np.abs(state.scalar.varphi))),
            float(np.sqrt(np.mean(scalar_constraint**2))),
        )
        if failed:
            raise FloatingPointError(
                "Nulloth multifield recovery failed: "
                f"cells={failed}, residual={report.maximum_residual:.3e}, "
                f"legendre={report.minimum_legendre_eigenvalue:.3e}, "
                f"thermo={report.minimum_thermodynamic_eigenvalue:.3e}, "
                f"m_eff2={report.minimum_effective_mass_squared:.3e}"
            )
        material_stress = master.stress
        if not self.production_parameters.atmosphere_gravitates:
            atmosphere = (
                state.matter.D
                <= sqrt_h
                * self.grhd.parameters.density_floor
                * (1.0 + 1.0e-10)
            ) & (
                carrier.number_density
                <= self.master_parameters.carrier_floor * (1.0 + 1.0e-10)
            )
            material_stress = type(master.stress)(
                np.where(atmosphere, 0.0, master.stress.rho),
                np.where(atmosphere[None, ...], 0.0, master.stress.momentum),
                np.where(atmosphere[None, None, ...], 0.0, master.stress.stress),
                np.where(atmosphere, 0.0, master.stress.trace),
            )
        total = _add_stress(_add_stress(material_stress, vector_stress), scalar_stress)
        sound = np.sqrt(
            np.clip(
                self.master_parameters.gamma_ad
                * self.master.thermal_pressure(
                    fluid.baryon_density, fluid.specific_internal_energy
                )
                / np.maximum(
                    self.master.thermal_energy(
                        fluid.baryon_density, fluid.specific_internal_energy
                    )
                    + self.master.thermal_pressure(
                        fluid.baryon_density, fluid.specific_internal_energy
                    ),
                    1.0e-300,
                ),
                0.0,
                1.0,
            )
        )
        return NullothRecovery(
            fluid,
            carrier,
            master,
            entropy_per_baryon,
            sound,
            material_stress,
            vector_stress,
            scalar_stress,
            total,
            master.source_charge_eulerian,
            master.source_current_up,
            report,
        )

    def _pack(self, state: NullothProductionState) -> tuple[Array, ...]:
        g = state.geometry
        m = state.matter
        return (
            g.conformal_metric,
            g.conformal_A,
            g.conformal_factor,
            g.trace_K,
            g.theta,
            g.gamma_hat,
            g.lapse,
            g.shift,
            g.shift_driver,
            m.D,
            m.momentum,
            m.energy,
            m.entropy,
            m.tracer,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            state.target_current,
            state.scalar.varphi,
            state.scalar.momentum,
            state.scalar.gradient,
        )

    def _unpack(
        self, values: tuple[Array, ...], time: float
    ) -> NullothProductionState:
        if len(values) != 27:
            raise ValueError(f"Nulloth state needs 27 array groups, got {len(values)}")
        geometry = CCZ4State(*values[:9], time)
        matter = Theory3MatterState(*values[9:14])
        scalar = NullothScalarState(values[24], values[25], values[26])
        return NullothProductionState(
            geometry,
            matter,
            values[14],
            values[15],
            values[16],
            values[17],
            values[18],
            values[19],
            values[20],
            values[21],
            values[22],
            values[23],
            scalar,
            time,
        )

    def rhs(self, time: float, values: tuple[Array, ...]) -> tuple[Array, ...]:
        state = self._unpack(values, time)
        recovery = self.recover(state)
        h, K = self.ccz4.physical_geometry(state.geometry)
        sources = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        geometry_rhs = self.ccz4.rhs(time, values[:9], sources)
        matter_rhs = list(self._conservative_rhs(state, recovery, h, K))
        target_charge_rhs, target_momentum_rhs = self._target_rhs(
            state, recovery, h, K
        )
        field_rhs = self.system.rhs_fields(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            recovery.target_charge_eulerian,
            recovery.target_current_up,
            state.geometry.lapse,
            state.geometry.shift,
            self.production_parameters.cleaning_damping,
        )
        scalar_rhs = self.scalar_system.rhs(
            h,
            state.geometry.trace_K,
            state.geometry.lapse,
            state.geometry.shift,
            state.scalar,
            recovery.master.scalar_equation_source,
        )
        return (
            geometry_rhs
            + tuple(matter_rhs)
            + field_rhs
            + (target_charge_rhs, target_momentum_rhs)
            + scalar_rhs
        )

    def _apply_carrier_drag(
        self, state: NullothProductionState, dt: float
    ) -> NullothProductionState:
        before = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sqrt_h = inverse_metric(h)[2]
        shape = self.grid.shape
        count = int(np.prod(shape))
        h_flat = h.reshape(3, 3, count)
        sqrt_flat = sqrt_h.reshape(count)
        D_N_flat = state.matter.D.reshape(count)
        D_D_flat = (
            state.target_charge / self.master_parameters.carrier_charge
        ).reshape(count)
        material_energy = (
            state.matter.energy / sqrt_h
            - before.vector_stress.rho
            - before.scalar_stress.rho
        ).reshape(count)
        material_momentum = (
            state.matter.momentum / sqrt_h[None, ...]
            - before.vector_stress.momentum
            - before.scalar_stress.momentum
        ).reshape(3, count)
        old_entropy = before.entropy_per_baryon.reshape(count)
        old_relative_gamma = before.master.relative_lorentz_factor.reshape(count)
        W_N = lorentz_factor(h, before.fluid.velocity).reshape(count)
        W_D = lorentz_factor(h, before.carrier.velocity).reshape(count)
        q_N_old = (W_N.reshape(shape)[None, ...] * before.fluid.velocity).reshape(3, count)
        q_D_old = (W_D.reshape(shape)[None, ...] * before.carrier.velocity).reshape(3, count)
        lapse_flat = state.geometry.lapse.reshape(count)
        phi_flat = state.scalar.varphi.reshape(count)
        solved = np.empty((9, count), dtype=float)
        entropy_increase = np.empty(count, dtype=float)
        for cell in range(count):
            if old_relative_gamma[cell] - 1.0 <= 1.0e-3:
                solved[:, cell] = np.concatenate(
                    (
                        [np.log(before.fluid.baryon_density.reshape(count)[cell])],
                        [old_entropy[cell]],
                        q_N_old[:, cell],
                        [np.log(before.carrier.number_density.reshape(count)[cell])],
                        q_D_old[:, cell],
                    )
                )
                entropy_increase[cell] = 0.0
                continue
            decay = float(
                np.exp(
                    -dt
                    * max(lapse_flat[cell], 0.0)
                    / self.master_parameters.relaxation_time
                )
            )
            relative_target = decay * (q_D_old[:, cell] - q_N_old[:, cell])
            target = np.concatenate(
                (
                    [D_N_flat[cell], D_D_flat[cell]],
                    material_momentum[:, cell],
                    [material_energy[cell]],
                    relative_target,
                )
            )
            energy_scale = max(
                abs(material_energy[cell]),
                np.linalg.norm(material_momentum[:, cell]),
                1.0e-10,
            )
            scale = np.array(
                [
                    max(abs(D_N_flat[cell]), 1.0e-10),
                    max(abs(D_D_flat[cell]), 1.0e-10),
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    energy_scale,
                    1.0,
                    1.0,
                    1.0,
                ]
            )
            x = np.concatenate(
                (
                    [np.log(before.fluid.baryon_density.reshape(count)[cell])],
                    [old_entropy[cell]],
                    q_N_old[:, cell],
                    [np.log(before.carrier.number_density.reshape(count)[cell])],
                    q_D_old[:, cell],
                )
            )

            def function(value: np.ndarray) -> np.ndarray:
                primitive = np.concatenate(
                    ([value[0]], value[2:5], [value[5]], value[6:9])
                )
                conservative = self._cell_predictions(
                    primitive,
                    h_flat[:, :, cell],
                    float(sqrt_flat[cell]),
                    float(value[1]),
                    float(phi_flat[cell]),
                )
                return np.concatenate((conservative[:6], value[6:9] - value[2:5]))

            for _ in range(self.production_parameters.recovery_iterations):
                residual = (function(x) - target) / scale
                norm = float(np.max(np.abs(residual)))
                if norm < 2.0e-9:
                    break
                jac = self._cell_jacobian(function, x) / scale[:, None]
                delta = np.linalg.solve(jac, -residual)
                accepted = False
                for power in range(12):
                    trial = x + delta * (0.5**power)
                    trial[0] = np.clip(trial[0], -700.0, 700.0)
                    trial[5] = np.clip(trial[5], -700.0, 700.0)
                    if np.max(np.abs((function(trial) - target) / scale)) < norm:
                        x = trial
                        accepted = True
                        break
                if not accepted:
                    break
            final = float(np.max(np.abs((function(x) - target) / scale)))
            if not np.isfinite(final) or final > 2.0e-7:
                raise FloatingPointError(
                    f"Nulloth implicit drag solve failed in cell {cell}: {final:.3e}"
                )
            entropy_increase[cell] = x[1] - old_entropy[cell]
            if entropy_increase[cell] < -2.0e-9:
                raise FloatingPointError(
                    "Nulloth drag collision map would decrease entropy: "
                    f"cell={cell}, delta_sigma={entropy_increase[cell]:.3e}"
                )
            solved[:, cell] = x

        density = np.exp(solved[0]).reshape(shape)
        entropy = solved[1].reshape(shape)
        q_N = solved[2:5].reshape((3,) + shape)
        carrier_density = np.exp(solved[5]).reshape(shape)
        q_D = solved[6:9].reshape((3,) + shape)
        W_N_new = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_N, q_N))
        W_D_new = np.sqrt(1.0 + np.einsum("ij...,i...,j...->...", h, q_D, q_D))
        internal = np.empty(shape, dtype=float)
        for index in np.ndindex(shape):
            internal[index] = self._entropy_internal(
                float(density[index]), float(entropy[index])
            )
        fluid = FluidPrimitive(
            density,
            internal,
            q_N / W_N_new[None, ...],
            before.fluid.sigma,
        )
        carrier = CarrierPrimitive(carrier_density, q_D / W_D_new[None, ...])
        master = self.master.evaluate(h, fluid, carrier, state.scalar.varphi)
        _, carrier_momentum = self.master.conserved_carrier(h, master)
        updated = NullothProductionState(
            state.geometry,
            Theory3MatterState(
                state.matter.D,
                state.matter.momentum,
                state.matter.energy,
                state.matter.D * entropy,
                state.matter.tracer,
            ),
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            carrier_momentum,
            state.scalar,
            state.time,
        )
        self._primitive_guess = np.concatenate(
            (
                np.log(density)[None, ...],
                q_N,
                np.log(carrier_density)[None, ...],
                q_D,
            ),
            axis=0,
        )
        self.recover(updated)
        entropy_change_density = state.matter.D * entropy_increase.reshape(shape)
        self.last_drag_entropy_change = float(self.grid.integrate(entropy_change_density))
        self.last_drag_heat = float(
            self.grid.integrate(
                sqrt_h
                * before.master.temperature
                * before.fluid.baryon_density
                * entropy_increase.reshape(shape)
            )
        )
        return updated

    def _apply_damping(
        self, state: NullothProductionState, dt: float
    ) -> NullothProductionState:
        p = self.theory_parameters
        if p.gamma_W == 0.0 and p.gamma_B == 0.0:
            self.last_damping_report = FiniteDampingReport()
            return state
        before = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sqrt_h = inverse_metric(h)[2]
        gauss_before = self.system.gauss_constraints(
            h, state.pi_A, state.pi_B, state.longitudinal_A, state.longitudinal_B
        )
        damping = self.system.damping_step(
            h,
            state.a,
            state.pi_A,
            state.b,
            state.pi_B,
            state.longitudinal_A,
            state.longitudinal_B,
            before.fluid,
            state.geometry.lapse,
            dt,
        )
        updated = NullothProductionState(
            state.geometry,
            Theory3MatterState(
                state.matter.D.copy(),
                state.matter.momentum.copy(),
                state.matter.energy.copy(),
                state.matter.entropy.copy(),
                state.matter.tracer.copy(),
            ),
            state.a,
            damping.pi_A,
            state.b,
            damping.pi_B,
            damping.longitudinal_A,
            damping.longitudinal_B,
            state.cleaning_A,
            state.cleaning_B,
            state.target_charge,
            state.target_current,
            state.scalar,
            state.time,
        )
        heated_internal = before.fluid.specific_internal_energy + damping.irreversible_heat / np.maximum(
            before.fluid.baryon_density, 1.0e-300
        )
        heated = FluidPrimitive(
            before.fluid.baryon_density,
            heated_internal,
            before.fluid.velocity,
            before.fluid.sigma,
        )
        updated.matter.entropy = updated.matter.D * self.master.specific_entropy(
            heated.baryon_density, heated.specific_internal_energy
        )
        after = self.recover(updated)
        gauss_after = self.system.gauss_constraints(
            h, updated.pi_A, updated.pi_B, updated.longitudinal_A, updated.longitudinal_B
        )
        self.last_damping_report = FiniteDampingReport(
            vector_energy_change=float(
                self.grid.integrate(sqrt_h * (after.vector_stress.rho - before.vector_stress.rho))
            ),
            vector_momentum_change=tuple(
                self.grid.integrate(
                    sqrt_h * (after.vector_stress.momentum[i] - before.vector_stress.momentum[i])
                )
                for i in range(3)
            ),
            irreversible_heat=float(self.grid.integrate(sqrt_h * damping.irreversible_heat)),
            total_energy_balance_error=0.0,
            maximum_gauss_change=max(
                float(np.max(np.abs(gauss_after[0] - gauss_before[0]))),
                float(np.max(np.abs(gauss_after[1] - gauss_before[1]))),
            ),
        )
        return updated

    def step(
        self, state: NullothProductionState, dt: float
    ) -> NullothProductionState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        entropy_before_step = float(self.grid.integrate(state.matter.entropy))
        self.last_projection_entropy_change = 0.0
        self.last_projection_minimum_change = 0.0
        values = self._pack(state)
        time = state.time

        def stage(
            base: tuple[Array, ...],
            slope: tuple[Array, ...],
            factor: float,
            stage_time: float,
        ) -> tuple[Array, ...]:
            candidate = self._unpack(
                tuple(y + factor * dt * k for y, k in zip(base, slope)),
                stage_time,
            )
            candidate.geometry = self.ccz4.project_algebraic(candidate.geometry)
            candidate.geometry.time = stage_time
            candidate.scalar.gradient = self.grid.gradient(candidate.scalar.varphi)
            candidate = self._project_to_master_manifold(candidate)
            return self._pack(candidate)

        k1 = tuple(self.rhs(time, values))
        y2 = stage(values, k1, 0.5, time + 0.5 * dt)
        k2 = tuple(self.rhs(time + 0.5 * dt, y2))
        y3 = stage(values, k2, 0.5, time + 0.5 * dt)
        k3 = tuple(self.rhs(time + 0.5 * dt, y3))
        y4 = stage(values, k3, 1.0, time + dt)
        k4 = tuple(self.rhs(time + dt, y4))
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
        result.scalar.gradient = self.grid.gradient(result.scalar.varphi)
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
        if self.last_step_entropy_change < -1.0e-10:
            raise FloatingPointError(
                "Nulloth projected finite-volume step decreased total entropy: "
                f"delta={self.last_step_entropy_change:.3e}"
            )
        return result

    def recommended_dt(
        self, state: NullothProductionState, cfl: float = 0.1
    ) -> float:
        self.recover(state)
        light_dt = cfl * min(self.grid.spacing) / np.sqrt(2.0 * self.grid.ndim)
        return min(light_dt, 0.2 * self.master_parameters.relaxation_time)

    def diagnostics(
        self, state: NullothProductionState
    ) -> dict[str, float | int]:
        recovery = self.recover(state)
        h, _ = self.ccz4.physical_geometry(state.geometry)
        sources = (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        )
        ccz = self.ccz4.diagnostics(state.geometry, sources)
        gauss_A, gauss_B = self.system.gauss_constraints(
            h, state.pi_A, state.pi_B, state.longitudinal_A, state.longitudinal_B
        )
        scalar_diag = self.scalar_system.diagnostics(
            h,
            state.scalar,
            self.master.scalar_potential(state.scalar.varphi),
        )
        D_D = state.target_charge / self.master_parameters.carrier_charge
        return {
            "time": state.time,
            "hamiltonian_l2": ccz.hamiltonian_l2,
            "momentum_l2": ccz.momentum_l2,
            "theta_l2": ccz.theta_l2,
            "z_l2": ccz.z_l2,
            "gauss_A_l2": float(np.sqrt(np.mean(gauss_A**2))),
            "gauss_B_l2": float(np.sqrt(np.mean(gauss_B**2))),
            "scalar_constraint_l2": scalar_diag.constraint_l2,
            "scalar_constraint_linf": scalar_diag.constraint_linf,
            "maximum_scalar_amplitude": scalar_diag.maximum_amplitude,
            "minimum_scalar_energy_density": scalar_diag.minimum_energy_density,
            "minimum_effective_mass_squared": recovery.report.minimum_effective_mass_squared,
            "maximum_coupling_coefficient": recovery.report.maximum_coupling_coefficient,
            "baryon_mass": float(np.sum(state.matter.D) * self.grid.cell_volume),
            "carrier_number": float(np.sum(D_D) * self.grid.cell_volume),
            "target_charge": float(np.sum(state.target_charge) * self.grid.cell_volume),
            "combined_energy": float(np.sum(state.matter.energy) * self.grid.cell_volume),
            "entropy_integral": float(np.sum(state.matter.entropy) * self.grid.cell_volume),
            "minimum_carrier_density": recovery.report.minimum_carrier_density,
            "maximum_relative_lorentz_factor": recovery.report.maximum_relative_lorentz_factor,
            "minimum_legendre_eigenvalue": recovery.report.minimum_legendre_eigenvalue,
            "minimum_thermodynamic_eigenvalue": recovery.report.minimum_thermodynamic_eigenvalue,
            "recovery_maximum_residual": recovery.report.maximum_residual,
            "recovery_failures": recovery.report.failed_cells,
            "damping_vector_energy_change": self.last_damping_report.vector_energy_change,
            "damping_heat": self.last_damping_report.irreversible_heat,
            "damping_gauss_change": self.last_damping_report.maximum_gauss_change,
            "drag_heat": self.last_drag_heat,
            "drag_entropy_change": self.last_drag_entropy_change,
            "projection_entropy_change": self.last_projection_entropy_change,
            "projection_minimum_entropy_density_change": self.last_projection_minimum_change,
            "step_entropy_change": self.last_step_entropy_change,
            "carrier_boundary_number_outflow": self.last_carrier_boundary_number_outflow,
            "carrier_number_change": self.last_carrier_number_change,
            "carrier_number_balance_residual": self.last_carrier_number_balance_residual,
        }


def save_nulloth_production_state(
    path: str | Path,
    state: NullothProductionState,
    *,
    parameters: NullothParameters | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    g = state.geometry
    m = state.matter
    payload: dict[str, Any] = {
        "format": "causal_repair.production.nulloth.v0.1",
        **(metadata or {}),
    }
    if parameters is not None:
        payload["nulloth_parameters"] = asdict(parameters)
    return save_checkpoint(
        path,
        metadata=payload,
        conformal_metric=g.conformal_metric,
        conformal_A=g.conformal_A,
        conformal_factor=g.conformal_factor,
        trace_K=g.trace_K,
        theta=g.theta,
        gamma_hat=g.gamma_hat,
        lapse=g.lapse,
        shift=g.shift,
        shift_driver=g.shift_driver,
        D=m.D,
        total_momentum=m.momentum,
        total_energy=m.energy,
        entropy=m.entropy,
        tracer=m.tracer,
        a=state.a,
        pi_A=state.pi_A,
        b=state.b,
        pi_B=state.pi_B,
        longitudinal_A=state.longitudinal_A,
        longitudinal_B=state.longitudinal_B,
        cleaning_A=state.cleaning_A,
        cleaning_B=state.cleaning_B,
        carrier_source_charge=state.target_charge,
        carrier_canonical_momentum=state.target_current,
        nulloth_varphi=state.scalar.varphi,
        nulloth_scalar_momentum=state.scalar.momentum,
        nulloth_scalar_gradient=state.scalar.gradient,
        time=state.time,
    )


def load_nulloth_production_state(
    path: str | Path,
) -> tuple[NullothProductionState, dict[str, Any]]:
    arrays, metadata = load_checkpoint(path)
    if metadata.get("format") != "causal_repair.production.nulloth.v0.1":
        raise ValueError("checkpoint is not a Nulloth v0.1 production state")
    time = float(arrays["time"])
    geometry = CCZ4State(
        arrays["conformal_metric"],
        arrays["conformal_A"],
        arrays["conformal_factor"],
        arrays["trace_K"],
        arrays["theta"],
        arrays["gamma_hat"],
        arrays["lapse"],
        arrays["shift"],
        arrays["shift_driver"],
        time,
    )
    matter = Theory3MatterState(
        arrays["D"],
        arrays["total_momentum"],
        arrays["total_energy"],
        arrays["entropy"],
        arrays["tracer"],
    )
    scalar = NullothScalarState(
        arrays["nulloth_varphi"],
        arrays["nulloth_scalar_momentum"],
        arrays["nulloth_scalar_gradient"],
    )
    return (
        NullothProductionState(
            geometry,
            matter,
            arrays["a"],
            arrays["pi_A"],
            arrays["b"],
            arrays["pi_B"],
            arrays["longitudinal_A"],
            arrays["longitudinal_B"],
            arrays["cleaning_A"],
            arrays["cleaning_B"],
            arrays["carrier_source_charge"],
            arrays["carrier_canonical_momentum"],
            scalar,
            time,
        ),
        metadata,
    )
