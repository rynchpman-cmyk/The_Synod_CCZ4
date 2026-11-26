"""Nulloth two-current master action with carrier-induced scalarization.

This module extends the Theory 3.3 M1 master function by a canonical scalar
order parameter ``varphi``.  The coupling is part of the invariant material
master action, not an external force or energy reservoir.

Conventions
-----------
Metric signature is ``(-,+,+,+)``.  The material master function is

    Lambda_N = Lambda_M1 + 1/2 C(n,d,W) varphi**2,

where ``W = x**2 - n*d >= 0``.  The canonical scalar sector has

    L_varphi = -1/2 grad(varphi)^2 - V0(varphi),
    V0 = 1/2 m_varphi**2 varphi**2 + 1/4 lambda_varphi varphi**4.

Consequently the scalar equation is

    box(varphi) = dV0/dvarphi - dLambda_N/dvarphi
                = (m_varphi**2 - C) varphi + lambda_varphi varphi**3.

The first runnable backend intentionally keeps the Einstein-Hilbert principal
part unchanged.  A non-minimal Jordan-frame extension requires a separate
frame-complete implementation and is not silently approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .adm import inverse_metric
from .grid import Array
from .matter import FluidPrimitive, lorentz_factor
from .theory2 import StressEnergy3p1
from .theory33 import CarrierPrimitive, Theory33MasterParameters


@dataclass(frozen=True)
class NullothParameters(Theory33MasterParameters):
    """Constitutive and scalar parameters for Nulloth.

    Units are inherited from the surrounding code's geometrized unit system.
    The three scalarization couplings are dimensionless after division by the
    declared scales.
    """

    scalar_mass: float = 0.5
    scalar_quartic: float = 1.0
    scalar_baryon_coupling: float = 0.0
    scalar_carrier_coupling: float = 2.0
    scalar_counterflow_coupling: float = 1.0
    scalar_baryon_scale: float = 1.0
    scalar_carrier_scale: float = 1.0
    scalar_counterflow_scale: float = 1.0
    scalar_amplitude_limit: float = 20.0
    effective_mass_floor: float = -1.0e6
    scalar_constraint_damping: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scalar_mass < 0.0:
            raise ValueError("scalar_mass must be nonnegative")
        if self.scalar_quartic <= 0.0:
            raise ValueError("scalar_quartic must be positive")
        if self.scalar_baryon_coupling < 0.0:
            raise ValueError("scalar_baryon_coupling must be nonnegative")
        if self.scalar_carrier_coupling < 0.0:
            raise ValueError("scalar_carrier_coupling must be nonnegative")
        if self.scalar_counterflow_coupling < 0.0:
            raise ValueError("scalar_counterflow_coupling must be nonnegative")
        if min(
            self.scalar_baryon_scale,
            self.scalar_carrier_scale,
            self.scalar_counterflow_scale,
        ) <= 0.0:
            raise ValueError("all Nulloth scalarization scales must be positive")
        if self.scalar_amplitude_limit <= 0.0:
            raise ValueError("scalar_amplitude_limit must be positive")
        if not np.isfinite(self.effective_mass_floor):
            raise ValueError("effective_mass_floor must be finite")
        if self.scalar_constraint_damping < 0.0:
            raise ValueError("scalar_constraint_damping must be nonnegative")


NULLOTH_DISCOVERY_PARAMETERS = NullothParameters(
    scalar_mass=0.4,
    scalar_quartic=50.0,
    scalar_baryon_coupling=0.0,
    scalar_carrier_coupling=12.0,
    scalar_counterflow_coupling=0.2,
    scalar_amplitude_limit=0.17,
)


@dataclass(frozen=True)
class NullothMasterState:
    entropy_per_baryon: Array
    temperature: Array
    relative_lorentz_factor: Array
    relative_invariant: Array
    lambda_mf: Array
    generalized_pressure: Array
    B_N: Array
    B_D: Array
    entrainment: Array
    chemical_N: Array
    chemical_D: Array
    thermodynamic_hessian_nn: Array
    thermodynamic_hessian_nd: Array
    thermodynamic_hessian_dd: Array
    stress: StressEnergy3p1
    carrier_momentum_down: Array
    carrier_momentum_normal: Array
    carrier_eulerian_density: Array
    source_charge_eulerian: Array
    source_current_up: Array
    drag_inertia: Array
    coupling_coefficient: Array
    effective_mass_squared: Array
    scalar_action_derivative: Array
    scalar_equation_source: Array


@dataclass(frozen=True)
class NullothCharacteristicScreen:
    """Necessary local characteristic screen, not a full qualification proof.

    The scalar has a canonical light-speed principal block.  Algebraic
    scalarization makes the complete principal matrix block triangular only in
    the frozen-coefficient limit; coincident-mode conditioning still requires
    an explicit coupled-symbol audit before a parameter domain is qualified.
    """

    speeds: np.ndarray
    maximum_absolute_speed: float
    eigenvector_condition: float
    legendre_minimum_eigenvalue: float
    thermodynamic_minimum_eigenvalue: float
    strongly_hyperbolic_screen: bool
    causal_screen: bool


@dataclass(frozen=True)
class NullothCoupledCharacteristicAudit:
    """Directional principal-symbol result for the complete physical sector.

    The 25 variables are eight two-current material primitives, ``varphi``,
    ``Pi``, the directional scalar gradient, and two seven-variable
    Proca/cleaning blocks.  Constant Proca masses, mass mixing, and current
    sources are derivative-free and therefore do not enter the principal
    matrices.  Their two Maxwell-cleaning blocks are retained explicitly so
    coincident light-speed eigenspaces are tested rather than assumed.
    """

    direction: np.ndarray
    time_matrix: np.ndarray
    space_matrix: np.ndarray
    speeds: np.ndarray
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    eigenvector_condition: float
    eigenvector_rank: int
    maximum_eigenspace_residual: float
    scalar_material_coupling_norm: float
    constraint_reduced_eigenvector_condition: float
    constraint_reduced_eigenvector_rank: int
    constraint_reduction_spectrum_error: float
    coupling_coefficient: float
    effective_mass_squared: float
    legendre_minimum_eigenvalue: float
    thermodynamic_minimum_eigenvalue: float
    complete_eigenbasis: bool
    strongly_hyperbolic: bool
    causal: bool


class NullothMasterFunction:
    """Invariant M1 two-current master function plus scalarization."""

    def __init__(self, parameters: NullothParameters | None = None) -> None:
        self.parameters = parameters or NullothParameters()

    def specific_entropy(self, density: Array, internal: Array) -> Array:
        p = self.parameters
        pressure = (p.gamma_ad - 1.0) * density * internal
        polytropic = pressure / np.maximum(density**p.gamma_ad, 1.0e-300)
        return np.log(np.maximum(polytropic, 1.0e-300)) / (p.gamma_ad - 1.0)

    def thermal_pressure(self, density: Array, internal: Array) -> Array:
        return (self.parameters.gamma_ad - 1.0) * density * internal

    def thermal_energy(self, density: Array, internal: Array) -> Array:
        return density * (1.0 + internal)

    def scalar_potential(self, varphi: Array) -> Array:
        p = self.parameters
        return (
            0.5 * p.scalar_mass**2 * varphi**2
            + 0.25 * p.scalar_quartic * varphi**4
        )

    def scalar_potential_derivative(self, varphi: Array) -> Array:
        p = self.parameters
        return p.scalar_mass**2 * varphi + p.scalar_quartic * varphi**3

    def coupling_coefficient(
        self, density: Array, carrier_density: Array, relative: Array
    ) -> Array:
        p = self.parameters
        return (
            p.scalar_baryon_coupling
            * (density / p.scalar_baryon_scale) ** 2
            + p.scalar_carrier_coupling
            * (carrier_density / p.scalar_carrier_scale) ** 2
            + p.scalar_counterflow_coupling
            * relative
            / p.scalar_counterflow_scale
        )

    def lambda_from_invariants(
        self,
        n_squared: Array,
        d_squared: Array,
        x_squared: Array,
        entropy: Array,
        varphi: Array,
    ) -> Array:
        """Evaluate the complete material master function from invariants."""
        p = self.parameters
        n = np.sqrt(np.maximum(n_squared, 1.0e-300))
        d = np.sqrt(np.maximum(d_squared, 1.0e-300))
        pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        thermal = n + pressure / (p.gamma_ad - 1.0)
        anchor = d - p.target_fraction * n
        rho0 = (
            thermal
            + p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
            + 0.5 * anchor**2 / p.chemical_susceptibility
        )
        relative = x_squared - n * d
        coupling = self.coupling_coefficient(n, d, relative)
        return (
            -rho0
            - p.entrainment_linear * relative
            - 0.5
            * p.entrainment_quadratic
            * relative**2
            / p.entrainment_scale_fourth
            + 0.5 * coupling * varphi**2
        )

    def _evaluate_terms(
        self,
        h: Array,
        density: Array,
        internal: Array,
        velocity: Array,
        carrier_density: Array,
        carrier_velocity: Array,
        varphi: Array,
    ) -> dict[str, Array | StressEnergy3p1]:
        p = self.parameters
        n = np.asarray(density, dtype=float)
        d = np.asarray(carrier_density, dtype=float)
        phi = np.asarray(varphi, dtype=float)
        if np.any(n <= 0.0) or np.any(d <= 0.0):
            raise ValueError("Nulloth master-function densities must be positive")
        if np.any(~np.isfinite(phi)) or np.any(np.abs(phi) > p.scalar_amplitude_limit):
            raise ValueError("Nulloth scalar left its declared amplitude domain")

        u = np.asarray(velocity, dtype=float)
        v = np.asarray(carrier_velocity, dtype=float)
        W_N = lorentz_factor(h, u)
        W_D = lorentz_factor(h, v)
        u_down = np.einsum("ij...,j...->i...", h, u)
        v_down = np.einsum("ij...,j...->i...", h, v)
        dot = np.einsum("ij...,i...,j...->...", h, u, v)
        gamma_rel = np.maximum(W_N * W_D * (1.0 - dot), 1.0)
        x2 = n * d * gamma_rel
        relative = x2 - n * d

        pressure = self.thermal_pressure(n, internal)
        thermal = self.thermal_energy(n, internal)
        entropy = self.specific_entropy(n, internal)
        anchor = d - p.target_fraction * n
        carrier_energy = p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
        chemical_energy = 0.5 * anchor**2 / p.chemical_susceptibility
        rho0 = thermal + carrier_energy + chemical_energy
        coupling = self.coupling_coefficient(n, d, relative)

        Q = (
            p.entrainment_linear
            + p.entrainment_quadratic
            * relative
            / p.entrainment_scale_fourth
            - 0.5
            * p.scalar_counterflow_coupling
            * phi**2
            / p.scalar_counterflow_scale
        )
        lambda_mf = (
            -rho0
            - p.entrainment_linear * relative
            - 0.5
            * p.entrainment_quadratic
            * relative**2
            / p.entrainment_scale_fourth
            + 0.5 * coupling * phi**2
        )

        rho_n = (
            1.0
            + p.gamma_ad
            * pressure
            / np.maximum((p.gamma_ad - 1.0) * n, 1.0e-300)
            - p.target_fraction * anchor / p.chemical_susceptibility
            - p.scalar_baryon_coupling
            * phi**2
            * n
            / p.scalar_baryon_scale**2
        )
        rho_d = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * d ** (p.carrier_sound_speed**2)
            + anchor / p.chemical_susceptibility
            - p.scalar_carrier_coupling
            * phi**2
            * d
            / p.scalar_carrier_scale**2
        )
        B_N = (rho_n - Q * d) / n
        B_D = (rho_d - Q * n) / d
        generalized_pressure = (
            lambda_mf + B_N * n**2 + B_D * d**2 + 2.0 * Q * x2
        )

        nW = n * W_N
        dW = d * W_D
        rho = (
            -generalized_pressure
            + B_N * nW**2
            + B_D * dW**2
            + 2.0 * Q * nW * dW
        )
        momentum = (
            B_N[None, ...] * nW[None, ...] ** 2 * u_down
            + B_D[None, ...] * dW[None, ...] ** 2 * v_down
            + Q[None, ...]
            * nW[None, ...]
            * dW[None, ...]
            * (u_down + v_down)
        )
        stress = generalized_pressure[None, None, ...] * h
        stress += (
            B_N[None, None, ...]
            * nW[None, None, ...] ** 2
            * u_down[:, None, ...]
            * u_down[None, :, ...]
        )
        stress += (
            B_D[None, None, ...]
            * dW[None, None, ...] ** 2
            * v_down[:, None, ...]
            * v_down[None, :, ...]
        )
        stress += (
            Q[None, None, ...]
            * nW[None, None, ...]
            * dW[None, None, ...]
            * (
                u_down[:, None, ...] * v_down[None, :, ...]
                + v_down[:, None, ...] * u_down[None, :, ...]
            )
        )
        h_inv = inverse_metric(h)[0]
        trace = np.einsum("ij...,ij...->...", h_inv, stress)

        chi_down = B_D[None, ...] * dW[None, ...] * v_down + Q[None, ...] * nW[None, ...] * u_down
        chi_normal = B_D * dW + Q * nW
        source_rho = p.carrier_charge * dW
        source_current = p.carrier_charge * dW[None, ...] * v

        rho_nn = (
            p.gamma_ad * pressure / np.maximum(n**2, 1.0e-300)
            + p.target_fraction**2 / p.chemical_susceptibility
            - p.scalar_baryon_coupling * phi**2 / p.scalar_baryon_scale**2
        )
        rho_nd = np.full_like(n, -p.target_fraction / p.chemical_susceptibility)
        rho_dd = (
            p.carrier_K
            * (1.0 + p.carrier_sound_speed**2)
            * p.carrier_sound_speed**2
            * d ** (p.carrier_sound_speed**2 - 1.0)
            + 1.0 / p.chemical_susceptibility
            - p.scalar_carrier_coupling * phi**2 / p.scalar_carrier_scale**2
        )
        temperature = pressure / np.maximum(n, 1.0e-300)
        drag_inertia = np.maximum(B_D * d**2, 0.0)
        effective_mass_squared = p.scalar_mass**2 - coupling
        scalar_action_derivative = coupling * phi
        scalar_equation_source = self.scalar_potential_derivative(phi) - scalar_action_derivative

        return {
            "entropy": entropy,
            "temperature": temperature,
            "gamma_rel": gamma_rel,
            "relative": relative,
            "lambda_mf": lambda_mf,
            "generalized_pressure": generalized_pressure,
            "B_N": B_N,
            "B_D": B_D,
            "entrainment": Q,
            "rho_n": rho_n,
            "rho_d": rho_d,
            "rho_nn": rho_nn,
            "rho_nd": rho_nd,
            "rho_dd": rho_dd,
            "stress": StressEnergy3p1(rho, momentum, stress, trace),
            "chi_down": chi_down,
            "chi_normal": chi_normal,
            "dW": dW,
            "source_rho": source_rho,
            "source_current": source_current,
            "drag_inertia": drag_inertia,
            "coupling": coupling,
            "effective_mass_squared": effective_mass_squared,
            "scalar_action_derivative": scalar_action_derivative,
            "scalar_equation_source": scalar_equation_source,
        }

    def evaluate(
        self,
        h: Array,
        fluid: FluidPrimitive,
        carrier: CarrierPrimitive,
        varphi: Array,
    ) -> NullothMasterState:
        t = self._evaluate_terms(
            h,
            fluid.baryon_density,
            fluid.specific_internal_energy,
            fluid.velocity,
            carrier.number_density,
            carrier.velocity,
            varphi,
        )
        return NullothMasterState(
            t["entropy"],
            t["temperature"],
            t["gamma_rel"],
            t["relative"],
            t["lambda_mf"],
            t["generalized_pressure"],
            t["B_N"],
            t["B_D"],
            t["entrainment"],
            t["rho_n"],
            t["rho_d"],
            t["rho_nn"],
            t["rho_nd"],
            t["rho_dd"],
            t["stress"],
            t["chi_down"],
            t["chi_normal"],
            t["dW"],
            t["source_rho"],
            t["source_current"],
            t["drag_inertia"],
            t["coupling"],
            t["effective_mass_squared"],
            t["scalar_action_derivative"],
            t["scalar_equation_source"],
        )

    def cell_predictions(
        self,
        primitive: np.ndarray,
        h: np.ndarray,
        sqrt_h: float,
        entropy_per_baryon: float,
        varphi: float,
    ) -> np.ndarray:
        """Map one primitive cell to nine independent material conserved values."""
        p = self.parameters
        n = float(np.exp(np.clip(primitive[0], -700.0, 700.0)))
        q_N = np.asarray(primitive[1:4], dtype=float)
        d = float(np.exp(np.clip(primitive[4], -700.0, 700.0)))
        q_D = np.asarray(primitive[5:8], dtype=float)
        W_N = float(np.sqrt(1.0 + q_N @ h @ q_N))
        W_D = float(np.sqrt(1.0 + q_D @ h @ q_D))
        u = q_N / W_N
        v = q_D / W_D
        internal = float(
            np.exp(np.clip((p.gamma_ad - 1.0) * entropy_per_baryon, -700.0, 700.0))
            * max(n, 1.0e-300) ** p.gamma_ad
            / ((p.gamma_ad - 1.0) * max(n, 1.0e-300))
        )
        terms = self._evaluate_terms(
            h,
            np.asarray(n),
            np.asarray(internal),
            u,
            np.asarray(d),
            v,
            np.asarray(varphi),
        )
        stress = terms["stress"]
        D_N = sqrt_h * n * W_N
        D_D = sqrt_h * d * W_D
        P_D = D_D * np.asarray(terms["chi_down"])
        return np.concatenate(
            ([D_N, D_D], np.asarray(stress.momentum), [float(stress.rho)], P_D)
        )

    def conserved_carrier(
        self, h: Array, state: NullothMasterState
    ) -> tuple[Array, Array]:
        sqrt_h = inverse_metric(h)[2]
        D_D = sqrt_h * state.carrier_eulerian_density
        return D_D, D_D[None, ...] * state.carrier_momentum_down

    def drag_force_down(
        self,
        h: Array,
        fluid: FluidPrimitive,
        carrier: CarrierPrimitive,
        state: NullothMasterState,
    ) -> tuple[Array, Array]:
        p = self.parameters
        W_N = lorentz_factor(h, fluid.velocity)
        W_D = lorentz_factor(h, carrier.velocity)
        u_down = np.einsum("ij...,j...->i...", h, fluid.velocity)
        v_down = np.einsum("ij...,j...->i...", h, carrier.velocity)
        resistance = state.drag_inertia / p.relaxation_time
        force = resistance[None, ...] * (
            W_N[None, ...] * u_down
            - state.relative_lorentz_factor[None, ...]
            * W_D[None, ...]
            * v_down
        )
        heat = resistance * (state.relative_lorentz_factor**2 - 1.0)
        return force, np.maximum(heat, 0.0)

    @staticmethod
    def _numerical_jacobian(function, point: np.ndarray) -> np.ndarray:
        base = np.asarray(function(point), dtype=float)
        jac = np.empty((base.size, point.size), dtype=float)
        for column in range(point.size):
            step = 2.0e-6 * max(1.0, abs(float(point[column])))
            plus = point.copy()
            minus = point.copy()
            plus[column] += step
            minus[column] -= step
            jac[:, column] = (function(plus) - function(minus)) / (2.0 * step)
        return jac

    @staticmethod
    def _clustered_eigenbasis(
        symbol: np.ndarray,
        cluster_tolerance: float = 1.0e-8,
    ) -> tuple[np.ndarray, int, float, float]:
        """Build complete repeated eigenspaces with SVD null bases.

        ``numpy.linalg.eig`` is allowed to return duplicate vectors for exact
        repeated eigenvalues.  That happened for the sevenfold ``-1``, zero,
        and ``+1`` sectors in the first Nulloth scan and falsely resembled a
        Jordan failure.  Here eigenvalues are clustered, and each geometric
        eigenspace is constructed from the null space of ``P-lambda I``.
        """
        eigenvalues = np.linalg.eigvals(symbol)
        order = np.lexsort((eigenvalues.imag, eigenvalues.real))
        ordered = eigenvalues[order]
        clusters: list[list[complex]] = []
        for value in ordered:
            if not clusters:
                clusters.append([complex(value)])
                continue
            center = sum(clusters[-1]) / len(clusters[-1])
            scale = max(1.0, abs(center), abs(value))
            if abs(value - center) <= cluster_tolerance * scale:
                clusters[-1].append(complex(value))
            else:
                clusters.append([complex(value)])

        size = symbol.shape[0]
        identity = np.eye(size, dtype=complex)
        basis_parts: list[np.ndarray] = []
        maximum_residual = 0.0
        for cluster in clusters:
            center = sum(cluster) / len(cluster)
            algebraic_multiplicity = len(cluster)
            _, singular_values, right = np.linalg.svd(
                symbol.astype(complex) - center * identity
            )
            scale = max(1.0, float(singular_values[0]))
            null_tolerance = cluster_tolerance * scale
            geometric_multiplicity = int(
                np.count_nonzero(singular_values <= null_tolerance)
            )
            if geometric_multiplicity:
                null_basis = right[-geometric_multiplicity:].conj().T
                residual = float(
                    np.linalg.norm(
                        (symbol - center * identity) @ null_basis,
                        ord=2,
                    )
                )
                maximum_residual = max(maximum_residual, residual)
                basis_parts.append(null_basis)
            if geometric_multiplicity < algebraic_multiplicity:
                maximum_residual = np.inf

        basis = (
            np.concatenate(basis_parts, axis=1)
            if basis_parts
            else np.empty((size, 0), dtype=complex)
        )
        rank = int(np.linalg.matrix_rank(basis, tol=cluster_tolerance))
        condition = (
            float(np.linalg.cond(basis))
            if basis.shape == (size, size) and rank == size
            else np.inf
        )
        return eigenvalues, rank, condition, maximum_residual

    def _flat_constitutive_vector(
        self, primitive: np.ndarray, entropy: float, varphi: float
    ) -> np.ndarray:
        log_n, rapid_N, log_d, rapid_D = primitive
        n = float(np.exp(log_n))
        d = float(np.exp(log_d))
        p = self.parameters
        pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        internal = pressure / ((p.gamma_ad - 1.0) * n)
        u = float(np.tanh(rapid_N))
        v = float(np.tanh(rapid_D))
        WN = float(np.cosh(rapid_N))
        WD = float(np.cosh(rapid_D))
        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([n]),
            np.array([internal]),
            np.array([[u], [0.0], [0.0]]),
            np.array([0.0]),
        )
        carrier = CarrierPrimitive(
            np.array([d]), np.array([[v], [0.0], [0.0]])
        )
        state = self.evaluate(h, fluid, carrier, np.array([varphi]))
        mu_x = state.B_N[0] * n * WN * u + state.entrainment[0] * d * WD * v
        mu_normal = state.B_N[0] * n * WN + state.entrainment[0] * d * WD
        return np.array(
            [
                n * WN,
                mu_x,
                d * WD,
                state.carrier_momentum_down[0, 0],
                n * WN * u,
                mu_normal,
                d * WD * v,
                state.carrier_momentum_normal[0],
            ],
            dtype=float,
        )

    def characteristic_screen(
        self,
        density: float,
        internal: float,
        velocity: float,
        carrier_density: float,
        carrier_velocity: float,
        varphi: float,
    ) -> NullothCharacteristicScreen:
        if density <= 0.0 or internal <= 0.0 or carrier_density <= 0.0:
            raise ValueError("characteristic screen needs positive matter state")
        if abs(velocity) >= 1.0 or abs(carrier_velocity) >= 1.0:
            raise ValueError("characteristic screen velocities must be subluminal")
        point = np.array(
            [
                np.log(density),
                np.arctanh(velocity),
                np.log(carrier_density),
                np.arctanh(carrier_velocity),
            ]
        )
        entropy = float(
            self.specific_entropy(np.array([density]), np.array([internal]))[0]
        )
        jac = self._numerical_jacobian(
            lambda value: self._flat_constitutive_vector(value, entropy, varphi),
            point,
        )
        time_matrix = jac[:4]
        space_matrix = jac[4:]
        symbol = np.linalg.solve(time_matrix, space_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(symbol)
        imag = float(np.max(np.abs(eigenvalues.imag)))
        material_speeds = np.sort(eigenvalues.real)
        speeds = np.sort(np.concatenate((material_speeds, np.array([-1.0, 1.0]))))
        condition = float(np.linalg.cond(eigenvectors))

        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([density]),
            np.array([internal]),
            np.array([[velocity], [0.0], [0.0]]),
            np.array([0.0]),
        )
        carrier = CarrierPrimitive(
            np.array([carrier_density]),
            np.array([[carrier_velocity], [0.0], [0.0]]),
        )
        state = self.evaluate(h, fluid, carrier, np.array([varphi]))
        legendre = np.array(
            [[state.B_N[0], state.entrainment[0]], [state.entrainment[0], state.B_D[0]]]
        )
        thermo = np.array(
            [
                [state.thermodynamic_hessian_nn[0], state.thermodynamic_hessian_nd[0]],
                [state.thermodynamic_hessian_nd[0], state.thermodynamic_hessian_dd[0]],
            ]
        )
        legendre_min = float(np.min(np.linalg.eigvalsh(legendre)))
        thermo_min = float(np.min(np.linalg.eigvalsh(thermo)))
        strong = (
            imag < 1.0e-8
            and np.isfinite(condition)
            and condition < self.parameters.characteristic_condition_limit
            and legendre_min > self.parameters.convexity_floor
            and thermo_min > self.parameters.convexity_floor
        )
        maximum = float(np.max(np.abs(speeds)))
        return NullothCharacteristicScreen(
            speeds,
            maximum,
            condition,
            legendre_min,
            thermo_min,
            strong,
            strong and maximum <= 1.0 + 1.0e-8,
        )

    @staticmethod
    def _proca_cleaning_principal_block() -> np.ndarray:
        """Return a symmetric directional Maxwell-Proca cleaning block.

        Variables are ``(E_n,E_t1,E_t2,B_n,B_t1,B_t2,psi)``.  The massive
        and mixed-Proca terms are algebraic.  This principal block has three
        incoming, three outgoing, and one zero-speed constraint mode.
        """
        block = np.zeros((7, 7), dtype=float)
        block[0, 6] = block[6, 0] = 1.0
        block[1, 5] = block[5, 1] = -1.0
        block[2, 4] = block[4, 2] = 1.0
        return block

    def _flat_coupled_fields(
        self,
        point: np.ndarray,
        entropy: float,
    ) -> np.ndarray:
        """Return currents and conjugate covectors for a flat local state."""
        n = float(np.exp(point[0]))
        q_N = np.asarray(point[1:4], dtype=float)
        d = float(np.exp(point[4]))
        q_D = np.asarray(point[5:8], dtype=float)
        varphi = float(point[8])
        W_N = float(np.sqrt(1.0 + q_N @ q_N))
        W_D = float(np.sqrt(1.0 + q_D @ q_D))
        velocity_N = q_N / W_N
        velocity_D = q_D / W_D
        p = self.parameters
        pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
        internal = pressure / ((p.gamma_ad - 1.0) * n)
        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([n]),
            np.array([internal]),
            velocity_N.reshape(3, 1),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([d]), velocity_D.reshape(3, 1)
        )
        state = self.evaluate(h, fluid, carrier, np.array([varphi]))
        B_N = float(state.B_N[0])
        B_D = float(state.B_D[0])
        entrainment = float(state.entrainment[0])
        mu_spatial = B_N * n * q_N + entrainment * d * q_D
        chi_spatial = B_D * d * q_D + entrainment * n * q_N
        mu_zero = -(B_N * n * W_N + entrainment * d * W_D)
        chi_zero = -(B_D * d * W_D + entrainment * n * W_N)
        return np.concatenate(
            (
                [n * W_N],
                n * q_N,
                [d * W_D],
                d * q_D,
                [mu_zero],
                mu_spatial,
                [chi_zero],
                chi_spatial,
            )
        )

    def coupled_characteristic_audit(
        self,
        density: float,
        internal: float,
        velocity: np.ndarray | tuple[float, float, float],
        carrier_density: float,
        carrier_velocity: np.ndarray | tuple[float, float, float],
        varphi: float,
        direction: np.ndarray | tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> NullothCoupledCharacteristicAudit:
        """Build and diagonalize the full frozen directional principal symbol.

        The multifluid rows are the two current-conservation equations and
        the six spatial projections of the two Carter vorticity equations.
        Their coefficients are differentiated with respect to all material
        primitives and ``varphi``.  The canonical scalar and both mixed-Proca
        cleaning blocks are then included explicitly in the same generalized
        eigenproblem.
        """
        if density <= 0.0 or internal <= 0.0 or carrier_density <= 0.0:
            raise ValueError("coupled audit needs positive matter state")
        velocity_N = np.asarray(velocity, dtype=float)
        velocity_D = np.asarray(carrier_velocity, dtype=float)
        normal = np.asarray(direction, dtype=float)
        if velocity_N.shape != (3,) or velocity_D.shape != (3,):
            raise ValueError("audit velocities must be three-vectors")
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("audit direction must be a finite three-vector")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("audit direction must be nonzero")
        normal /= norm
        speed_N_squared = float(velocity_N @ velocity_N)
        speed_D_squared = float(velocity_D @ velocity_D)
        if speed_N_squared >= 1.0 or speed_D_squared >= 1.0:
            raise ValueError("audit velocities must be subluminal")
        if abs(varphi) > self.parameters.scalar_amplitude_limit:
            raise ValueError("audit scalar amplitude leaves its declared domain")

        W_N = 1.0 / np.sqrt(1.0 - speed_N_squared)
        W_D = 1.0 / np.sqrt(1.0 - speed_D_squared)
        entropy = float(
            self.specific_entropy(
                np.array([density]), np.array([internal])
            )[0]
        )
        point = np.concatenate(
            (
                [np.log(density)],
                W_N * velocity_N,
                [np.log(carrier_density)],
                W_D * velocity_D,
                [varphi],
            )
        )
        fields = self._flat_coupled_fields(point, entropy)
        jacobian = self._numerical_jacobian(
            lambda value: self._flat_coupled_fields(value, entropy), point
        )

        # Field layout: N0, N^i, C0, C^i, mu_0, mu_i, chi_0, chi_i.
        N_zero = fields[0]
        N_spatial = fields[1:4]
        C_zero = fields[4]
        C_spatial = fields[5:8]
        mu_zero_jac = jacobian[8]
        mu_spatial_jac = jacobian[9:12]
        chi_zero_jac = jacobian[12]
        chi_spatial_jac = jacobian[13:16]
        N_directional = float(normal @ N_spatial)
        C_directional = float(normal @ C_spatial)

        material_time = np.zeros((8, 11), dtype=float)
        material_space = np.zeros((8, 11), dtype=float)
        material_time[0, :9] = jacobian[0]
        material_space[0, :9] = normal @ jacobian[1:4]
        for component in range(3):
            material_time[1 + component, :9] = (
                N_zero * mu_spatial_jac[component]
            )
            material_space[1 + component, :9] = (
                N_directional * mu_spatial_jac[component]
                - normal[component]
                * (
                    N_zero * mu_zero_jac
                    + N_spatial @ mu_spatial_jac
                )
            )
        material_time[4, :9] = jacobian[4]
        material_space[4, :9] = normal @ jacobian[5:8]
        for component in range(3):
            material_time[5 + component, :9] = (
                C_zero * chi_spatial_jac[component]
            )
            material_space[5 + component, :9] = (
                C_directional * chi_spatial_jac[component]
                - normal[component]
                * (
                    C_zero * chi_zero_jac
                    + C_spatial @ chi_spatial_jac
                )
            )

        physical_time = np.zeros((11, 11), dtype=float)
        physical_space = np.zeros((11, 11), dtype=float)
        physical_time[:8] = material_time
        physical_space[:8] = material_space
        physical_time[8:, 8:] = np.eye(3)
        physical_space[9, 10] = 1.0
        physical_space[10, 9] = 1.0

        proca = self._proca_cleaning_principal_block()
        total_size = 25
        time_matrix = np.eye(total_size, dtype=float)
        space_matrix = np.zeros((total_size, total_size), dtype=float)
        time_matrix[:11, :11] = physical_time
        space_matrix[:11, :11] = physical_space
        space_matrix[11:18, 11:18] = proca
        space_matrix[18:25, 18:25] = proca

        symbol = np.linalg.solve(time_matrix, space_matrix)
        eigenvalues, rank, condition, eigenspace_residual = (
            self._clustered_eigenbasis(symbol)
        )
        maximum_imaginary = float(np.max(np.abs(eigenvalues.imag)))
        speeds = np.sort(eigenvalues.real)
        maximum = float(np.max(np.abs(speeds)))

        # In the explicit first-order scalar reduction, phi_t=-Pi and
        # D_s(phi)=Phi_s replace derivatives of phi in the material equations.
        # Those terms are lower order, leaving phi as a decoupled zero-speed
        # variable.  The reduced symbol must retain the same spectrum and a
        # complete basis; this is an independent regular-reduction check.
        reduced_time = time_matrix.copy()
        reduced_space = space_matrix.copy()
        reduced_time[:8, 8] = 0.0
        reduced_space[:8, 8] = 0.0
        reduced_symbol = np.linalg.solve(reduced_time, reduced_space)
        (
            reduced_eigenvalues,
            reduced_rank,
            reduced_condition,
            reduced_residual,
        ) = self._clustered_eigenbasis(reduced_symbol)
        spectrum_error = float(
            np.max(
                np.abs(
                    np.sort_complex(eigenvalues)
                    - np.sort_complex(reduced_eigenvalues)
                )
            )
        )
        eigenspace_residual = max(eigenspace_residual, reduced_residual)

        h = np.eye(3).reshape(3, 3, 1)
        fluid = FluidPrimitive(
            np.array([density]),
            np.array([internal]),
            velocity_N.reshape(3, 1),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([carrier_density]), velocity_D.reshape(3, 1)
        )
        state = self.evaluate(h, fluid, carrier, np.array([varphi]))
        legendre = np.array(
            [
                [state.B_N[0], state.entrainment[0]],
                [state.entrainment[0], state.B_D[0]],
            ]
        )
        thermo = np.array(
            [
                [
                    state.thermodynamic_hessian_nn[0],
                    state.thermodynamic_hessian_nd[0],
                ],
                [
                    state.thermodynamic_hessian_nd[0],
                    state.thermodynamic_hessian_dd[0],
                ],
            ]
        )
        legendre_min = float(np.min(np.linalg.eigvalsh(legendre)))
        thermo_min = float(np.min(np.linalg.eigvalsh(thermo)))
        coupling_norm = float(
            np.linalg.norm(material_time[:, 8])
            + np.linalg.norm(material_space[:, 8])
        )
        reduction_regular = (
            reduced_rank == total_size
            and np.isfinite(reduced_condition)
            and spectrum_error < 1.0e-8
        )
        complete = (
            rank == total_size
            and np.isfinite(condition)
            and reduction_regular
        )
        strong = (
            maximum_imaginary < 1.0e-8
            and complete
            and condition < self.parameters.characteristic_condition_limit
            and reduced_condition
            < self.parameters.characteristic_condition_limit
            and eigenspace_residual < 1.0e-7
            and legendre_min > self.parameters.convexity_floor
            and thermo_min > self.parameters.convexity_floor
        )
        return NullothCoupledCharacteristicAudit(
            normal,
            time_matrix,
            space_matrix,
            speeds,
            maximum,
            maximum_imaginary,
            condition,
            rank,
            eigenspace_residual,
            coupling_norm,
            reduced_condition,
            reduced_rank,
            spectrum_error,
            float(state.coupling_coefficient[0]),
            float(state.effective_mass_squared[0]),
            legendre_min,
            thermo_min,
            complete,
            strong,
            strong and maximum <= 1.0 + 1.0e-8,
        )
