"""Finite-difference audit of every Nulloth constitutive derivative."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction, NullothParameters
from causal_repair.theory33 import CarrierPrimitive


@dataclass(frozen=True)
class ConstitutiveDerivativeAudit:
    samples: int
    maximum_density_first_relative_error: float
    maximum_carrier_first_relative_error: float
    maximum_counterflow_first_relative_error: float
    maximum_scalar_first_relative_error: float
    maximum_density_hessian_relative_error: float
    maximum_cross_hessian_relative_error: float
    maximum_carrier_hessian_relative_error: float
    maximum_pressure_identity_error: float
    maximum_scalar_source_identity_error: float
    nonfinite_results: int


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(actual), abs(expected), 1.0e-12)


def run_derivative_audit(
    *, seed: int = 20260722, samples: int = 128
) -> ConstitutiveDerivativeAudit:
    if samples <= 0:
        raise ValueError("samples must be positive")
    parameters = NullothParameters(
        entrainment_linear=0.03,
        entrainment_quadratic=0.02,
        scalar_baryon_coupling=0.7,
        scalar_carrier_coupling=1.3,
        scalar_counterflow_coupling=0.9,
        scalar_quartic=3.0,
    )
    master = NullothMasterFunction(parameters)
    rng = np.random.default_rng(seed)
    h = np.eye(3).reshape(3, 3, 1)
    maxima = np.zeros(9)
    nonfinite = 0

    for _ in range(samples):
        density = rng.uniform(0.45, 0.85)
        carrier_density = rng.uniform(0.13, 0.23)
        internal = rng.uniform(0.12, 0.32)
        varphi = rng.uniform(-0.14, 0.14)
        fluid_velocity = rng.normal(size=3)
        fluid_velocity *= rng.uniform(0.03, 0.18) / np.linalg.norm(fluid_velocity)
        carrier_velocity = rng.normal(size=3)
        carrier_velocity *= rng.uniform(0.04, 0.25) / np.linalg.norm(
            carrier_velocity
        )
        fluid = FluidPrimitive(
            np.array([density]),
            np.array([internal]),
            fluid_velocity.reshape(3, 1),
            np.zeros(1),
        )
        carrier = CarrierPrimitive(
            np.array([carrier_density]), carrier_velocity.reshape(3, 1)
        )
        state = master.evaluate(h, fluid, carrier, np.array([varphi]))
        entropy = float(state.entropy_per_baryon[0])
        counterflow = float(state.relative_invariant[0])

        def energy(n: float, d: float, w: float, phi: float) -> float:
            return -float(
                master.lambda_from_invariants(
                    np.array(n * n),
                    np.array(d * d),
                    np.array(n * d + w),
                    np.array(entropy),
                    np.array(phi),
                )
            )

        step_n = 2.0e-4 * density
        step_d = 2.0e-4 * carrier_density
        step_w = 2.0e-5 * max(density * carrier_density, 1.0e-3)
        step_phi = 2.0e-6
        center = energy(density, carrier_density, counterflow, varphi)

        derivative_n = (
            energy(density + step_n, carrier_density, counterflow, varphi)
            - energy(density - step_n, carrier_density, counterflow, varphi)
        ) / (2.0 * step_n)
        derivative_d = (
            energy(density, carrier_density + step_d, counterflow, varphi)
            - energy(density, carrier_density - step_d, counterflow, varphi)
        ) / (2.0 * step_d)
        derivative_w = (
            energy(density, carrier_density, counterflow + step_w, varphi)
            - energy(density, carrier_density, counterflow - step_w, varphi)
        ) / (2.0 * step_w)
        lambda_phi = (
            -energy(density, carrier_density, counterflow, varphi + step_phi)
            + energy(density, carrier_density, counterflow, varphi - step_phi)
        ) / (2.0 * step_phi)

        second_n = (
            energy(density + step_n, carrier_density, counterflow, varphi)
            - 2.0 * center
            + energy(density - step_n, carrier_density, counterflow, varphi)
        ) / step_n**2
        second_d = (
            energy(density, carrier_density + step_d, counterflow, varphi)
            - 2.0 * center
            + energy(density, carrier_density - step_d, counterflow, varphi)
        ) / step_d**2
        cross_nd = (
            energy(
                density + step_n,
                carrier_density + step_d,
                counterflow,
                varphi,
            )
            - energy(
                density + step_n,
                carrier_density - step_d,
                counterflow,
                varphi,
            )
            - energy(
                density - step_n,
                carrier_density + step_d,
                counterflow,
                varphi,
            )
            + energy(
                density - step_n,
                carrier_density - step_d,
                counterflow,
                varphi,
            )
        ) / (4.0 * step_n * step_d)

        pressure_identity = (
            float(state.lambda_mf[0])
            + float(state.B_N[0]) * density**2
            + float(state.B_D[0]) * carrier_density**2
            + 2.0
            * float(state.entrainment[0])
            * density
            * carrier_density
            * float(state.relative_lorentz_factor[0])
        )
        scalar_source_identity = (
            master.scalar_potential_derivative(np.array(varphi))
            - float(state.scalar_action_derivative[0])
        )
        observed = np.array(
            [
                _relative_error(derivative_n, float(state.chemical_N[0])),
                _relative_error(derivative_d, float(state.chemical_D[0])),
                _relative_error(derivative_w, float(state.entrainment[0])),
                _relative_error(
                    lambda_phi, float(state.scalar_action_derivative[0])
                ),
                _relative_error(
                    second_n, float(state.thermodynamic_hessian_nn[0])
                ),
                _relative_error(
                    cross_nd, float(state.thermodynamic_hessian_nd[0])
                ),
                _relative_error(
                    second_d, float(state.thermodynamic_hessian_dd[0])
                ),
                abs(pressure_identity - float(state.generalized_pressure[0])),
                abs(
                    float(scalar_source_identity)
                    - float(state.scalar_equation_source[0])
                ),
            ]
        )
        if not np.all(np.isfinite(observed)):
            nonfinite += 1
        else:
            maxima = np.maximum(maxima, observed)

    return ConstitutiveDerivativeAudit(
        samples,
        *map(float, maxima),
        nonfinite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    result = run_derivative_audit(seed=args.seed, samples=args.samples)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

