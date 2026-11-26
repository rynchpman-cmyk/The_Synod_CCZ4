"""Qualify the first bounded Nulloth scalarization discovery campaign."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.nulloth import (
    NULLOTH_DISCOVERY_PARAMETERS,
    NullothMasterFunction,
)


@dataclass(frozen=True)
class DiscoveryDomainQualification:
    samples: int
    failures: int
    activated_samples: int
    sampled_domain_qualified: bool
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    maximum_eigenvector_condition: float
    maximum_constraint_reduced_condition: float
    maximum_eigenspace_residual: float
    maximum_reduction_spectrum_error: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    minimum_effective_mass_squared: float
    maximum_equilibrium_amplitude: float
    first_failure: dict[str, object] | None


def qualify_discovery_domain(
    *, seed: int = 20260721, random_samples: int = 1024, quick: bool = False
) -> DiscoveryDomainQualification:
    """Scan the declared discovery box, including corners and 3-D states."""
    if random_samples < 0:
        raise ValueError("random_samples must be nonnegative")
    if quick:
        random_samples = min(random_samples, 24)
    master = NullothMasterFunction(NULLOTH_DISCOVERY_PARAMETERS)
    rng = np.random.default_rng(seed)
    cases: list[tuple[float, float, np.ndarray, float, np.ndarray, float, np.ndarray]] = []

    directions = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )
    if quick:
        corners = itertools.product(
            (0.4, 0.9),
            (0.1,),
            (0.12, 0.24),
            (0.0, 0.16),
            (0.0,),
            directions[:1],
        )
    else:
        corners = itertools.product(
            (0.4, 0.9),
            (0.1, 0.35),
            (0.12, 0.24),
            (-0.16, 0.0, 0.16),
            (-0.3, 0.0, 0.3),
            directions,
        )
    for density, internal, carrier_density, varphi, counterflow, direction in corners:
        cases.append(
            (
                density,
                internal,
                np.zeros(3),
                carrier_density,
                np.array([counterflow, 0.0, 0.0]),
                varphi,
                direction,
            )
        )

    for _ in range(random_samples):
        velocity = rng.normal(size=3)
        velocity *= rng.uniform(0.0, 0.2) / np.linalg.norm(velocity)
        carrier_velocity = rng.normal(size=3)
        carrier_velocity *= rng.uniform(0.0, 0.3) / np.linalg.norm(
            carrier_velocity
        )
        direction = rng.normal(size=3)
        cases.append(
            (
                rng.uniform(0.4, 0.9),
                rng.uniform(0.1, 0.35),
                velocity,
                rng.uniform(0.12, 0.24),
                carrier_velocity,
                rng.uniform(-0.16, 0.16),
                direction,
            )
        )

    failures = 0
    activated = 0
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    maximum_reduced_condition = 0.0
    maximum_residual = 0.0
    maximum_spectrum_error = 0.0
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    minimum_effective_mass = np.inf
    maximum_equilibrium_amplitude = 0.0
    first_failure: dict[str, object] | None = None
    for index, case in enumerate(cases):
        audit = master.coupled_characteristic_audit(*case)
        maximum_speed = max(maximum_speed, audit.maximum_absolute_speed)
        maximum_imaginary = max(
            maximum_imaginary, audit.maximum_imaginary_speed
        )
        maximum_condition = max(
            maximum_condition, audit.eigenvector_condition
        )
        maximum_reduced_condition = max(
            maximum_reduced_condition,
            audit.constraint_reduced_eigenvector_condition,
        )
        maximum_residual = max(
            maximum_residual, audit.maximum_eigenspace_residual
        )
        maximum_spectrum_error = max(
            maximum_spectrum_error,
            audit.constraint_reduction_spectrum_error,
        )
        minimum_legendre = min(
            minimum_legendre, audit.legendre_minimum_eigenvalue
        )
        minimum_thermo = min(
            minimum_thermo, audit.thermodynamic_minimum_eigenvalue
        )
        minimum_effective_mass = min(
            minimum_effective_mass, audit.effective_mass_squared
        )
        if audit.effective_mass_squared < 0.0:
            activated += 1
            maximum_equilibrium_amplitude = max(
                maximum_equilibrium_amplitude,
                np.sqrt(
                    -audit.effective_mass_squared
                    / NULLOTH_DISCOVERY_PARAMETERS.scalar_quartic
                ),
            )
        if not (audit.strongly_hyperbolic and audit.causal):
            failures += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "density": case[0],
                    "internal_energy": case[1],
                    "velocity": case[2].tolist(),
                    "carrier_density": case[3],
                    "carrier_velocity": case[4].tolist(),
                    "varphi": case[5],
                    "direction": case[6].tolist(),
                    "rank": audit.eigenvector_rank,
                    "condition": audit.eigenvector_condition,
                    "maximum_speed": audit.maximum_absolute_speed,
                }

    qualified = failures == 0 and activated > 0
    return DiscoveryDomainQualification(
        len(cases),
        failures,
        activated,
        qualified,
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
        maximum_reduced_condition,
        maximum_residual,
        maximum_spectrum_error,
        float(minimum_legendre),
        float(minimum_thermo),
        float(minimum_effective_mass),
        float(maximum_equilibrium_amplitude),
        first_failure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--random-samples", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = qualify_discovery_domain(
        seed=args.seed,
        random_samples=args.random_samples,
        quick=args.quick,
    )
    payload = {
        "parameters": asdict(NULLOTH_DISCOVERY_PARAMETERS),
        "declared_bounds": {
            "density": [0.4, 0.9],
            "specific_internal_energy": [0.1, 0.35],
            "carrier_density": [0.12, 0.24],
            "fluid_speed": [0.0, 0.2],
            "carrier_speed": [0.0, 0.3],
            "varphi": [-0.16, 0.16],
        },
        "qualification": asdict(result),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
