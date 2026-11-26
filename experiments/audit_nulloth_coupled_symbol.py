"""Scan the full Nulloth multifluid-scalar-Proca principal symbol."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np

from causal_repair.nulloth import NullothMasterFunction, NullothParameters


@dataclass(frozen=True)
class CoupledSymbolScanResult:
    samples: int
    failures: int
    sampled_domain_passed: bool
    maximum_absolute_speed: float
    maximum_imaginary_speed: float
    maximum_eigenvector_condition: float
    maximum_constraint_reduced_condition: float
    maximum_eigenspace_residual: float
    maximum_constraint_reduction_spectrum_error: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_scalar_material_coupling_norm: float
    first_failure: dict[str, object] | None


def run_symbol_scan(*, quick: bool = False) -> CoupledSymbolScanResult:
    """Audit a declared finite state-and-direction grid.

    A passing finite scan is evidence for the sampled domain, not a proof for
    unsampled states.  Campaign qualification must record and preserve the
    exact grids used here (or denser replacements).
    """
    parameters = NullothParameters(
        scalar_baryon_coupling=0.2,
        scalar_carrier_coupling=0.5,
        scalar_counterflow_coupling=0.3,
    )
    master = NullothMasterFunction(parameters)
    densities = (0.8,) if quick else (0.4, 0.8)
    internal_energies = (0.2,) if quick else (0.1, 0.3)
    carrier_densities = (0.2,) if quick else (0.1, 0.2)
    amplitudes = (0.0, 0.1) if quick else (0.0, 0.1, 0.25)
    counterflows = (0.0, 0.2) if quick else (-0.2, 0.0, 0.2)
    directions = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    samples = 0
    failures = 0
    maximum_speed = 0.0
    maximum_imaginary = 0.0
    maximum_condition = 0.0
    maximum_reduced_condition = 0.0
    maximum_residual = 0.0
    maximum_spectrum_error = 0.0
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    maximum_coupling = 0.0
    first_failure: dict[str, object] | None = None
    for density, internal, carrier_density, amplitude, counterflow, direction in itertools.product(
        densities,
        internal_energies,
        carrier_densities,
        amplitudes,
        counterflows,
        directions,
    ):
        samples += 1
        audit = master.coupled_characteristic_audit(
            density,
            internal,
            np.zeros(3),
            carrier_density,
            np.array([counterflow, 0.0, 0.0]),
            amplitude,
            direction,
        )
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
        maximum_coupling = max(
            maximum_coupling, audit.scalar_material_coupling_norm
        )
        if not (audit.strongly_hyperbolic and audit.causal):
            failures += 1
            if first_failure is None:
                first_failure = {
                    "density": density,
                    "internal_energy": internal,
                    "carrier_density": carrier_density,
                    "varphi": amplitude,
                    "carrier_velocity": counterflow,
                    "direction": direction.tolist(),
                    "maximum_absolute_speed": audit.maximum_absolute_speed,
                    "maximum_imaginary_speed": audit.maximum_imaginary_speed,
                    "eigenvector_condition": audit.eigenvector_condition,
                    "eigenvector_rank": audit.eigenvector_rank,
                }

    return CoupledSymbolScanResult(
        samples,
        failures,
        failures == 0,
        maximum_speed,
        maximum_imaginary,
        maximum_condition,
        maximum_reduced_condition,
        maximum_residual,
        maximum_spectrum_error,
        float(minimum_legendre),
        float(minimum_thermo),
        maximum_coupling,
        first_failure,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asdict(run_symbol_scan(quick=args.quick)), indent=2))


if __name__ == "__main__":
    main()
