"""Executable derivation and characteristic gates for Theory 3.3."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from causal_repair.theory33 import Theory33MasterFunction, Theory33MasterParameters


def hilbert_stress_error(master: Theory33MasterFunction) -> float:
    """Compare the Carter stress with a metric finite variation.

    The densitized currents are fixed during the variation, matching the
    pullback-current convention in the Theory 3.3 specification.
    """
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    N = np.array([0.12, 0.018, -0.004, 0.0])
    C = np.array([0.008, -0.001, 0.0005, 0.0])
    entropy = 0.3

    def density_lagrangian(g: np.ndarray) -> float:
        root = float(np.sqrt(-np.linalg.det(g)))
        current_N = N / root
        current_C = C / root
        n2 = -float(current_N @ g @ current_N)
        d2 = -float(current_C @ g @ current_C)
        x2 = -float(current_N @ g @ current_C)
        return root * float(
            master.lambda_from_invariants(
                np.array(n2), np.array(d2), np.array(x2), np.array(entropy)
            )
        )

    root = float(np.sqrt(-np.linalg.det(metric)))
    n2 = -float(N @ metric @ N)
    d2 = -float(C @ metric @ C)
    x2 = -float(N @ metric @ C)
    n = np.sqrt(n2)
    d = np.sqrt(d2)
    relative = x2 - n * d
    p = master.parameters
    pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
    anchor = d - p.target_fraction * n
    rho0 = (
        n
        + pressure / (p.gamma_ad - 1.0)
        + p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
        + 0.5 * anchor**2 / p.chemical_susceptibility
    )
    Q = p.entrainment_linear + p.entrainment_quadratic * relative / p.entrainment_scale_fourth
    Lambda = -rho0 - p.entrainment_linear * relative - 0.5 * p.entrainment_quadratic * relative**2 / p.entrainment_scale_fourth
    rho_n = (
        1.0
        + p.gamma_ad * pressure / ((p.gamma_ad - 1.0) * n)
        - p.target_fraction * anchor / p.chemical_susceptibility
    )
    rho_d = (
        p.carrier_K * (1.0 + p.carrier_sound_speed**2) * d ** p.carrier_sound_speed**2
        + anchor / p.chemical_susceptibility
    )
    B_N = (rho_n - Q * d) / n
    B_D = (rho_d - Q * n) / d
    Psi = Lambda + B_N * n2 + B_D * d2 + 2.0 * Q * x2
    inverse = np.linalg.inv(metric)
    expected = (
        Psi * inverse
        + B_N * np.outer(N, N)
        + B_D * np.outer(C, C)
        + Q * (np.outer(N, C) + np.outer(C, N))
    )
    numerical = np.empty((4, 4), dtype=float)
    step = 2.0e-6
    for mu in range(4):
        for nu in range(mu, 4):
            plus = metric.copy()
            minus = metric.copy()
            plus[mu, nu] += step
            minus[mu, nu] -= step
            if mu != nu:
                plus[nu, mu] += step
                minus[nu, mu] -= step
            derivative = (density_lagrangian(plus) - density_lagrangian(minus)) / (2.0 * step)
            value = derivative / root if mu != nu else 2.0 * derivative / root
            numerical[mu, nu] = value
            numerical[nu, mu] = value
    return float(np.max(np.abs(numerical - expected)))


def run_audit(parameters: Theory33MasterParameters) -> dict[str, object]:
    master = Theory33MasterFunction(parameters)
    density_samples = (1.0e-3, 1.0e-2, 1.0e-1)
    internal_samples = (0.1, 1.0)
    velocity_samples = np.linspace(0.0, 0.57, 9)
    records: list[dict[str, object]] = []
    failures = 0
    maximum_speed = 0.0
    maximum_condition = 0.0
    minimum_legendre = np.inf
    minimum_thermo = np.inf
    for density in density_samples:
        for internal in internal_samples:
            carrier = max(parameters.target_fraction * density, parameters.carrier_floor * 10.0)
            for velocity in velocity_samples:
                audit = master.characteristic_audit(
                    density, internal, velocity, carrier, -velocity
                )
                gamma_rel = (1.0 + velocity**2) / (1.0 - velocity**2)
                inside = bool(
                    gamma_rel
                    <= parameters.maximum_relative_lorentz_factor + 1.0e-12
                )
                passed = bool(audit.strongly_hyperbolic and audit.causal)
                if inside and not passed:
                    failures += 1
                maximum_speed = max(maximum_speed, audit.maximum_absolute_speed)
                maximum_condition = max(maximum_condition, audit.eigenvector_condition)
                minimum_legendre = min(minimum_legendre, audit.legendre_minimum_eigenvalue)
                minimum_thermo = min(minimum_thermo, audit.thermodynamic_minimum_eigenvalue)
                records.append(
                    {
                        "density": density,
                        "internal_energy": internal,
                        "counterflow_velocity": float(velocity),
                        "relative_lorentz_factor": float(gamma_rel),
                        "inside_declared_domain": inside,
                        "speeds": audit.speeds.tolist(),
                        "condition": audit.eigenvector_condition,
                        "strongly_hyperbolic": audit.strongly_hyperbolic,
                        "causal": audit.causal,
                    }
                )
    stress_error = hilbert_stress_error(master)
    passed = bool(failures == 0 and stress_error < 2.0e-8)
    return {
        "theory": "3.3",
        "parameters": asdict(parameters),
        "passed": passed,
        "domain_failures": failures,
        "hilbert_stress_max_error": stress_error,
        "maximum_characteristic_speed_sampled": maximum_speed,
        "maximum_eigenvector_condition_sampled": maximum_condition,
        "minimum_legendre_eigenvalue_sampled": minimum_legendre,
        "minimum_thermodynamic_eigenvalue_sampled": minimum_thermo,
        "samples": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/theory33_closure/derivation_audit.json"),
    )
    args = parser.parse_args()
    result = run_audit(Theory33MasterParameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
