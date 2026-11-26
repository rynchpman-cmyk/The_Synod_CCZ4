"""Audit the Nulloth material stress against fixed-current metric variation."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from causal_repair.nulloth import NullothMasterFunction, NullothParameters


def hilbert_material_stress_error(
    master: NullothMasterFunction, varphi: float
) -> float:
    """Compare Carter stress with metric variation at fixed densitized currents."""
    metric = np.diag([-1.0, 1.0, 1.0, 1.0])
    densitized_baryon = np.array([0.12, 0.018, -0.004, 0.0])
    densitized_carrier = np.array([0.008, -0.001, 0.0005, 0.0])
    entropy = 0.3

    def density_lagrangian(g: np.ndarray) -> float:
        root = float(np.sqrt(-np.linalg.det(g)))
        current_baryon = densitized_baryon / root
        current_carrier = densitized_carrier / root
        n_squared = -float(current_baryon @ g @ current_baryon)
        d_squared = -float(current_carrier @ g @ current_carrier)
        x_squared = -float(current_baryon @ g @ current_carrier)
        return root * float(
            master.lambda_from_invariants(
                np.array(n_squared),
                np.array(d_squared),
                np.array(x_squared),
                np.array(entropy),
                np.array(varphi),
            )
        )

    root = float(np.sqrt(-np.linalg.det(metric)))
    baryon = densitized_baryon / root
    carrier = densitized_carrier / root
    n_squared = -float(baryon @ metric @ baryon)
    d_squared = -float(carrier @ metric @ carrier)
    x_squared = -float(baryon @ metric @ carrier)
    n = float(np.sqrt(n_squared))
    d = float(np.sqrt(d_squared))
    relative = x_squared - n * d
    p = master.parameters
    pressure = np.exp((p.gamma_ad - 1.0) * entropy) * n**p.gamma_ad
    anchor = d - p.target_fraction * n
    rho_zero = (
        n
        + pressure / (p.gamma_ad - 1.0)
        + p.carrier_K * d ** (1.0 + p.carrier_sound_speed**2)
        + 0.5 * anchor**2 / p.chemical_susceptibility
    )
    coupling = (
        p.scalar_baryon_coupling * (n / p.scalar_baryon_scale) ** 2
        + p.scalar_carrier_coupling * (d / p.scalar_carrier_scale) ** 2
        + p.scalar_counterflow_coupling
        * relative
        / p.scalar_counterflow_scale
    )
    entrainment = (
        p.entrainment_linear
        + p.entrainment_quadratic
        * relative
        / p.entrainment_scale_fourth
        - 0.5
        * p.scalar_counterflow_coupling
        * varphi**2
        / p.scalar_counterflow_scale
    )
    lagrangian = (
        -rho_zero
        - p.entrainment_linear * relative
        - 0.5
        * p.entrainment_quadratic
        * relative**2
        / p.entrainment_scale_fourth
        + 0.5 * coupling * varphi**2
    )
    rho_n = (
        1.0
        + p.gamma_ad * pressure / ((p.gamma_ad - 1.0) * n)
        - p.target_fraction * anchor / p.chemical_susceptibility
        - p.scalar_baryon_coupling
        * varphi**2
        * n
        / p.scalar_baryon_scale**2
    )
    rho_d = (
        p.carrier_K
        * (1.0 + p.carrier_sound_speed**2)
        * d ** p.carrier_sound_speed**2
        + anchor / p.chemical_susceptibility
        - p.scalar_carrier_coupling
        * varphi**2
        * d
        / p.scalar_carrier_scale**2
    )
    coefficient_baryon = (rho_n - entrainment * d) / n
    coefficient_carrier = (rho_d - entrainment * n) / d
    generalized_pressure = (
        lagrangian
        + coefficient_baryon * n_squared
        + coefficient_carrier * d_squared
        + 2.0 * entrainment * x_squared
    )
    inverse = np.linalg.inv(metric)
    expected = (
        generalized_pressure * inverse
        + coefficient_baryon * np.outer(baryon, baryon)
        + coefficient_carrier * np.outer(carrier, carrier)
        + entrainment
        * (
            np.outer(baryon, carrier)
            + np.outer(carrier, baryon)
        )
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
            derivative = (
                density_lagrangian(plus) - density_lagrangian(minus)
            ) / (2.0 * step)
            value = derivative / root if mu != nu else 2.0 * derivative / root
            numerical[mu, nu] = value
            numerical[nu, mu] = value
    return float(np.max(np.abs(numerical - expected)))


def run_action_audit() -> dict[str, object]:
    parameters = NullothParameters(
        entrainment_linear=0.03,
        entrainment_quadratic=0.02,
        scalar_baryon_coupling=0.7,
        scalar_carrier_coupling=1.3,
        scalar_counterflow_coupling=0.9,
    )
    master = NullothMasterFunction(parameters)
    amplitudes = (0.0, 0.23, -0.31)
    errors = tuple(
        hilbert_material_stress_error(master, amplitude)
        for amplitude in amplitudes
    )
    return {
        "parameters": asdict(parameters),
        "amplitudes": amplitudes,
        "cases": len(amplitudes),
        "hilbert_material_stress_errors": errors,
        "hilbert_material_stress_max_error": max(errors),
    }


def main() -> None:
    print(json.dumps(run_action_audit(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

