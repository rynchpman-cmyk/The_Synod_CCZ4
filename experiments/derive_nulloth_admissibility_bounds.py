"""Derive high-precision sufficient convexity bounds for Nulloth."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import mpmath as mp

from causal_repair.nulloth import NULLOTH_DISCOVERY_PARAMETERS


@dataclass(frozen=True)
class AdmissibilityBounds:
    precision_decimal_digits: int
    production_scalar_amplitude_cap: float
    thermodynamic_sufficient_lower_bound_at_cap: float
    legendre_sufficient_lower_bound_at_cap: float
    global_thermodynamic_guaranteed_amplitude: float
    global_legendre_guaranteed_amplitude: float
    central_condition_denial_amplitude: float
    central_thermodynamic_zero_amplitude: float
    central_legendre_zero_amplitude: float
    condition_to_thermodynamic_gap: float
    thermodynamic_to_legendre_gap: float
    thermodynamic_cap_margin_fraction: float
    legendre_cap_margin_fraction: float
    binary64_thermodynamic_root_error: float
    binary64_legendre_root_error: float


def _first_positive_root(function, upper: mp.mpf = mp.mpf("1")) -> mp.mpf:
    lower = mp.mpf("0")
    lower_value = function(lower)
    if lower_value <= 0:
        raise ValueError("root function is not positive at zero")
    bracket_upper = None
    for index in range(1, 1001):
        point = upper * index / 1000
        if function(point) <= 0:
            bracket_upper = point
            break
        lower = point
    if bracket_upper is None:
        raise ValueError("no positive root in the requested interval")
    for _ in range(256):
        midpoint = (lower + bracket_upper) / 2
        if function(midpoint) > 0:
            lower = midpoint
        else:
            bracket_upper = midpoint
    return (lower + bracket_upper) / 2


def derive_bounds() -> AdmissibilityBounds:
    digits = 80
    mp.mp.dps = digits
    p = NULLOTH_DISCOVERY_PARAMETERS
    gamma = mp.mpf(str(p.gamma_ad))
    target = mp.mpf(str(p.target_fraction))
    susceptibility = mp.mpf(str(p.chemical_susceptibility))
    carrier_k = mp.mpf(str(p.carrier_K))
    carrier_cs = mp.mpf(str(p.carrier_sound_speed))
    carrier_cs2 = carrier_cs**2
    scalar_n = mp.mpf(str(p.scalar_baryon_coupling))
    scalar_d = mp.mpf(str(p.scalar_carrier_coupling))
    scalar_w = mp.mpf(str(p.scalar_counterflow_coupling))
    amplitude_cap = mp.mpf(str(p.scalar_amplitude_limit))
    n_min, n_max = mp.mpf("0.4"), mp.mpf("0.9")
    e_min = mp.mpf("0.1")
    d_min, d_max = mp.mpf("0.12"), mp.mpf("0.24")

    thermo_nn_zero = (
        gamma * (gamma - 1) * e_min / n_max
        + target**2 / susceptibility
    )
    thermo_dd_zero = (
        carrier_k
        * (1 + carrier_cs2)
        * carrier_cs2
        * d_max ** (carrier_cs2 - 1)
        + 1 / susceptibility
    )
    thermo_offdiag = target / susceptibility

    def thermodynamic_lower(phi: mp.mpf) -> mp.mpf:
        nn = thermo_nn_zero - scalar_n * phi**2
        dd = thermo_dd_zero - scalar_d * phi**2
        return min(nn, dd) - thermo_offdiag

    thermodynamic_guarantee = min(
        mp.inf
        if scalar_n == 0
        else mp.sqrt((thermo_nn_zero - thermo_offdiag) / scalar_n),
        mp.sqrt((thermo_dd_zero - thermo_offdiag) / scalar_d),
    )

    legendre_n_zero = (
        (1 + gamma * e_min) / n_max
        - target
        / susceptibility
        * (d_max / n_min - target)
    )
    legendre_d_zero = (
        carrier_k * (1 + carrier_cs2) * d_max ** (carrier_cs2 - 1)
        + (1 - target * n_max / d_min) / susceptibility
    )

    def legendre_lower(phi: mp.mpf) -> mp.mpf:
        mixing = mp.mpf("0.5") * scalar_w * phi**2
        nn = legendre_n_zero - scalar_n * phi**2
        dd = legendre_d_zero - scalar_d * phi**2
        return min(nn, dd) - mixing

    legendre_guarantee = min(
        mp.sqrt(
            legendre_n_zero
            / (scalar_n + mp.mpf("0.5") * scalar_w)
        ),
        mp.sqrt(
            legendre_d_zero
            / (scalar_d + mp.mpf("0.5") * scalar_w)
        ),
    )

    n = mp.mpf("0.65")
    internal = mp.mpf("0.225")
    d = mp.mpf("0.18")
    thermo_nn = (
        gamma * (gamma - 1) * internal / n
        + target**2 / susceptibility
    )
    thermo_dd_base = (
        carrier_k
        * (1 + carrier_cs2)
        * carrier_cs2
        * d ** (carrier_cs2 - 1)
        + 1 / susceptibility
    )

    def central_thermo_determinant(phi: mp.mpf) -> mp.mpf:
        nn = thermo_nn - scalar_n * phi**2
        dd = thermo_dd_base - scalar_d * phi**2
        return nn * dd - (target / susceptibility) ** 2

    anchor = d - target * n

    def central_legendre_determinant(phi: mp.mpf) -> mp.mpf:
        coupling = -mp.mpf("0.5") * scalar_w * phi**2
        rho_n = (
            1
            + gamma * internal
            - target * anchor / susceptibility
            - scalar_n * phi**2 * n
        )
        rho_d = (
            carrier_k * (1 + carrier_cs2) * d**carrier_cs2
            + anchor / susceptibility
            - scalar_d * phi**2 * d
        )
        b_n = (rho_n - coupling * d) / n
        b_d = (rho_d - coupling * n) / d
        return b_n * b_d - coupling**2

    thermo_root = _first_positive_root(central_thermo_determinant)
    legendre_root = _first_positive_root(central_legendre_determinant)
    condition_boundary = mp.mpf("0.20536348521709444")

    # Independently evaluate the same roots with ordinary binary64 algebra.
    float_thermo_root = (
        (
            float(thermo_dd_base)
            - float(target / susceptibility) ** 2 / float(thermo_nn)
        )
        / float(scalar_d)
    ) ** 0.5

    def float_legendre(phi: float) -> float:
        return float(central_legendre_determinant(mp.mpf(str(phi))))

    float_lower, float_upper = 0.0, 1.0
    for _ in range(80):
        midpoint = 0.5 * (float_lower + float_upper)
        if float_legendre(midpoint) > 0.0:
            float_lower = midpoint
        else:
            float_upper = midpoint
    float_legendre_root = 0.5 * (float_lower + float_upper)

    return AdmissibilityBounds(
        digits,
        float(amplitude_cap),
        float(thermodynamic_lower(amplitude_cap)),
        float(legendre_lower(amplitude_cap)),
        float(thermodynamic_guarantee),
        float(legendre_guarantee),
        float(condition_boundary),
        float(thermo_root),
        float(legendre_root),
        float(thermo_root - condition_boundary),
        float(legendre_root - thermo_root),
        float((thermodynamic_guarantee - amplitude_cap) / amplitude_cap),
        float((legendre_guarantee - amplitude_cap) / amplitude_cap),
        abs(float_thermo_root - float(thermo_root)),
        abs(float_legendre_root - float(legendre_root)),
    )


def main() -> None:
    print(json.dumps(asdict(derive_bounds()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
