"""Reproduce the complete promotion campaign for the Carrier Coherence Brake."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from promote_nulloth_spell_candidates import (
    SpellPromotionResult,
    promote_candidate,
)


@dataclass(frozen=True)
class Spell001Qualification:
    resolution_results: tuple[SpellPromotionResult, ...]
    long_run: SpellPromotionResult
    resolutions_count: int
    promoted_resolutions: int
    long_run_promoted: int
    nonlinear_turnover: int
    scalar_amplitude_relative_change_16_to_32: float
    carrier_velocity_effect_relative_change_16_to_32: float
    high_resolution_kinetic_effect: float
    suppressive_sign_consistent_16_to_32: int
    maximum_number_relative_drift: float
    maximum_gauss_l2: float
    maximum_scalar_constraint_l2: float
    maximum_gravitational_constraint_l2: float
    minimum_legendre_eigenvalue: float
    minimum_thermodynamic_eigenvalue: float
    maximum_restart_error: float
    long_peak_scalar_amplitude: float
    long_peak_to_final_drop: float


def _relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-30)


def qualify_spell_001() -> Spell001Qualification:
    resolutions = (4, 8, 16, 32)
    results = tuple(
        promote_candidate(points=points, duration=2.0, dt=1.0e-2)
        for points in resolutions
    )
    long_run = promote_candidate(points=8, duration=6.0, dt=1.0e-2)
    medium = results[2]
    high = results[3]
    all_results = results + (long_run,)

    return Spell001Qualification(
        results,
        long_run,
        len(results),
        sum(result.promoted for result in results),
        int(long_run.promoted),
        int(long_run.active.nonlinear_turnover),
        _relative_change(
            medium.active.final_scalar_amplitude,
            high.active.final_scalar_amplitude,
        ),
        _relative_change(
            medium.carrier_velocity_effect_l2,
            high.carrier_velocity_effect_l2,
        ),
        high.carrier_kinetic_effect,
        int(
            medium.carrier_kinetic_effect < 0.0
            and high.carrier_kinetic_effect < 0.0
        ),
        max(
            max(
                result.active.baryon_relative_drift,
                result.active.carrier_relative_drift,
            )
            for result in all_results
        ),
        max(result.active.gauss_max_l2 for result in all_results),
        max(result.active.scalar_constraint_l2 for result in all_results),
        max(
            max(result.active.hamiltonian_l2, result.active.momentum_l2)
            for result in all_results
        ),
        min(result.active.minimum_legendre_eigenvalue for result in all_results),
        min(
            result.active.minimum_thermodynamic_eigenvalue
            for result in all_results
        ),
        max(result.active.restart_maximum_error for result in all_results),
        long_run.active.peak_scalar_amplitude,
        long_run.active.peak_to_final_drop,
    )


def main() -> None:
    print(json.dumps(asdict(qualify_spell_001()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
