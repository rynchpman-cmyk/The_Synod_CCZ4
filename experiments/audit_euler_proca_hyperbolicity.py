"""Characteristic audit for the Euler--density-dependent Proca coupling.

The pointwise generalized Jacobian uses the exact total conserved densities and
fluxes implemented by ``ProductionSolver`` on a flat, homogeneous slice while
freezing the local Proca/target amplitudes.  It is therefore a diagnostic of
the total-matter block, not a substitute for a first-order reduction of the
full constrained Proca system.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from causal_repair.adm import flat_metric
from causal_repair.grhd import GRHDParameters, ValenciaGRHD
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive


@dataclass(frozen=True)
class LocalBackground:
    mass: float = 0.5
    coupling: float = 30.0
    Z: float = 1.0
    gamma_ad: float = 5.0 / 3.0
    a_x: float = 0.0
    astar_x: float = 0.0
    pi_x: float = 0.0


def primitive_coordinates(density: float, internal: float, velocity: np.ndarray) -> np.ndarray:
    speed_sq = float(velocity @ velocity)
    if density <= 0.0 or internal <= 0.0 or speed_sq >= 1.0:
        raise ValueError("state must have positive thermodynamics and subluminal velocity")
    spatial_four_velocity = velocity / np.sqrt(1.0-speed_sq)
    return np.concatenate(([np.log(density), np.log(internal)], spatial_four_velocity))


def state_and_flux(z: np.ndarray, background: LocalBackground) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (D,S_i,E) and its x flux for a local homogeneous state."""
    density, internal = np.exp(z[:2])
    spatial_u = z[2:5]
    lorentz = np.sqrt(1.0 + spatial_u @ spatial_u)
    velocity = spatial_u / lorentz
    pressure = (background.gamma_ad-1.0)*density*internal
    enthalpy = 1.0 + background.gamma_ad*internal
    D = density*lorentz
    fluid_momentum = density*enthalpy*lorentz**2*velocity
    fluid_energy = density*enthalpy*lorentz**2-pressure

    a = np.array([background.a_x, 0.0, 0.0])
    astar = np.array([background.astar_x, 0.0, 0.0])
    pi = np.array([background.pi_x, 0.0, 0.0])
    coefficient = background.coupling*density/background.Z
    effective_mass_squared = background.mass**2+coefficient
    phi_star = float(velocity @ astar)
    # Homogeneous form of the production code's algebraic Gauss reconstruction.
    phi = coefficient*phi_star/effective_mass_squared
    electric = -pi/background.Z
    identity = np.eye(3)

    a_squared = float(a @ a)
    electric_squared = float(electric @ electric)
    proca_rho = background.Z*(
        0.5*electric_squared
        + 0.5*background.mass**2*(phi**2+a_squared)
    )
    proca_momentum = background.Z*background.mass**2*phi*a
    proca_stress = background.Z*(
        -np.outer(electric, electric)
        + 0.5*identity*electric_squared
        + background.mass**2*(
            np.outer(a, a)-0.5*identity*(a_squared-phi**2)
        )
    )

    mismatch = a-astar
    mismatch_phi = phi-phi_star
    mismatch_squared = float(mismatch @ mismatch)
    interaction_coefficient = background.coupling*density
    interaction_rho = 0.5*interaction_coefficient*(
        mismatch_phi**2+mismatch_squared
    )
    interaction_momentum = interaction_coefficient*mismatch_phi*mismatch
    interaction_stress = interaction_coefficient*(
        np.outer(mismatch, mismatch)
        + 0.5*identity*(mismatch_phi**2-mismatch_squared)
    )

    repair_rho = proca_rho+interaction_rho
    repair_momentum = proca_momentum+interaction_momentum
    repair_stress = proca_stress+interaction_stress
    conserved = np.concatenate((
        [D], fluid_momentum+repair_momentum, [fluid_energy+repair_rho]
    ))
    flux = np.concatenate((
        [D*velocity[0]],
        fluid_momentum*velocity[0]+np.array([pressure, 0.0, 0.0])+repair_stress[0],
        [fluid_momentum[0]+repair_momentum[0]],
    ))
    return conserved, flux, effective_mass_squared


def numerical_derivative(z: np.ndarray, background: LocalBackground, output: int) -> np.ndarray:
    columns = []
    for index in range(len(z)):
        step = 2.0e-6*max(1.0, abs(float(z[index])))
        plus, minus = z.copy(), z.copy()
        plus[index] += step
        minus[index] -= step
        upper = state_and_flux(plus, background)[output]
        lower = state_and_flux(minus, background)[output]
        columns.append((upper-lower)/(2.0*step))
    return np.stack(columns, axis=1)


def characteristic_data(density: float, internal: float, velocity: np.ndarray,
                        background: LocalBackground) -> dict[str, object]:
    z = primitive_coordinates(density, internal, velocity)
    conserved_jacobian = numerical_derivative(z, background, 0)
    flux_jacobian = numerical_derivative(z, background, 1)
    condition_conserved = float(np.linalg.cond(conserved_jacobian))
    try:
        jacobian = np.linalg.solve(conserved_jacobian.T, flux_jacobian.T).T
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        condition_eigenvectors = float(np.linalg.cond(eigenvectors))
    except np.linalg.LinAlgError:
        eigenvalues = np.full(5, complex(np.nan, np.nan))
        condition_eigenvectors = float("inf")
    effective_mass_squared = state_and_flux(z, background)[2]
    maximum_imaginary = float(np.nanmax(np.abs(eigenvalues.imag)))
    maximum_speed = float(np.nanmax(np.abs(eigenvalues.real)))
    return {
        "eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag)}
            for value in eigenvalues[np.argsort(eigenvalues.real)]
        ],
        "maximum_imaginary_part": maximum_imaginary,
        "maximum_characteristic_speed": maximum_speed,
        "eigenvector_condition": condition_eigenvectors,
        "conserved_map_condition": condition_conserved,
        "effective_mass_squared": float(effective_mass_squared),
        "real_diagonalizable": bool(
            maximum_imaginary < 1.0e-7
            and np.isfinite(condition_eigenvectors)
            and condition_eigenvectors < 1.0e8
        ),
    }


def evolved_shock(points: int, final_time: float) -> tuple[np.ndarray, FluidPrimitive]:
    grid = PeriodicGrid((points,), (1.0,), "fd2")
    h = flat_metric(grid)
    solver = ValenciaGRHD(
        grid, GRHDParameters(density_floor=1.0e-10, internal_energy_floor=1.0e-9)
    )
    (x,) = grid.coordinates()
    fluid = FluidPrimitive(
        np.where(x < 0.5, 1.0, 0.125),
        np.where(x < 0.5, 1.5, 1.2),
        grid.zeros((3,)), grid.zeros(),
    )
    state = solver.primitives_to_conserved(fluid, h)
    dt_limit = 0.12*grid.spacing[0]
    while state.time < final_time-1.0e-14:
        state = solver.step(
            state, min(dt_limit, final_time-state.time), h, grid.zeros((3, 3)),
            np.ones(grid.shape), grid.zeros((3,)),
        )
    recovered, report = solver.conserved_to_primitives(state, h)
    if report.failed_cells:
        raise RuntimeError("shock evolution primitive recovery failed")
    return x, recovered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--time", type=float, default=0.12)
    parser.add_argument("--output", type=Path, default=Path("runs/hyperbolicity_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x, shock = evolved_shock(args.points, args.time)

    amplitudes = np.linspace(0.0, 3.0, 61)
    couplings = (0.0, 10.0, 30.0, 100.0)
    sample_indices = np.arange(0, args.points, max(1, args.points//64))
    scans = []
    for coupling in couplings:
        for amplitude in amplitudes:
            background = LocalBackground(
                coupling=coupling, a_x=0.5*amplitude, astar_x=amplitude
            )
            local = [
                characteristic_data(
                    float(shock.baryon_density[i]),
                    float(shock.specific_internal_energy[i]),
                    shock.velocity[:, i], background,
                )
                for i in sample_indices
            ]
            scans.append({
                "coupling": coupling,
                "target_amplitude": float(amplitude),
                "nonhyperbolic_fraction": float(np.mean([
                    not item["real_diagonalizable"] for item in local
                ])),
                "superluminal_fraction": float(np.mean([
                    item["maximum_characteristic_speed"] > 1.0+1.0e-7 for item in local
                ])),
                "maximum_imaginary_part": max(item["maximum_imaginary_part"] for item in local),
                "maximum_characteristic_speed": max(item["maximum_characteristic_speed"] for item in local),
                "maximum_eigenvector_condition": max(item["eigenvector_condition"] for item in local),
            })

    representative = []
    for name, index in (
        ("left_state", int(np.argmin(np.abs(x-0.25)))),
        ("shock_front", int(np.argmax(np.abs(np.gradient(shock.baryon_density))))),
        ("maximum_velocity", int(np.argmax(np.abs(shock.velocity[0])))),
        ("right_state", int(np.argmin(np.abs(x-0.75)))),
    ):
        states = []
        for coupling, amplitude in ((0.0, 0.0), (30.0, 0.1), (30.0, 0.5), (30.0, 2.0)):
            background = LocalBackground(coupling=coupling, a_x=0.5*amplitude, astar_x=amplitude)
            states.append({
                "background": asdict(background),
                "characteristics": characteristic_data(
                    float(shock.baryon_density[index]),
                    float(shock.specific_internal_energy[index]),
                    shock.velocity[:, index], background,
                ),
            })
        representative.append({
            "name": name, "index": index, "x": float(x[index]),
            "density": float(shock.baryon_density[index]),
            "specific_internal_energy": float(shock.specific_internal_energy[index]),
            "velocity": shock.velocity[:, index].tolist(), "states": states,
        })

    thresholds = []
    for coupling in couplings[1:]:
        rows = [row for row in scans if row["coupling"] == coupling]
        nonhyperbolic = next((row["target_amplitude"] for row in rows if row["nonhyperbolic_fraction"] > 0), None)
        superluminal = next((row["target_amplitude"] for row in rows if row["superluminal_fraction"] > 0), None)
        thresholds.append({
            "coupling": coupling,
            "first_sampled_nonhyperbolic_amplitude": nonhyperbolic,
            "first_sampled_superluminal_amplitude": superluminal,
        })

    # A shock's cold upstream state can be highly boosted even when the
    # standard Sod-like tube above is only mildly relativistic.  Scan that
    # independent state envelope to expose velocity/longitudinal-field mixing.
    boost_velocities = np.linspace(0.0, 0.95, 39)
    boost_amplitudes = np.linspace(0.0, 1.5, 61)
    boosted_scan = []
    for coupling in couplings[1:]:
        for amplitude in boost_amplitudes:
            background = LocalBackground(
                coupling=coupling, a_x=0.5*amplitude, astar_x=amplitude
            )
            local = [
                characteristic_data(
                    0.125, 0.10, np.array([velocity, 0.0, 0.0]), background
                )
                for velocity in boost_velocities
            ]
            complex_mask = np.array([
                item["maximum_imaginary_part"] > 1.0e-7 for item in local
            ])
            boosted_scan.append({
                "coupling": coupling,
                "target_amplitude": float(amplitude),
                "first_nonhyperbolic_velocity": (
                    float(boost_velocities[np.argmax(complex_mask)])
                    if np.any(complex_mask) else None
                ),
                "nonhyperbolic_velocity_fraction": float(np.mean(complex_mask)),
                "maximum_imaginary_part": max(
                    item["maximum_imaginary_part"] for item in local
                ),
            })
    boosted_thresholds = []
    for coupling in couplings[1:]:
        rows = [row for row in boosted_scan if row["coupling"] == coupling]
        first = next(
            (row for row in rows if row["first_nonhyperbolic_velocity"] is not None),
            None,
        )
        boosted_thresholds.append({
            "coupling": coupling,
            "first_sampled_nonhyperbolic_amplitude": (
                None if first is None else first["target_amplitude"]
            ),
            "velocity_at_first_sampled_loss": (
                None if first is None else first["first_nonhyperbolic_velocity"]
            ),
        })

    payload = {
        "scope": "frozen-Proca total-matter generalized Jacobian on an evolved relativistic shock",
        "warning": "not a proof for the full constrained mixed-order PDE system",
        "points": args.points, "shock_time": args.time,
        "shock_maximum_density_gradient": float(np.max(np.abs(np.gradient(shock.baryon_density, x)))),
        "shock_maximum_speed": float(np.max(np.abs(shock.velocity[0]))),
        "shock_density_range": [
            float(np.min(shock.baryon_density)), float(np.max(shock.baryon_density))
        ],
        "effective_mass_squared_ranges": {
            f"g={coupling:g}": [
                float(0.5**2+coupling*np.min(shock.baryon_density)),
                float(0.5**2+coupling*np.max(shock.baryon_density)),
            ]
            for coupling in couplings
        },
        "thresholds": thresholds, "representative_states": representative,
        "scan": scans,
        "boosted_cold_state": {
            "density": 0.125, "specific_internal_energy": 0.10,
            "thresholds": boosted_thresholds, "scan": boosted_scan,
        },
    }
    maximum_velocity_index = int(np.argmax(np.abs(shock.velocity[0])))
    payload["aligned_target_controls"] = [
        {
            "coupling": coupling, "a_x_equals_astar_x": amplitude,
            "characteristics": characteristic_data(
                float(shock.baryon_density[maximum_velocity_index]),
                float(shock.specific_internal_energy[maximum_velocity_index]),
                shock.velocity[:, maximum_velocity_index],
                LocalBackground(
                    coupling=coupling, a_x=amplitude, astar_x=amplitude
                ),
            ),
        }
        for coupling in (30.0, 100.0)
        for amplitude in (0.5, 2.0)
    ]
    (args.output/"characteristic_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    for coupling in couplings[1:]:
        rows = [row for row in scans if row["coupling"] == coupling]
        axes[0].plot(amplitudes, [r["maximum_characteristic_speed"] for r in rows], label=f"g={coupling:g}")
        axes[1].plot(amplitudes, [r["nonhyperbolic_fraction"] for r in rows], label=f"g={coupling:g}")
    axes[0].axhline(1.0, color="black", ls="--", lw=1)
    axes[0].set(xlabel=r"target amplitude $|A_*|$ (with $A=A_*/2$)", ylabel="max |local eigenvalue|", title="Characteristic-speed deformation")
    axes[1].set(xlabel=r"target amplitude $|A_*|$", ylabel="fraction of sampled shock states", title="Complex-eigenvalue incidence", ylim=(-0.02, 1.02))
    for coupling in couplings[1:]:
        rows = [row for row in boosted_scan if row["coupling"] == coupling]
        onset = [
            np.nan if row["first_nonhyperbolic_velocity"] is None
            else row["first_nonhyperbolic_velocity"]
            for row in rows
        ]
        axes[2].plot(boost_amplitudes, onset, label=f"g={coupling:g}")
    axes[2].set(
        xlabel=r"target amplitude $|A_*|$", ylabel="first non-real velocity",
        title="Cold boosted upstream state", ylim=(0.0, 1.0),
    )
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(frameon=False)
    figure.savefig(args.output/"characteristic_audit.png", dpi=180)
    print(json.dumps({
        "thresholds": thresholds,
        "boosted_cold_thresholds": boosted_thresholds,
        "shock_maximum_speed": payload["shock_maximum_speed"],
        "shock_maximum_density_gradient": payload["shock_maximum_density_gradient"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
