"""Explore and rank visually interesting fixed-background repair dynamics.

Run from the repository root with ``PYTHONPATH=src``.  Results are JSON plus
compressed trajectories for candidates selected with ``--case``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from causal_repair.grid import PeriodicGrid
from causal_repair.initial_data import TwoLobeParameters, rotating_two_lobe_data
from causal_repair.theory2 import Theory2Parameters, interaction_stress_energy
from causal_repair.fixed_theory2 import Theory2FixedSolver


@dataclass(frozen=True)
class Case:
    mass: float
    tau_star: float
    coupling: float
    peak_density: float
    beta_target: float = 1.76

    @property
    def name(self) -> str:
        return (
            f"m{self.mass:g}_tau{self.tau_star:g}_g{self.coupling:g}_"
            f"rho{self.peak_density:g}_b{self.beta_target:g}"
        ).replace(".", "p")


def centered_coordinate(grid: PeriodicGrid) -> np.ndarray:
    return grid.coordinates()[0] - 0.5 * grid.lengths[0]


def count_zero_crossings(values: np.ndarray, threshold: float = 0.03) -> int:
    scale = max(float(np.max(np.abs(values))), 1e-30)
    signs = np.sign(np.where(np.abs(values) >= threshold * scale, values, 0.0))
    signs = signs[signs != 0]
    return int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0


def run_case(
    case: Case,
    points: int,
    length: float,
    final_time: float,
    samples: int,
    store_fields: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    grid = PeriodicGrid((points,), (length,), "fd4")
    theory = Theory2Parameters(
        mass=case.mass,
        coupling=case.coupling,
        tau_star=case.tau_star,
    )
    benchmark = TwoLobeParameters(
        separation=3.0,
        width=0.9,
        peak_density=case.peak_density,
        matter_speed=0.2,
        beta_target=case.beta_target,
    )
    initial = rotating_two_lobe_data(grid, theory, benchmark)
    solver = Theory2FixedSolver(grid, initial.fluid, theory, h=initial.geometry.h)
    state = initial.repair
    initial_profile = state.a[2].copy()
    profile_norm = max(grid.integrate(initial_profile**2), 1e-30)
    x = centered_coordinate(grid)
    exterior = np.abs(x) > 3.5
    sample_times = np.linspace(0.0, final_time, samples)
    dt_limit = min(0.10 * min(grid.spacing), 0.08 / case.mass, 0.08 * case.tau_star)

    series: dict[str, list[float]] = {
        key: []
        for key in (
            "time", "a_l2", "astar_l2", "mismatch_l2", "pi_l2", "phi_l2",
            "proca_energy", "interaction_energy", "exterior_energy_fraction",
            "profile_projection", "spectral_entropy", "minimum_causal_margin",
        )
    }
    fields: dict[str, list[np.ndarray]] | None = (
        {"a_z": [], "astar_z": [], "pi_z": [], "phi": [], "energy": []}
        if store_fields else None
    )

    def record() -> None:
        phi = solver.phi(state)
        proca = solver.proca.stress_energy_theory2(
            solver.h, state.a, state.pi, solver.fluid, state.astar
        )
        interaction = interaction_stress_energy(
            solver.h, solver.fluid, state.a, phi, state.astar, theory
        )
        repair_energy = proca[0] + interaction[0]
        total_repair = max(grid.integrate(repair_energy), 1e-30)
        spectrum = np.abs(np.fft.rfft(state.a[2])) ** 2
        spectrum[0] = 0.0
        probabilities = spectrum / max(float(np.sum(spectrum)), 1e-30)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
        entropy /= np.log(max(len(probabilities), 2))
        _, margin = solver.minkowski_current(state)
        series["time"].append(state.time)
        series["a_l2"].append(float(np.sqrt(np.mean(np.sum(state.a**2, axis=0)))))
        series["astar_l2"].append(float(np.sqrt(np.mean(np.sum(state.astar**2, axis=0)))))
        series["mismatch_l2"].append(float(np.sqrt(np.mean(np.sum((state.a-state.astar)**2, axis=0)))))
        series["pi_l2"].append(float(np.sqrt(np.mean(np.sum(state.pi**2, axis=0)))))
        series["phi_l2"].append(float(np.sqrt(np.mean(phi**2))))
        series["proca_energy"].append(grid.integrate(proca[0]))
        series["interaction_energy"].append(grid.integrate(interaction[0]))
        series["exterior_energy_fraction"].append(
            float(grid.integrate(np.where(exterior, repair_energy, 0.0)) / total_repair)
        )
        series["profile_projection"].append(
            float(grid.integrate(state.a[2] * initial_profile) / profile_norm)
        )
        series["spectral_entropy"].append(float(entropy))
        series["minimum_causal_margin"].append(float(np.min(margin)))
        if fields is not None:
            fields["a_z"].append(state.a[2].copy())
            fields["astar_z"].append(state.astar[2].copy())
            fields["pi_z"].append(state.pi[2].copy())
            fields["phi"].append(phi.copy())
            fields["energy"].append(repair_energy.copy())

    record()
    for target_time in sample_times[1:]:
        while state.time < target_time - 1e-13:
            dt = min(dt_limit, target_time - state.time)
            state = solver.step(state, dt)
        record()

    arrays = {key: np.asarray(value) for key, value in series.items()}
    initial_a = max(arrays["a_l2"][0], 1e-30)
    late = arrays["time"] >= min(2.0 * case.tau_star, 0.7 * final_time)
    metrics = {
        "case": asdict(case),
        "name": case.name,
        "stable": bool(all(np.all(np.isfinite(value)) for value in arrays.values())),
        "peak_mismatch_over_initial_a": float(np.max(arrays["mismatch_l2"]) / initial_a),
        "peak_interaction_fraction": float(
            np.max(arrays["interaction_energy"] / np.maximum(arrays["proca_energy"], 1e-30))
        ),
        "maximum_exterior_energy_fraction": float(np.max(arrays["exterior_energy_fraction"])),
        "final_exterior_energy_fraction": float(arrays["exterior_energy_fraction"][-1]),
        "final_core_retention_fraction": float(1.0 - arrays["exterior_energy_fraction"][-1]),
        "exterior_half_time": (
            float(arrays["time"][np.flatnonzero(arrays["exterior_energy_fraction"] >= 0.5)[0]])
            if np.any(arrays["exterior_energy_fraction"] >= 0.5)
            else None
        ),
        "late_field_fraction": float(np.max(arrays["a_l2"][late]) / initial_a),
        "profile_zero_crossings": count_zero_crossings(arrays["profile_projection"]),
        "maximum_spectral_entropy": float(np.max(arrays["spectral_entropy"])),
        "minimum_causal_margin": float(np.min(arrays["minimum_causal_margin"])),
        "final_gauss_l2": solver.diagnostics(state)["gauss_l2"],
        "dt_limit": dt_limit,
        "points": points,
        "length": length,
        "final_time": final_time,
    }
    trajectory = None
    if fields is not None:
        trajectory = {"x": x, **arrays, **{key: np.asarray(value) for key, value in fields.items()}}
    return metrics, trajectory


def cases() -> list[Case]:
    return [
        Case(mass, tau, coupling, density)
        for mass, tau, coupling, density in product(
            (0.25, 0.6, 1.2),
            (0.15, 0.6, 2.4),
            (0.0, 5.0, 30.0),
            (0.002, 0.08),
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/repair_exploration"))
    parser.add_argument("--points", type=int, default=96)
    parser.add_argument("--length", type=float, default=24.0)
    parser.add_argument("--final-time", type=float, default=8.0)
    parser.add_argument("--samples", type=int, default=81)
    parser.add_argument("--case", type=int, help="run only this case index and save fields")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = cases()
    if args.case is not None:
        selected = [selected[args.case]]
    summaries = []
    for index, case in enumerate(selected):
        metrics, trajectory = run_case(
            case, args.points, args.length, args.final_time, args.samples, args.case is not None
        )
        summaries.append(metrics)
        print(json.dumps({"index": args.case if args.case is not None else index, **metrics}, sort_keys=True))
        if trajectory is not None:
            np.savez_compressed(args.output / f"{case.name}.npz", **trajectory)
    destination = args.output / ("single_summary.json" if args.case is not None else "sweep_summary.json")
    destination.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
