"""Run the Theory 3.3 counterflow compact-collision campaign.

The inexpensive campaign is a one-dimensional periodic planar analogue.  A
periodic constant-mean-curvature solve supplies constraint-compatible strong
gravity, while two compact baryon pulses and their faster carrier currents
collide around the origin.  Every saved trajectory contains dashboard-ready
primitive and geometry fields.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grid import PeriodicGrid
from causal_repair.initial_constraints import (
    CTTParameters,
    PeriodicCMCInitialDataSolver,
)
from causal_repair.matter import FluidPrimitive
from causal_repair.production3 import Theory3ProductionParameters
from causal_repair.production33 import (
    Theory33ProductionSolver,
    save_theory33_production_state,
)
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import (
    CarrierPrimitive,
    Theory33MasterFunction,
    Theory33MasterParameters,
)


@dataclass(frozen=True)
class CollisionCase:
    peak_density: float
    counterflow: float
    relaxation_time: float
    collision_speed: float = 0.18
    separation: float = 1.1
    width: float = 0.75
    density_floor_fraction: float = 0.01
    internal_energy: float = 0.25
    target_fraction: float = 0.05
    chemical_susceptibility: float = 10.0
    carrier_sound_speed: float = 0.9
    carrier_K: float = 0.2

    @property
    def name(self) -> str:
        fraction = (
            ""
            if abs(self.target_fraction - 0.05) < 1.0e-14
            else f"_f{self.target_fraction:.2f}"
        )
        sound = (
            ""
            if abs(self.carrier_sound_speed - 0.9) < 1.0e-14
            else f"_cs{self.carrier_sound_speed:.2f}"
        )
        carrier_scale = (
            ""
            if abs(self.carrier_K - 0.2) < 1.0e-14
            else f"_k{self.carrier_K:.2f}"
        )
        return (
            f"rho{self.peak_density:.3f}_"
            f"cf{self.counterflow:.2f}_tau{self.relaxation_time:.2f}"
            f"{fraction}{sound}{carrier_scale}"
        ).replace(".", "p")


DEFAULT_SWEEP = tuple(
    CollisionCase(peak, counterflow, relaxation)
    for peak in (0.012, 0.025)
    for counterflow in (0.0, 0.28)
    for relaxation in (0.03, 0.30)
)


FRACTION_SWEEP = tuple(
    CollisionCase(
        0.025,
        0.28,
        relaxation,
        target_fraction=fraction,
    )
    for fraction in (0.05, 0.10, 0.20, 0.35, 0.50, 1.00, 2.00)
    for relaxation in (0.03, 0.30)
)


def constitutive_gate(
    case: CollisionCase, master: Theory33MasterParameters
) -> dict[str, float | bool]:
    function = Theory33MasterFunction(master)
    audits = []
    floor = case.density_floor_fraction * case.peak_density
    carrier_speed = min(case.collision_speed + case.counterflow, 0.8)
    for density in (floor, case.peak_density, 2.0 * case.peak_density):
        carrier_density = max(case.target_fraction * density, master.carrier_floor)
        for baryon_velocity, target_velocity in (
            (0.0, 0.0),
            (case.collision_speed, carrier_speed),
            (-case.collision_speed, -carrier_speed),
        ):
            audits.append(
                function.characteristic_audit(
                    density,
                    case.internal_energy,
                    baryon_velocity,
                    carrier_density,
                    target_velocity,
                )
            )
    passed = all(audit.strongly_hyperbolic and audit.causal for audit in audits)
    result: dict[str, float | bool] = {
        "passed": passed,
        "maximum_characteristic_speed": max(
            audit.maximum_absolute_speed for audit in audits
        ),
        "maximum_eigenvector_condition": max(
            audit.eigenvector_condition for audit in audits
        ),
        "minimum_legendre_eigenvalue": min(
            audit.legendre_minimum_eigenvalue for audit in audits
        ),
        "minimum_thermodynamic_eigenvalue": min(
            audit.thermodynamic_minimum_eigenvalue for audit in audits
        ),
    }
    if not passed:
        raise FloatingPointError(
            f"carrier-fraction constitutive gate failed: {result}"
        )
    return result


def collision_primitives(
    grid: PeriodicGrid, case: CollisionCase
) -> tuple[FluidPrimitive, CarrierPrimitive]:
    (coordinate,) = grid.coordinates()
    x = coordinate - 0.5 * grid.lengths[0]
    left = np.exp(-((x + 0.5 * case.separation) / case.width) ** 2)
    right = np.exp(-((x - 0.5 * case.separation) / case.width) ** 2)
    compact = left + right
    localization = compact / (compact + case.density_floor_fraction)
    inward = (left - right) / np.maximum(compact, 1.0e-300)
    density_floor = case.density_floor_fraction * case.peak_density
    density = density_floor + case.peak_density * compact
    velocity = grid.zeros((3,))
    velocity[0] = case.collision_speed * inward * localization
    fluid = FluidPrimitive(
        density,
        np.full(grid.shape, case.internal_energy),
        velocity,
        grid.zeros(),
    )
    carrier_velocity = velocity.copy()
    carrier_velocity[0] += case.counterflow * inward * localization
    speed = np.abs(carrier_velocity[0])
    if float(np.max(speed)) >= 0.8:
        carrier_velocity[0] *= 0.8 / float(np.max(speed))
    carrier = CarrierPrimitive(
        np.maximum(case.target_fraction * density, 1.0e-10),
        carrier_velocity,
    )
    return fluid, carrier


def build_initial_state(
    case: CollisionCase,
    *,
    points: int,
    length: float,
    derivatives: str,
    ctt_tolerance: float,
    ctt_iterations: int,
    picard_iterations: int,
) -> tuple[PeriodicGrid, Theory33ProductionSolver, object, object, dict[str, float | bool]]:
    grid = PeriodicGrid((points,), (length,), method=derivatives)
    master = Theory33MasterParameters(
        carrier_K=case.carrier_K,
        carrier_sound_speed=case.carrier_sound_speed,
        target_fraction=case.target_fraction,
        chemical_susceptibility=case.chemical_susceptibility,
        relaxation_time=case.relaxation_time,
        maximum_relative_lorentz_factor=2.0,
    )
    gate = constitutive_gate(case, master)
    vector = Theory3Parameters(
        target_speed=max(
            case.carrier_sound_speed,
            np.sqrt(master.gamma_ad - 1.0) + 0.05,
        ),
        target_relaxation_time=case.relaxation_time,
        target_strength=0.0,
        gamma_ad=master.gamma_ad,
    )
    ccz4 = CCZ4Parameters(kappa_gravity=8.0 * np.pi)
    production = Theory3ProductionParameters(recovery_iterations=24)
    solver = Theory33ProductionSolver(
        grid, master, vector, ccz4, production=production
    )
    fluid, carrier = collision_primitives(grid, case)
    state = solver.initialize(
        solver.ccz4.flat_state(), fluid, carrier=carrier
    )
    ctt_parameters = CTTParameters(
        kappa=ccz4.kappa_gravity,
        maximum_iterations=ctt_iterations,
        tolerance=ctt_tolerance,
        relaxation=0.3,
        report_interval=20,
    )
    ctt_solver = PeriodicCMCInitialDataSolver(
        grid, ctt_parameters, expanding=False
    )
    psi = vector_potential = None
    result = None
    for _ in range(picard_iterations):
        recovery = solver.recover(state)
        result = ctt_solver.solve(
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            psi,
            vector_potential,
        )
        if not result.report.converged:
            raise FloatingPointError(
                f"periodic CMC solve did not converge: {result.report}"
            )
        psi = result.conformal_factor
        vector_potential = result.vector_potential
        state = solver.initialize(
            solver.ccz4.from_adm(result.geometry), fluid, carrier=carrier
        )
    assert result is not None
    return grid, solver, state, result.report, gate


def _sample(solver: Theory33ProductionSolver, state: object) -> dict[str, object]:
    recovery = solver.recover(state)
    diagnostic = solver.diagnostics(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    center = tuple(size // 2 for size in solver.grid.shape)
    return {
        **diagnostic,
        "peak_baryon_density": float(np.max(recovery.fluid.baryon_density)),
        "central_baryon_density": float(recovery.fluid.baryon_density[center]),
        "peak_carrier_density": float(np.max(recovery.carrier.number_density)),
        "minimum_lapse": float(np.min(state.geometry.lapse)),
        "minimum_physical_metric": float(np.min(h[0, 0])),
        "maximum_physical_metric": float(np.max(h[0, 0])),
    }


def run_case(
    case: CollisionCase,
    destination: Path,
    *,
    points: int,
    length: float,
    derivatives: str,
    final_time: float,
    dt: float,
    samples: int,
    ctt_tolerance: float,
    ctt_iterations: int,
    picard_iterations: int,
) -> dict[str, object]:
    grid, solver, state, ctt_report, gate = build_initial_state(
        case,
        points=points,
        length=length,
        derivatives=derivatives,
        ctt_tolerance=ctt_tolerance,
        ctt_iterations=ctt_iterations,
        picard_iterations=picard_iterations,
    )
    steps = max(1, int(np.ceil(final_time / dt)))
    dt = final_time / steps
    sample_steps = set(np.linspace(0, steps, samples, dtype=int).tolist())
    records: list[dict[str, object]] = []
    fields: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "time",
            "baryon_density",
            "carrier_density",
            "baryon_velocity",
            "carrier_velocity",
            "relative_lorentz_factor",
            "specific_entropy",
            "lapse",
            "conformal_factor",
            "theta",
            "longitudinal_A",
            "longitudinal_B",
        )
    }
    cumulative_projection_entropy = 0.0
    cumulative_drag_entropy = 0.0
    cumulative_step_entropy = 0.0
    minimum_projection_change = 0.0
    initial_baryons = solver.diagnostics(state)["baryon_mass"]
    initial_carriers = solver.diagnostics(state)["carrier_number"]

    def record() -> None:
        recovery = solver.recover(state)
        records.append(_sample(solver, state))
        fields["time"].append(float(state.time))
        fields["baryon_density"].append(recovery.fluid.baryon_density.copy())
        fields["carrier_density"].append(recovery.carrier.number_density.copy())
        fields["baryon_velocity"].append(recovery.fluid.velocity[0].copy())
        fields["carrier_velocity"].append(recovery.carrier.velocity[0].copy())
        fields["relative_lorentz_factor"].append(
            recovery.master.relative_lorentz_factor.copy()
        )
        fields["specific_entropy"].append(recovery.entropy_per_baryon.copy())
        fields["lapse"].append(state.geometry.lapse.copy())
        fields["conformal_factor"].append(
            state.geometry.conformal_factor.copy()
        )
        fields["theta"].append(state.geometry.theta.copy())
        fields["longitudinal_A"].append(state.longitudinal_A.copy())
        fields["longitudinal_B"].append(state.longitudinal_B.copy())

    for step in range(steps + 1):
        if step in sample_steps:
            record()
        if step == steps:
            break
        state = solver.step(state, dt)
        cumulative_projection_entropy += solver.last_projection_entropy_change
        cumulative_drag_entropy += solver.last_drag_entropy_change
        cumulative_step_entropy += solver.last_step_entropy_change
        minimum_projection_change = min(
            minimum_projection_change,
            solver.last_projection_minimum_change,
        )

    final = solver.diagnostics(state)
    initial_peak = float(records[0]["peak_baryon_density"])
    maximum_peak = max(float(row["peak_baryon_density"]) for row in records)
    minimum_lapse = min(float(row["minimum_lapse"]) for row in records)
    maximum_gamma = max(
        float(row["maximum_relative_lorentz_factor"]) for row in records
    )
    final_gamma = float(records[-1]["maximum_relative_lorentz_factor"])
    compression = maximum_peak / initial_peak
    if minimum_lapse < 0.6 or compression > 2.5:
        outcome = "collapse_candidate"
    elif case.counterflow == 0.0 and compression > 1.03:
        outcome = "gravity_compression"
    elif compression > 1.03 and final_gamma - 1.0 < 0.4 * max(maximum_gamma - 1.0, 1.0e-12):
        outcome = "drag_locked_compression"
    elif compression > 1.03:
        outcome = "counterflow_compression"
    else:
        outcome = "mild_deformation"
    summary = {
        "case": asdict(case),
        "name": case.name,
        "outcome": outcome,
        "points": points,
        "length": length,
        "dt": dt,
        "steps": steps,
        "final_time": final_time,
        "ctt": asdict(ctt_report),
        "constitutive_gate": gate,
        "initial_peak_baryon_density": initial_peak,
        "maximum_peak_baryon_density": maximum_peak,
        "peak_compression_ratio": compression,
        "minimum_lapse": minimum_lapse,
        "maximum_relative_lorentz_factor": maximum_gamma,
        "final_relative_lorentz_factor": final_gamma,
        "cumulative_projection_entropy_change": cumulative_projection_entropy,
        "minimum_projection_entropy_density_change": minimum_projection_change,
        "cumulative_drag_entropy_change": cumulative_drag_entropy,
        "cumulative_step_entropy_change": cumulative_step_entropy,
        "relative_baryon_error": (float(final["baryon_mass"]) - float(initial_baryons))
        / max(abs(float(initial_baryons)), 1.0e-300),
        "relative_carrier_error": (float(final["carrier_number"]) - float(initial_carriers))
        / max(abs(float(initial_carriers)), 1.0e-300),
        "final_hamiltonian_l2": final["hamiltonian_l2"],
        "final_momentum_l2": final["momentum_l2"],
        "final_recovery_residual": final["recovery_maximum_residual"],
        "recovery_failures": final["recovery_failures"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    arrays["x"] = grid.coordinates()[0] - 0.5 * length
    np.savez_compressed(destination / "trajectory.npz", **arrays)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_theory33_production_state(
        destination / "final_checkpoint.npz",
        state,
        metadata={"experiment": "theory33_counterflow_collision", **summary},
    )
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/theory33_collision"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--fraction-sweep",
        action="store_true",
        help="scan causal carrier fractions 0.05 through 2.00 at both drag times",
    )
    parser.add_argument(
        "--eos-loading-sweep",
        action="store_true",
        help="scan common fractions 0.10, 0.20, 0.50, 1.00, and 2.00 at the selected carrier sound speed",
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse completed case summaries"
    )
    parser.add_argument("--peak-density", type=float, default=0.025)
    parser.add_argument("--counterflow", type=float, default=0.28)
    parser.add_argument("--relaxation-time", type=float, default=0.30)
    parser.add_argument("--collision-speed", type=float, default=0.18)
    parser.add_argument("--target-fraction", type=float, default=0.05)
    parser.add_argument("--carrier-sound-speed", type=float, default=0.9)
    parser.add_argument("--carrier-K", type=float, default=0.2)
    parser.add_argument("--points", type=int, default=16)
    parser.add_argument("--length", type=float, default=8.0)
    parser.add_argument("--derivatives", choices=("fd2", "fd4"), default="fd2")
    parser.add_argument("--final-time", type=float, default=0.04)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--ctt-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--ctt-iterations", type=int, default=3000)
    parser.add_argument("--picard-iterations", type=int, default=1)
    args = parser.parse_args()
    eos_cases = tuple(
        CollisionCase(
            0.025,
            0.28,
            relaxation,
            target_fraction=fraction,
            carrier_sound_speed=args.carrier_sound_speed,
            carrier_K=args.carrier_K,
        )
        for fraction in (0.10, 0.20, 0.50, 1.00, 2.00)
        for relaxation in (0.03, 0.30)
    )
    cases = (
        eos_cases
        if args.eos_loading_sweep
        else FRACTION_SWEEP
        if args.fraction_sweep
        else DEFAULT_SWEEP
        if args.sweep
        else (
            CollisionCase(
                args.peak_density,
                args.counterflow,
                args.relaxation_time,
                collision_speed=args.collision_speed,
                target_fraction=args.target_fraction,
                carrier_sound_speed=args.carrier_sound_speed,
                carrier_K=args.carrier_K,
            ),
        )
    )
    summaries = []
    for case in cases:
        summary_path = args.output / case.name / "summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if case.counterflow == 0.0 and summary["outcome"] == "counterflow_compression":
                summary["outcome"] = "gravity_compression"
                summary_path.write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
            summaries.append(summary)
            print(json.dumps({"resumed": case.name}, sort_keys=True))
            continue
        summaries.append(
            run_case(
                case,
                args.output / case.name,
                points=args.points,
                length=args.length,
                derivatives=args.derivatives,
                final_time=args.final_time,
                dt=args.dt,
                samples=args.samples,
                ctt_tolerance=args.ctt_tolerance,
                ctt_iterations=args.ctt_iterations,
                picard_iterations=args.picard_iterations,
            )
        )
    if len(summaries) > 1:
        champion = max(
            summaries,
            key=lambda row: (
                float(row["peak_compression_ratio"])
                + 2.0 * (1.0 - float(row["minimum_lapse"]))
                + 0.5 * (float(row["maximum_relative_lorentz_factor"]) - 1.0)
            ),
        )
        counterflow_cases = [
            row for row in summaries if float(row["case"]["counterflow"]) > 0.0
        ]
        drag_locked = min(
            counterflow_cases,
            key=lambda row: float(row["final_relative_lorentz_factor"]),
        )
        penetrating = max(
            counterflow_cases,
            key=lambda row: float(row["final_relative_lorentz_factor"]),
        )
        scan = {
            "cases": summaries,
            "champion": champion["name"],
            "paired_champions": {
                "drag_locked": drag_locked["name"],
                "penetrating": penetrating["name"],
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "scan_summary.json").write_text(
            json.dumps(scan, indent=2), encoding="utf-8"
        )
        print(json.dumps({"champion": champion["name"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
