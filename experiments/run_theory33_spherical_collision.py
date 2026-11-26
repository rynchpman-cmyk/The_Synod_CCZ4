"""Run an isolated 3-D head-on Theory 3.3 two-current sphere collision."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from causal_repair.adm import ADMSolver, ADMParameters, flat_metric
from causal_repair.ccz4 import CCZ4Parameters
from causal_repair.grhd import GRHDParameters
from causal_repair.grid import CartesianGrid
from causal_repair.initial_constraints import CTTInitialDataSolver, CTTParameters
from causal_repair.matter import FluidPrimitive
from causal_repair.nr_diagnostics import HawkingMassDiagnostic
from causal_repair.production3 import Theory3ProductionParameters
from causal_repair.production33 import (
    Theory33ProductionSolver,
    load_theory33_production_state,
    save_theory33_production_state,
)
from causal_repair.theory3 import Theory3Parameters
from causal_repair.theory33 import CarrierPrimitive, Theory33MasterParameters


@dataclass(frozen=True)
class SphericalCollisionCase:
    peak_density: float = 0.005
    density_floor: float = 1.0e-7
    internal_energy: float = 0.25
    separation: float = 1.6
    width: float = 1.0
    baryon_speed: float = 0.18
    carrier_speed: float = 0.46
    target_fraction: float = 2.0
    carrier_K: float = 0.4
    carrier_sound_speed: float = 0.7
    relaxation_time: float = 0.03
    maximum_relative_lorentz_factor: float = 4.0

    @property
    def name(self) -> str:
        return (
            f"sphere_tau{self.relaxation_time:.3f}_f{self.target_fraction:.2f}_"
            f"k{self.carrier_K:.2f}"
        ).replace(".", "p")


def primitives(
    grid: CartesianGrid, case: SphericalCollisionCase
) -> tuple[FluidPrimitive, CarrierPrimitive, np.ndarray]:
    x, y, z = grid.coordinates()
    half = 0.5 * case.separation
    left = np.exp(-((x + half) ** 2 + y**2 + z**2) / case.width**2)
    right = np.exp(-((x - half) ** 2 + y**2 + z**2) / case.width**2)
    profile = left + right
    direction = (left - right) / np.maximum(profile, 1.0e-300)
    localization = profile / (profile + 0.01)
    density = case.density_floor + case.peak_density * profile
    velocity = grid.zeros((3,))
    velocity[0] = case.baryon_speed * direction * localization
    carrier_velocity = grid.zeros((3,))
    carrier_velocity[0] = case.carrier_speed * direction * localization
    fluid = FluidPrimitive(
        density,
        np.full(grid.shape, case.internal_energy),
        velocity,
        grid.zeros(),
    )
    carrier = CarrierPrimitive(
        np.maximum(case.target_fraction * density, 1.0e-10),
        carrier_velocity,
    )
    radius = np.sqrt(x * x + y * y + z * z)
    return fluid, carrier, radius


def constitutive_gate(
    solver: Theory33ProductionSolver, case: SphericalCollisionCase
) -> dict[str, float | bool]:
    audits = []
    for density in (
        case.density_floor,
        case.peak_density,
        2.0 * case.peak_density,
    ):
        carrier_density = max(
            case.target_fraction * density,
            solver.master_parameters.carrier_floor,
        )
        for baryon_velocity, carrier_velocity in (
            (0.0, 0.0),
            (case.baryon_speed, case.carrier_speed),
            (-case.baryon_speed, -case.carrier_speed),
        ):
            audits.append(
                solver.master.characteristic_audit(
                    density,
                    case.internal_energy,
                    baryon_velocity,
                    carrier_density,
                    carrier_velocity,
                )
            )
    # The evolution can generate much larger counterflow than is present on
    # the initial slice.  Audit symmetric counter-moving rapidities, several
    # off-target carrier fractions, and the full configured domain cap.
    gamma_samples = np.unique(
        np.concatenate(
            (
                np.array([1.0, 1.5, 2.0, 3.0]),
                np.array([case.maximum_relative_lorentz_factor]),
            )
        )
    )
    for density in (case.density_floor, case.peak_density, 2.0 * case.peak_density):
        for density_ratio in (0.5, 1.0, case.target_fraction, 3.0, 4.0):
            carrier_density = max(
                density_ratio * density, solver.master_parameters.carrier_floor
            )
            for relative_gamma in gamma_samples:
                rapidity = float(np.arccosh(relative_gamma))
                baryon_velocity = -float(np.tanh(0.5 * rapidity))
                carrier_velocity = -baryon_velocity
                audits.append(
                    solver.master.characteristic_audit(
                        density,
                        case.internal_energy,
                        baryon_velocity,
                        carrier_density,
                        carrier_velocity,
                    )
                )
    return {
        "passed": bool(
            all(audit.strongly_hyperbolic and audit.causal for audit in audits)
        ),
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


def build_initial_state(
    grid: CartesianGrid,
    case: SphericalCollisionCase,
    *,
    ctt_tolerance: float,
    ctt_iterations: int,
    picard_iterations: int,
) -> tuple[Theory33ProductionSolver, object, object, FluidPrimitive, CarrierPrimitive, np.ndarray, dict[str, float | bool]]:
    solver = make_solver(grid, case)
    gate = constitutive_gate(solver, case)
    if not bool(gate["passed"]):
        raise RuntimeError(f"constitutive gate failed: {gate}")
    fluid, carrier, radius = primitives(grid, case)
    elliptic = CTTInitialDataSolver(
        grid,
        CTTParameters(
            maximum_iterations=ctt_iterations,
            tolerance=ctt_tolerance,
            report_interval=20,
            relaxation=0.35,
        ),
    )
    h = flat_metric(grid)
    conformal_factor = None
    vector_potential = None
    zero_vector = grid.zeros((3,))
    zero_scalar = grid.zeros()
    result = None
    for _ in range(picard_iterations):
        material = solver.master.evaluate(h, fluid, carrier)
        vector_stress = solver.system.vector_stress(
            h,
            zero_vector,
            zero_vector,
            zero_vector,
            zero_vector,
            zero_scalar,
            zero_scalar,
            material.source_charge_eulerian,
        )
        result = elliptic.solve(
            material.stress.rho + vector_stress.rho,
            material.stress.momentum + vector_stress.momentum,
            conformal_factor,
            vector_potential,
        )
        h = result.geometry.h
        conformal_factor = result.conformal_factor
        vector_potential = result.vector_potential
    assert result is not None
    if not result.report.converged:
        raise RuntimeError(f"isolated CTT solve failed: {result.report}")
    state = solver.initialize(
        solver.ccz4.from_adm(result.geometry),
        fluid,
        carrier=carrier,
    )
    return solver, state, result, fluid, carrier, radius, gate


def make_solver(
    grid: CartesianGrid, case: SphericalCollisionCase
) -> Theory33ProductionSolver:
    """Construct the evolution system independently of initial-data generation."""
    master = Theory33MasterParameters(
        target_fraction=case.target_fraction,
        carrier_K=case.carrier_K,
        carrier_sound_speed=case.carrier_sound_speed,
        relaxation_time=case.relaxation_time,
        maximum_relative_lorentz_factor=case.maximum_relative_lorentz_factor,
    )
    vector = Theory3Parameters(
        target_speed=max(case.carrier_sound_speed, 0.9),
        target_relaxation_time=case.relaxation_time,
        target_strength=0.0,
        gamma_ad=master.gamma_ad,
    )
    solver = Theory33ProductionSolver(
        grid,
        master,
        vector,
        CCZ4Parameters(radiative_boundaries=True),
        GRHDParameters(
            gamma_ad=master.gamma_ad,
            density_floor=case.density_floor,
            internal_energy_floor=1.0e-8,
        ),
        Theory3ProductionParameters(atmosphere_gravitates=False),
    )
    return solver


def core_constraints(
    solver: Theory33ProductionSolver, state: object
) -> tuple[float, float]:
    recovery = solver.recover(state)
    geometry = solver.ccz4.to_adm(state.geometry)
    hamiltonian, momentum = ADMSolver(
        solver.grid, ADMParameters(solver.ccz4.parameters.kappa_gravity)
    ).constraints(
        geometry,
        (
            recovery.total_stress.rho,
            recovery.total_stress.momentum,
            recovery.total_stress.stress,
        ),
    )
    core = (slice(2, -2),) * 3
    return (
        float(np.sqrt(np.mean(hamiltonian[core] ** 2))),
        float(np.sqrt(np.mean(np.sum(momentum[(slice(None),) + core] ** 2, axis=0)))),
    )


def _same_float(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1.0e-12, atol=1.0e-14))


def _load_restart_bundle(
    restart_from: Path,
    *,
    case: SphericalCollisionCase,
    points: int,
    length: float,
    dt: float,
    extraction_radii: tuple[float, ...],
) -> tuple[object, dict[str, object], dict[str, np.ndarray], Path]:
    """Load and cross-check the checkpoint, summary, and append history."""
    restart_directory = restart_from if restart_from.is_dir() else restart_from.parent
    checkpoint_path = (
        restart_from
        if restart_from.is_file()
        else restart_directory / "final_checkpoint.npz"
    )
    summary_path = restart_directory / "summary.json"
    trajectory_path = restart_directory / "trajectory.npz"
    missing = [
        str(path)
        for path in (checkpoint_path, summary_path, trajectory_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"restart bundle is incomplete: {missing}")

    state, metadata = load_theory33_production_state(checkpoint_path)
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: archive[name].copy() for name in archive.files}

    if tuple(state.matter.D.shape) != (points,) * 3:
        raise ValueError(
            f"restart grid {state.matter.D.shape} does not match requested {(points,) * 3}"
        )
    if int(previous["points"]) != points or not _same_float(
        float(previous["length"]), length
    ):
        raise ValueError("restart grid metadata does not match the requested grid")
    if not _same_float(float(previous["dt"]), dt):
        raise ValueError("restart timestep does not match the existing trajectory")
    if not np.allclose(
        np.asarray(previous["extraction_radii"], dtype=float),
        extraction_radii,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("restart extraction radii do not match")
    requested_case = asdict(case)
    stored_case = previous.get("case", {})
    if set(stored_case) != set(requested_case):
        raise ValueError("restart case fields do not match the current specification")
    for name, value in requested_case.items():
        stored = stored_case[name]
        if isinstance(value, float):
            matches = _same_float(float(stored), value)
        else:
            matches = stored == value
        if not matches:
            raise ValueError(
                f"restart case mismatch for {name}: stored={stored!r}, requested={value!r}"
            )

    required = {
        "time",
        "extraction_radii",
        "hawking_mass",
        "areal_radius",
        "outgoing_expansion",
        "ingoing_expansion",
        "peak_baryon_density",
        "peak_carrier_density",
        "relative_lorentz_factor",
        "collision_scale",
        "inner_baryon_mass",
        "inner_carrier_number",
        "baryon_density_slice",
        "carrier_density_slice",
        "conformal_factor_slice",
    }
    if missing_keys := required.difference(trajectory):
        raise ValueError(f"restart trajectory is missing arrays: {sorted(missing_keys)}")
    times = np.asarray(trajectory["time"], dtype=float)
    if times.ndim != 1 or len(times) < 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError("restart trajectory time must be a nonempty, increasing vector")
    for name in required.difference({"extraction_radii"}):
        if trajectory[name].shape[0] != len(times):
            raise ValueError(f"restart trajectory array {name!r} has the wrong length")
    for name in (
        "boundary_flux_instrumented",
        "carrier_boundary_number_outflow_rate",
        "cumulative_carrier_boundary_number_outflow",
        "carrier_number_change_since_boundary_instrumentation",
        "carrier_number_balance_residual",
    ):
        if name in trajectory and trajectory[name].shape[0] != len(times):
            raise ValueError(f"restart trajectory array {name!r} has the wrong length")
    if not np.allclose(
        np.asarray(trajectory["extraction_radii"], dtype=float),
        extraction_radii,
        rtol=1.0e-12,
        atol=1.0e-14,
    ):
        raise ValueError("restart trajectory extraction radii do not match")
    join_time = float(times[-1])
    if not _same_float(float(state.time), join_time):
        raise ValueError("checkpoint time does not match the trajectory endpoint")
    if not _same_float(float(previous["final_time"]), join_time):
        raise ValueError("summary time does not match the trajectory endpoint")
    if "final_time" in metadata and not _same_float(
        float(metadata["final_time"]), join_time
    ):
        raise ValueError("checkpoint metadata time does not match its state")
    if len(times) > 1 and not np.allclose(
        np.diff(times), dt, rtol=1.0e-11, atol=1.0e-13
    ):
        raise ValueError("restart trajectory does not use the requested fixed timestep")
    return state, previous, trajectory, restart_directory


def _records_from_trajectory(trajectory: dict[str, np.ndarray]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    sample_count = len(trajectory["time"])
    optional = {
        "boundary_flux_instrumented": np.zeros(sample_count, dtype=bool),
        "carrier_boundary_number_outflow_rate": np.zeros(sample_count),
        "cumulative_carrier_boundary_number_outflow": np.zeros(sample_count),
        "carrier_number_change_since_boundary_instrumentation": np.zeros(sample_count),
        "carrier_number_balance_residual": np.zeros(sample_count),
    }
    optional.update(
        {name: trajectory[name] for name in optional if name in trajectory}
    )
    for index, time in enumerate(trajectory["time"]):
        records.append(
            {
                "time": float(time),
                "hawking_mass": trajectory["hawking_mass"][index].tolist(),
                "areal_radius": trajectory["areal_radius"][index].tolist(),
                "mean_outgoing_expansion": trajectory["outgoing_expansion"][index].tolist(),
                "mean_ingoing_expansion": trajectory["ingoing_expansion"][index].tolist(),
                "peak_baryon_density": float(trajectory["peak_baryon_density"][index]),
                "peak_carrier_density": float(trajectory["peak_carrier_density"][index]),
                "maximum_relative_lorentz_factor": float(
                    trajectory["relative_lorentz_factor"][index]
                ),
                "collision_scale": float(trajectory["collision_scale"][index]),
                "inner_baryon_mass": float(trajectory["inner_baryon_mass"][index]),
                "inner_carrier_number": float(
                    trajectory["inner_carrier_number"][index]
                ),
                "boundary_flux_instrumented": bool(
                    optional["boundary_flux_instrumented"][index]
                ),
                "carrier_boundary_number_outflow_rate": float(
                    optional["carrier_boundary_number_outflow_rate"][index]
                ),
                "cumulative_carrier_boundary_number_outflow": float(
                    optional["cumulative_carrier_boundary_number_outflow"][index]
                ),
                "carrier_number_change_since_boundary_instrumentation": float(
                    optional[
                        "carrier_number_change_since_boundary_instrumentation"
                    ][index]
                ),
                "carrier_number_balance_residual": float(
                    optional["carrier_number_balance_residual"][index]
                ),
            }
        )
    return records


def run_case(
    case: SphericalCollisionCase,
    destination: Path,
    *,
    points: int,
    length: float,
    final_time: float,
    dt: float,
    extraction_radii: tuple[float, ...],
    ctt_tolerance: float,
    ctt_iterations: int,
    picard_iterations: int,
    restart_from: Path | None = None,
) -> dict[str, object]:
    grid = CartesianGrid((points,) * 3, (length,) * 3, method="fd2")
    previous: dict[str, object] | None = None
    previous_trajectory: dict[str, np.ndarray] | None = None
    if restart_from is None:
        solver, state, ctt, _, _, coordinate_radius, gate = build_initial_state(
            grid,
            case,
            ctt_tolerance=ctt_tolerance,
            ctt_iterations=ctt_iterations,
            picard_iterations=picard_iterations,
        )
        ctt_report = asdict(ctt.report)
    else:
        state, previous, previous_trajectory, restart_directory = _load_restart_bundle(
            restart_from,
            case=case,
            points=points,
            length=length,
            dt=dt,
            extraction_radii=extraction_radii,
        )
        if destination.resolve() != restart_directory.resolve():
            raise ValueError(
                "restart append is in-place: --output must name the restart directory"
            )
        solver = make_solver(grid, case)
        gate = constitutive_gate(solver, case)
        if not bool(gate["passed"]):
            raise RuntimeError(f"constitutive gate failed: {gate}")
        solver.recover(state)
        x, y, z = grid.coordinates()
        coordinate_radius = np.sqrt(x * x + y * y + z * z)
        ctt_report = dict(previous["ctt"])

    diagnostic = HawkingMassDiagnostic(grid, n_theta=10, n_phi=20)
    join_time = float(state.time)
    if restart_from is None:
        added_steps = max(1, int(np.ceil(final_time / dt)))
        dt = final_time / added_steps
    else:
        remaining = final_time - join_time
        if remaining <= 0.0:
            raise ValueError(
                f"restart final time {final_time} must exceed checkpoint time {join_time}"
            )
        added_steps = int(round(remaining / dt))
        if added_steps < 1 or not _same_float(added_steps * dt, remaining):
            raise ValueError(
                "restart final time must lie on the existing fixed-timestep lattice"
            )

    current = solver.diagnostics(state)
    if previous is None:
        initial_baryons = float(current["baryon_mass"])
        initial_carriers = float(current["carrier_number"])
        initial_core_constraints = core_constraints(solver, state)
        records: list[dict[str, object]] = []
        density_slices: list[np.ndarray] = []
        carrier_slices: list[np.ndarray] = []
        conformal_slices: list[np.ndarray] = []
        cumulative_drag_entropy = 0.0
        cumulative_step_entropy = 0.0
        previous_steps = 0
        restart_count = 0
        restart_join_times: list[float] = []
        boundary_flux_instrumentation_time = float(state.time)
        carrier_number_at_boundary_instrumentation = float(
            current["carrier_number"]
        )
        cumulative_carrier_boundary_outflow = 0.0
        carrier_number_balance_residual = 0.0
        newly_instrumented_boundary_flux = True
    else:
        assert previous_trajectory is not None
        baryon_change = float(previous["relative_baryon_change"])
        carrier_change = float(previous["relative_carrier_change"])
        initial_baryons = float(
            previous.get(
                "initial_baryon_mass",
                float(current["baryon_mass"]) / (1.0 + baryon_change),
            )
        )
        initial_carriers = float(
            previous.get(
                "initial_carrier_number",
                float(current["carrier_number"]) / (1.0 + carrier_change),
            )
        )
        initial_core_constraints = (
            float(previous["initial_core_hamiltonian_l2"]),
            float(previous["initial_core_momentum_l2"]),
        )
        records = _records_from_trajectory(previous_trajectory)
        density_slices = list(previous_trajectory["baryon_density_slice"])
        carrier_slices = list(previous_trajectory["carrier_density_slice"])
        conformal_slices = list(previous_trajectory["conformal_factor_slice"])
        cumulative_drag_entropy = float(previous["cumulative_drag_entropy_change"])
        cumulative_step_entropy = float(previous["cumulative_step_entropy_change"])
        previous_steps = int(previous["steps"])
        if previous_steps != len(records) - 1:
            raise ValueError("restart summary step count does not match its trajectory")
        restart_count = int(previous.get("restart_count", 0)) + 1
        restart_join_times = [
            *[float(value) for value in previous.get("restart_join_times", [])],
            join_time,
        ]
        if "boundary_flux_instrumentation_time" in previous:
            boundary_flux_instrumentation_time = float(
                previous["boundary_flux_instrumentation_time"]
            )
            carrier_number_at_boundary_instrumentation = float(
                previous["carrier_number_at_boundary_flux_instrumentation"]
            )
            cumulative_carrier_boundary_outflow = float(
                previous["cumulative_carrier_boundary_number_outflow"]
            )
            carrier_number_balance_residual = float(
                previous["carrier_number_balance_residual"]
            )
            newly_instrumented_boundary_flux = False
        else:
            boundary_flux_instrumentation_time = join_time
            carrier_number_at_boundary_instrumentation = float(
                current["carrier_number"]
            )
            cumulative_carrier_boundary_outflow = 0.0
            carrier_number_balance_residual = 0.0
            newly_instrumented_boundary_flux = True
    middle = points // 2

    def record() -> None:
        recovery = solver.recover(state)
        values = solver.diagnostics(state)
        geometry = solver.ccz4.to_adm(state.geometry)
        masses = [
            diagnostic.evaluate(geometry, radius) for radius in extraction_radii
        ]
        D = state.matter.D
        total_D = max(grid.integrate(D), 1.0e-300)
        collision_scale = float(grid.integrate(np.abs(grid.coordinates()[0]) * D) / total_D)
        inner_mask = coordinate_radius <= extraction_radii[0]
        inner_baryons = float(np.sum(state.matter.D[inner_mask]) * grid.cell_volume)
        inner_carriers = float(
            np.sum(
                state.target_charge[inner_mask]
                / solver.master_parameters.carrier_charge
            )
            * grid.cell_volume
        )
        records.append(
            {
                "time": float(state.time),
                "hawking_mass": [mass.mass for mass in masses],
                "areal_radius": [mass.areal_radius for mass in masses],
                "mean_outgoing_expansion": [
                    mass.mean_outgoing_expansion for mass in masses
                ],
                "mean_ingoing_expansion": [
                    mass.mean_ingoing_expansion for mass in masses
                ],
                "peak_baryon_density": float(
                    np.max(recovery.fluid.baryon_density)
                ),
                "peak_carrier_density": float(
                    np.max(recovery.carrier.number_density)
                ),
                "maximum_relative_lorentz_factor": float(
                    recovery.report.maximum_relative_lorentz_factor
                ),
                "collision_scale": collision_scale,
                "inner_baryon_mass": inner_baryons,
                "inner_carrier_number": inner_carriers,
                "boundary_flux_instrumented": True,
                "carrier_boundary_number_outflow_rate": float(
                    solver.carrier_boundary_number_outflow_rate(state, recovery)
                ),
                "cumulative_carrier_boundary_number_outflow": (
                    cumulative_carrier_boundary_outflow
                ),
                "carrier_number_change_since_boundary_instrumentation": (
                    float(values["carrier_number"])
                    - carrier_number_at_boundary_instrumentation
                ),
                "carrier_number_balance_residual": carrier_number_balance_residual,
                "hamiltonian_l2": float(values["hamiltonian_l2"]),
                "momentum_l2": float(values["momentum_l2"]),
                "recovery_failures": int(values["recovery_failures"]),
            }
        )
        density_slices.append(recovery.fluid.baryon_density[:, :, middle].copy())
        carrier_slices.append(recovery.carrier.number_density[:, :, middle].copy())
        conformal_slices.append(state.geometry.conformal_factor[:, :, middle].copy())

    if newly_instrumented_boundary_flux and records:
        records[-1]["boundary_flux_instrumented"] = True
        records[-1]["carrier_boundary_number_outflow_rate"] = float(
            solver.carrier_boundary_number_outflow_rate(state)
        )
        records[-1]["cumulative_carrier_boundary_number_outflow"] = 0.0
        records[-1]["carrier_number_change_since_boundary_instrumentation"] = 0.0
        records[-1]["carrier_number_balance_residual"] = 0.0
    if not records:
        record()
    for _ in range(added_steps):
        state = solver.step(state, dt)
        cumulative_drag_entropy += solver.last_drag_entropy_change
        cumulative_step_entropy += solver.last_step_entropy_change
        cumulative_carrier_boundary_outflow += (
            solver.last_carrier_boundary_number_outflow
        )
        current_carrier_number = (
            grid.integrate(state.target_charge)
            / solver.master_parameters.carrier_charge
        )
        carrier_number_balance_residual = (
            current_carrier_number
            - carrier_number_at_boundary_instrumentation
            + cumulative_carrier_boundary_outflow
        )
        record()

    final = solver.diagnostics(state)
    final_core_constraints = core_constraints(solver, state)
    mass_history = np.asarray([row["hawking_mass"] for row in records])
    summary = {
        "name": case.name,
        "case": asdict(case),
        "points": points,
        "length": length,
        "dt": dt,
        "steps": previous_steps + added_steps,
        "final_time": float(state.time),
        "restart_count": restart_count,
        "restart_join_times": restart_join_times,
        "extraction_radii": list(extraction_radii),
        "constitutive_gate": gate,
        "ctt": ctt_report,
        "initial_core_hamiltonian_l2": initial_core_constraints[0],
        "initial_core_momentum_l2": initial_core_constraints[1],
        "final_core_hamiltonian_l2": final_core_constraints[0],
        "final_core_momentum_l2": final_core_constraints[1],
        "initial_hawking_mass": mass_history[0].tolist(),
        "final_hawking_mass": mass_history[-1].tolist(),
        "maximum_hawking_mass": np.max(mass_history, axis=0).tolist(),
        "final_peak_baryon_density": records[-1]["peak_baryon_density"],
        "final_peak_carrier_density": records[-1]["peak_carrier_density"],
        "final_relative_lorentz_factor": records[-1][
            "maximum_relative_lorentz_factor"
        ],
        "initial_collision_scale": records[0]["collision_scale"],
        "final_collision_scale": records[-1]["collision_scale"],
        "final_inner_baryon_mass": records[-1]["inner_baryon_mass"],
        "final_inner_carrier_number": records[-1]["inner_carrier_number"],
        "cumulative_drag_entropy_change": cumulative_drag_entropy,
        "cumulative_step_entropy_change": cumulative_step_entropy,
        "boundary_flux_instrumentation_time": boundary_flux_instrumentation_time,
        "carrier_number_at_boundary_flux_instrumentation": (
            carrier_number_at_boundary_instrumentation
        ),
        "cumulative_carrier_boundary_number_outflow": (
            cumulative_carrier_boundary_outflow
        ),
        "carrier_number_change_since_boundary_flux_instrumentation": (
            float(final["carrier_number"])
            - carrier_number_at_boundary_instrumentation
        ),
        "carrier_number_balance_residual": carrier_number_balance_residual,
        "relative_carrier_number_balance_residual": (
            carrier_number_balance_residual
            / max(abs(carrier_number_at_boundary_instrumentation), 1.0e-300)
        ),
        "final_carrier_boundary_number_outflow_rate": records[-1][
            "carrier_boundary_number_outflow_rate"
        ],
        "initial_baryon_mass": initial_baryons,
        "initial_carrier_number": initial_carriers,
        "relative_baryon_change": (
            float(final["baryon_mass"]) - initial_baryons
        )
        / initial_baryons,
        "relative_carrier_change": (
            float(final["carrier_number"]) - initial_carriers
        )
        / initial_carriers,
        "recovery_failures": int(final["recovery_failures"]),
    }
    destination.mkdir(parents=True, exist_ok=True)
    pending_summary = destination / "summary.pending.json"
    pending_trajectory = destination / "trajectory.pending.npz"
    pending_checkpoint = destination / "final_checkpoint.pending.npz"
    pending_paths = (pending_summary, pending_trajectory, pending_checkpoint)
    try:
        for path in pending_paths:
            path.unlink(missing_ok=True)
        pending_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez_compressed(
            pending_trajectory,
            time=np.asarray([row["time"] for row in records]),
            extraction_radii=np.asarray(extraction_radii),
            hawking_mass=mass_history,
            areal_radius=np.asarray([row["areal_radius"] for row in records]),
            outgoing_expansion=np.asarray(
                [row["mean_outgoing_expansion"] for row in records]
            ),
            ingoing_expansion=np.asarray(
                [row["mean_ingoing_expansion"] for row in records]
            ),
            peak_baryon_density=np.asarray(
                [row["peak_baryon_density"] for row in records]
            ),
            peak_carrier_density=np.asarray(
                [row["peak_carrier_density"] for row in records]
            ),
            relative_lorentz_factor=np.asarray(
                [row["maximum_relative_lorentz_factor"] for row in records]
            ),
            collision_scale=np.asarray(
                [row["collision_scale"] for row in records]
            ),
            inner_baryon_mass=np.asarray(
                [row["inner_baryon_mass"] for row in records]
            ),
            inner_carrier_number=np.asarray(
                [row["inner_carrier_number"] for row in records]
            ),
            boundary_flux_instrumented=np.asarray(
                [row["boundary_flux_instrumented"] for row in records],
                dtype=bool,
            ),
            carrier_boundary_number_outflow_rate=np.asarray(
                [row["carrier_boundary_number_outflow_rate"] for row in records]
            ),
            cumulative_carrier_boundary_number_outflow=np.asarray(
                [
                    row["cumulative_carrier_boundary_number_outflow"]
                    for row in records
                ]
            ),
            carrier_number_change_since_boundary_instrumentation=np.asarray(
                [
                    row["carrier_number_change_since_boundary_instrumentation"]
                    for row in records
                ]
            ),
            carrier_number_balance_residual=np.asarray(
                [row["carrier_number_balance_residual"] for row in records]
            ),
            baryon_density_slice=np.asarray(density_slices),
            carrier_density_slice=np.asarray(carrier_slices),
            conformal_factor_slice=np.asarray(conformal_slices),
        )
        save_theory33_production_state(
            pending_checkpoint,
            state,
            metadata={
                "experiment": "isolated_spherical_two_current_collision",
                **summary,
            },
        )
        # Publish the summary last: readers can treat it as the bundle commit marker.
        pending_trajectory.replace(destination / "trajectory.npz")
        pending_checkpoint.replace(destination / "final_checkpoint.npz")
        pending_summary.replace(destination / "summary.json")
    finally:
        for path in pending_paths:
            path.unlink(missing_ok=True)
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/theory33_spheres"))
    parser.add_argument(
        "--restart-from",
        type=Path,
        help="append in place from this run directory or its final checkpoint",
    )
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--length", type=float, default=8.0)
    parser.add_argument("--final-time", type=float, default=0.016)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--extraction-radii", default="1.5,2.5,3.0")
    parser.add_argument("--relaxation-time", type=float, default=0.03)
    parser.add_argument("--peak-density", type=float, default=0.005)
    parser.add_argument("--separation", type=float, default=1.6)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--baryon-speed", type=float, default=0.18)
    parser.add_argument("--carrier-speed", type=float, default=0.46)
    parser.add_argument("--target-fraction", type=float, default=2.0)
    parser.add_argument("--carrier-K", type=float, default=0.4)
    parser.add_argument("--carrier-sound-speed", type=float, default=0.7)
    parser.add_argument(
        "--maximum-relative-lorentz-factor", type=float, default=4.0
    )
    parser.add_argument("--ctt-tolerance", type=float, default=3.0e-5)
    parser.add_argument("--ctt-iterations", type=int, default=5000)
    parser.add_argument("--picard-iterations", type=int, default=3)
    args = parser.parse_args()
    radii = tuple(float(value) for value in args.extraction_radii.split(","))
    if not radii or min(radii) <= 0.0 or max(radii) >= 0.5 * args.length:
        raise ValueError("extraction radii must lie inside the finite box")
    case = SphericalCollisionCase(
        peak_density=args.peak_density,
        separation=args.separation,
        width=args.width,
        baryon_speed=args.baryon_speed,
        carrier_speed=args.carrier_speed,
        target_fraction=args.target_fraction,
        carrier_K=args.carrier_K,
        carrier_sound_speed=args.carrier_sound_speed,
        relaxation_time=args.relaxation_time,
        maximum_relative_lorentz_factor=args.maximum_relative_lorentz_factor,
    )
    run_case(
        case,
        args.output,
        points=args.points,
        length=args.length,
        final_time=args.final_time,
        dt=args.dt,
        extraction_radii=radii,
        ctt_tolerance=args.ctt_tolerance,
        ctt_iterations=args.ctt_iterations,
        picard_iterations=args.picard_iterations,
        restart_from=args.restart_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
