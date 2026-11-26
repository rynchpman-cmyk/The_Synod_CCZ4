"""Command-line experiments for the causal-repair package."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .adm import ADMParameters, ADMSolver
from .coupled import CoupledADMProcaSolver
from .curved_proca import CurvedProcaSystem
from .grid import CartesianGrid, PeriodicGrid
from .io import save_checkpoint
from .proca import ProcaParameters, ProcaSolver, ProcaState
from .relaxation import RelaxationParameters, RelaxationSolver
from .fixed_theory2 import Theory2FixedSolver
from .initial_data import TwoLobeParameters, rotating_two_lobe_data
from .initial_data import constraint_solved_rotating_two_lobe_data
from .initial_constraints import CTTParameters
from .theory2 import Theory2Parameters
from .ccz4 import CCZ4Parameters
from .grhd import GRHDParameters
from .matter import FluidPrimitive
from .production import ProductionParameters, ProductionSolver, save_production_state
from .theory3 import Theory3Parameters
from .production3 import (
    Theory3ProductionParameters,
    Theory3ProductionSolver,
    save_theory3_production_state,
)
from .theory33 import CarrierPrimitive, Theory33MasterParameters
from .production33 import (
    Theory33ProductionSolver,
    save_theory33_production_state,
)
from .adm import ADMState, flat_metric
from .tov import PolytropicEOS, solve_tov


def _grid(args: argparse.Namespace) -> PeriodicGrid:
    if getattr(args, "isolated", False):
        method = "fd4" if args.derivatives == "spectral" else args.derivatives
        return CartesianGrid((args.points,), (args.length,), method=method)
    return PeriodicGrid((args.points,), (args.length,), args.derivatives)


def _emit(payload: dict[str, object]) -> None:
    # Strict JSON has no NaN/Infinity literals. Undefined diagnostics become null.
    clean = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in payload.items()
    }
    print(json.dumps(clean, sort_keys=True, allow_nan=False))


def run_proca(args: argparse.Namespace) -> int:
    grid = _grid(args)
    parameters = ProcaParameters(args.Z, args.mass)
    solver = ProcaSolver(grid, parameters)
    x = grid.coordinates()[0]
    a = grid.zeros((3,))
    pi = grid.zeros((3,))
    a[args.component] = args.amplitude * np.exp(
        -0.5 * (((x - 0.5 * args.length + 0.5 * args.length) % args.length - 0.5 * args.length) / args.width) ** 2
    )
    state = ProcaState(a, pi)
    dt = args.dt or solver.recommended_dt(args.cfl)
    initial_energy = solver.diagnostics(state).energy
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    previous = state
    for step in range(1, steps + 1):
        previous, state = state, solver.step(state, dt)
        if step % args.report_every == 0 or step == steps:
            diag = asdict(solver.diagnostics(state, previous))
            diag["relative_energy_error"] = (
                (diag["energy"] - initial_energy) / initial_energy
                if initial_energy != 0.0
                else float("nan")
            )
            _emit(diag)
    if args.output:
        save_checkpoint(
            args.output,
            metadata={"branch": "proca", "parameters": asdict(parameters)},
            a=state.a,
            pi=state.pi,
            time=state.time,
            phi=solver.phi(state),
        )
    return 0


def run_relaxation(args: argparse.Namespace) -> int:
    grid = _grid(args)
    x = grid.coordinates()[0]
    target_field = grid.zeros((3,))
    target_field[0] = args.target_amplitude * np.cos(2.0 * np.pi * args.mode * x / args.length)
    parameters = RelaxationParameters(args.mobility, args.susceptibility, args.temperature)
    solver = RelaxationSolver(grid, lambda _time: target_field, parameters, seed=args.seed)
    state = solver.zero_state()
    dt = args.dt or min(args.cfl * min(grid.spacing), 0.1 * parameters.tau)
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(1, steps + 1):
        state = solver.step(state, dt, stochastic=not args.deterministic)
        if step % args.report_every == 0 or step == steps:
            _emit(asdict(solver.diagnostics(state)))
    if args.output:
        save_checkpoint(
            args.output,
            metadata={"branch": "relaxation", "parameters": asdict(parameters)},
            xi=state.xi,
            target=target_field,
            time=state.time,
        )
    return 0


def run_coupled(args: argparse.Namespace) -> int:
    grid = _grid(args)
    proca_parameters = ProcaParameters(args.Z, args.mass)
    adm = ADMSolver(grid, ADMParameters(args.kappa, args.cosmological_constant))
    curved = CurvedProcaSystem(grid, proca_parameters)
    solver = CoupledADMProcaSolver(adm, curved)
    state = solver.flat_state()
    state.a[0] = args.amplitude
    rho, _, _ = curved.stress_energy(state.h, state.a, state.pi)
    expansion = np.sqrt(np.maximum((args.kappa * rho + args.cosmological_constant) / 3.0, 0.0))
    for i in range(3):
        state.K[i, i] = -expansion
    _emit({"time": state.time, **solver.constraint_norms(state)})
    dt = args.dt or args.cfl * min(grid.spacing)
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(1, steps + 1):
        state = solver.step(state, dt)
        if step % args.report_every == 0 or step == steps:
            _emit({"time": state.time, **solver.constraint_norms(state)})
    if args.output:
        save_checkpoint(
            args.output,
            metadata={
                "branch": "coupled_adm_proca",
                "warning": "experimental raw-ADM free evolution",
                "proca_parameters": asdict(proca_parameters),
            },
            h=state.h,
            K=state.K,
            a=state.a,
            pi=state.pi,
            time=state.time,
        )
    return 0


def run_theory2(args: argparse.Namespace) -> int:
    grid = _grid(args)
    parameters = Theory2Parameters(
        Z=args.Z,
        mass=args.mass,
        coupling=args.coupling,
        tau_star=args.tau_star,
        ell_sigma=args.ell_sigma,
        gamma_ad=args.gamma_ad,
    )
    benchmark = TwoLobeParameters(
        separation=args.separation,
        width=args.width,
        peak_density=args.peak_density,
        density_floor=args.density_floor,
        matter_speed=args.matter_speed,
        beta_target=args.beta_target,
    )
    initial = rotating_two_lobe_data(grid, parameters, benchmark)
    solver = Theory2FixedSolver(grid, initial.fluid, parameters, h=initial.geometry.h)
    state = initial.repair
    dt = args.dt or min(
        args.cfl * min(grid.spacing),
        args.cfl / parameters.mass,
        args.cfl * parameters.tau_star,
    )
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(steps + 1):
        if step % args.report_every == 0 or step == steps:
            diagnostics = solver.diagnostics(state)
            _, margin = solver.minkowski_current(state)
            diagnostics["minimum_causal_margin"] = float(np.min(margin))
            _emit(diagnostics)
        if step < steps:
            state = solver.step(state, dt)
    if args.output:
        save_checkpoint(
            args.output,
            metadata={
                "branch": "theory2_fixed_background",
                "parameters": asdict(parameters),
                "gr_constraints_solved": initial.gravitational_constraints_solved,
            },
            a=state.a,
            pi=state.pi,
            astar=state.astar,
            phi=solver.phi(state),
            baryon_density=initial.fluid.baryon_density,
            internal_energy=initial.fluid.specific_internal_energy,
            velocity=initial.fluid.velocity,
            sigma=initial.fluid.sigma,
            time=state.time,
        )
    return 0


def run_production(args: argparse.Namespace) -> int:
    grid = _grid(args)
    theory = Theory2Parameters(
        Z=args.Z,
        mass=args.mass,
        coupling=args.coupling,
        tau_star=args.tau_star,
        gamma_ad=args.gamma_ad,
    )
    ccz4_parameters = CCZ4Parameters(
        kappa_gravity=args.kappa,
        kappa1=args.kappa1,
        kappa2=args.kappa2,
        kappa3=args.kappa3,
        shift_speed=args.shift_speed,
        eta_shift=args.eta_shift,
    )
    grhd_parameters = GRHDParameters(
        gamma_ad=args.gamma_ad,
        density_floor=args.density_floor,
        internal_energy_floor=args.internal_energy_floor,
    )
    production_parameters = ProductionParameters(
        proca_cleaning_damping=args.proca_cleaning_damping
    )
    solver = ProductionSolver(
        grid, theory, ccz4_parameters, grhd_parameters, production_parameters
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, args.density),
        np.full(grid.shape, args.internal_energy),
        grid.zeros((3,)),
        grid.zeros(),
    )
    astar = grid.zeros((3,))
    astar[1] = args.target_amplitude
    a = astar.copy()
    pi = grid.zeros((3,))

    # Homogeneous CMC data solve both GR constraints algebraically.
    provisional = solver.initialize(solver.ccz4.flat_state(), fluid, a, pi, astar)
    recovery = solver.recover(provisional)
    rho = recovery.total_stress.rho
    expansion = np.sqrt(
        np.maximum(
            (args.kappa * rho + ccz4_parameters.cosmological_constant) / 3.0,
            0.0,
        )
    )
    h = flat_metric(grid)
    K = grid.zeros((3, 3))
    for i in range(3):
        K[i, i] = -expansion
    geometry = solver.ccz4.from_adm(ADMState(h, K))
    state = solver.initialize(geometry, fluid, a, pi, astar)
    dt = args.dt or solver.recommended_dt(state, args.cfl)
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(steps + 1):
        if step % args.report_every == 0 or step == steps:
            _emit(solver.diagnostics(state))
        if step < steps:
            state = solver.step(state, dt)
    if args.output:
        save_production_state(
            args.output,
            state,
            metadata={
                "branch": "production_ccz4_grhd_theory2",
                "theory": asdict(theory),
                "ccz4": asdict(ccz4_parameters),
                "grhd": asdict(grhd_parameters),
                "production": asdict(production_parameters),
            },
        )
    return 0


def run_theory3_production(args: argparse.Namespace) -> int:
    grid = _grid(args)
    theory = Theory3Parameters(
        Z_A=args.Z_A,
        Z_B=args.Z_B,
        mass_A=args.mass_A,
        mass_B=args.mass_B,
        mixing_mass_squared=args.mixing_mass_squared,
        target_speed=args.target_speed,
        target_relaxation_time=args.target_relaxation_time,
        target_strength=args.target_strength,
        sound_speed_margin=args.sound_speed_margin,
        gamma_ad=args.gamma_ad,
        gamma_W=args.gamma_W,
        gamma_B=args.gamma_B,
    )
    ccz4_parameters = CCZ4Parameters(
        kappa_gravity=args.kappa,
        kappa1=args.kappa1,
        kappa2=args.kappa2,
        kappa3=args.kappa3,
        shift_speed=args.shift_speed,
        eta_shift=args.eta_shift,
    )
    grhd_parameters = GRHDParameters(
        gamma_ad=args.gamma_ad,
        density_floor=args.density_floor,
        internal_energy_floor=args.internal_energy_floor,
    )
    production_parameters = Theory3ProductionParameters(
        cleaning_damping=args.cleaning_damping
    )
    solver = Theory3ProductionSolver(
        grid, theory, ccz4_parameters, grhd_parameters, production_parameters
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, args.density),
        np.full(grid.shape, args.internal_energy),
        grid.zeros((3,)),
        grid.zeros(),
    )
    a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_A = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    a[1] = args.A_amplitude
    b[1] = args.B_amplitude
    pi_A[0] = -args.Z_A * args.electric_A
    pi_B[0] = -args.Z_B * args.electric_B
    target_charge = np.full(grid.shape, args.target_charge)

    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid=fluid,
        a=a,
        pi_A=pi_A,
        b=b,
        pi_B=pi_B,
        target_charge_eulerian=target_charge,
        reservoir_energy=args.reservoir_energy,
    )
    recovery = solver.recover(provisional)
    expansion = np.sqrt(
        np.maximum(
            (
                args.kappa * recovery.total_stress.rho
                + ccz4_parameters.cosmological_constant
            )
            / 3.0,
            0.0,
        )
    )
    h = flat_metric(grid)
    K = grid.zeros((3, 3))
    for i in range(3):
        K[i, i] = -expansion
    geometry = solver.ccz4.from_adm(ADMState(h, K))
    state = solver.initialize(
        geometry,
        fluid=fluid,
        a=a,
        pi_A=pi_A,
        b=b,
        pi_B=pi_B,
        target_charge_eulerian=target_charge,
        reservoir_energy=args.reservoir_energy,
    )
    dt = args.dt or solver.recommended_dt(state, args.cfl)
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(steps + 1):
        if step % args.report_every == 0 or step == steps:
            _emit(solver.diagnostics(state))
        if step < steps:
            state = solver.step(state, dt)
    if args.output:
        save_theory3_production_state(
            args.output,
            state,
            metadata={
                "branch": "production_ccz4_grhd_theory3_2",
                "theory": asdict(theory),
                "ccz4": asdict(ccz4_parameters),
                "grhd": asdict(grhd_parameters),
                "production": asdict(production_parameters),
            },
        )
    return 0


def run_theory33_production(args: argparse.Namespace) -> int:
    grid = _grid(args)
    master = Theory33MasterParameters(
        gamma_ad=args.gamma_ad,
        carrier_charge=args.carrier_charge,
        carrier_K=args.carrier_K,
        carrier_sound_speed=args.carrier_sound_speed,
        chemical_susceptibility=args.chemical_susceptibility,
        target_fraction=args.target_fraction,
        entrainment_linear=args.entrainment_linear,
        entrainment_quadratic=args.entrainment_quadratic,
        relaxation_time=args.target_relaxation_time,
        maximum_relative_lorentz_factor=args.maximum_relative_lorentz_factor,
    )
    legacy_speed_floor = np.sqrt(args.gamma_ad - 1.0) + 0.05
    vector = Theory3Parameters(
        Z_A=args.Z_A,
        Z_B=args.Z_B,
        mass_A=args.mass_A,
        mass_B=args.mass_B,
        mixing_mass_squared=args.mixing_mass_squared,
        target_speed=max(args.carrier_sound_speed, legacy_speed_floor),
        target_relaxation_time=args.target_relaxation_time,
        target_strength=0.0,
        sound_speed_margin=0.05,
        gamma_ad=args.gamma_ad,
        gamma_W=args.gamma_W,
        gamma_B=args.gamma_B,
    )
    ccz4_parameters = CCZ4Parameters(
        kappa_gravity=args.kappa,
        kappa1=args.kappa1,
        kappa2=args.kappa2,
        kappa3=args.kappa3,
        shift_speed=args.shift_speed,
        eta_shift=args.eta_shift,
    )
    grhd_parameters = GRHDParameters(
        gamma_ad=args.gamma_ad,
        density_floor=args.density_floor,
        internal_energy_floor=args.internal_energy_floor,
    )
    production_parameters = Theory3ProductionParameters(
        cleaning_damping=args.cleaning_damping
    )
    solver = Theory33ProductionSolver(
        grid,
        master,
        vector,
        ccz4_parameters,
        grhd_parameters,
        production_parameters,
    )
    fluid = FluidPrimitive(
        np.full(grid.shape, args.density),
        np.full(grid.shape, args.internal_energy),
        grid.zeros((3,)),
        grid.zeros(),
    )
    carrier_density = (
        args.carrier_density
        if args.carrier_density is not None
        else args.target_fraction * args.density
    )
    carrier_velocity = grid.zeros((3,))
    carrier_velocity[0] = args.carrier_velocity
    carrier = CarrierPrimitive(
        np.full(grid.shape, max(carrier_density, master.carrier_floor)),
        carrier_velocity,
    )
    a = grid.zeros((3,))
    b = grid.zeros((3,))
    pi_A = grid.zeros((3,))
    pi_B = grid.zeros((3,))
    a[1] = args.A_amplitude
    b[1] = args.B_amplitude
    pi_A[0] = -args.Z_A * args.electric_A
    pi_B[0] = -args.Z_B * args.electric_B

    provisional = solver.initialize(
        solver.ccz4.flat_state(),
        fluid,
        carrier=carrier,
        a=a,
        pi_A=pi_A,
        b=b,
        pi_B=pi_B,
    )
    recovery = solver.recover(provisional)
    expansion = np.sqrt(
        np.maximum(
            (
                args.kappa * recovery.total_stress.rho
                + ccz4_parameters.cosmological_constant
            )
            / 3.0,
            0.0,
        )
    )
    h = flat_metric(grid)
    K = grid.zeros((3, 3))
    for i in range(3):
        K[i, i] = -expansion
    geometry = solver.ccz4.from_adm(ADMState(h, K))
    state = solver.initialize(
        geometry,
        fluid,
        carrier=carrier,
        a=a,
        pi_A=pi_A,
        b=b,
        pi_B=pi_B,
    )
    dt = args.dt or solver.recommended_dt(state, args.cfl)
    steps = int(np.ceil(args.final_time / dt))
    dt = args.final_time / steps
    for step in range(steps + 1):
        if step % args.report_every == 0 or step == steps:
            _emit(solver.diagnostics(state))
        if step < steps:
            state = solver.step(state, dt)
    if args.output:
        save_theory33_production_state(
            args.output,
            state,
            metadata={
                "branch": "production_ccz4_multifluid_theory3_3",
                "master": asdict(master),
                "vector": asdict(vector),
                "ccz4": asdict(ccz4_parameters),
                "grhd": asdict(grhd_parameters),
                "production": asdict(production_parameters),
            },
        )
    return 0


def run_tov(args: argparse.Namespace) -> int:
    solution = solve_tov(
        args.central_density,
        PolytropicEOS(args.polytropic_K, args.gamma),
        dr=args.dr,
        maximum_radius=args.maximum_radius,
    )
    _emit(
        {
            "central_density": args.central_density,
            "surface_radius": solution.surface_radius,
            "gravitational_mass": solution.gravitational_mass,
            "baryonic_mass": solution.baryonic_mass,
            "compactness": solution.gravitational_mass / solution.surface_radius,
        }
    )
    if args.output:
        save_checkpoint(
            args.output,
            metadata={"branch": "tov", "K": args.polytropic_K, "gamma": args.gamma},
            radius=solution.radius,
            density=solution.density,
            pressure=solution.pressure,
            energy_density=solution.energy_density,
            enclosed_mass=solution.enclosed_mass,
            lapse=solution.lapse,
            radial_metric=solution.radial_metric,
        )
    return 0


def run_initial_data(args: argparse.Namespace) -> int:
    grid = CartesianGrid(
        (args.points,) * 3, (args.length,) * 3, method=args.derivatives
    )
    theory = Theory2Parameters(
        Z=args.Z,
        mass=args.mass,
        coupling=args.coupling,
        tau_star=args.tau_star,
        gamma_ad=args.gamma_ad,
    )
    benchmark = TwoLobeParameters(
        separation=args.separation,
        width=args.width,
        peak_density=args.peak_density,
        density_floor=args.density_floor,
        matter_speed=args.matter_speed,
        beta_target=args.beta_target,
    )
    ctt = CTTParameters(
        kappa=args.kappa,
        maximum_iterations=args.maximum_iterations,
        tolerance=args.tolerance,
    )
    data = constraint_solved_rotating_two_lobe_data(
        grid, theory, benchmark, ctt, args.picard_iterations
    )
    report = data.constraint_report
    assert report is not None
    _emit(asdict(report))
    if args.output:
        save_checkpoint(
            args.output,
            metadata={
                "branch": "isolated_ctt_theory2_two_lobe",
                "theory": asdict(theory),
                "benchmark": asdict(benchmark),
                "ctt": asdict(ctt),
                "constraint_report": asdict(report),
            },
            h=data.geometry.h,
            K=data.geometry.K,
            a=data.repair.a,
            pi=data.repair.pi,
            astar=data.repair.astar,
            phi=data.phi,
            baryon_density=data.fluid.baryon_density,
            internal_energy=data.fluid.specific_internal_energy,
            velocity=data.fluid.velocity,
            sigma=data.fluid.sigma,
        )
    return 0 if report.converged else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causal-repair")
    subcommands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--points", type=int, default=64)
        command.add_argument("--length", type=float, default=2.0 * np.pi)
        command.add_argument("--derivatives", choices=("spectral", "fd2", "fd4"), default="spectral")
        command.add_argument("--final-time", type=float, default=2.0)
        command.add_argument("--dt", type=float)
        command.add_argument("--cfl", type=float, default=0.1)
        command.add_argument("--report-every", type=int, default=20)
        command.add_argument("--output", type=Path)

    proca = subcommands.add_parser("proca", help="evolve the conservative Proca branch")
    common(proca)
    proca.add_argument("--Z", type=float, default=1.0)
    proca.add_argument("--mass", type=float, default=1.0)
    proca.add_argument("--amplitude", type=float, default=0.1)
    proca.add_argument("--width", type=float, default=0.5)
    proca.add_argument("--component", type=int, choices=(0, 1, 2), default=1)
    proca.set_defaults(handler=run_proca)

    relaxation = subcommands.add_parser("relax", help="evolve deterministic or noisy relaxation")
    common(relaxation)
    relaxation.add_argument("--mobility", type=float, default=1.0)
    relaxation.add_argument("--susceptibility", type=float, default=1.0)
    relaxation.add_argument("--temperature", type=float, default=0.0)
    relaxation.add_argument("--target-amplitude", type=float, default=1.0)
    relaxation.add_argument("--mode", type=int, default=1)
    relaxation.add_argument("--seed", type=int, default=1234)
    relaxation.add_argument("--deterministic", action="store_true")
    relaxation.set_defaults(handler=run_relaxation)

    coupled = subcommands.add_parser("coupled", help="experimental homogeneous ADM--Proca run")
    common(coupled)
    coupled.set_defaults(final_time=0.1, cfl=0.02)
    coupled.add_argument("--Z", type=float, default=1.0)
    coupled.add_argument("--mass", type=float, default=1.0)
    coupled.add_argument("--amplitude", type=float, default=1.0e-3)
    coupled.add_argument("--kappa", type=float, default=8.0 * np.pi)
    coupled.add_argument("--cosmological-constant", type=float, default=0.0)
    coupled.set_defaults(handler=run_coupled)

    theory2 = subcommands.add_parser(
        "theory2", help="run the fixed-background Theory 2.0 two-lobe benchmark"
    )
    common(theory2)
    theory2.set_defaults(final_time=0.5, cfl=0.1)
    theory2.add_argument("--Z", type=float, default=1.0)
    theory2.add_argument("--mass", type=float, default=0.5)
    theory2.add_argument("--coupling", type=float, default=0.1)
    theory2.add_argument("--tau-star", type=float, default=1.0)
    theory2.add_argument("--ell-sigma", type=float, default=1.0)
    theory2.add_argument("--gamma-ad", type=float, default=5.0 / 3.0)
    theory2.add_argument("--separation", type=float, default=2.0)
    theory2.add_argument("--width", type=float, default=0.8)
    theory2.add_argument("--peak-density", type=float, default=1.0e-3)
    theory2.add_argument("--density-floor", type=float, default=1.0e-11)
    theory2.add_argument("--matter-speed", type=float, default=0.25)
    theory2.add_argument("--beta-target", type=float, default=1.76)
    theory2.set_defaults(handler=run_theory2)

    production = subcommands.add_parser(
        "production", help="run the coupled CCZ4 + GRHD + Theory 2.0 system"
    )
    common(production)
    production.set_defaults(points=16, final_time=0.01, cfl=0.05)
    production.add_argument("--Z", type=float, default=1.0)
    production.add_argument("--mass", type=float, default=0.5)
    production.add_argument("--coupling", type=float, default=0.1)
    production.add_argument("--tau-star", type=float, default=1.0)
    production.add_argument("--gamma-ad", type=float, default=5.0 / 3.0)
    production.add_argument("--kappa", type=float, default=8.0 * np.pi)
    production.add_argument("--kappa1", type=float, default=0.1)
    production.add_argument("--kappa2", type=float, default=0.0)
    production.add_argument("--kappa3", type=float, default=1.0)
    production.add_argument("--shift-speed", type=float, default=1.0)
    production.add_argument("--eta-shift", type=float, default=1.0)
    production.add_argument("--density", type=float, default=1.0e-4)
    production.add_argument("--internal-energy", type=float, default=0.05)
    production.add_argument("--density-floor", type=float, default=1.0e-12)
    production.add_argument("--internal-energy-floor", type=float, default=1.0e-10)
    production.add_argument("--target-amplitude", type=float, default=1.0e-3)
    production.add_argument("--proca-cleaning-damping", type=float, default=1.0)
    production.add_argument("--isolated", action="store_true")
    production.set_defaults(handler=run_production)

    production3 = subcommands.add_parser(
        "production3",
        help="run the overhauled CCZ4 + entropy-GRHD + Theory 3.2 system",
    )
    common(production3)
    production3.set_defaults(points=16, final_time=0.01, cfl=0.05)
    production3.add_argument("--Z-A", dest="Z_A", type=float, default=1.0)
    production3.add_argument("--Z-B", dest="Z_B", type=float, default=1.0)
    production3.add_argument("--mass-A", dest="mass_A", type=float, default=0.5)
    production3.add_argument("--mass-B", dest="mass_B", type=float, default=0.5)
    production3.add_argument("--mixing-mass-squared", type=float, default=0.25)
    production3.add_argument("--target-speed", type=float, default=0.95)
    production3.add_argument("--target-relaxation-time", type=float, default=1.0)
    production3.add_argument("--target-strength", type=float, default=0.0)
    production3.add_argument("--target-charge", type=float, default=1.0e-4)
    production3.add_argument("--sound-speed-margin", type=float, default=0.05)
    production3.add_argument("--gamma-W", dest="gamma_W", type=float, default=0.0)
    production3.add_argument("--gamma-B", dest="gamma_B", type=float, default=0.0)
    production3.add_argument("--gamma-ad", type=float, default=5.0 / 3.0)
    production3.add_argument("--kappa", type=float, default=8.0 * np.pi)
    production3.add_argument("--kappa1", type=float, default=0.1)
    production3.add_argument("--kappa2", type=float, default=0.0)
    production3.add_argument("--kappa3", type=float, default=1.0)
    production3.add_argument("--shift-speed", type=float, default=1.0)
    production3.add_argument("--eta-shift", type=float, default=1.0)
    production3.add_argument("--density", type=float, default=1.0e-3)
    production3.add_argument("--internal-energy", type=float, default=0.1)
    production3.add_argument("--reservoir-energy", type=float, default=1.0e-3)
    production3.add_argument("--density-floor", type=float, default=1.0e-12)
    production3.add_argument("--internal-energy-floor", type=float, default=1.0e-10)
    production3.add_argument("--A-amplitude", dest="A_amplitude", type=float, default=0.0)
    production3.add_argument("--B-amplitude", dest="B_amplitude", type=float, default=0.0)
    production3.add_argument("--electric-A", dest="electric_A", type=float, default=0.0)
    production3.add_argument("--electric-B", dest="electric_B", type=float, default=0.0)
    production3.add_argument("--cleaning-damping", type=float, default=1.0)
    production3.add_argument("--isolated", action="store_true")
    production3.set_defaults(handler=run_theory3_production)

    production33 = subcommands.add_parser(
        "production33",
        help="run the CCZ4 + two-current master-action Theory 3.3 system",
    )
    common(production33)
    production33.set_defaults(points=16, final_time=0.001, cfl=0.025)
    production33.add_argument("--Z-A", dest="Z_A", type=float, default=1.0)
    production33.add_argument("--Z-B", dest="Z_B", type=float, default=1.0)
    production33.add_argument("--mass-A", dest="mass_A", type=float, default=0.5)
    production33.add_argument("--mass-B", dest="mass_B", type=float, default=0.5)
    production33.add_argument("--mixing-mass-squared", type=float, default=0.25)
    production33.add_argument("--carrier-charge", type=float, default=1.0)
    production33.add_argument("--carrier-K", type=float, default=0.2)
    production33.add_argument("--carrier-sound-speed", type=float, default=0.9)
    production33.add_argument("--chemical-susceptibility", type=float, default=10.0)
    production33.add_argument("--target-fraction", type=float, default=0.05)
    production33.add_argument("--carrier-density", type=float)
    production33.add_argument("--carrier-velocity", type=float, default=0.0)
    production33.add_argument("--entrainment-linear", type=float, default=0.0)
    production33.add_argument("--entrainment-quadratic", type=float, default=0.0)
    production33.add_argument("--maximum-relative-lorentz-factor", type=float, default=2.0)
    production33.add_argument("--target-relaxation-time", type=float, default=1.0)
    production33.add_argument("--gamma-W", dest="gamma_W", type=float, default=0.0)
    production33.add_argument("--gamma-B", dest="gamma_B", type=float, default=0.0)
    production33.add_argument("--gamma-ad", type=float, default=5.0 / 3.0)
    production33.add_argument("--kappa", type=float, default=8.0 * np.pi)
    production33.add_argument("--kappa1", type=float, default=0.1)
    production33.add_argument("--kappa2", type=float, default=0.0)
    production33.add_argument("--kappa3", type=float, default=1.0)
    production33.add_argument("--shift-speed", type=float, default=1.0)
    production33.add_argument("--eta-shift", type=float, default=1.0)
    production33.add_argument("--density", type=float, default=1.0e-3)
    production33.add_argument("--internal-energy", type=float, default=0.1)
    production33.add_argument("--density-floor", type=float, default=1.0e-12)
    production33.add_argument("--internal-energy-floor", type=float, default=1.0e-10)
    production33.add_argument("--A-amplitude", dest="A_amplitude", type=float, default=0.0)
    production33.add_argument("--B-amplitude", dest="B_amplitude", type=float, default=0.0)
    production33.add_argument("--electric-A", dest="electric_A", type=float, default=0.0)
    production33.add_argument("--electric-B", dest="electric_B", type=float, default=0.0)
    production33.add_argument("--cleaning-damping", type=float, default=1.0)
    production33.add_argument("--isolated", action="store_true")
    production33.set_defaults(handler=run_theory33_production)

    tov = subcommands.add_parser("tov", help="generate a spherical TOV equilibrium")
    tov.add_argument("--central-density", type=float, default=1.28e-3)
    tov.add_argument("--polytropic-K", type=float, default=100.0)
    tov.add_argument("--gamma", type=float, default=2.0)
    tov.add_argument("--dr", type=float, default=0.01)
    tov.add_argument("--maximum-radius", type=float, default=100.0)
    tov.add_argument("--output", type=Path)
    tov.set_defaults(final_time=1.0, report_every=1, handler=run_tov)

    initial = subcommands.add_parser(
        "initial-data", help="solve isolated CTT data for the Theory 2.0 two-lobe benchmark"
    )
    initial.set_defaults(final_time=1.0, report_every=1)
    initial.add_argument("--points", type=int, default=16)
    initial.add_argument("--length", type=float, default=12.0)
    initial.add_argument("--derivatives", choices=("fd2", "fd4"), default="fd2")
    initial.add_argument("--Z", type=float, default=1.0)
    initial.add_argument("--mass", type=float, default=0.5)
    initial.add_argument("--coupling", type=float, default=0.1)
    initial.add_argument("--tau-star", type=float, default=1.0)
    initial.add_argument("--gamma-ad", type=float, default=5.0 / 3.0)
    initial.add_argument("--kappa", type=float, default=8.0 * np.pi)
    initial.add_argument("--separation", type=float, default=2.0)
    initial.add_argument("--width", type=float, default=0.8)
    initial.add_argument("--peak-density", type=float, default=1.0e-3)
    initial.add_argument("--density-floor", type=float, default=1.0e-11)
    initial.add_argument("--matter-speed", type=float, default=0.05)
    initial.add_argument("--beta-target", type=float, default=0.02)
    initial.add_argument("--maximum-iterations", type=int, default=8000)
    initial.add_argument("--tolerance", type=float, default=2e-7)
    initial.add_argument("--picard-iterations", type=int, default=3)
    initial.add_argument("--output", type=Path)
    initial.set_defaults(handler=run_initial_data)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.final_time <= 0.0 or args.report_every <= 0:
        raise SystemExit("final-time and report-every must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
