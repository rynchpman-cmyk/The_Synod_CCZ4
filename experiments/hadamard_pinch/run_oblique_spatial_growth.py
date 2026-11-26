"""Replay optimized oblique Fourier perturbations using the native PDE step."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig, expm, svd

from audit_transit_corridor import BACKGROUND, EntropyPencil, normal
from run_homogeneous_departure import FixedMinkowski
from validate_pinch import D, INTERNAL, N, PARAMETERS
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory33 import CarrierPrimitive


def average_primitive(solver, state, rotation, carrier_background, phase):
    recovered = solver.recover(state)
    velocities = [rotation.T @ fluid.velocity for fluid in (recovered.fluid, recovered.carrier)]
    proper = [velocity/np.sqrt(1 - np.sum(velocity**2, axis=0)) for velocity in velocities]
    prim = np.vstack((np.log(recovered.fluid.baryon_density), proper[0],
                      np.log(recovered.carrier.number_density), proper[1], recovered.entropy_per_baryon))
    mean = np.mean(prim, axis=1)
    v = mean[5]/np.sqrt(1 + mean[5]**2)
    u = mean[1]/np.sqrt(1 + mean[1]**2)
    updated = (v - u)/(1 - u*v)
    epsilon = float(np.mean(recovered.fluid.specific_internal_energy))
    base = (np.exp(mean[0]), np.exp(mean[4]), epsilon)
    signal = 2*np.mean((prim - mean[:, None])*np.exp(-1j*phase), axis=1)
    return signal, base, updated, float(np.mean(state.scalar.varphi)), float(np.linalg.norm(mean[5:8] - carrier_background))


def run(phi, velocity, angle, points, wavenumber, amplitude, drag, dt, unstable_mode, duration):
    length = 2*np.pi/wavenumber
    grid = PeriodicGrid((points,), (length,))
    parameters = PARAMETERS if drag else replace(PARAMETERS, relaxation_time=float('inf'))
    solver = NullothProductionSolver(grid, parameters)
    solver.ccz4 = FixedMinkowski(grid, solver.ccz4.parameters)
    direction = normal(angle)
    rotation = np.array([direction, [-direction[1], direction[0], 0], [0, 0, 1]])
    pencil = EntropyPencil(solver.master, BACKGROUND, phi, velocity)
    time, space = pencil.material(direction)
    symbol = np.linalg.solve(time, space)
    roots, vectors = eig(symbol)
    growth_index = np.argmax(roots.imag)
    if unstable_mode:
        if roots[growth_index].imag <= 1e-8:
            raise ValueError('The requested Hadamard mode has no positive growth rate')
        optimal = vectors[:, growth_index]
    else:
        _, _, right = svd(expm(-1j*wavenumber*symbol*duration))
        optimal = right[0].conj()
    phase = 2*np.pi*np.arange(points)/points
    seed = amplitude*np.real(optimal[:, None]*np.exp(1j*phase))
    reference_proper = np.array([velocity/np.sqrt(1 - velocity**2), 0.0, 0.0])
    n = N*np.exp(seed[0])
    d = D*np.exp(seed[4])
    entropy = solver.master.specific_entropy(N, INTERNAL) + seed[8]
    pressure = np.exp((PARAMETERS.gamma_ad - 1)*entropy)*n**PARAMETERS.gamma_ad
    internal = pressure/((PARAMETERS.gamma_ad - 1)*n)
    q_n, q_d = rotation @ seed[1:4], rotation @ (reference_proper[:, None] + seed[5:8])
    u, v = (q/np.sqrt(1 + np.sum(q**2, axis=0)) for q in (q_n, q_d))
    state = solver.initialize(solver.ccz4.flat_state(), FluidPrimitive(n, internal, u, grid.zeros()),
                              carrier=CarrierPrimitive(d, v), varphi=phi)
    rows, failure = [], None
    evolving_signal = optimal.copy()
    preceding_symbol = symbol
    checkpoints = round(duration/dt)
    for index in range(checkpoints + 1):
        if index:
            try:
                state = solver.step(state, dt)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                failure = {'attempted_time': index*dt, 'reason': str(exc)}
                break
        signal, background, relative, scalar, drift = average_primitive(solver, state, rotation, reference_proper, phase)
        pencil = EntropyPencil(solver.master, background, scalar, relative)
        time, space = pencil.material(direction)
        current_symbol = np.linalg.solve(time, space)
        if index:
            evolving_signal = expm(-.5j*wavenumber*dt*(preceding_symbol + current_symbol)) @ evolving_signal
        preceding_symbol = current_symbol
        frozen_signal = expm(-1j*wavenumber*symbol*state.time) @ optimal
        rows.append([state.time, np.linalg.norm(signal)/amplitude, np.linalg.norm(frozen_signal),
                     np.linalg.norm(evolving_signal), scalar, relative, drift])
    print(points, wavenumber, len(rows)-1, rows[-1], failure or 'completed', flush=True)
    metadata = {'points': points, 'wavenumber': wavenumber, 'length': length, 'dt': dt, 'amplitude': amplitude,
                'cfl_light': dt/grid.spacing[0], 'direction_degrees': angle, 'native_drag': drag, 'failure': failure,
                'seed': 'growing eigenmode' if unstable_mode else 'optimal primitive norm perturbation',
                'initial_maximum_modal_growth_rate': float(wavenumber*roots[growth_index].imag), 'duration': duration}
    return np.asarray(rows), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-drag', action='store_true')
    parser.add_argument('--refine', choices=('time', 'space', 'amplitude', 'early-growth', 'long-transverse'))
    parser.add_argument('--hadamard', action='store_true', help='Evolve the transverse unstable reference mode')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    summary = json.loads((directory / 'hadamard_pinch_summary.json').read_text())
    phi, velocity, _ = summary['pinch_phi_v_speed']
    angle = summary['entropy_directional_audit']['classifications'][0]['collision']['angle_degrees']
    if args.hadamard:
        phi, velocity, angle = .15, .85, 90.0
    cases = {
        None: [(16, 128, 1e-5, .00025), (16, 256, 1e-5, .00025), (32, 256, 1e-5, .00025)],
        'time': [(32, 256, 1e-5, .000125)],
        'space': [(40, 256, 1e-5, .00025)],
        'amplitude': [(32, 256, 1e-3, .00025)],
        'early-growth': [(16, 256, 1e-4, .000125), (32, 256, 1e-4, .000125), (64, 256, 1e-4, .000125)],
        'long-transverse': [(64, 256, 1e-5, .000125)],
    }
    duration = .0025 if args.refine == 'early-growth' else .016
    results = [run(phi, velocity, angle, n, k, amplitude, args.native_drag, dt, args.hadamard, duration)
               for n, k, amplitude, dt in cases[args.refine]]
    columns = ['time', 'native_primitive_gain', 'frozen_principal_gain', 'evolving_principal_gain',
               'mean_phi', 'mean_v_rel', 'carrier_proper_velocity_drift']
    suffix = '_drag' if args.native_drag else ''
    if args.hadamard:
        suffix += '_hadamard'
    if args.refine:
        suffix += '_' + args.refine
    np.savez_compressed(directory / f'hadamard_pinch_spatial_growth{suffix}.npz', columns=columns,
                        metadata_json=json.dumps([row[1] for row in results]),
                        **{f'run_{i}': row[0] for i, row in enumerate(results)})


if __name__ == '__main__':
    main()
