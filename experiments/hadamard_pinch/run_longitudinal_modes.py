"""Evolve the longitudinal acoustic doublet and deterministic broadband data."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

from run_transverse_tangent import build_solver, material_symbol, physical_fields, velocity_from_proper
from validate_pinch import D, INTERNAL, N
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.theory33 import CarrierPrimitive


def acoustic_basis(master, phi, velocity, speed):
    def symbol(v):
        return material_symbol(master, N, D, INTERNAL, np.zeros(3), np.array([v, 0., 0.]), phi)

    matrix = symbol(velocity)
    left, _, right = np.linalg.svd(matrix - speed*np.eye(9))
    right = right[-2:].T
    left = left[:, -2:]
    derivative = (symbol(velocity + 1e-5) - symbol(velocity - 1e-5))/(2e-5)
    split = np.linalg.solve(left.T @ right, left.T @ derivative @ right)
    slopes, rotation = eig(split)
    if np.max(abs(slopes.imag)) > 1e-8:
        raise ArithmeticError('Velocity perturbation failed to resolve real acoustic branches')
    modes = right @ rotation[:, np.argsort(slopes.real)].real
    modes /= np.linalg.norm(modes, axis=0)
    modes *= np.sign(modes[np.argmax(abs(modes), axis=0), np.arange(2)])
    return modes, np.linalg.pinv(modes), np.linalg.norm((matrix-speed*np.eye(9)) @ modes, axis=0)


def perturbation(label, modes, phase):
    seed = np.zeros((11, len(phase)))
    if label.startswith('broadband'):
        rng = np.random.default_rng(20260724)
        for harmonic in (1, 2, 3):
            weights = rng.normal(size=11) + 1j*rng.normal(size=11)
            seed += np.real(weights[:, None]*np.exp(1j*harmonic*phase))/harmonic
    else:
        coefficients = {'mode_1': (1, 0), 'mode_2': (0, 1), 'in_phase': (1, 1), 'opposite_phase': (1, -1)}
        seed[:9] = (modes @ coefficients[label])[:, None]*np.cos(phase)
    seed /= np.sqrt(2*np.mean(np.sum(seed**2, axis=0)))
    if label == 'broadband_material':
        seed[9:] = 0.0
    if label == 'broadband_scalar':
        seed[:9] = 0.0
    return seed


def run(case, pinch, duration):
    label, points, k, amplitude, dt = case
    phi, velocity, speed = pinch
    grid = PeriodicGrid((points,), (2*np.pi/k,))
    solver = build_solver(grid)
    modes, dual, residual = acoustic_basis(solver.master, *pinch)
    phase = 2*np.pi*np.arange(points)/points
    base = np.zeros((11, 1))
    base[0, 0], base[4, 0] = np.log(N), np.log(D)
    base[5, 0] = velocity/np.sqrt(1-velocity**2)
    base[8, 0] = solver.master.specific_entropy(N, INTERNAL)
    base[9, 0] = phi
    seed = perturbation(label, modes, phase)
    values = base + amplitude*seed
    gamma = solver.master_parameters.gamma_ad
    n, d = np.exp(values[0]), np.exp(values[4])
    eps = np.exp((gamma-1)*values[8])*n**(gamma-1)/(gamma-1)
    fluid = FluidPrimitive(n, eps, velocity_from_proper(values[1:4]), grid.zeros())
    carrier = CarrierPrimitive(d, velocity_from_proper(values[5:8]))
    state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier,
                              varphi=values[9], scalar_momentum=values[10])
    dt = float(dt)
    rows, modal, spectra, failure = [], [], [], None
    for step in range(round(duration/dt)+1):
        if step:
            try:
                state = solver.step(state, dt)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                failure = {'attempted_time': step*dt, 'reason': str(exc)}
                break
        fields, recovered = physical_fields(solver, state)
        planes = np.array([np.exp(-1j*harmonic*phase) for harmonic in (1, 2, 3)])
        coefficients = 2*(fields[:11] @ planes.T)/points/amplitude
        pair = dual @ coefficients[:9, 0]
        signal = coefficients[:9, 0]
        rows.append([state.time, np.linalg.norm(signal), np.linalg.norm(signal-modes @ pair),
                     np.linalg.norm(coefficients[:, 1:]), recovered.report.maximum_residual])
        modal.append(pair)
        spectra.append(coefficients)
    rows, modal = np.asarray(rows), np.asarray(modal)
    print(case, 't', rows[-1, 0], 'pair', abs(modal[-1]), failure or 'completed', flush=True)
    return rows, modal, np.asarray(spectra), {
        'seed': label, 'points': points, 'k': k, 'length': grid.lengths[0], 'amplitude': amplitude,
        'dt': dt, 'light_cfl': dt/grid.spacing[0], 'failure': failure, 'mode_residuals': residual.tolist(),
        'initial_modal_amplitudes': abs(modal[0]).tolist(), 'final_modal_amplitudes': abs(modal[-1]).tolist(),
        'endtime': float(rows[-1, 0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('pairs', 'refine', 'small', 'extended', 'broadband_weak',
                                         'broadband_material', 'broadband_scalar', 'broadband_fine'))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    pinch = json.loads((folder/'hadamard_pinch_summary.json').read_text())['pinch_phi_v_speed']
    suites = {
        'pairs': [(label, 16, 128, 1e-5, .00025) for label in ('mode_1', 'mode_2', 'in_phase', 'opposite_phase', 'broadband')],
        'refine': [('mode_1', 32, 128, 1e-5, .000125), ('mode_2', 32, 128, 1e-5, .000125),
                   ('in_phase', 16, 64, 1e-3, .00025), ('opposite_phase', 16, 256, 1e-3, .000125),
                   ('broadband', 32, 128, 1e-5, .000125)],
        'small': [('mode_1', 16, 128, amplitude, .00025) for amplitude in (1e-3, 1e-7, 1e-9)],
        'extended': [(label, 32, 128, 1e-3, .00025)
                     for label in ('mode_1', 'mode_2', 'in_phase', 'opposite_phase', 'broadband')],
        'broadband_weak': [('broadband', 32, 128, 1e-5, .00025)],
        'broadband_material': [('broadband_material', 32, 128, 1e-5, .00025)],
        'broadband_scalar': [('broadband_scalar', 32, 128, 1e-5, .00025)],
        'broadband_fine': [('broadband', 40, 128, 1e-5, .00025)],
    }
    duration = .004 if args.suite in ('pairs', 'refine', 'small') else .016
    results = [run(case, pinch, duration) for case in suites[args.suite]]
    columns = ('time', 'fundamental_material_norm', 'outside_initial_acoustic_subspace',
               'higher_harmonic_norm', 'recovery_residual')
    arrays = {f'{prefix}_{i}': result[j] for i, result in enumerate(results)
              for j, prefix in enumerate(('diagnostics', 'modal', 'fourier'))}
    np.savez_compressed(folder/f'hadamard_pinch_longitudinal_{args.suite}.npz', columns=columns,
                        metadata_json=json.dumps([row[-1] for row in results]), **arrays)


if __name__ == '__main__':
    main()
