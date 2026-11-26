"""Measure frozen-background growth near the oblique Nulloth collision."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals, expm, svd

from audit_transit_corridor import BACKGROUND, EntropyPencil, normal
from validate_pinch import PARAMETERS
from causal_repair.nulloth import NullothMasterFunction


def fractional_current_map(relative_velocity):
    """Map log density, proper velocity and entropy to fractional currents."""
    boost = 1/np.sqrt(1 - relative_velocity**2)
    transform = np.eye(9)
    transform[4, 4] = boost
    transform[4, 5] = relative_velocity
    transform[5, 4] = boost*relative_velocity
    transform[8, 8] = PARAMETERS.gamma_ad - 1
    return transform


def main():
    directory = Path(__file__).resolve().parent
    summary = json.loads((directory / 'hadamard_pinch_summary.json').read_text())
    phi, velocity, _ = summary['pinch_phi_v_speed']
    collision = summary['entropy_directional_audit']['classifications'][0]['collision']['angle_degrees']
    backgrounds = [
        ('pinch_longitudinal', phi, velocity, 0), ('oblique_collision', phi, velocity, collision),
        ('oblique_minus_1_degree', phi, velocity, collision - 1),
        ('oblique_plus_1_degree', phi, velocity, collision + 1),
        ('oblique_minus_0.01_degree', phi, velocity, collision - .01),
        ('oblique_plus_0.01_degree', phi, velocity, collision + .01),
        ('low_flow_control', phi, .2, collision),
    ]
    names = ['primitive_coordinates', 'fractional_current_and_entropy']
    wavenumbers = np.array([1., 8., 32., 128.])
    times = np.linspace(0, 2, 201)
    samples = np.empty((len(backgrounds), len(names), len(wavenumbers), len(times)))
    optimal = np.empty(samples.shape[:-1] + (9,), dtype=np.complex128)
    symbols, maximum_imaginary, max_initial_rank = [], [], []
    master = NullothMasterFunction(PARAMETERS)
    for state_id, (label, state_phi, relative, angle) in enumerate(backgrounds):
        pencil = EntropyPencil(master, BACKGROUND, state_phi, relative)
        time, space = pencil.material(normal(angle))
        symbol = np.linalg.solve(time, space)
        symbols.append(symbol)
        maximum_imaginary.append(float(max(abs(eigvals(symbol).imag))))
        max_initial_rank.append(master._clustered_eigenbasis(symbol)[1])
        for norm, transform in enumerate([np.eye(9), fractional_current_map(relative)]):
            scaled = transform @ np.linalg.solve(transform.T, symbol.T).T
            for mode, k in enumerate(wavenumbers):
                for i, sample_time in enumerate(times):
                    values = svd(expm(-1j*k*scaled*sample_time), compute_uv=False)
                    samples[state_id, norm, mode, i] = values[0]
                maximum = int(np.argmax(samples[state_id, norm, mode]))
                _, _, right = svd(expm(-1j*k*scaled*times[maximum]))
                optimal[state_id, norm, mode] = np.linalg.solve(transform, right[0].conj())
        print(label, maximum_imaginary[-1], samples[state_id, :, :, -1].tolist(), flush=True)
    np.savez_compressed(directory / 'hadamard_pinch_transient_growth.npz',
                        states=[row[0] for row in backgrounds], state_phi_v_angle=[row[1:] for row in backgrounds],
                        maximum_imaginary=maximum_imaginary, native_eigenbasis_rank=max_initial_rank,
                        norms=names, wavenumbers=wavenumbers, time=times, max_singular_value=samples,
                        optimal_initial_primitive_perturbations=optimal, symbols=symbols)


if __name__ == '__main__':
    main()
