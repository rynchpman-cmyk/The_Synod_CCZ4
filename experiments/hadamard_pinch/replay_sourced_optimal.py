"""Test the sourced variational prediction with a nonlinear Fourier replay."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_sourced_variation import K, SIZE, STEP, initialize_full, scale_current, screen_cell
from run_transverse_tangent import HarmonicProbeGrid, build_solver, physical_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('low_flow', 'transverse'))
    parser.add_argument('--amplitude', type=float, default=2e-4)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    path = folder/f'hadamard_pinch_sourced_{args.case}_5_0.0002.npz'
    with np.load(path) as model:
        times = model['times']
        model_id = list(model['labels']).index('evolving_full')
        initial = model['means'][0]
        optimal = model['optimal'][model_id, 0]
        expected = model['propagators'][:,model_id,:9] @ optimal
    samples = 8
    grid = HarmonicProbeGrid(samples, 64, 2*np.pi/K, 0)
    solver = build_solver(grid)
    phase = 2*np.pi*np.arange(samples)/samples
    seed = np.zeros((SIZE,samples))
    seed[:9] = np.real(optimal[:,None]*np.exp(1j*phase))
    state = initialize_full(solver,initial[:,None]+args.amplitude*seed)
    transform = scale_current(initial[:9],solver.master_parameters.gamma_ad)
    signal, failure, guards = [],None,[]
    pinch_speed = json.loads((folder/'hadamard_pinch_summary.json').read_text())['pinch_phi_v_speed'][2]
    for index,_ in enumerate(times):
        if index:
            try:
                state = solver.step(state, STEP)
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                failure = {'time': float(times[index]),'reason': str(exc)}
                break
        fields, recovery = physical_fields(solver,state)
        values = fields[:9]
        signal.append((values-np.roll(values,samples//2,axis=1)) @ np.exp(-1j*phase)/(samples*args.amplitude))
        if args.case == 'low_flow' and index % 16 == 0:
            guards.append({'time': float(times[index]), 'cells': [screen_cell(solver,fields,recovery,pinch_speed,i)
                                                                 for i in range(samples)]})
    signal = np.asarray(signal)
    expected = expected[:len(signal)]
    euclidean = np.linalg.norm(signal,axis=1)
    weighted = np.linalg.norm(signal @ transform.T,axis=1)/np.linalg.norm(transform @ optimal)
    output = {'case':args.case, 'amplitude':args.amplitude,'failure':failure,
              'last_time':float(times[len(signal)-1]),'measured_peak_gain':float(max(euclidean)),
              'model_peak_gain':float(max(np.linalg.norm(expected,axis=1))), 'measured_end_gain':float(euclidean[-1]),
              'model_end_gain':float(np.linalg.norm(expected[-1])),
              'maximum_primitive_signal_difference':float(np.max(np.linalg.norm(signal-expected,axis=1))),
              'directional_screen': guards}
    suffix=f'{args.case}_{args.amplitude:.1g}'
    np.savez_compressed(folder/f'hadamard_pinch_optimal_{suffix}.npz',time=times[:len(signal)],signal=signal,
                        predicted=expected,euclidean_gain=euclidean,weighted_gain=weighted,
                        metadata_json=json.dumps(output,allow_nan=False))
    print(output,flush=True)


if __name__=='__main__':
    main()
