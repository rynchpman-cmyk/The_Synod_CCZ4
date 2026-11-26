"""Linearize the native Nulloth step on the constraint-compatible Fourier sector."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm, svd

from analyze_transverse_tangent import state_from_primitive
from run_transit_shooting import screen
from run_transverse_tangent import HarmonicProbeGrid, K, build_solver, material_symbol, physical_fields, velocity_from_proper
from validate_pinch import D, INTERNAL, N
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.theory33 import CarrierPrimitive


SIZE = 23
END = .016
STEP = .000125


def initialize_full(solver, values):
    gamma = solver.master_parameters.gamma_ad
    n, d = np.exp(values[0]), np.exp(values[4])
    internal = np.exp((gamma-1)*values[8])*n**(gamma-1)/(gamma-1)
    fluid = FluidPrimitive(n, internal, velocity_from_proper(values[1:4]), solver.grid.zeros())
    carrier = CarrierPrimitive(d, velocity_from_proper(values[5:8]))
    return solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier,
                             varphi=values[9], scalar_momentum=values[10],
                             a=values[11:14], pi_A=values[14:17], b=values[17:20], pi_B=values[20:23])


def tangent(solver, base, epsilon, operation, homogeneous=False):
    points = solver.grid.shape[0]
    phase = 2*np.pi*np.arange(points)/points
    factor = np.exp(-1j*phase)
    background = np.broadcast_to(base[:SIZE, None], (SIZE, points)).copy()
    matrix = np.empty((SIZE, SIZE), dtype=complex)
    max_auxiliary = 0.0
    for column in range(SIZE):
        positive = background.copy()
        positive[column] += epsilon*(1 if homogeneous else np.cos(phase))
        out = physical_fields(solver, operation(initialize_full(solver, positive)))[0]
        if homogeneous:
            negative = background.copy()
            negative[column] -= epsilon
            opposite = physical_fields(solver, operation(initialize_full(solver, negative)))[0]
            derivative = np.mean(out-opposite, axis=1)/(2*epsilon)
        else:
            derivative = ((out - np.roll(out, points//2, axis=1)) @ factor)/(points*epsilon)
        matrix[:, column] = derivative[:SIZE]
        max_auxiliary = max(max_auxiliary, float(np.linalg.norm(derivative[25:27])))
    return logm(matrix)/STEP, max_auxiliary


def screen_cell(solver, fields, recovered, pinch_speed, index):
    gamma = float(recovered.master.relative_lorentz_factor[index])
    report = recovered.report
    return screen(solver, {
        'n': float(recovered.fluid.baryon_density[index]),
        'd': float(recovered.carrier.number_density[index]),
        'eps': float(recovered.fluid.specific_internal_energy[index]),
        'phi': float(fields[9,index]), 'v_rel': float(np.sqrt(max(0.,1-gamma**-2))),
        'minimum_legendre_eigenvalue': report.minimum_legendre_eigenvalue,
        'minimum_thermodynamic_eigenvalue': report.minimum_thermodynamic_eigenvalue,
        'recovery_failures': report.failed_cells,
    }, pinch_speed)


def background(case, folder, pinch):
    if case == 'transverse':
        data = folder/'hadamard_pinch_transverse_tangent_n064_dt0.00013_a0.0001_harmonic8_production.npz'
        with np.load(data) as saved:
            return saved['time'], saved['mean'][:, :SIZE], saved['initial_primitive_mode'], None
    solver = build_solver(PeriodicGrid((4,), (1.0,)))
    phi = pinch[0]
    angle = np.deg2rad(64.4761802656)
    v = .2*np.array([np.cos(angle), -np.sin(angle), 0.])
    base = np.zeros((SIZE,4))
    base[0], base[4], base[8], base[9] = np.log(N), np.log(D), solver.master.specific_entropy(N, INTERNAL), phi
    base[5:8] = (v/np.sqrt(1-v @ v))[:, None]
    state = initialize_full(solver, base)
    summaries, means, times = [], [], []
    for i in range(round(END/STEP)+1):
        if i:
            state = solver.step(state, STEP)
        fields, recovery = physical_fields(solver, state)
        means.append(np.mean(fields[:SIZE], axis=1))
        times.append(state.time)
        if i % 4 == 0:
            check = screen_cell(solver, fields, recovery, pinch[2], 0)
            summaries.append({'time': state.time, **check})
    return np.array(times), np.array(means), None, summaries


def scale_current(p, gamma):
    transform = np.eye(9)
    for start in (0,4):
        q = p[start+1:start+4]
        boost = np.sqrt(1+q @ q)
        transform[start,start] = boost
        transform[start,start+1:start+4] = q/boost
        transform[start+1:start+4,start] = q
    transform[8,8] = gamma-1
    return transform


def cross_block_stopped(matrix, sector):
    index = np.arange(SIZE)
    inside = np.isin(index, sector)
    output = matrix.copy()
    output[inside[:, None] != inside[None, :]] = 0
    return output


def model_histories(master, times, means, nodes, generators, drag, seed):
    labels = ('frozen_full', 'evolving_full', 'evolving_principal', 'without_scalar_cross_blocks',
              'without_vector_cross_blocks', 'without_native_drag_linear_term')
    prop = np.broadcast_to(np.eye(SIZE,9, dtype=complex), (len(labels),SIZE,9)).copy()
    maps = np.empty((len(times),*prop.shape), dtype=complex)
    maps[0] = prop
    for i, dt in enumerate(np.diff(times), start=1):
        middle = .5*(times[i]+times[i-1])
        left = np.clip(np.searchsorted(nodes,middle)-1,0,len(nodes)-2)
        weight = (middle-nodes[left])/(nodes[left+1]-nodes[left])
        generator = (1-weight)*generators[left]+weight*generators[left+1]
        drag_now = (1-weight)*drag[left]+weight*drag[left+1]
        reference = .5*(means[i]+means[i-1])
        mat = np.zeros((SIZE,SIZE),dtype=complex)
        n,d,internal,u,v = state_from_primitive(reference[:9])
        mat[:9,:9] = -1j*K*material_symbol(master,n,d,internal,u,v,reference[9])
        variants = (generators[0],generator, mat, cross_block_stopped(generator, (9,10)),
                    cross_block_stopped(generator, range(11,23)), generator-drag_now)
        for j, model in enumerate(variants):
            prop[j] = expm(dt*model) @ prop[j]
        maps[i] = prop
    transform = scale_current(means[0,:9],master.parameters.gamma_ad)
    material = maps[:,:,:9,:]
    scaled = transform @ material @ np.linalg.inv(transform)
    gains = np.stack((np.linalg.svd(material,compute_uv=False)[...,0],np.linalg.svd(scaled,compute_uv=False)[...,0]),axis=-1)
    optimal = np.empty((len(labels),2,9),dtype=complex)
    summaries = {}
    for j, name in enumerate(labels):
        peak = np.argmax(gains[:,j],axis=0)
        _,_,euclidean = svd(material[peak[0],j])
        _,_,weighted = svd(scaled[peak[1],j])
        optimal[j] = [euclidean[0].conj(),np.linalg.solve(transform,weighted[0].conj())]
        summaries[name] = {'peak_gains': gains[peak,j,np.arange(2)].tolist(), 'peak_times': times[peak].tolist(),
                           'endpoint_gains': gains[-1,j].tolist()}
        if seed is not None:
            summaries[name]['seed_end_gain'] = float(np.linalg.norm(material[-1,j] @ seed))
    return gains,optimal,maps,summaries,labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('transverse','low_flow'))
    parser.add_argument('--nodes',type=int,default=5,choices=(3,5,9))
    parser.add_argument('--epsilon',type=float,default=2e-4)
    parser.add_argument('--audit-background',action='store_true',help='Recheck the stored low-flow background with the directional screen')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    pinch = json.loads((folder/'hadamard_pinch_summary.json').read_text())['pinch_phi_v_speed']
    times,means,seed,guards = background(args.case,folder,pinch)
    suffix = f'{args.case}_{args.nodes}_{args.epsilon:.1g}'
    report_path = folder/f'hadamard_pinch_sourced_{suffix}.json'
    if args.audit_background:
        if guards is None:
            parser.error('--audit-background applies to low_flow')
        with np.load(folder/f'hadamard_pinch_sourced_{suffix}.npz') as stored:
            np.testing.assert_allclose(means,stored['means'],rtol=1e-9,atol=1e-11)
        report = json.loads(report_path.read_text())
        report['screened_background'] = guards
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('Directional checkpoints',len(guards),'rejected',sum(bool(row['reasons']) for row in guards),flush=True)
        return
    nodes = np.linspace(0,len(times)-1,args.nodes,dtype=int)
    wave = build_solver(HarmonicProbeGrid(4,64,2*np.pi/K,0))
    zero = build_solver(PeriodicGrid((4,), (1.0,)))
    full,drag,source,auxiliary = [],[],[],[]
    for i in nodes:
        row = means[i]
        full_now, leak = tangent(wave,row,args.epsilon,lambda state: wave.step(state,STEP))
        drag_now, _ = tangent(wave,row,args.epsilon,lambda state: wave._apply_carrier_drag(state,STEP))
        source_now, _ = tangent(zero,row,args.epsilon,lambda state: zero.step(state,STEP),homogeneous=True)
        full.append(full_now)
        drag.append(drag_now)
        source.append(source_now)
        auxiliary.append(leak)
        print(args.case,float(times[i]),'source norm',np.linalg.norm(source_now,2),'cleaner tangent',leak,flush=True)
    gains,optimal,maps,summaries,labels = model_histories(wave.master,times,means,times[nodes],full,drag,seed)
    report = {'case': args.case,'linearization_step': STEP,'amplitude': args.epsilon,
              'wavenumber': K,'equivalent_cells': 64,'fourier_samples': 4,'nodes': times[nodes].tolist(),
              'field_count': SIZE,'norms': ['primitive','fractional current and entropy'], 'models': summaries,
              'k0_source_matrix_norm': [float(np.linalg.norm(matrix,2)) for matrix in source],
              'drag_matrix_norm': [float(np.linalg.norm(matrix,2)) for matrix in drag],
              'maximum_cleaner_response': max(auxiliary),'screened_background': guards}
    if args.case=='transverse':
        with np.load(folder/'hadamard_pinch_transverse_tangent_n064_dt0.00013_a0.0001_harmonic8_production.npz') as saved:
            reference = saved['tangent'][:,:9]
            predicted = maps[:,1,:9] @ seed
            report['native_seed_end_gain'] = float(np.linalg.norm(reference[-1]))
            report['maximum_seed_signal_difference'] = float(np.max(np.linalg.norm(reference-predicted,axis=1)))
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(folder/f'hadamard_pinch_sourced_{suffix}.npz',times=times,means=means,nodes=nodes,full=full,drag=drag,source=source,
                        labels=labels,gains=gains,optimal=optimal,propagators=maps)
    print(args.case,summaries,flush=True)


if __name__ == '__main__':
    main()
