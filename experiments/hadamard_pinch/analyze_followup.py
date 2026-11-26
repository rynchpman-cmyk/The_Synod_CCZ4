"""Summarize the controlled, longitudinal, and sourced follow-up experiments."""

from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze_longitudinal(folder):
    metadata, superpositions, broadband = {},{},{}
    for suite in ('pairs','refine','small','extended','broadband_weak','broadband_material','broadband_scalar','broadband_fine'):
        with np.load(folder/f'hadamard_pinch_longitudinal_{suite}.npz') as data:
            cases = json.loads(str(data['metadata_json']))
            for i, case in enumerate(cases):
                rows, modes = data[f'diagnostics_{i}'], data[f'modal_{i}']
                case['final_outside_subspace_norm'] = float(rows[-1,2])
                case['final_higher_harmonic_norm'] = float(rows[-1,3])
                if case['seed'] in ('mode_1','mode_2'):
                    unseeded = 1 if case['seed']=='mode_1' else 0
                    case['maximum_unseeded_mode_amplitude'] = float(max(abs(modes[:,unseeded])))
                elif case['seed'].startswith('broadband'):
                    broadband[suite] = data[f'fourier_{i}'][:, :9]
            if suite in ('pairs','extended'):
                a,b = data['fourier_0'], data['fourier_1']
                for case in (2,3):
                    projection = data[f'modal_{case}'][0]
                    prediction = projection[0]*a + projection[1]*b
                    observed = data[f'fourier_{case}']
                    superpositions[f'{suite}_{cases[case]["seed"]}'] = float(np.max(np.linalg.norm(
                        (prediction-observed)[:,:9,0],axis=1)))
            metadata[suite] = cases
    weak = broadband['broadband_weak']
    baseline = np.linalg.norm(weak[0],axis=0)
    broadband_endpoints = {name:(np.linalg.norm(history[-1],axis=0)/baseline).tolist()
                           for name, history in broadband.items()}
    material,scalar = broadband['broadband_material'],broadband['broadband_scalar']
    count = min(len(weak),len(material),len(scalar))
    difference = material[:count]+scalar[:count]-weak[:count]
    split_error = float(np.max(np.linalg.norm(difference.reshape(count,-1),axis=1)))
    return {'cases':metadata, 'maximum_superposition_fundamental_error':superpositions,
            'broadband_endpoint_material_gain_per_harmonic':broadband_endpoints,
            'broadband_decomposition_maximum_difference':split_error}


def plots(folder, control, pinch):
    figure, axes = plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for result in control['cases']:
        rows=result['rows']
        axes[0].plot([row['v_rel'] for row in rows],[row['phi'] for row in rows],alpha=.7)
        axes[0].scatter(rows[-1]['v_rel'],rows[-1]['phi'],color='#a5462b',s=17)
    axes[0].scatter(pinch[1],pinch[0],marker='*',color='#34383b',s=80,label='Reference pinch')
    axes[0].scatter([],[],color='#a5462b',s=17,label='First rejected state')
    axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Relative velocity',ylabel='Scalar amplitude',title='Controlled trajectories')
    with np.load(folder/'hadamard_pinch_longitudinal_extended.npz') as data:
        for index,label in enumerate(('Acoustic mode 1 seed','Acoustic mode 2 seed')):
            time=data[f'diagnostics_{index}'][:,0]
            modes=abs(data[f'modal_{index}'])
            axes[1].plot(time,modes[:,1-index],label=label)
    axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Time',ylabel='Initially unseeded mode amplitude',title='Transfer within the longitudinal doublet')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(folder/'controlled_and_longitudinal_followup.png',dpi=180)
    plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for axis,case,title in zip(axes,('low_flow','transverse'),('Low-flow background','High-flow transverse background')):
        with np.load(folder/f'hadamard_pinch_optimal_{case}_0.0002.npz') as replay:
            time=replay['time']
            axis.plot(time,np.linalg.norm(replay['predicted'],axis=1),':',color='#34383b',label='Sourced variational prediction')
            axis.plot(time,replay['euclidean_gain'],color='#185d82',label='Native nonlinear replay')
        axis.set(xlabel='Time',ylabel='Primitive perturbation gain',title=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.savefig(folder/'sourced_optimal_replay.png',dpi=180)
    plt.close(figure)

    figure,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    with np.load(folder/'hadamard_pinch_longitudinal_broadband_weak.npz') as run:
        baseline=np.linalg.norm(run['fourier_0'][0,:9,0])
    for axis,suites in zip(axes,((('extended','32 cells, larger seed'),('broadband_weak','32 cells, smaller seed'),
                                 ('broadband_fine','40 cells, smaller seed')),
                                (('broadband_weak','All initial perturbations'),('broadband_material','Material seed'),
                                 ('broadband_scalar','Scalar seed')))):
        for suite,label in suites:
            index=4 if suite=='extended' else 0
            with np.load(folder/f'hadamard_pinch_longitudinal_{suite}.npz') as run:
                time=run[f'diagnostics_{index}'][:,0]
                amplitude=np.linalg.norm(run[f'fourier_{index}'][:,:9,0],axis=1)
            axis.plot(time,amplitude/baseline,label=label)
        axis.set(xlabel='Time',ylabel='Fundamental material response / initial full-seed response')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_title('Broadband amplitude and grid checks')
    axes[1].set_title('Material and scalar initial seeds')
    figure.savefig(folder/'longitudinal_broadband_growth.png',dpi=180)
    plt.close(figure)


def main():
    folder=Path(__file__).resolve().parent
    pinch=read_json(folder/'hadamard_pinch_summary.json')['pinch_phi_v_speed']
    control=read_json(folder/'hadamard_pinch_controlled_transit.json')
    detours=read_json(folder/'hadamard_pinch_thermodynamic_detours.json')
    target_outcomes=Counter(status for row in detours['slices'] for v,status in row['tests'] if abs(v-pinch[1])<=.011)
    sourced={path.stem.removeprefix('hadamard_pinch_sourced_'):read_json(path)
             for path in sorted(folder.glob('hadamard_pinch_sourced_*.json'))}
    replays={}
    for path in sorted(folder.glob('hadamard_pinch_optimal_*.npz')):
        with np.load(path) as data:
            replays[path.stem.removeprefix('hadamard_pinch_optimal_')]=json.loads(str(data['metadata_json']))
    summary={
        'controlled': {
            'cases': [{key:value for key,value in row.items() if key!='rows'} for row in control['cases']],
            'target_velocity_outcomes': dict(target_outcomes),
            'detour_state_count': sum(len(row['tests']) for row in detours['slices']),
            'maximum_screened_detour_velocity': max(row['highest_screened_v'] for row in detours['slices']
                                                    if row['highest_screened_v'] is not None),
        },
        'longitudinal': analyze_longitudinal(folder), 'sourced': sourced, 'nonlinear_optimal_replays': replays,
    }
    (folder/'hadamard_pinch_followup_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    plots(folder,control,pinch)


if __name__=='__main__':
    main()
