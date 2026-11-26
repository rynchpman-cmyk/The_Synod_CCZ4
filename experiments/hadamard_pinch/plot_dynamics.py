"""Summarize and plot the homogeneous, spatial, and frozen-propagator data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#185d82', '#a5462b', '#3e7868', '#735286']
LABELS = ['Minkowski, no drag', 'Minkowski, native drag', 'CCZ4, weak gravity', 'CCZ4, default gravity']


def save(figure, folder, name):
    for axis in figure.axes:
        axis.grid(alpha=.2)
    figure.savefig(folder / name, dpi=180)
    plt.close(figure)


def homogeneous(folder, summary):
    with np.load(folder / 'hadamard_pinch_dynamics.npz') as data:
        columns = list(data['columns'])
        runs = [dict(zip(columns, data[row['array_key']].T)) for row in summary['homogeneous_dynamics']['runs']
                if row['dt'] == .0005]
    figure, axis = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    for label, color, run in zip(LABELS, COLORS, runs):
        axis.plot(run['v_rel'], run['phi'], color=color, label=label)
        axis.plot(run['v_rel'][-1], run['phi'][-1], 'o', color=color)
    axis.set(xlabel='Relative velocity', ylabel='Scalar amplitude', title='Homogeneous departure from the pinch')
    axis.legend(fontsize=8)
    save(figure, folder, '09_homogeneous_trajectories.png')

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4), layout='constrained')
    for label, color, run in zip(LABELS, COLORS, runs):
        for axis, field in zip(axes, ('phi', 'v_rel')):
            axis.plot(run['time'], run[field], color=color, label=label)
    axes[0].set_ylabel('Scalar amplitude')
    axes[1].set_ylabel('Relative velocity')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Coordinate time')
    figure.suptitle('Scalar growth and relative velocity')
    save(figure, folder, '10_scalar_and_velocity.png')

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4), layout='constrained')
    for label, color, run in zip(LABELS, COLORS, runs):
        convexity = np.minimum(run['minimum_thermodynamic_eigenvalue'], run['minimum_legendre_eigenvalue'])
        axes[0].plot(run['time'], convexity, color=color, label=label)
        axes[1].plot(run['time'], run['oblique_collision_angle_degrees'], color=color)
    axes[0].set_ylabel('Minimum convexity eigenvalue')
    axes[1].set_ylabel('Angle of defective zero-speed mode (degrees)')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Coordinate time')
    figure.suptitle('Convexity margins and persistence of the oblique defect')
    save(figure, folder, '11_admissibility_history.png')

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for label, color, run in zip(LABELS, COLORS, runs):
        axes[0].plot(run['time'], run['energy_balance_error'], label=label, color=color)
        axes[1].plot(run['time'], run['total_momentum_x'] - run['total_momentum_x'][0], color=color)
        axes[2].plot(run['time'], run['entropy_integral'] - run['entropy_integral'][0], color=color)
    axes[0].set_ylabel('Energy change minus geometric work')
    axes[1].set_ylabel('Change in integrated momentum')
    axes[2].set_ylabel('Change in integrated entropy')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Coordinate time')
    figure.suptitle('Homogeneous energy, momentum, and entropy accounting')
    save(figure, folder, '14_conservation_accounting.png')


def spatial(folder):
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.2), layout='constrained')
    files = ['hadamard_pinch_spatial_growth_drag.npz', 'hadamard_pinch_spatial_growth_drag_space.npz']
    for filename in files:
        with np.load(folder / filename) as data:
            for i, meta in enumerate(json.loads(str(data['metadata_json']))):
                times, native, _, evolving, *_ = data[f'run_{i}'].T
                axis = axes[0 if meta['wavenumber'] == 128 else 1]
                first = len(axis.lines) == 0
                axis.plot(times, native, label=f"Native, {meta['points']} cells")
                if first:
                    axis.plot(times, evolving, ':', color='#34383b', label='Evolving principal symbol')
    for axis, k in zip(axes, (128, 256)):
        axis.set(title=f'Wavenumber {k}', xlabel='Coordinate time', ylabel='Fourier perturbation gain')
        axis.legend(fontsize=8)
    figure.suptitle('Oblique spatial amplification with native drag')
    save(figure, folder, '12_spatial_amplification.png')


def transverse(folder):
    base = folder / 'hadamard_pinch_spatial_growth_drag_hadamard.npz'
    early = folder / 'hadamard_pinch_spatial_growth_drag_hadamard_early-growth.npz'
    long = folder / 'hadamard_pinch_spatial_growth_drag_hadamard_long-transverse.npz'
    if not base.exists() or not early.exists():
        return
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.2), layout='constrained')
    for axis, paths in zip(axes, ((base, long), (early,))):
        for path in paths:
            if not path.exists():
                continue
            with np.load(path) as data:
                metadata = json.loads(str(data['metadata_json']))
                for i, case in enumerate(metadata):
                    if case['wavenumber'] != 256:
                        continue
                    t, native, _, evolving, *_ = data[f'run_{i}'].T
                    first = not axis.lines
                    axis.plot(t, native, label=f"Native, {case['points']} cells")
                    if first:
                        axis.plot(t, evolving, ':', color='#34383b', label='Evolving principal symbol')
        axis.set(xlabel='Coordinate time', ylabel='Fourier perturbation gain')
        axis.legend(fontsize=8)
    axes[0].set_title('Full sampled trajectory')
    axes[1].set_title('Early growth with grid refinement')
    figure.suptitle('Transverse Hadamard mode at wavenumber 256')
    save(figure, folder, '12_transverse_hadamard.png')


def propagator(folder):
    with np.load(folder / 'hadamard_pinch_transient_growth.npz') as data:
        t, k, gain = data['time'], data['wavenumbers'][-1], data['max_singular_value'][:, :, -1]
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.2), layout='constrained')
    for i, label, color in zip((0, 1, 3, 6), ('Longitudinal pinch', 'Oblique collision', 'One degree off collision', 'Low-flow control'), COLORS):
        for j, axis in enumerate(axes):
            axis.plot(t*k, gain[i, j], label=label, color=color)
    for axis, title in zip(axes, ('Primitive coordinates', 'Fractional current and entropy')):
        axis.set(title=title, xlabel=r'$k t$', ylabel='Largest propagator singular value')
    axes[0].legend(fontsize=8)
    figure.suptitle('Frozen-background amplification near the oblique collision')
    save(figure, folder, '13_frozen_propagator.png')


def summarize_spatial_and_frozen(folder, summary):
    spatial_cases = []
    for path in sorted(folder.glob('hadamard_pinch_spatial_growth*.npz')):
        with np.load(path) as data:
            for i, metadata in enumerate(json.loads(str(data['metadata_json']))):
                t, native, frozen, evolving, *_ = data[f'run_{i}'].T
                peak = int(np.argmax(native))
                spatial_cases.append({**metadata, 'data_file': path.name, 'array_key': f'run_{i}',
                                      'last_time': float(t[-1]), 'last_native_gain': float(native[-1]),
                                      'peak_native_gain': float(native[peak]), 'peak_time': float(t[peak]),
                                      'last_frozen_principal_gain': float(frozen[-1]),
                                      'last_evolving_principal_gain': float(evolving[-1])})
    summary['spatial_growth'] = spatial_cases
    with np.load(folder / 'hadamard_pinch_transient_growth.npz') as data:
        gains = data['max_singular_value'][:, :, -1]
        peak = np.argmax(gains, axis=2)
        summary['frozen_propagator'] = {
            'wavenumber': float(data['wavenumbers'][-1]), 'norms': data['norms'].tolist(),
            'backgrounds': [{'state': label, 'peak_gains': np.max(gains[i], axis=1).tolist(),
                             'peak_times': data['time'][peak[i]].tolist()}
                            for i, label in enumerate(data['states'].tolist())],
        }
    summary['scope'] = 'Static validation, native homogeneous and spatial evolution, autonomous shooting, and frozen principal propagators.'


def main():
    folder = Path(__file__).resolve().parent
    summary_path = folder / 'hadamard_pinch_summary.json'
    summary = json.loads(summary_path.read_text())
    homogeneous(folder, summary)
    spatial(folder)
    transverse(folder)
    propagator(folder)
    summarize_spatial_and_frozen(folder, summary)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
