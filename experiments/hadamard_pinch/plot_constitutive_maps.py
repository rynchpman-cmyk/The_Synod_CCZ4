"""Render the global longitudinal map and local pinch geometry."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals

from validate_pinch import D, INTERNAL, N, PARAMETERS, native_pencil
from causal_repair.nulloth import NullothMasterFunction


def local(folder, reference):
    phi, velocity, _ = map(float, reference['arbitrary_precision']['precision_ladder'][-1]['root_phi_v_speed'])
    samples = reference['arbitrary_precision']['gap_samples']
    figure, axis = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    for side, color in [(-1, '#185d82'), (1, '#a5462b')]:
        states = [row for row in samples if side*float(row['delta_phi']) > 0]
        offsets = np.append(0, [float(row['delta_phi']) for row in states][::-1])
        low, high = (np.append(0, [float(row[key]) - velocity for row in states][::-1]) for key in ('low', 'high'))
        axis.fill_between(offsets, low, high, alpha=.24, color=color)
        axis.plot(offsets, low, color=color)
        axis.plot(offsets, high, color=color)
    axis.plot(0, 0, 'o', color='#34383b', label='Semisimple pinch')
    axis.set(xlabel=rf'$\phi - {phi:.9f}$', ylabel=rf'$v_\mathrm{{rel}} - {velocity:.9f}$',
             title='Longitudinal Hadamard gap near the pinch')
    axis.set_xticks(np.linspace(min(float(row['delta_phi']) for row in samples),
                                max(float(row['delta_phi']) for row in samples), 5))
    axis.grid(alpha=.2)
    axis.legend()
    figure.savefig(folder / '02_pinch_geometry.png', dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    for side, color, label in [(-1, '#185d82', 'Negative offset'), (1, '#a5462b', 'Positive offset')]:
        states = [row for row in samples if side*float(row['delta_phi']) > 0]
        axis.loglog([abs(float(row['delta_phi'])) for row in states],
                    [float(row['center_pair_splitting_modulus']) for row in states],
                    'o-', color=color, label=label, markerfacecolor='none')
    axis.set(xlabel=r'$|\Delta\phi|$', ylabel='Magnitude of acoustic pair splitting',
             title='Acoustic eigenvalue separation at the gap center')
    axis.grid(alpha=.2)
    axis.legend()
    figure.savefig(folder / '04_eigenvalue_splitting.png', dpi=180)
    plt.close(figure)


def global_map(folder):
    master = NullothMasterFunction(PARAMETERS)
    scalars = np.linspace(0, .18, 37)
    velocities = np.linspace(0, .92, 61)
    data = np.empty((len(scalars), len(velocities), 4))
    for i, phi in enumerate(scalars):
        for j, v in enumerate(velocities):
            time, space = native_pencil(master, phi, v)
            roots = eigvals(space, time)
            screen = master.characteristic_screen(N, INTERNAL, 0, D, v, phi)
            data[i, j] = (max(abs(roots.imag)), max(abs(roots)), screen.legendre_minimum_eigenvalue,
                          screen.thermodynamic_minimum_eigenvalue)
    columns = ('max_imag', 'max_material_modulus', 'legendre_min', 'thermodynamic_min')
    np.savez_compressed(folder / 'hadamard_pinch_global_longitudinal.npz', phi=scalars, v_rel=velocities,
                        values=data, columns=columns)
    figure, axis = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    image = axis.pcolormesh(velocities, scalars, data[:, :, 0], shading='nearest', cmap='magma')
    figure.colorbar(image, ax=axis, label=r'Maximum $|\mathrm{Im}\,\lambda|$')
    axis.axvline(np.sqrt(1 - PARAMETERS.maximum_relative_lorentz_factor**-2), color='#a6dbfc',
                linestyle=':', label='Production counterflow limit')
    axis.plot(.7608844075272453, .010884443568990347, 'o', color='#a6dbfc', label='Longitudinal pinch')
    axis.set(xlabel='Relative velocity', ylabel='Scalar amplitude', title='Longitudinal instability across the constitutive slice')
    axis.legend(fontsize=8, loc='upper left')
    figure.savefig(folder / '01_longitudinal_phase_diagram.png', dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.1), layout='constrained')
    for axis, column, title, threshold in zip(axes, (2, 3, 1), ('Legendre minimum', 'Thermodynamic minimum', 'Material speed modulus'),
                                             (0, 0, 1)):
        values = data[:, :, column]
        image = axis.pcolormesh(velocities, scalars, values, shading='nearest', cmap='viridis')
        if values.min() < threshold < values.max():
            axis.contour(velocities, scalars, values, [threshold], colors='white', linewidths=1)
        figure.colorbar(image, ax=axis)
        axis.set(title=title, xlabel='Relative velocity')
    axes[0].set_ylabel('Scalar amplitude')
    figure.suptitle('Convexity and characteristic speed in the longitudinal slice')
    figure.savefig(folder / '07_convexity_and_causality.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    directory = Path(__file__).resolve().parent
    local(directory, json.loads((directory / 'hadamard_pinch_reference.json').read_text()))
    global_map(directory)
