"""Render the gap-width and pilot directional plots from campaign data."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).resolve().parent
    reference = json.loads((directory / 'hadamard_pinch_reference.json').read_text())
    precision = reference['arbitrary_precision']
    samples = precision['gap_samples']
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), layout='constrained')
    slope = precision['limiting_width_slope']
    for sign, color in [(-1, '#185d82'), (1, '#a5462b')]:
        subset = [row for row in samples if sign*float(row['delta_phi']) > 0]
        offsets = np.array([abs(float(row['delta_phi'])) for row in subset])
        widths = np.array([float(row['width']) for row in subset])
        label = r'$\Delta\phi<0$' if sign < 0 else r'$\Delta\phi>0$'
        axes[0].loglog(offsets, widths, 'o', markerfacecolor='none', color=color, label=label)
        axes[1].semilogx(offsets, widths/offsets, 'o-', color=color, label=label)
    axes[0].loglog(offsets, slope*offsets, ':', color='#34383b', label='Linear limit')
    axes[1].axhline(slope, linestyle=':', color='#34383b')
    axes[0].set_ylabel('Width in relative velocity')
    axes[1].set_ylabel(r'Width / $|\Delta\phi|$')
    axes[0].legend()
    for axis in axes:
        axis.set_xlabel(r'Scalar offset $|\Delta\phi|$')
        axis.grid(alpha=0.2)
    figure.suptitle('Reopening of the longitudinal instability gap')
    figure.savefig(directory / '03_gap_width_scaling.png', dpi=180)
    plt.close(figure)

    with np.load(directory / 'hadamard_pinch_directional_scan.npz') as scan:
        angles, values, states = (scan[key] for key in ('angle_degrees', 'values', 'states'))
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout='constrained')
    labels = ['Pinch', 'High-flow comparison', 'Gap center (−)', 'Below gap (−)',
              'Above gap (−)', 'Gap center (+)', 'Below gap (+)', 'Above gap (+)']
    for index, (label, state) in enumerate(zip(labels, states, strict=True)):
        if state == 'transverse_reference':
            axes[1].plot(angles, values[index, :, 0], label=label, color='#a5462b')
        else:
            axes[0].plot(angles, values[index, :, 0], label=label,
                         linewidth=2.0 if state == 'pinch' else 1.05)
    axes[0].set_title('Pinch and nearby gap states')
    axes[1].set_title(r'$\phi=0.15,\; v_\mathrm{rel}=0.85$')
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel('Propagation angle (degrees)')
        axis.set_ylabel(r'Maximum $|\mathrm{Im}\,\lambda|$')
        axis.grid(alpha=0.2)
    axes[0].set_xlim(0, 8)
    axes[1].set_xticks([0, 45, 90, 135, 180])
    figure.suptitle('Directional instability in the frozen-entropy principal symbol')
    figure.savefig(directory / '06_directional_instability.png', dpi=180)
    plt.close(figure)

    summary = json.loads((directory / 'hadamard_pinch_summary.json').read_text())
    collision = summary['oblique_zero_speed_collision']['native_centered_difference']
    figure, axis = plt.subplots(figsize=(6.8, 4.3), layout='constrained')
    for sign, color in [(-1, '#185d82'), (1, '#a5462b')]:
        rows = [row for row in collision['angle_approach'] if sign*row['angle_offset_degrees'] > 0]
        offsets = [abs(row['angle_offset_degrees']) for row in rows]
        label = 'Approach from below' if sign < 0 else 'Approach from above'
        axis.loglog(offsets, [row['reduced_condition'] for row in rows], 'o',
                    markerfacecolor='none', markersize=8 if sign < 0 else 5, color=color, label=label)
    scale = np.median([abs(row['angle_offset_degrees'])*row['reduced_condition']
                       for row in collision['angle_approach'] if row['reduced_condition'] is not None])
    axis.loglog(offsets, scale/np.array(offsets), ':', color='#34383b', label='Inverse angular offset')
    axis.set_title('Eigenbasis conditioning near the oblique zero-speed collision')
    axis.set_xlabel(r'Angular distance from the collision $|\theta-\theta_*|$ (degrees)')
    axis.set_ylabel('Constraint-reduced eigenbasis condition number')
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(directory / '05_eigenbasis_conditioning.png', dpi=180)
    plt.close(figure)

    continuation_path = directory / 'hadamard_pinch_continuation.npz'
    if continuation_path.exists():
        plot_continuation(directory, continuation_path)


def plot_continuation(directory, continuation_path):
    with np.load(continuation_path) as arrays:
        relative = arrays['relative_background_offsets']
        states = arrays['states']
        angles = arrays['collision_angles_degrees']
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    for i, (label, color) in enumerate(zip(('Ordinary density', 'Carrier density', 'Internal energy'),
                                           ('#185d82', '#a5462b', '#3e7868'))):
        other = np.delete(relative, i, axis=1)
        indices = np.flatnonzero(np.all(other == 0, axis=1))
        indices = indices[np.argsort(relative[indices, i])]
        x = relative[indices, i]*100
        for axis, values in zip(axes, (states[:, 3], states[:, 4], angles)):
            axis.plot(x, values[indices] - values[0], 'o-', markersize=4, color=color, label=label)
    for axis, label in zip(axes, (r'Change in pinch $\phi$', r'Change in pinch $v_\mathrm{rel}$',
                                'Change in oblique collision angle (degrees)')):
        axis.set_xlabel('Background change (%)')
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axes[0].legend(fontsize=8)
    figure.suptitle('Continuation of the pinch and oblique collision')
    figure.savefig(directory / '08_pinch_continuation.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
