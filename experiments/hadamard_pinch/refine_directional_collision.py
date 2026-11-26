"""Refine the zero-speed oblique collision in the static directional scan."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import brentq

from validate_pinch import D, INTERNAL, N, PARAMETERS
from causal_repair.nulloth import NullothMasterFunction


class RichardsonMaster(NullothMasterFunction):
    """Alternate fourth-order finite-difference stencil for the diagnostic."""

    @staticmethod
    def _numerical_jacobian(function, point):
        columns = []
        for delta in np.diag(2e-4*np.maximum(1, abs(point))):
            span = np.max(abs(delta))
            coarse = (function(point + delta) - function(point - delta))/(2*span)
            fine = (function(point + delta/2) - function(point - delta/2))/span
            columns.append((4*fine - coarse)/3)
        return np.column_stack(columns)


def audit_at(master, phi, v, angle):
    theta = np.deg2rad(angle)
    return master.coupled_characteristic_audit(
        N, INTERNAL, (0, 0, 0), D, (v, 0, 0), phi, (np.cos(theta), np.sin(theta), 0),
    )


def finite(value):
    return float(value) if np.isfinite(value) else None


def analyze(master, phi, v):
    def zero_branch(angle):
        result = audit_at(master, phi, v, angle)
        roots = eigvals(result.space_matrix[:8, :8], result.time_matrix[:8, :8])
        # Two advective roots are stationary here. Their cluster trace tracks
        # the crossing even arbitrarily close to zero in float64.
        return float(sum(sorted(roots, key=abs)[:3]).real)

    angle = brentq(zero_branch, 64.45, 64.5, xtol=1e-12)
    result = audit_at(master, phi, v, angle)
    time, space = result.time_matrix.copy(), result.space_matrix.copy()
    material = np.linalg.solve(time[:8, :8], space[:8, :8])
    time[:8, 8] = space[:8, 8] = 0.0
    reduced = np.linalg.solve(time, space)
    systems = {'material': material, 'constraint_reduced': reduced}
    multiplicities = {}
    for name, matrix in systems.items():
        roots = eigvals(matrix)
        singular = np.linalg.svd(matrix, compute_uv=False)
        multiplicities[name] = {
            'singular_values_at_zero': singular.tolist(),
            'near_zero_eigenvalues_real_imag': [[float(z.real), float(z.imag)]
                                               for z in sorted(roots, key=abs)[:8]],
            'thresholds': [
                {'threshold': tol, 'algebraic_cluster_size': int(np.count_nonzero(abs(roots) < tol)),
                 'svd_nullity': int(np.count_nonzero(singular < tol))}
                for tol in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
            ],
        }
    approach = []
    for offset in (-1e-3, -1e-4, -1e-5, 0, 1e-5, 1e-4, 1e-3):
        sample = audit_at(master, phi, v, angle + offset)
        approach.append({'angle_offset_degrees': offset, 'crossing_speed': zero_branch(angle + offset),
                         'full_rank': sample.eigenvector_rank,
                         'full_condition': finite(sample.eigenvector_condition),
                         'reduced_rank': sample.constraint_reduced_eigenvector_rank,
                         'reduced_condition': finite(sample.constraint_reduced_eigenvector_condition)})
    return {
        'angle_degrees': angle, 'mirror_angle_degrees': 180 - angle,
        'crossing_speed': zero_branch(angle), 'multiplicities': multiplicities,
        'angle_approach': approach,
    }


def main():
    folder = Path(__file__).resolve().parent
    summary = folder / 'hadamard_pinch_summary.json'
    data = json.loads(summary.read_text())
    phi, v, _ = data['pinch_phi_v_speed']
    data['oblique_zero_speed_collision'] = {
        'command_from_solver_root': 'python -B experiments/hadamard_pinch/refine_directional_collision.py',
        'native_centered_difference': analyze(NullothMasterFunction(PARAMETERS), phi, v),
        'fourth_order_richardson_difference': analyze(RichardsonMaster(PARAMETERS), phi, v),
    }
    summary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
