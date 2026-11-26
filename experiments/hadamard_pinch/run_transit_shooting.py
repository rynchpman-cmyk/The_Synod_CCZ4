"""Shoot autonomous native trajectories from outside the oblique defect band."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals

from audit_transit_corridor import EntropyPencil, normal
from run_homogeneous_departure import FixedMinkowski, diagnostics, initial_state
from validate_pinch import D, INTERNAL, N, PARAMETERS
from causal_repair.grid import PeriodicGrid
from causal_repair.production_nulloth import NullothProductionSolver


def screen(solver, row, pinch_speed):
    pencil = EntropyPencil(solver.master, (row['n'], row['d'], row['eps']), row['phi'], row['v_rel'])
    collision = pencil.collision((0, 89.9))
    worst_imag, maximum_speed, condition = 0.0, 1.0, 0.0
    incomplete = False
    minimum_separation = None
    for angle in np.linspace(0, 90, 46):
        time, space = pencil.material(normal(angle))
        roots = eigvals(space, time)
        _, rank, basis_condition, _ = solver.master._clustered_eigenbasis(np.linalg.solve(time, space))
        worst_imag = max(worst_imag, float(max(abs(roots.imag))))
        maximum_speed = max(maximum_speed, float(max(abs(roots))))
        incomplete |= rank < 9 or not np.isfinite(basis_condition)
        if np.isfinite(basis_condition):
            condition = max(condition, basis_condition)
        if angle == 0:
            pair = sorted(roots, key=lambda value: abs(value - pinch_speed))[:2]
            minimum_separation = float(abs(pair[1] - pair[0]))
    margin = min(row['minimum_legendre_eigenvalue'], row['minimum_thermodynamic_eigenvalue'])
    reasons = []
    if collision['defective']:
        reasons.append('oblique zero-speed defect')
    if incomplete:
        reasons.append('incomplete or unresolved sampled eigenspace')
    if worst_imag > 1e-8:
        reasons.append('complex characteristic')
    if maximum_speed > 1 + 1e-8:
        reasons.append('superluminal characteristic')
    if margin <= PARAMETERS.convexity_floor or row['recovery_failures']:
        reasons.append('convexity or recovery failure')
    if condition >= PARAMETERS.characteristic_condition_limit:
        reasons.append('sampled directional eigenbasis condition limit')
    return {'oblique_collision_angle': collision['angle_degrees'], 'maximum_imaginary': worst_imag,
            'maximum_characteristic_modulus': maximum_speed, 'maximum_finite_basis_condition': condition,
            'longitudinal_pinch_pair_separation': minimum_separation, 'convexity_margin': margin,
            'reasons': reasons}


def run(seed, pinch, background=(N, D, INTERNAL)):
    phi, v, pi = seed
    dt, duration = .001, .06
    grid = PeriodicGrid((4,), (1.0,))
    solver = NullothProductionSolver(grid, PARAMETERS)
    solver.ccz4 = FixedMinkowski(grid, solver.ccz4.parameters)
    state = initial_state(solver, phi, v, constrained=False, scalar_momentum=pi, background=background)
    rows, failure = [], None
    for step in range(round(duration/dt) + 1):
        try:
            if step:
                state = solver.step(state, dt)
            row = diagnostics(solver, state, solver.recover(state), False, pinch)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure = {'time': step*dt, 'reasons': [str(exc)]}
            break
        checks = screen(solver, row, pinch[2])
        rows.append({**row, **checks})
        if checks['reasons']:
            failure = {'time': float(state.time), 'reasons': checks['reasons']}
            break
    print(seed, background, len(rows), failure, rows[-1]['relative_pinch_distance_phi_v'], flush=True)
    return {'initial_seed_phi_v_pi': seed, 'initial_n_d_eps': background, 'dt': dt, 'failure': failure, 'rows': rows,
            'minimum_distance_phi_v': min(row['relative_pinch_distance_phi_v'] for row in rows),
            'minimum_sampled_convexity_margin': min(row['convexity_margin'] for row in rows),
            'minimum_sampled_longitudinal_pinch_pair_separation': min(row['longitudinal_pinch_pair_separation'] for row in rows),
            'maximum_sampled_imaginary': max(row['maximum_imaginary'] for row in rows),
            'maximum_sampled_finite_basis_condition': max(row['maximum_finite_basis_condition'] for row in rows),
            'energy_balance_error': max(abs(row['combined_energy'] - rows[0]['combined_energy']) for row in rows),
            'entropy_change': rows[-1]['entropy_integral'] - rows[0]['entropy_integral']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thermodynamic-sweep', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    path = folder / 'hadamard_pinch_summary.json'
    summary = json.loads(path.read_text())
    pinch = summary['pinch_phi_v_speed']
    if args.thermodynamic_sweep:
        perturbations = [(1.05, .2375, .285), (.95, .2625, .315), (1.05, .2625, .285), (.95, .2375, .315)]
        seeds = [(pinch[0], .44, .25), (pinch[0], .44, .25), (.015, .4, .23), (.015, .4, .23)]
        summary['autonomous_shooting']['thermodynamic_sweep'] = [
            run(seed, pinch, background) for seed, background in zip(seeds, perturbations)
        ]
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return
    seeds = [(pinch[0], .43, pi) for pi in (-.25, 0, .25)]
    seeds.extend([(.02, .32, -.25), (.02, .32, .25), (.03, .22, .45), (pinch[0], .49, .25)])
    runs = [run(seed, pinch) for seed in seeds]
    summary['autonomous_shooting'] = {
        'command': 'python -B experiments/hadamard_pinch/run_transit_shooting.py',
        'grid': [4], 'length': 1.0, 'geometry': 'fixed Minkowski', 'drag': 'native', 'scope': 'Local exploratory shooting',
        'entered_phi_v_radius_0.01': sum(result['minimum_distance_phi_v'] <= .01 for result in runs),
        'runs': runs,
    }
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
