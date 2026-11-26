"""Extend the static audit with entropy and continue the local pinch manifold."""

from itertools import product
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import brentq

from validate_pinch import D, INTERNAL, N, PARAMETERS, native_pencil, pinch_equations, state_sample
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory33 import CarrierPrimitive
from refine_directional_collision import RichardsonMaster


BACKGROUND = np.array([N, D, INTERNAL])
STATE_FIELDS = ('n', 'd', 'eps', 'phi', 'v_rel', 'lambda_star')
DIRECTION_FIELDS = ('max_imag', 'max_modulus', 'material_condition', 'material_rank', 'A0_condition')


class EntropyPencil:
    """Augment the native symbol with specific entropy in a collinear rest frame."""

    def __init__(self, master, background, phi, v):
        n, d, eps = background
        self.master = master
        self.v = v
        self.native = [master.coupled_characteristic_audit(
            n, eps, (0, 0, 0), d, (v, 0, 0), phi, direction,
        ) for direction in np.eye(3)]
        self.time = np.eye(26)
        self.time[:25, :25] = self.native[0].time_matrix
        # At fixed currents, entropy changes only the ordinary chemical
        # potential. In the ordinary rest frame d_s(mu_i)=0 and
        # d_s(mu_0)=-gamma*T. The vortical momentum RHS is n*T*grad(s).
        temperature = (master.parameters.gamma_ad - 1)*eps
        self.entropy_force = n*(master.parameters.gamma_ad - 1)*temperature
        self.space_axes = np.stack([audit.space_matrix[:8, :9] for audit in self.native])

    def matrices(self, direction, reduced=True):
        time = self.time.copy()
        space = np.zeros_like(time)
        space[:25, :25] = self.native[0].space_matrix
        space[:8, :9] = np.einsum('i,ijk->jk', direction, self.space_axes)
        space[1:4, 25] = self.entropy_force*direction
        # In the matter rest frame the entropy transport equation is s_t=0.
        if reduced:
            time[:8, 8] = space[:8, 8] = 0
        return time, space

    def material(self, direction):
        time, space = self.matrices(direction)
        material = np.r_[0:8, 25]
        return time[np.ix_(material, material)], space[np.ix_(material, material)]

    def collision(self, bracket=(58, 71)):
        """Find the material acoustic root crossing the stationary modes."""
        def speed(angle):
            direction = normal(angle)
            time, space = self.material(direction)
            roots = list(eigvals(space[:8, :8], time[:8, :8]))
            carrier_speed = self.v*direction[0]
            for advective_speed in (0, 0, carrier_speed, carrier_speed):
                roots.pop(min(range(len(roots)), key=lambda i: abs(roots[i] - advective_speed)))
            return float(sorted(roots, key=lambda root: root.real)[1].real)

        endpoints = [speed(angle) for angle in bracket]
        if endpoints[0]*endpoints[1] >= 0:
            return {'bracket_speeds': endpoints, 'angle_degrees': None, 'defective': None}
        angle = brentq(speed, *bracket, xtol=1e-11)
        direction = normal(angle)
        time_mat, space_mat = self.material(direction)
        time, space = self.matrices(direction)
        pencils = {
            'frozen_material': (time_mat[:8, :8], space_mat[:8, :8]),
            'entropy_material': (time_mat, space_mat),
            'entropy_scalar_vector_reduced': (time, space),
        }
        counts = {}
        for name, (a0, a) in pencils.items():
            roots = eigvals(a, a0)
            singular = np.linalg.svd(np.linalg.solve(a0, a), compute_uv=False)
            counts[name] = {
                'cluster_size_1e-6': int(np.count_nonzero(abs(roots) < 1e-6)),
                'cluster_size_1e-7': int(np.count_nonzero(abs(roots) < 1e-7)),
                'svd_nullity_1e-8': int(np.count_nonzero(singular < 1e-8)),
                'singular_values': singular.tolist(),
            }
        full = counts['entropy_scalar_vector_reduced']
        return {'bracket_speeds': endpoints, 'angle_degrees': angle, 'counts': counts,
                'defective': full['cluster_size_1e-6'] > full['svd_nullity_1e-8']}


def normal(angle):
    theta = np.deg2rad(angle)
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def directional_check(solver, summary, old_scan, output):
    master = solver.master
    values = np.empty((len(old_scan['states']), len(old_scan['angle_degrees']), len(DIRECTION_FIELDS)))
    classifications = []
    for i, (name, (phi, v)) in enumerate(zip(old_scan['states'], old_scan['phi_v'])):
        pencil = EntropyPencil(master, BACKGROUND, phi, v)
        for j, angle in enumerate(old_scan['angle_degrees']):
            time, space = pencil.material(normal(angle))
            roots = eigvals(space, time)
            _, rank, condition, _ = master._clustered_eigenbasis(np.linalg.solve(time, space))
            values[i, j] = max(abs(roots.imag)), max(1, max(abs(roots))), condition, rank, np.linalg.cond(time)
        collision = pencil.collision()
        recovery = state_sample(solver, float(phi), float(v))
        max_imag = float(np.max(values[i, :, 0]))
        finite = np.isfinite(values[i, :, 2])
        audit = pencil.native[0]
        classifications.append({
            'state': str(name), 'collision': collision, 'max_imag': max_imag,
            'REAL_SPECTRUM_at_sampled_angles': max_imag <= 1e-8,
            'COMPLETE_EIGENBASIS': False if collision['defective'] else None,
            'DIRECTIONAL_CONDITIONING_bounded': False if collision['defective'] else None,
            'maximum_finite_sampled_material_condition': float(np.max(values[i, finite, 2])) if finite.any() else None,
            'CAUSAL_SPEEDS_at_sampled_angles': bool(np.max(values[i, :, 1]) <= 1 + 1e-8),
            'LEGENDRE_POSITIVITY': audit.legendre_minimum_eigenvalue > PARAMETERS.convexity_floor,
            'THERMODYNAMIC_CONVEXITY': audit.thermodynamic_minimum_eigenvalue > PARAMETERS.convexity_floor,
            'PRIMITIVE_RECOVERABILITY': recovery['primitive_recovery_failed_cells'] == 0,
            'primitive_recovery_residual': recovery['primitive_recovery_residual'],
            'FULL_ADMISSIBILITY': False if collision['defective'] or max_imag > 1e-8 else None,
        })
    # The scalar and two Proca principal blocks are canonical direct summands
    # after the constraint reduction. They add only speeds -1, 0, and +1.
    phi, v, _ = summary['pinch_phi_v_speed']
    pencil = EntropyPencil(master, BACKGROUND, phi, v)
    native_time, native_space = native_pencil(master, phi, v)
    roots = eigvals(native_space, native_time)
    a0, a = pencil.material(normal(0))
    material_roots = eigvals(a, a0)
    longitudinal_error = max(min(abs(value - material_roots)) for value in roots)
    direction = normal(37)
    a0, a = pencil.matrices(direction, reduced=False)
    native = master.coupled_characteristic_audit(N, INTERNAL, (0, 0, 0), D, (v, 0, 0), phi, direction)
    native_error = max(np.max(abs(a0[:25, :25] - native.time_matrix)),
                       np.max(abs(a[:25, :25] - native.space_matrix)))
    np.savez_compressed(output / 'hadamard_pinch_entropy_directional_scan.npz', states=old_scan['states'],
                        phi_v=old_scan['phi_v'], angle_degrees=old_scan['angle_degrees'],
                        columns=DIRECTION_FIELDS, values=values)
    return {'variables': 26, 'longitudinal_spectrum_crosscheck_error': float(longitudinal_error),
            'native_frozen_subpencil_crosscheck_error': float(native_error),
            'classifications': classifications}


def continue_pinch(master, summary, output):
    with mp.workdps(50):
        base = tuple(map(mp.mpf, ('1', '.25', '.30')))
        seed = tuple(map(mp.mpf, map(str, summary['pinch_phi_v_speed'])))

        def solve(relative, guess):
            bg = [value*(1 + mp.mpf(str(delta))) for value, delta in zip(base, relative)]
            root = mp.findroot(lambda *x: pinch_equations(*x, background=bg), tuple(guess))
            residual = max(abs(value) for value in pinch_equations(*root, background=bg))
            return bg, root, residual

        _, center, _ = solve((0, 0, 0), seed)
        combined = list(center) + list(base)

        def partial(row, column):
            def equation(value):
                point = combined.copy()
                point[column] = value
                return pinch_equations(*point[:3], background=point[3:])[row]
            return float(mp.diff(equation, combined[column]))

        jacobian = np.array([[partial(row, column) for column in range(6)] for row in range(3)])
        tangent = -np.linalg.solve(jacobian[:, :3], jacobian[:, 3:])
        relative_states = [np.zeros(3)]
        for axis in np.eye(3):
            for magnitude in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
                relative_states.extend([axis*magnitude, -axis*magnitude])
        relative_states.extend(np.array(corner)*0.01 for corner in product((-1, 1), repeat=3))
        states, collisions, residuals = [], [], []
        for relative in relative_states:
            bg, root, residual = solve(relative, center)
            state = [float(value) for value in [*bg, *root]]
            states.append(state)
            collisions.append(EntropyPencil(master, state[:3], *state[3:5]).collision())
            residuals.append(float(residual))
    print(f'Continued {len(states)} thermodynamic states', flush=True)
    offsets = list(product((-1e-3, -5e-4, 0, 5e-4, 1e-3), (-0.01, -0.005, 0, 0.005, 0.01)))
    neighborhood = []
    for i in [0, *range(len(states) - 8, len(states))]:
        state = states[i]
        for delta_phi, delta_v in offsets:
            phi, v = state[3] + delta_phi, state[4] + delta_v
            result = EntropyPencil(master, state[:3], phi, v).collision()
            neighborhood.append({'anchor': i, 'delta_phi': delta_phi, 'delta_v': delta_v, 'phi': phi,
                                 'v_rel': v, **result})
    print(f'Audited the oblique collision in {len(neighborhood)} neighboring states', flush=True)
    np.savez_compressed(output / 'hadamard_pinch_continuation.npz', states=states, state_fields=STATE_FIELDS,
                        relative_background_offsets=relative_states, residuals=residuals,
                        collision_angles_degrees=[row['angle_degrees'] for row in collisions],
                        pinch_jacobian=jacobian, pinch_tangent=tangent,
                        neighborhood=np.array([[row[key] for key in ('anchor', 'delta_phi', 'delta_v', 'phi', 'v_rel', 'angle_degrees')]
                                               for row in neighborhood], dtype=float),
                        neighborhood_fields=('anchor', 'delta_phi', 'delta_v', 'phi', 'v_rel', 'angle_degrees'))
    return {
        'states': [{'state': dict(zip(STATE_FIELDS, state)), 'relative_background_offset': relative.tolist(),
                    'pinch_residual': residual, 'collision': collision}
                   for state, relative, residual, collision in zip(states, relative_states, residuals, collisions)],
        'pinch_jacobian_column_order': ['phi', 'v_rel', 'lambda_star', 'n', 'd', 'eps'],
        'pinch_jacobian': jacobian.tolist(),
        'pinch_jacobian_rank': int(np.linalg.matrix_rank(jacobian)),
        'pinch_jacobian_singular_values': np.linalg.svd(jacobian, compute_uv=False).tolist(),
        'fixed_background_jacobian_singular_values': np.linalg.svd(jacobian[:, :3], compute_uv=False).tolist(),
        'pinch_tangent_d_phi_v_speed_by_n_d_eps': tangent.tolist(),
        'neighborhood_samples': neighborhood,
    }


def frozen_control(solver, summary):
    """Record the homogeneous initial state without taking a time step."""
    phi, v, _ = summary['pinch_phi_v_speed']
    grid = solver.grid
    fluid = FluidPrimitive(np.full(grid.shape, N), np.full(grid.shape, INTERNAL),
                           grid.zeros((3,)), grid.zeros())
    velocity = grid.zeros((3,))
    velocity[0] = v
    carrier = CarrierPrimitive(np.full(grid.shape, D), velocity)
    state = solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=phi)
    recovered = solver.recover(state)
    stress = recovered.total_stress
    hamiltonian, momentum = solver.ccz4.constraints(
        state.geometry, (stress.rho, stress.momentum, stress.stress),
    )
    return {
        'classification': 'E1 frozen homogeneous initial-state control',
        'time_steps': 0,
        'scalar_source_initial_acceleration': -float(recovered.master.scalar_equation_source[0]),
        'native_full_recovery_residual': recovered.report.maximum_residual,
        'flat_with_sources_hamiltonian_constraint_max_abs': float(np.max(abs(hamiltonian))),
        'flat_with_sources_momentum_constraint_max_abs': float(np.max(abs(momentum))),
        'production_transit_initial_state_admissible': False,
    }


def main():
    output = Path(__file__).resolve().parent
    path = output / 'hadamard_pinch_summary.json'
    summary = json.loads(path.read_text())
    solver = NullothProductionSolver(PeriodicGrid((4,), (1.0,)), PARAMETERS)
    master = solver.master
    print('Auditing the 26-variable entropy, scalar, and vector pencil', flush=True)
    with np.load(output / 'hadamard_pinch_directional_scan.npz') as scan:
        entropy = directional_check(solver, summary, scan, output)
    alternative = RichardsonMaster(PARAMETERS)
    entropy['pinch_richardson_collision'] = EntropyPencil(
        alternative, BACKGROUND, *summary['pinch_phi_v_speed'][:2],
    ).collision()
    summary['entropy_directional_audit'] = entropy
    summary['constitutive_continuation'] = continue_pinch(master, summary, output)
    summary['homogeneous_frozen_control'] = frozen_control(solver, summary)
    summary['scope'] = 'Static reproduction, entropy-inclusive directional audit, and local constitutive continuation. No time evolution.'
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
