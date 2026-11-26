"""Run the pinch state through the native homogeneous Nulloth step."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import brentq

from audit_transit_corridor import EntropyPencil, normal
from validate_pinch import D, INTERNAL, N, PARAMETERS
from causal_repair.adm import inverse_metric
from causal_repair.ccz4 import CCZ4Parameters, CCZ4Solver
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory33 import CarrierPrimitive


class FixedMinkowski(CCZ4Solver):
    """Hold the geometry constant at every native Runge–Kutta stage."""

    def rhs(self, time, values, matter=None):
        return tuple(np.zeros_like(value) for value in values)


def initial_state(solver, phi, relative_speed, constrained, scalar_momentum=0.0, background=(N, D, INTERNAL)):
    grid = solver.grid
    n, d, internal = background
    geometry = solver.ccz4.flat_state()
    h, _ = solver.ccz4.physical_geometry(geometry)

    def primitives(u):
        v = (relative_speed + u)/(1 + relative_speed*u)
        u_field, v_field = grid.zeros((3,)), grid.zeros((3,))
        u_field[0], v_field[0] = u, v
        fluid = FluidPrimitive(np.full(grid.shape, n), np.full(grid.shape, internal), u_field, grid.zeros())
        return fluid, CarrierPrimitive(np.full(grid.shape, d), v_field)

    if constrained:
        # The initial vector electric, magnetic and spatial potential fields
        # vanish. Native Gauss initialization supplies the charged normal
        # potentials. With zero scalar momentum, only matter carries momentum.
        def frame_momentum(u):
            fluid, carrier = primitives(u)
            result = solver.master.evaluate(h, fluid, carrier, np.full(grid.shape, phi))
            return float(result.stress.momentum[0, 0])

        u = brentq(frame_momentum, -relative_speed, 0, xtol=1e-15)
    else:
        u = 0.0
    fluid, carrier = primitives(u)
    state = solver.initialize(geometry, fluid, carrier=carrier, varphi=phi, scalar_momentum=scalar_momentum)
    if constrained:
        # Flat spatial torus, isotropic initial K_ij and zero total current
        # give H = (2/3) K^2 - 2 kappa rho and a zero momentum constraint.
        rho = solver.recover(state).total_stress.rho
        state.geometry.trace_K[:] = -np.sqrt(3*solver.ccz4.parameters.kappa_gravity*rho)
        checks = solver.diagnostics(state)
        if max(checks[key] for key in ('hamiltonian_l2', 'momentum_l2', 'gauss_A_l2', 'gauss_B_l2')) > 1e-10:
            raise FloatingPointError('Initial CCZ4 or Gauss constraints failed')
    return state


def diagnostics(solver, state, recovery, full, pinch):
    n = float(recovery.fluid.baryon_density[0])
    d = float(recovery.carrier.number_density[0])
    eps = float(recovery.fluid.specific_internal_energy[0])
    phi = float(state.scalar.varphi[0])
    pi = float(state.scalar.momentum[0])
    h, k = solver.ccz4.physical_geometry(state.geometry)
    a = float(np.sqrt(h[0, 0, 0]))
    u = float(recovery.fluid.velocity[0, 0])*a
    v = float(recovery.carrier.velocity[0, 0])*a
    relative = (v - u)/(1 - u*v)
    stress = recovery.total_stress
    h_inv, _, sqrt_h = inverse_metric(h)
    stress_up = np.einsum('ik...,jl...,kl...->ij...', h_inv, h_inv, stress.stress)
    work = sqrt_h*state.geometry.lapse*np.einsum('ij...,ij...->...', k, stress_up)
    scalar_acceleration = -state.geometry.lapse[0]**2*(
        recovery.master.scalar_equation_source[0] + state.geometry.trace_K[0]*pi
    )
    row = {
        **solver.diagnostics(state), 'n': n, 'd': d, 'eps': eps, 'phi': phi,
        'scalar_momentum': pi, 'scalar_acceleration_proper_time': float(-recovery.master.scalar_equation_source[0] - state.geometry.trace_K[0]*pi),
        'scalar_acceleration_coordinate_without_lapse_derivative': float(scalar_acceleration),
        'v_N_orthonormal': u, 'v_D_orthonormal': v, 'v_rel': relative,
        'relative_pinch_distance_phi_v': float(np.hypot(phi - pinch[0], relative - pinch[1])),
        'Q': float(recovery.master.entrainment[0]), 'C': float(recovery.master.coupling_coefficient[0]),
        'total_momentum_x': solver.grid.integrate(state.matter.momentum[0]),
        'geometry_energy_work_rate': solver.grid.integrate(work),
        'spatial_stress_orthonormal_xx': float(stress.stress[0, 0, 0]/h[0, 0, 0]),
        'volume_factor': float(sqrt_h[0]), 'lapse': float(state.geometry.lapse[0]),
        'vector_spatial_max': float(max(np.max(abs(state.a)), np.max(abs(state.b)))),
    }
    if full:
        pencil = EntropyPencil(solver.master, (n, d, eps), phi, relative)
        collision = pencil.collision((35, 78))
        angular_max_imag, angular_max_modulus, max_finite_cond = 0.0, 0.0, 0.0
        for angle in np.linspace(0, 90, 181):
            time, space = pencil.material(normal(angle))
            roots = eigvals(space, time)
            _, _, condition, _ = solver.master._clustered_eigenbasis(np.linalg.solve(time, space))
            angular_max_imag = max(angular_max_imag, max(abs(roots.imag)))
            angular_max_modulus = max(angular_max_modulus, max(abs(roots)))
            if np.isfinite(condition):
                max_finite_cond = max(max_finite_cond, condition)
            if angle == 0:
                closest = sorted(roots, key=lambda root: abs(root - pinch[2]))[:2]
                row['longitudinal_pinch_pair_separation'] = float(abs(closest[0] - closest[1]))
                row['longitudinal_max_imag'] = float(max(abs(roots.imag)))
        row.update(max_directional_imag=float(angular_max_imag), max_characteristic_modulus=float(max(1, angular_max_modulus)),
                   sampled_max_finite_material_eigenbasis_condition=float(max_finite_cond),
                   oblique_collision_angle_degrees=collision['angle_degrees'], oblique_defect=collision['defective'])
    return row


def run(label, pinch, dt, steps, gravity, drag):
    constrained = gravity is not None
    grid = PeriodicGrid((4,), (1.0,))
    parameters = PARAMETERS if drag else replace(PARAMETERS, relaxation_time=float('inf'))
    gravity_parameters = CCZ4Parameters() if gravity in (None, 'default') else CCZ4Parameters(kappa_gravity=gravity)
    solver = NullothProductionSolver(grid, parameters, ccz4=gravity_parameters)
    if not constrained:
        solver.ccz4 = FixedMinkowski(grid, gravity_parameters)
    state = initial_state(solver, pinch[0], pinch[1], constrained)
    rows = []
    failure = None
    cumulative_work = cumulative_drag_heat = 0.0
    previous = None
    for index in range(steps + 1):
        if index:
            try:
                state = solver.step(state, dt)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                failure = {'attempted_time': index*dt, 'reason': str(exc)}
                break
        sampled = index in (0, 1, 2, steps) or index % max(1, round(.005/dt)) == 0
        row = diagnostics(solver, state, solver.recover(state), sampled, pinch)
        if previous is not None:
            cumulative_work += dt*(previous['geometry_energy_work_rate'] + row['geometry_energy_work_rate'])/2
            cumulative_drag_heat += row['drag_heat']
        row['cumulative_drag_heat'] = cumulative_drag_heat
        row['geometry_energy_work_trapezoidal'] = cumulative_work
        row['energy_balance_error'] = row['combined_energy'] - (rows[0]['combined_energy'] if rows else row['combined_energy']) - cumulative_work
        rows.append(row)
        previous = row
    print(label, dt, len(rows)-1, failure or 'completed', flush=True)
    return {'label': label, 'dt': dt, 'steps_requested': steps, 'rows': rows, 'failure': failure,
            'geometry': 'CCZ4' if constrained else 'fixed Minkowski',
            'actual_kappa': gravity_parameters.kappa_gravity if constrained else 0,
            'drag_relaxation': PARAMETERS.relaxation_time if drag else 'infinite', 'charged_vectors_evolved': True}


def refine_history(directory, summary, summary_path):
    """Re-evaluate every accepted state without replaying the evolution."""
    path = directory / 'hadamard_pinch_dynamics.npz'
    with np.load(path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    columns = list(arrays['columns'])
    column = {name: columns.index(name) for name in columns}
    master = NullothMasterFunction(PARAMETERS)
    early_angles = np.linspace(0, 10, 501)
    for run in summary['homogeneous_dynamics']['runs']:
        rows = arrays[run['array_key']]
        peak = {'max_imag': 0.0, 'time': 0.0, 'angle': 0.0}
        for row in rows:
            read = lambda name: row[column[name]]
            background = tuple(read(name) for name in ('n', 'd', 'eps'))
            time = read('time')
            pencil = EntropyPencil(master, background, read('phi'), read('v_rel'))
            collision = pencil.collision((35, 78))
            row[column['oblique_collision_angle_degrees']] = np.nan if collision['angle_degrees'] is None else collision['angle_degrees']
            row[column['oblique_defect']] = np.nan if collision['defective'] is None else float(collision['defective'])
            if time <= .005 + 1e-12:
                for angle in early_angles:
                    a0, a = pencil.material(normal(angle))
                    imag = float(max(abs(eigvals(a, a0).imag)))
                    if imag > peak['max_imag']:
                        peak = {'max_imag': imag, 'time': float(time), 'angle': float(angle)}
        run['adaptive_history'] = {
            'defective_accepted_states': int(np.count_nonzero(rows[:, column['oblique_defect']] == 1)),
            'states_evaluated': len(rows), 'early_directional_peak': peak,
        }
        for name, index in [('initial', 0), ('final', -1)]:
            for field in ('oblique_collision_angle_degrees', 'oblique_defect'):
                value = rows[index, column[field]]
                run[name][field] = (bool(value) if field == 'oblique_defect' else float(value)) if np.isfinite(value) else None
    np.savez_compressed(path, **arrays)
    summary['scope'] = 'Static reproduction, continuation, and native homogeneous trajectories at the obstructed pinch.'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reaudit', action='store_true', help='Refine the directional analysis of saved trajectories')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    summary_path = directory / 'hadamard_pinch_summary.json'
    summary = json.loads(summary_path.read_text())
    if args.reaudit:
        refine_history(directory, summary, summary_path)
        return
    pinch = summary['pinch_phi_v_speed']
    cases = [('fixed_no_drag', None, False), ('fixed_native_drag', None, True),
             ('ccz4_weak_gravity_native_drag', 1e-3, True), ('ccz4_default_gravity_native_drag', 'default', True)]
    runs = []
    for label, gravity, drag in cases:
        for dt in (.001, .0005):
            runs.append(run(label, pinch, dt, round(.05/dt), gravity, drag))
    columns = sorted(set().union(*(row.keys() for result in runs for row in result['rows'])))
    arrays = {f'run_{i}': np.array([[row.get(key, np.nan) if row.get(key) is not None else np.nan
                                   for key in columns] for row in result['rows']]) for i, result in enumerate(runs)}
    np.savez_compressed(directory / 'hadamard_pinch_dynamics.npz', columns=columns, **arrays)
    summary['homogeneous_dynamics'] = {'scope': 'Native homogeneous trajectories at the obstructed pinch',
                                       'command': 'python -B experiments/hadamard_pinch/run_homogeneous_departure.py',
                                       'runs': [{**{key: value for key, value in result.items() if key != 'rows'},
                                                 'array_key': f'run_{i}', 'initial': result['rows'][0], 'final': result['rows'][-1],
                                                 'maximum_energy_balance_error': max(abs(row['energy_balance_error']) for row in result['rows'])}
                                                for i, result in enumerate(runs)]}
    summary['scope'] = 'Static reproduction, continuation, and native homogeneous trajectories at the obstructed pinch.'
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
