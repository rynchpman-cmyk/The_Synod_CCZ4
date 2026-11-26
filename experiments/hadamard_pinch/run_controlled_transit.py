"""Test homogeneous transit with budgeted reversible carrier and scalar impulses."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from run_homogeneous_departure import diagnostics, initial_state
from run_transverse_tangent import build_solver
from run_transit_shooting import screen
from validate_pinch import D, INTERNAL, N
from causal_repair.grid import PeriodicGrid
from causal_repair.theory33 import CarrierPrimitive


def impulse(solver, state, elapsed, carrier_rate, scalar_rate):
    """Kick carrier proper velocity and scalar momentum at fixed lab counts.

    The ordinary primitive state, specific entropy, scalar coordinate and
    vector fields are held during the kick. Changes of energy, total momentum
    and carrier canonical momentum are the applied external work and impulse.
    """
    recovered = solver.recover(state)
    h, _ = solver.ccz4.physical_geometry(state.geometry)
    old_velocity = recovered.carrier.velocity
    q = old_velocity/np.sqrt(1-np.sum(old_velocity**2, axis=0))
    q[0] += elapsed*carrier_rate
    boost = np.sqrt(1+np.sum(q**2, axis=0))
    carrier_number = state.target_charge/solver.master_parameters.carrier_charge
    carrier = CarrierPrimitive(carrier_number/boost, q/boost)
    material = solver.master.evaluate(h, recovered.fluid, carrier, state.scalar.varphi)
    energy_material = material.stress.rho - recovered.master.stress.rho
    momentum = material.stress.momentum - recovered.master.stress.momentum
    pi = state.scalar.momentum + elapsed*scalar_rate
    energy_scalar = .5*(pi**2-state.scalar.momentum**2)
    _, canonical = solver.master.conserved_carrier(h, material)
    state = replace(state, matter=replace(state.matter,
                                          energy=state.matter.energy + energy_material + energy_scalar,
                                          momentum=state.matter.momentum + momentum),
                    scalar=replace(state.scalar, momentum=pi), target_current=canonical)
    budget = np.array([solver.grid.integrate(energy_material), solver.grid.integrate(energy_scalar),
                       solver.grid.integrate(momentum[0]), 0.0])
    return state, budget


def run(case, pinch):
    label, phi, v, pi, carrier_rate, scalar_rate, dt, background = case
    solver = build_solver(PeriodicGrid((4,), (1.0,)))
    state = initial_state(solver, phi, v, False, scalar_momentum=pi, background=background)
    rows, cumulative, failure = [], np.zeros(4), None

    def inspect(time, stage):
        row = diagnostics(solver, state, solver.recover(state), False, pinch)
        checks = screen(solver, row, pinch[2])
        row.update(checks, control_time=time, stage=stage, applied_work_carrier=cumulative[0],
                   applied_work_scalar=cumulative[1], applied_momentum_x=cumulative[2],
                   injected_carrier_number=cumulative[3])
        baseline = rows[0] if rows else row
        row['energy_balance_error'] = row['combined_energy']-baseline['combined_energy']-sum(cumulative[:2])
        row['momentum_balance_error'] = row['total_momentum_x']-baseline['total_momentum_x']-cumulative[2]
        row['carrier_balance_error'] = row['carrier_number']-baseline['carrier_number']-cumulative[3]
        rows.append(row)
        return {'time': time, 'stage': stage, 'reason': checks['reasons']} if checks['reasons'] else None

    failure = inspect(0, 'initial')
    for index in range(round(.06/dt)):
        if failure:
            break
        time = index*dt
        stages = ('first half impulse', 'native evolution', 'second half impulse')
        stage_times = (time, time+dt, time+dt)
        try:
            for stage, stage_time in zip(stages, stage_times):
                if stage == 'native evolution':
                    state = solver.step(state, dt)
                else:
                    state, increment = impulse(solver, state, .5*dt, carrier_rate, scalar_rate)
                    cumulative += increment
                failure = inspect(stage_time, stage)
                if failure:
                    break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            failure = {'time': stage_time, 'stage': stage, 'reason': [str(exc)]}
    admissible = [row for row in rows if not row['reasons']]
    result = {'label': label, 'initial_phi_v_pi': [phi, v, pi], 'background_n_d_eps': background,
              'rates_carrier_q_and_scalar_pi': [carrier_rate, scalar_rate], 'dt': dt, 'failure': failure,
              'minimum_screened_pinch_distance': min((row['relative_pinch_distance_phi_v'] for row in admissible), default=None),
              'maximum_energy_balance_error': max(abs(row['energy_balance_error']) for row in rows),
              'maximum_momentum_balance_error': max(abs(row['momentum_balance_error']) for row in rows),
              'maximum_carrier_balance_error': max(abs(row['carrier_balance_error']) for row in rows), 'rows': rows}
    print(label, failure, result['minimum_screened_pinch_distance'], result['maximum_energy_balance_error'], flush=True)
    return result


def main():
    folder = Path(__file__).resolve().parent
    pinch = json.loads((folder/'hadamard_pinch_summary.json').read_text())['pinch_phi_v_speed']
    base = (N,D,INTERNAL)
    cases = [
        ('carrier_weak', pinch[0], .43, .05, 3., 0., .001, base),
        ('carrier_strong', pinch[0], .43, .05, 12., 0., .001, base),
        ('carrier_strong_refined', pinch[0], .43, .05, 12., 0., .0005, base),
        ('scalar_redirect', .02, .32, .10, 7., 4., .001, base),
        ('reverse_scalar', .02, .32, 0., 7., -3., .001, base),
        ('warm_redirect', pinch[0], .43, .1, 8., 3., .001, (.95,.2625,.315)),
        ('warm_redirect_refined', pinch[0], .43, .1, 8., 3., .0005, (.95,.2625,.315)),
        ('cool_redirect', .012, .42, .1, 8., 3., .001, (1.05,.2375,.285)),
    ]
    results = [run(case, pinch) for case in cases]
    (folder/'hadamard_pinch_controlled_transit.json').write_text(json.dumps({
        'geometry': 'fixed Minkowski', 'native_drag': True, 'duration': .06,
        'actuator': 'split reversible carrier proper-velocity and canonical scalar momentum impulses', 'cases': results,
    }, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
