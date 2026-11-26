"""Reproduce the static Nulloth pinch without changing the production solver."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import eig, eigvals
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothParameters
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory33 import CarrierPrimitive


PARAMETERS = NullothParameters(
    carrier_K=1.2, carrier_sound_speed=0.45, target_fraction=0.25,
    chemical_susceptibility=6.0, entrainment_linear=0.04,
    entrainment_quadratic=0.01, scalar_baryon_coupling=0.20,
    scalar_carrier_coupling=0.20, scalar_counterflow_coupling=10.0,
    scalar_carrier_scale=0.25, scalar_counterflow_scale=0.0025,
    scalar_mass=1.0, scalar_quartic=2500.0,
)
N, D, INTERNAL = 1.0, 0.25, 0.30
SEED = ("0.01088444356899034629", "0.760884407527245", "0.471084089380828")
OFFSETS = ("1e-3", "1e-4", "1e-5", "1e-6", "1e-8", "1e-10", "1e-12")


def native_pencil(master, phi, v, u=0.0, step=1.0):
    """Include the passive entropy characteristic in the longitudinal pencil.

    The entropy column comes from differentiating the native constitutive
    fluxes. In the ordinary momentum equation, subtract T ds/W_N from
    the spatial side of the isentropic vorticity equation.
    """
    entropy = float(master.specific_entropy(N, INTERNAL))
    point = np.array([np.log(N), np.arctanh(u), np.log(D), np.arctanh(v), entropy])

    def fluxes(values):
        return master._flat_constitutive_vector(values[:4], values[4], phi)

    if step == 1.0:
        jacobian = master._numerical_jacobian(fluxes, point)
    else:
        deltas = np.diag(2e-6 * step * np.maximum(1.0, np.abs(point)))
        jacobian = np.column_stack([
            (fluxes(point + delta) - fluxes(point - delta)) / (2 * delta[i])
            for i, delta in enumerate(deltas)
        ])
    time = np.vstack((jacobian[:4], [0, 0, 0, 0, 1.0]))
    space = np.vstack((jacobian[4:], [0, 0, 0, 0, u]))
    space[1, 4] -= (PARAMETERS.gamma_ad - 1) * INTERNAL * np.sqrt(1 - u*u)
    return time, space


def split_squared(master, phi, v):
    time, space = native_pencil(master, phi, v)
    roots = sorted(eigvals(space[:4, :4], time[:4, :4]), key=lambda root: root.real)
    return float(((roots[2] - roots[1]) ** 2).real)


def acoustic_hessian(phi, v, background=('1', '.25', '.30')):
    """Differentiate Lambda via (n, d, W) and the two contravariant currents."""
    n, d, eps = map(mp.mpf, background)
    gamma = mp.mpf(5) / 3
    p = {key: mp.mpf(str(value)) for key, value in asdict(PARAMETERS).items()}
    power = p['carrier_sound_speed']**2
    anchor = d - p['target_fraction'] * n
    chi = p['chemical_susceptibility']
    c_n = p['scalar_baryon_coupling'] / p['scalar_baryon_scale']**2
    c_d = p['scalar_carrier_coupling'] / p['scalar_carrier_scale']**2
    c_w = p['scalar_counterflow_coupling'] / p['scalar_counterflow_scale']
    a2 = p['entrainment_quadratic'] / p['entrainment_scale_fourth']
    boost = 1 / mp.sqrt(1 - v*v)
    rho_n = 1 + gamma*eps - p['target_fraction']*anchor/chi - c_n*n*phi**2
    rho_d = p['carrier_K']*(1 + power)*d**power + anchor/chi - c_d*d*phi**2
    q = p['entrainment_linear'] + a2*n*d*(boost - 1) - c_w*phi**2/2
    rho_nn = gamma*(gamma - 1)*eps/n + p['target_fraction']**2/chi - c_n*phi**2
    rho_dd = p['carrier_K']*(1 + power)*power*d**(power - 1) + 1/chi - c_d*phi**2
    invariant_hessian = mp.matrix([
        [-rho_nn, p['target_fraction']/chi, 0],
        [p['target_fraction']/chi, -rho_dd, 0],
        [0, 0, -a2],
    ])
    dn = mp.matrix([1, 0, 0, 0])
    dd = mp.matrix([0, 0, boost, -boost*v])
    dx = mp.matrix([d*boost, -d*boost*v, n, 0])
    dw = dx - d*dn - n*dd
    jacobian = mp.matrix([list(dn), list(dd), list(dw)])
    hn = (mp.diag([1, -1, 0, 0]) - dn*dn.T)/n
    hd = (mp.diag([0, 0, 1, -1]) - dd*dd.T)/d
    hx = mp.matrix([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])
    hw = hx - d*hn - n*hd - dn*dd.T - dd*dn.T
    return jacobian.T*invariant_hessian*jacobian - rho_n*hn - rho_d*hd - q*hw


def acoustic(hessian, speed):
    # Current conservation fixes delta J^x = speed * delta J^t.
    basis = mp.matrix([[1, 0], [speed, 0], [0, 1], [0, speed]])
    return basis.T*hessian*basis


def pinch_equations(phi, v, speed, background=('1', '.25', '.30')):
    pencil = acoustic(acoustic_hessian(phi, v, background), speed)
    return pencil[0, 0], pencil[1, 1], pencil[0, 1]


def det2(matrix):
    return matrix[0, 0]*matrix[1, 1] - matrix[0, 1]*matrix[1, 0]


def gap_boundary(phi, v0, speed0, delta, v_slope, speed_slope):
    """Solve for a repeated root in coordinates scaled by the offset."""
    def residual(v_shift, speed_shift):
        hessian = acoustic_hessian(phi, v0 + delta*v_shift)
        speed = speed0 + delta*speed_shift

        def determinant(value):
            return det2(acoustic(hessian, value))

        return determinant(speed)/delta**2, mp.diff(determinant, speed)/delta

    shift = mp.findroot(residual, (v_slope, speed_slope), tol=mp.mpf('1e-55'))
    return v0 + delta*shift[0], speed0 + delta*shift[1], shift, residual(*shift)


def high_precision():
    ladder = []
    for digits in (30, 50, 80, 100):
        with mp.workdps(digits):
            root = mp.findroot(pinch_equations, tuple(map(mp.mpf, SEED)))
            ladder.append({
                'decimal_digits': digits,
                'root_phi_v_speed': [mp.nstr(value, digits) for value in root],
                'acoustic_entries': [mp.nstr(value, 8) for value in pinch_equations(*root)],
            })
    with mp.workdps(100):
        phi0, v0, speed0 = root
        hessian0 = acoustic_hessian(phi0, v0)
        det_second = mp.diff(lambda speed: det2(acoustic(hessian0, speed)), speed0, 2)
        # Recover both first-order boundary directions from the linearized pencil.
        jacobian = mp.matrix([
            [mp.diff(lambda x: pinch_equations(*[x if i == axis else root[i] for i in range(3)])[j], root[axis])
             for axis in range(3)] for j in range(3)
        ])

        def linear_det(v_slope, speed_slope):
            nn, dd, nd = jacobian * mp.matrix([1, v_slope, speed_slope])
            return nn*dd - nd**2

        def speed_polynomial(v_slope):
            constant = linear_det(v_slope, 0)
            plus, minus = linear_det(v_slope, 1), linear_det(v_slope, -1)
            return (plus + minus)/2 - constant, (plus - minus)/2, constant

        def discriminant(v_slope):
            a, b, c = speed_polynomial(v_slope)
            return b*b - 4*a*c

        c = discriminant(0)
        plus, minus = discriminant(1), discriminant(-1)
        a, b = (plus + minus)/2 - c, (plus - minus)/2
        slopes = [(-b + sign*mp.sqrt(b*b - 4*a*c))/(2*a) for sign in (-1, 1)]
        seeds = [(v, -speed_polynomial(v)[1]/(2*speed_polynomial(v)[0])) for v in sorted(slopes)]
        gaps = []
        for sign in (-1, 1):
            for offset in OFFSETS:
                delta = sign*mp.mpf(offset)
                phi = phi0 + delta
                edges = sorted((gap_boundary(phi, v0, speed0, delta, *seed) for seed in seeds), key=lambda edge: edge[0])
                low, high = edges[0][0], edges[1][0]
                center = (low + high)/2
                hessian = acoustic_hessian(phi, center)
                center_speed = (edges[0][1] + edges[1][1])/2

                def scaled_determinant(z):
                    return det2(acoustic(hessian, center_speed + abs(delta)*z))/delta**2

                constant = scaled_determinant(0)
                linear = mp.diff(scaled_determinant, 0)
                quadratic = mp.diff(scaled_determinant, 0, 2)/2
                center_seed = (-linear + mp.sqrt(linear**2 - 4*quadratic*constant))/(2*quadratic)
                center_root = center_speed + abs(delta)*mp.findroot(
                    scaled_determinant, center_seed, solver='newton', tol=mp.mpf('1e-65'),
                )
                if abs(center_root.imag) < abs(delta)*mp.mpf('1e-8'):
                    raise ArithmeticError(f"Gap center lacked a complex root at {delta}")
                gaps.append({
                    'delta_phi': mp.nstr(delta, 12),
                    'low': mp.nstr(low, 50), 'high': mp.nstr(high, 50),
                    'center': mp.nstr(center, 50), 'width': mp.nstr(high - low, 35),
                    'width_over_abs_offset': float((high - low)/abs(delta)),
                    'center_imaginary_speed': mp.nstr(abs(center_root.imag), 35),
                    'center_pair_splitting_modulus': mp.nstr(2*abs(center_root.imag), 35),
                    'scaled_boundary_residual_max': mp.nstr(max(abs(r) for edge in edges for r in edge[3]), 8),
                })
        return root, {
            'precision_ladder': ladder,
            'determinant_second_speed_derivative': mp.nstr(det_second, 35),
            'limiting_boundary_slopes': [float(seed[0]) for seed in seeds],
            'limiting_width_slope': float(abs(seeds[0][0] - seeds[1][0])),
            'gap_samples': gaps,
        }


def state_sample(solver, phi, v, u=0.0):
    master = solver.master
    grid = solver.grid
    metric = np.broadcast_to(np.eye(3)[:, :, None], (3, 3, *grid.shape))
    fluid = FluidPrimitive(np.full(grid.shape, N), np.full(grid.shape, INTERNAL),
                           np.broadcast_to([[u], [0.0], [0.0]], (3, *grid.shape)), grid.zeros())
    carrier = CarrierPrimitive(np.full(grid.shape, D), np.broadcast_to([[v], [0.0], [0.0]], (3, *grid.shape)))
    scalar = np.full(grid.shape, phi)
    state = master.evaluate(metric, fluid, carrier, scalar)
    u_proper, v_proper = np.sinh(np.arctanh([u, v]))
    point = np.array([np.log(N), u_proper, 0, 0, np.log(D), v_proper, 0, 0])

    def conserved(values):
        return master.cell_predictions(values, np.eye(3), 1.0, state.entropy_per_baryon[0], phi)

    jac = master._numerical_jacobian(conserved, point)
    screen = master.characteristic_screen(N, INTERNAL, u, D, v, phi)
    time, space = native_pencil(master, phi, v, u)
    roots, vectors = eig(space, time)
    order = np.lexsort((roots.imag, roots.real))
    gamma_rel = float(state.relative_lorentz_factor[0])
    solver._primitive_guess = None
    targets = np.repeat(conserved(point)[:, None], grid.shape[0], axis=1)
    _, _, residual, failed = solver._recover_cells(
        metric, targets[0], targets[1], targets[2:5], targets[5], targets[6:],
        state.entropy_per_baryon, scalar,
    )
    result = {
        'n': N, 'd': D, 'eps': INTERNAL, 'phi': phi, 'v_N': u, 'v_D': v,
        'v_rel': (v - u)/(1 - u*v), 'Gamma_rel': gamma_rel,
        'W': float(state.relative_invariant[0]), 'Q': float(state.entrainment[0]),
        'B_N': float(state.B_N[0]), 'B_D': float(state.B_D[0]),
        'Psi': float(state.generalized_pressure[0]), 'C': float(state.coupling_coefficient[0]),
        'effective_scalar_mass_squared': float(state.effective_mass_squared[0]),
        'minimum_legendre_eigenvalue': screen.legendre_minimum_eigenvalue,
        'minimum_thermodynamic_eigenvalue': screen.thermodynamic_minimum_eigenvalue,
        'condition_number_A0': float(np.linalg.cond(time)),
        'characteristic_eigenvalues_real': roots[order].real.tolist(),
        'characteristic_eigenvalues_imag': roots[order].imag.tolist(),
        'maximum_characteristic_modulus': float(np.max(abs(roots))),
        'eigenbasis_condition_number': float(np.linalg.cond(vectors)),
        'primitive_recovery_jacobian': jac.tolist(),
        'primitive_recovery_residual': float(np.max(residual)),
        'primitive_recovery_failed_cells': failed,
        'within_production_counterflow_domain': gamma_rel <= PARAMETERS.maximum_relative_lorentz_factor,
        'scalar_acceleration': -float(state.scalar_equation_source[0]),
        'fixed_current_scalar_equilibrium': float(np.sqrt(max(0, -state.effective_mass_squared[0]) / PARAMETERS.scalar_quartic)),
    }
    return result


def native_nullspace(master, phi, v, speed):
    results = []
    for step in (0.25, 0.5, 1.0, 2.0, 4.0):
        time, space = native_pencil(master, phi, v, step=step)
        shifted = np.linalg.solve(time, space) - speed*np.eye(5)
        _, singular, vectors = np.linalg.svd(shifted)
        roots = eigvals(space, time)
        results.append({
            'native_derivative_step_multiplier': step,
            'singular_values': singular.tolist(),
            'reconstructed_vectors': vectors[-2:].tolist(),
            'reconstructed_residuals': np.linalg.norm(shifted @ vectors[-2:].T, axis=0).tolist(),
            'cluster_distances': sorted(abs(roots - speed).tolist())[:2],
            'tolerance_sweep': [
                {'tolerance': tol, 'nullity': int(np.count_nonzero(singular <= tol)),
                 'cluster_size': int(np.count_nonzero(abs(roots - speed) <= tol))}
                for tol in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
            ],
        })
    return results


def directional_scan(master, states):
    """Pilot scan of the solver's frozen-entropy 25-variable pencil."""
    angles = np.unique(np.r_[np.linspace(0, 180, 1801), 10.0**np.arange(-7, -1),
                             180 - 10.0**np.arange(-7, -1)])
    data = np.empty((len(states), angles.size, 6))
    for i, (_, phi, v) in enumerate(states):
        for j, theta in enumerate(np.deg2rad(angles)):
            audit = master.coupled_characteristic_audit(
                N, INTERNAL, (0, 0, 0), D, (v, 0, 0), phi, (np.cos(theta), np.sin(theta), 0),
            )
            roots = eigvals(audit.space_matrix, audit.time_matrix)
            data[i, j] = (max(abs(roots.imag)), max(abs(roots)), audit.eigenvector_condition,
                          audit.eigenvector_rank, audit.maximum_eigenspace_residual,
                          np.linalg.cond(audit.time_matrix))
    return angles, data


def main():
    output = Path(__file__).resolve().parent
    solver = NullothProductionSolver(PeriodicGrid((4,), (1.0,)), PARAMETERS)
    master = solver.master
    print('Reconstructing the acoustic pencil at 30 to 100 digits', flush=True)
    root, precision = high_precision()
    phi0, v0, speed0 = map(float, root)
    velocities = np.linspace(0, 0.95, 191)
    splits = [split_squared(master, 0.15, velocity) for velocity in velocities]
    edges = [brentq(lambda v: split_squared(master, 0.15, v), a, b, xtol=1e-12)
             for a, b, fa, fb in zip(velocities[:-1], velocities[1:], splits[:-1], splits[1:])
             if fa*fb < 0]
    print(f'Native longitudinal edges at phi=0.15: {edges}', flush=True)
    print('Recording native recovery and longitudinal diagnostics', flush=True)
    band = [state_sample(solver, 0.15, float(v)) for v in velocities]
    band_edges = [state_sample(solver, 0.15, v) for v in edges]
    pinch = state_sample(solver, phi0, v0)
    laboratory = state_sample(solver, phi0, (v0 + 0.6)/(1 + v0*0.6), 0.6)
    nullspace = native_nullspace(master, phi0, v0, speed0)
    reference = {
        'inputs': {'n': N, 'd': D, 'eps': INTERNAL, 'constitutive': asdict(PARAMETERS),
                   'vector': asdict(solver.theory_parameters), 'ccz4': asdict(solver.ccz4.parameters),
                   'grhd': asdict(solver.grhd.parameters), 'production': asdict(solver.production_parameters)},
        'command_from_solver_root': 'python -B experiments/hadamard_pinch/validate_pinch.py',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'mpmath': mp.__version__},
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__).resolve(), ROOT / 'Nulloth exp.txt',
                                       *sorted((ROOT / 'src' / 'causal_repair').glob('*.py'))]},
        'band_phi_0.15': {'edges': edges, 'samples': band, 'edge_samples': band_edges},
        'pinch_rest_frame': pinch, 'pinch_laboratory_frame': laboratory,
        'native_nullspace': nullspace, 'arbitrary_precision': precision,
    }
    (output / 'hadamard_pinch_reference.json').write_text(json.dumps(reference, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Scanning directions with the native frozen-entropy pencil', flush=True)
    states = [('pinch', phi0, v0), ('transverse_reference', 0.15, 0.85)]
    for gap in precision['gap_samples']:
        if abs(float(gap['delta_phi'])) == 1e-4:
            width = float(gap['width'])
            phi = phi0 + float(gap['delta_phi'])
            suffix = gap['delta_phi']
            states.extend([(f'gap_center_{suffix}', phi, float(gap['center'])),
                           (f'below_gap_{suffix}', phi, float(gap['low']) - width),
                           (f'above_gap_{suffix}', phi, float(gap['high']) + width)])
    angles, data = directional_scan(master, states)
    columns = ['max_imag', 'max_modulus', 'eigenbasis_condition', 'rank', 'eigenspace_residual', 'A0_condition']
    np.savez_compressed(output / 'hadamard_pinch_directional_scan.npz', angle_degrees=angles,
                        states=np.array([state[0] for state in states]),
                        phi_v=np.array([state[1:] for state in states]), columns=columns, values=data)
    directional = {}
    for i, (name, phi, v) in enumerate(states):
        index = int(np.argmax(data[i, :, 0]))
        finite_conditions = data[i, np.isfinite(data[i, :, 2]), 2]
        directional[name] = {
            'phi': phi, 'v_rel': v, 'max_imag': data[i, index, 0],
            'max_imag_angle_degrees': float(angles[index]), 'longitudinal_max_imag': data[i, 0, 0],
            'transverse_max_imag': data[i, np.flatnonzero(angles == 90)[0], 0],
            'max_modulus': float(np.max(data[i, :, 1])), 'minimum_rank': int(np.min(data[i, :, 3])),
            'max_finite_eigenbasis_condition': float(max(finite_conditions)) if finite_conditions.size else None,
            'nonfinite_eigenbasis_condition_count': int(np.count_nonzero(~np.isfinite(data[i, :, 2]))),
        }
    summary = {
        'scope': 'Static reproduction and frozen-entropy directional pilot. No time evolution.',
        'band_phi_0.15_edges': edges, 'pinch_phi_v_speed': [phi0, v0, speed0],
        'rest_frame': pinch, 'laboratory_frame': laboratory,
        'native_svd': next(entry for entry in nullspace if entry['native_derivative_step_multiplier'] == 1.0),
        'limiting_width_slope': precision['limiting_width_slope'],
        'limiting_boundary_slopes': precision['limiting_boundary_slopes'],
        'directional_variables': 25, 'directional_samples_per_state': int(angles.size),
        'directional_frozen_entropy': directional,
    }
    (output / 'hadamard_pinch_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(directional, indent=2), flush=True)


if __name__ == '__main__':
    main()
