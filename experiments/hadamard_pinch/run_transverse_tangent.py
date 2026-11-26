"""Replay an antisymmetric Fourier pair and retain the native tangent history."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eig

from run_homogeneous_departure import FixedMinkowski
from validate_pinch import D, INTERNAL, N, PARAMETERS
from causal_repair.adm import inverse_metric
from causal_repair.grid import PeriodicGrid
from causal_repair.matter import FluidPrimitive
from causal_repair.production_nulloth import NullothProductionSolver
from causal_repair.theory33 import CarrierPrimitive


K = 256.0
PHI = 0.15
VELOCITY = 0.85
DIRECTION = np.array([1.0, 0.0, 0.0])
MATERIAL = np.r_[0:8, 25]
PRIMITIVE_COLUMNS = ('log_n', 'q_Nx', 'q_Ny', 'q_Nz', 'log_d', 'q_Dx', 'q_Dy', 'q_Dz', 'entropy')


class HarmonicProbeGrid(PeriodicGrid):
    """Sample one Fourier period while evaluating shifts at a target spacing.

    This experiment grid reproduces the target lattice's stencil on a small
    Fourier subspace. Its convergence claim is only for the first harmonic
    in the small-amplitude limit, with native periodic replays as checks.
    """

    def __init__(self, samples, target_points, length, transverse_cells):
        shape, lengths = ((samples, transverse_cells), (length, 1.0)) if transverse_cells else ((samples,), (length,))
        super().__init__(shape, lengths)
        object.__setattr__(self, 'target_points', target_points)

    @property
    def spacing(self):
        return (self.lengths[0]/self.target_points, *super().spacing[1:])

    @property
    def cell_volume(self):
        return self.volume/int(np.prod(self.shape))

    def shift(self, field, offset, axis):
        if axis:
            return super().shift(field, offset, axis)
        array_axis = self._spatial_axis(field, axis)
        samples = self.shape[0]
        modes = np.arange(samples//2 + 1)
        phase = np.exp(-2j*np.pi*offset*modes/self.target_points)
        multiplier_shape = [1]*field.ndim
        multiplier_shape[array_axis] = len(modes)
        return np.fft.irfft(phase.reshape(multiplier_shape)*np.fft.rfft(field, axis=array_axis), n=samples, axis=array_axis)

    def _differentiate(self, field, axis, order):
        array_axis = self._spatial_axis(field, axis)
        samples = self.shape[0]
        k = 2*np.pi*np.arange(samples//2 + 1)/self.lengths[0]
        multiplier_shape = [1]*field.ndim
        multiplier_shape[array_axis] = len(k)
        return np.fft.irfft(((1j*k)**order).reshape(multiplier_shape)*np.fft.rfft(field, axis=array_axis), n=samples, axis=array_axis)

    def derivative(self, field, axis):
        return super().derivative(field, axis) if axis else self._differentiate(field, axis, 1)

    def second_derivative(self, field, axis):
        return super().second_derivative(field, axis) if axis else self._differentiate(field, axis, 2)


class LegacyVorticityProbe(NullothProductionSolver):
    """Reproduce the pre-fix contraction for the historical comparison."""

    def _target_rhs(self, state, recovery, h, k):
        charge, momentum = super()._target_rhs(state, recovery, h, k)
        sqrt_h = inverse_metric(h)[2]
        covector = recovery.master.carrier_momentum_down
        transport = state.geometry.lapse[None, ...]*recovery.carrier.velocity - state.geometry.shift
        for component in range(self.grid.ndim):
            for axis in range(self.grid.ndim, 3):
                coefficient = recovery.master.carrier_eulerian_density*transport[axis]
                momentum[component] -= sqrt_h*self._path_product(coefficient, covector[axis], component)
        return charge, momentum


def velocity_from_proper(q):
    return q / np.sqrt(1 + np.sum(q*q, axis=0))


def material_symbol(master, n, d, internal, u, v, phi):
    """Entropy-inclusive principal matrix in the fixed simulation frame."""
    native = master.coupled_characteristic_audit(n, internal, u, d, v, phi, DIRECTION)
    time, space = np.eye(26), np.zeros((26, 26))
    time[:25, :25], space[:25, :25] = native.time_matrix, native.space_matrix
    # d_s(mu_i) = gamma*T*W*u_i, d_s(mu_0) = -gamma*T*W.
    # Subtract n*T*grad(s) in the spatial material equation.
    gamma = master.parameters.gamma_ad
    temperature = (gamma - 1)*internal
    lorentz_squared = 1/(1 - u @ u)
    directional_speed = float(u @ DIRECTION)
    entropy_time = n*gamma*temperature*lorentz_squared*u
    time[1:4, 25] = entropy_time
    space[1:4, 25] = directional_speed*entropy_time + n*(gamma - 1)*temperature*DIRECTION
    space[25, 25] = directional_speed
    a0 = time[np.ix_(MATERIAL, MATERIAL)]
    a = space[np.ix_(MATERIAL, MATERIAL)]
    return np.linalg.solve(a0, a)


def seed_mode(master):
    base = np.zeros(9)
    base[0], base[4] = np.log(N), np.log(D)
    base[6] = -VELOCITY/np.sqrt(1 - VELOCITY**2)
    base[8] = master.specific_entropy(N, INTERNAL)
    symbol = material_symbol(master, N, D, INTERNAL, np.zeros(3), np.array([0., -VELOCITY, 0.]), PHI)
    roots, vectors = eig(symbol)
    index = np.argmax(roots.imag)
    vector = vectors[:, index]
    pivot = vector[np.argmax(abs(vector))]
    return base, vector*(pivot.conjugate()/abs(pivot)), roots[index]


def initialize(solver, primitives):
    n, d = np.exp(primitives[0]), np.exp(primitives[4])
    gamma = solver.master_parameters.gamma_ad
    pressure = np.exp((gamma - 1)*primitives[8])*n**gamma
    fluid = FluidPrimitive(n, pressure/((gamma - 1)*n), velocity_from_proper(primitives[1:4]), solver.grid.zeros())
    carrier = CarrierPrimitive(d, velocity_from_proper(primitives[5:8]))
    return solver.initialize(solver.ccz4.flat_state(), fluid, carrier=carrier, varphi=PHI)


def physical_fields(solver, state):
    recovery = solver.recover(state)
    u, v = recovery.fluid.velocity, recovery.carrier.velocity
    q_n, q_d = (velocity/np.sqrt(1 - np.sum(velocity**2, axis=0)) for velocity in (u, v))
    values = np.concatenate((np.log(recovery.fluid.baryon_density)[None], q_n,
                             np.log(recovery.carrier.number_density)[None], q_d, recovery.entropy_per_baryon[None],
                             state.scalar.varphi[None], state.scalar.momentum[None], state.a, state.pi_A, state.b, state.pi_B,
                             state.longitudinal_A[None], state.longitudinal_B[None], state.cleaning_A[None], state.cleaning_B[None]))
    return values, recovery


def build_solver(grid, legacy_contraction=False):
    solver_type = LegacyVorticityProbe if legacy_contraction else NullothProductionSolver
    solver = solver_type(grid, PARAMETERS)
    solver.ccz4 = FixedMinkowski(grid, solver.ccz4.parameters)
    return solver


def replay(args):
    if args.points % 2 or args.points < 4 or args.amplitude <= 0 or args.dt <= 0:
        raise ValueError('Provide an even grid size of at least four and positive amplitude and dt')
    steps = round(args.duration/args.dt)
    if steps < 1 or not np.isclose(steps*args.dt, args.duration, atol=1e-14, rtol=0):
        raise ValueError('Duration must be a positive integral number of timesteps')
    samples = args.harmonic_samples or args.points
    if samples % 2 or samples < 4:
        raise ValueError('Fourier probe requires an even number of at least four samples')
    shape, lengths = ((samples, args.transverse_cells), (2*np.pi/K, 1.0)) if args.transverse_cells else ((samples,), (2*np.pi/K,))
    grid = (HarmonicProbeGrid(samples, args.points, 2*np.pi/K, args.transverse_cells) if args.harmonic_samples
            else PeriodicGrid(shape, lengths))
    solver = build_solver(grid, args.legacy_contraction)
    base, vector, root = seed_mode(solver.master)
    phase = 2*np.pi*np.arange(samples)/samples
    fourier = np.exp(-1j*phase)
    seed = args.amplitude*np.real(vector[:, None]*np.exp(1j*phase))
    positive_seed, negative_seed = base[:, None] + seed, base[:, None] - seed
    if args.transverse_cells:
        positive_seed, negative_seed = (np.broadcast_to(item[:, :, None], (9, *shape))
                                        for item in (positive_seed, negative_seed))
    pos = initialize(solver, positive_seed)
    paired = build_solver(grid, args.legacy_contraction) if args.explicit_pair else None
    neg = initialize(paired, negative_seed) if paired else None
    case = f'n{args.points:03d}_dt{args.dt:.2g}_a{args.amplitude:.2g}' + ('_pair' if paired else '')
    if args.harmonic_samples:
        case += f'_harmonic{samples}'
    if args.transverse_cells:
        case += f'_y{args.transverse_cells}'
    if args.complete_contraction:
        case += '_full-contraction'
    elif not args.legacy_contraction:
        case += '_production'
    directory = Path(__file__).resolve().parent
    output = directory / f'hadamard_pinch_transverse_tangent_{case}.npz'
    times, means, signals, symmetry, recovery_residual, y_variation = [], [], [], [], [], []
    failure = None
    for index in range(steps + 1):
        if index:
            try:
                pos = solver.step(pos, args.dt)
                if paired:
                    neg = paired.step(neg, args.dt)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                failure = {'attempted_time': index*args.dt, 'reason': str(exc)}
                break
        positive, recovery = physical_fields(solver, pos)
        shifted = np.roll(positive, samples//2, axis=1)
        negative = physical_fields(paired, neg)[0] if paired else shifted
        symmetry.append(float(np.max(abs(negative - shifted))) if paired else 0.0)
        if args.transverse_cells:
            y_variation.append(float(np.max(np.ptp(positive, axis=2))))
            positive, negative = [np.mean(value, axis=2) for value in (positive, negative)]
        means.append(np.mean((positive + negative)/2, axis=1))
        signals.append(np.mean((positive - negative)*fourier, axis=1)/args.amplitude)
        times.append(pos.time)
        recovery_residual.append(recovery.report.maximum_residual)
        if index and index % 32 == 0:
            print(f'{case} step {index}/{steps}', flush=True)
    metadata = {'points': args.points, 'dt': args.dt, 'amplitude': args.amplitude, 'duration': args.duration,
                'wavenumber': K, 'harmonic_samples': args.harmonic_samples,
                'transverse_cells': args.transverse_cells, 'completed_carrier_contraction': not args.legacy_contraction,
                'maximum_transverse_variation': max(y_variation, default=0.0),
                'phase_pair': 'independent' if paired else 'periodic half-period translation',
                'initial_characteristic': [float(root.real), float(root.imag)], 'failure': failure}
    fields = (*PRIMITIVE_COLUMNS, 'phi', 'scalar_momentum', 'a_x', 'a_y', 'a_z', 'pi_Ax', 'pi_Ay', 'pi_Az',
              'b_x', 'b_y', 'b_z', 'pi_Bx', 'pi_By', 'pi_Bz', 'longitudinal_A', 'longitudinal_B', 'cleaning_A', 'cleaning_B')
    np.savez_compressed(output, metadata_json=json.dumps(metadata), time=times, mean=means,
                        tangent=signals, fields=fields, translation_error=symmetry,
                        recovery_residual=recovery_residual, initial_primitive_mode=vector)
    print(f'{case} reached {times[-1]:.8g} and gain {np.linalg.norm(signals[-1][:9]):.8g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points', type=int, default=32)
    parser.add_argument('--dt', type=float, default=.000125)
    parser.add_argument('--amplitude', type=float, default=1e-4)
    parser.add_argument('--duration', type=float, default=.016)
    parser.add_argument('--explicit-pair', action='store_true')
    parser.add_argument('--harmonic-samples', type=int, help='Measure the first harmonic on this compact probe grid')
    parser.add_argument('--transverse-cells', type=int, default=0, help='Add an active uniform Cartesian axis')
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument('--complete-contraction', action='store_true', help='Save under the historical full-contraction filename')
    replay_mode.add_argument('--legacy-contraction', action='store_true', help='Reproduce the historical reduced-grid result')
    replay(parser.parse_args())
