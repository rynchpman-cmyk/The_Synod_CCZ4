"""Compare the transverse tangent with grid-frame transport and grid damping."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from run_transverse_tangent import K, material_symbol, velocity_from_proper
from validate_pinch import PARAMETERS
from causal_repair.matter import FluidPrimitive
from causal_repair.nulloth import NullothMasterFunction
from causal_repair.theory33 import CarrierPrimitive


def state_from_primitive(p):
    n, d = np.exp(p[[0, 4]])
    gamma = PARAMETERS.gamma_ad
    pressure = np.exp((gamma - 1)*p[8])*n**gamma
    return n, d, pressure/((gamma - 1)*n), velocity_from_proper(p[1:4]), velocity_from_proper(p[5:8])


def conservative_material_symbol(master, p, phi, full_contraction):
    """Assemble transport from the native material flux and Carter path product."""
    metric = np.eye(3)[:, :, None]

    def evaluate(values):
        n, d, eps, u, v = state_from_primitive(values)
        fluid = FluidPrimitive(np.array([n]), np.array([eps]), u[:, None], np.zeros(1))
        carrier = CarrierPrimitive(np.array([d]), v[:, None])
        state = master.evaluate(metric, fluid, carrier, np.array([phi]))
        density_n = n/np.sqrt(1 - u @ u)
        density_d = float(state.carrier_eulerian_density[0])
        momentum = state.stress.momentum[:, 0]
        carrier_momentum = density_d*state.carrier_momentum_down[:, 0]
        conserved = np.r_[density_n, density_d, momentum, state.stress.rho[0], carrier_momentum]
        flux = np.r_[density_n*u[0], density_d*v[0], state.stress.stress[0, :, 0], momentum[0], carrier_momentum*v[0]]
        return np.r_[conserved, flux, -state.carrier_momentum_normal[0], state.carrier_momentum_down[:, 0]]

    jac = master._numerical_jacobian(evaluate, p)
    values = evaluate(p)
    carrier_velocity = velocity_from_proper(p[5:8])
    components = slice(None) if full_contraction else slice(1)
    jac[9+6] -= values[1]*(jac[18] + carrier_velocity[components] @ jac[19:22][components])
    return np.linalg.solve(jac[:9], jac[9:18])


def cumulative_trapezoid(values, time):
    return np.r_[0, np.cumsum(.5*(values[1:] + values[:-1])*np.diff(time))]


def compare(source, master):
    with np.load(source) as arrays:
        meta = json.loads(str(arrays['metadata_json']))
        time = arrays['time']
        mean = arrays['mean']
        signal = arrays['tangent']
        translation_error = float(np.max(arrays['translation_error']))
        max_recovery_residual = float(np.max(arrays['recovery_residual']))
    symbols, conservative_full, conservative_limited = [], [], []
    for row in mean:
        n, d, eps, u, v = state_from_primitive(row[:9])
        matrix = material_symbol(master, n, d, eps, u, v, row[9])
        symbols.append(matrix)
        conservative_full.append(conservative_material_symbol(master, row[:9], row[9], True))
        conservative_limited.append(conservative_material_symbol(master, row[:9], row[9], False))
    symbols = np.array(symbols)
    conservative_full, conservative_limited = np.array(conservative_full), np.array(conservative_limited)
    has_full_contraction = meta.get('completed_carrier_contraction') or meta.get('transverse_cells')
    conservative = conservative_full if has_full_contraction else conservative_limited
    full_error = np.linalg.norm(symbols - conservative_full, ord=2, axis=(1, 2))
    limited_error = np.linalg.norm(symbols - conservative_limited, ord=2, axis=(1, 2))
    spacing = 2*np.pi/(K*meta['points'])
    modified_k = np.sin(K*spacing)/spacing
    # The native Rusanov face-speed bound in fixed Minkowski is one here.
    diffusion = (1 - np.cos(K*spacing))/spacing
    material = signal[:, :9]
    norm = np.linalg.norm(material, axis=1)
    initial = material[0]
    models = np.empty((4, len(time), len(initial)), dtype=np.complex128)
    models[:, 0] = initial
    for i, dt in enumerate(np.diff(time), start=1):
        symbol = .5*(symbols[i-1] + symbols[i])
        native_material = .5*(conservative[i-1] + conservative[i])
        models[0, i] = expm(-1j*K*dt*symbol) @ models[0, i-1]
        models[1, i] = np.exp(-diffusion*dt)*(expm(-1j*modified_k*dt*symbol) @ models[1, i-1])
        models[2, i] = expm(-1j*K*dt*native_material) @ models[2, i-1]
        models[3, i] = np.exp(-diffusion*dt)*(expm(-1j*modified_k*dt*native_material) @ models[3, i-1])
    model_gains = np.linalg.norm(models, axis=-1)/norm[0]
    similarity = abs(np.vdot(material[-1], models[3, -1]))/(norm[-1]*np.linalg.norm(models[3, -1]))
    power = np.real(np.einsum('ni,nij,nj->n', material.conj(), -1j*modified_k*conservative, material))/(norm**2)
    principal_budget = cumulative_trapezoid(power, time)
    diffusion_budget = -diffusion*time
    log_gain = np.log(norm/norm[0])
    unmodeled_budget = log_gain - principal_budget - diffusion_budget
    return {'metadata': meta, 'file': source.name, 'time': time, 'mean': mean,
            'material': material, 'native_gain': norm/norm[0],
            'model_gains': model_gains, 'translation_error': translation_error, 'recovery_residual_max': max_recovery_residual,
            'full_conservative_error_max': float(max(full_error)), 'limited_conservative_error_max': float(max(limited_error)),
            'principal_budget': principal_budget,
            'diffusion_budget': diffusion_budget, 'unmodeled_budget': unmodeled_budget, 'log_gain': log_gain,
            'modal_similarity_final': float(similarity), 'field_final_norm': float(np.linalg.norm(signal[-1, 9:]))}


def history_distance(a, b):
    count = min(len(a['time']), len(b['time']))
    if np.max(abs(a['time'][:count] - b['time'][:count])) > 1e-12:
        raise ValueError('History comparison requires matched time levels')
    return float(np.max(np.linalg.norm(a['material'][:count] - b['material'][:count], axis=1)))


def summary_entry(item):
    peak = int(np.argmax(item['native_gain']))
    return {'file': item['file'], **item['metadata'], 'last_time': float(item['time'][-1]),
            'native_final_gain': float(item['native_gain'][-1]), 'native_peak_gain': float(item['native_gain'][peak]),
            'peak_time': float(item['time'][peak]), 'grid_frame_principal_final_gain': float(item['model_gains'][0, -1]),
            'grid_frame_rusanov_model_final_gain': float(item['model_gains'][1, -1]),
            'case_material_principal_final_gain': float(item['model_gains'][2, -1]),
            'case_material_rusanov_final_gain': float(item['model_gains'][3, -1]),
            'max_explicit_translation_error': item['translation_error'], 'recovery_residual_max': item['recovery_residual_max'],
            'full_conservative_principal_error': item['full_conservative_error_max'],
            'limited_conservative_principal_error': item['limited_conservative_error_max'],
            'path_log_gain_principal': float(item['principal_budget'][-1]),
            'path_log_gain_rusanov': float(item['diffusion_budget'][-1]),
            'path_log_gain_residual': float(item['unmodeled_budget'][-1]),
            'rusanov_model_final_modal_similarity': item['modal_similarity_final'],
            'scalar_and_vector_tangent_norm_final': item['field_final_norm']}


def plot(directory, direct, compact, corrected, transverse, fit, full_fit):
    native = direct[64]
    colors = ('#185d82', '#a5462b', '#3e7868', '#75518b')
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), layout='constrained')
    ax = axes[0]
    ax.plot(native['time'], native['model_gains'][0], ':', color='#34383b', label='Grid-frame principal symbol')
    ax.plot(native['time'], native['native_gain'], color=colors[0], label='Paired native replay, 64 cells')
    ax.plot(transverse['time'], transverse['native_gain'], '--', color=colors[1],
            label='Uniform 2D replay, 64-cell-equivalent spacing')
    limit = compact[max(compact)]
    ax.plot(limit['time'], limit['native_gain'], color=colors[2], label=f'1D probe, {max(compact):,}-cell equivalent')
    corrected_limit = corrected[max(corrected)]
    ax.plot(corrected_limit['time'], corrected_limit['native_gain'], color=colors[3],
            label=f'Full contraction, {max(corrected):,}-cell equivalent')
    ax.set(xlabel='Coordinate time', ylabel='Primitive Fourier gain', title='Native response and transport models')
    ax.legend(fontsize=7.8)
    ax = axes[1]
    ax.plot([1/n for n in sorted(direct)], [np.log(direct[n]['native_gain'][-1]) for n in sorted(direct)],
            'o', markerfacecolor='none', color=colors[0], label='Full-grid replay')
    ax.plot([1/n for n in sorted(compact)], [np.log(compact[n]['native_gain'][-1]) for n in sorted(compact)],
            'x', color=colors[2], label='Compact probe')
    ax.plot([1/n for n in sorted(corrected)], [np.log(corrected[n]['native_gain'][-1]) for n in sorted(corrected)],
            '+', color=colors[3], label='Full contraction probe')
    x = np.linspace(0, 1/min(compact), 60)
    ax.plot(x, np.polyval(fit, x), ':', color=colors[2], label='Fine-grid fits')
    ax.plot(x[x <= 1/min(corrected)], np.polyval(full_fit, x[x <= 1/min(corrected)]), ':', color=colors[3])
    ax.set(xlabel='Inverse cell count', ylabel='Endpoint log gain', title='Grid refinement at the final time')
    ax.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(directory / 'transverse_tangent_replay.png', dpi=180)
    plt.close(figure)


def main():
    directory = Path(__file__).resolve().parent
    master = NullothMasterFunction(PARAMETERS)
    cases = [compare(path, master) for path in sorted(directory.glob('hadamard_pinch_transverse_tangent_*.npz'))
             if not path.stem.endswith('_production')]
    baseline = [x for x in cases if x['metadata']['dt'] == .000125 and x['metadata']['amplitude'] == 1e-4
                and x['metadata']['duration'] == .016 and x['metadata']['failure'] is None]
    direct = {x['metadata']['points']: x for x in baseline if not x['metadata'].get('harmonic_samples')
              and not x['metadata'].get('transverse_cells') and not x['metadata'].get('completed_carrier_contraction')}
    compact = {x['metadata']['points']: x for x in baseline if x['metadata'].get('harmonic_samples') == 8
               and not x['metadata'].get('transverse_cells') and not x['metadata'].get('completed_carrier_contraction')}
    corrected = {x['metadata']['points']: x for x in baseline if x['metadata'].get('harmonic_samples') == 8
                 and x['metadata'].get('completed_carrier_contraction')}
    fine = [n for n in sorted(compact) if n >= 128]
    full_fine = [n for n in sorted(corrected) if n >= 128]
    if not {16, 32, 64} <= direct.keys() or len(fine) < 3 or len(full_fine) < 3 or 64 not in corrected:
        raise ValueError('Analysis needs the full-grid, fine compact, and full-contraction replays')
    fit = np.polyfit([1/n for n in fine], [np.log(compact[n]['native_gain'][-1]) for n in fine], 1)
    fit_subset = np.polyfit([1/n for n in fine[1:]], [np.log(compact[n]['native_gain'][-1]) for n in fine[1:]], 1)
    full_fit = np.polyfit([1/n for n in full_fine], [np.log(corrected[n]['native_gain'][-1]) for n in full_fine], 1)
    full_subset = np.polyfit([1/n for n in full_fine[1:]], [np.log(corrected[n]['native_gain'][-1]) for n in full_fine[1:]], 1)
    cases_2d = [x for x in cases if x['metadata'].get('transverse_cells')]
    transverse = next(x for x in cases_2d if x['metadata']['points'] == 64 and x['metadata']['duration'] == .016)
    dimension_crosschecks = {}
    for case in cases_2d:
        candidates = [x for x in cases if x['metadata'].get('completed_carrier_contraction')
                      and x['metadata']['points'] == case['metadata']['points']
                      and x['metadata']['dt'] == case['metadata']['dt']]
        dimension_crosschecks[case['file']] = {x['file']: history_distance(x, case) for x in candidates}
    prior_path = directory / 'hadamard_pinch_spatial_growth_drag_hadamard_long-transverse.npz'
    with np.load(prior_path) as prior:
        prior_model_endpoint = float(prior['run_0'][-1, 3])
    paired = [x for x in cases if x['metadata']['phase_pair'] == 'independent']
    if not cases_2d or not paired:
        raise ValueError('Analysis needs an independent sign pair, full-contraction refinement, and two-dimensional validation')
    report = {
        'cases': [summary_entry(item) for item in cases],
        'native_compact_max_complex_difference': {str(n): history_distance(direct[n], compact[n]) for n in direct if n in compact},
        'native_2d_completed_1d_contraction_max_complex_difference': dimension_crosschecks,
        'old_collinear_rest_frame_surrogate_endpoint_gain': prior_model_endpoint,
        'fine_compact_fit_points': fine, 'compact_limit_endpoint_gain': float(np.exp(fit[-1])),
        'compact_limit_gain_fit_sensitivity': float(abs(np.exp(fit[-1]) - np.exp(fit_subset[-1]))),
        'compact_log_gain_slope_per_inverse_cell': float(fit[0]),
        'full_contraction_fit_points': full_fine,
        'full_contraction_limit_endpoint_gain': float(np.exp(full_fit[-1])),
        'full_contraction_limit_gain_fit_sensitivity': float(abs(np.exp(full_fit[-1]) - np.exp(full_subset[-1]))),
        'maximum_independent_pair_translation_error': max(x['translation_error'] for x in paired),
        'maximum_y_variation': max(x['metadata']['maximum_transverse_variation'] for x in cases_2d),
        'maximum_64_cell_background_difference_when_completing_contraction': float(np.max(abs(compact[64]['mean'] - corrected[64]['mean']))),
    }
    columns = ('time', 'native_gain', 'covariant_principal_gain', 'covariant_rusanov_gain', 'case_material_principal_gain',
               'case_material_rusanov_gain', 'log_gain', 'principal_log_gain', 'rusanov_log_gain', 'residual_log_gain')
    histories = {f'case_{i}': np.column_stack((item['time'], item['native_gain'], *item['model_gains'], item['log_gain'],
                                              item['principal_budget'], item['diffusion_budget'], item['unmodeled_budget']))
                 for i, item in enumerate(cases)}
    np.savez_compressed(directory / 'transverse_tangent_comparison.npz', columns=columns,
                        files=[item['file'] for item in cases], **histories)
    (directory / 'hadamard_pinch_transverse_tangent_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    plot(directory, direct, compact, corrected, transverse, fit, full_fit)


if __name__ == '__main__':
    main()
