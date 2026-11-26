# Hadamard pinch campaign

## Static reproduction

The native Nulloth equations reproduce the longitudinal pinch. The repeated longitudinal characteristic is semisimple. The same state has a separate defective oblique zero-speed collision in the native directional symbol.

| Measurement | Result |
| --- | ---: |
| Instability band at \(\phi=0.15\), rest frame | \(0.162091636280 < |v_\mathrm{rel}| < 0.759728089199\) |
| Pinch scalar amplitude | \(0.01088444356899034629\) |
| Pinch relative velocity | \(0.76088440752724528412\) |
| Repeated longitudinal speed | \(0.47108408938082803353\) |
| Gap-width coefficient as \(\Delta\phi\to0\) | \(6.991101441746\) |
| Transverse \(\max |\operatorname{Im}\lambda|\) at \(\phi=0.15,\ v_\mathrm{rel}=0.85\) | \(0.2944750330\) |

The float64 calculation used the native constitutive fluxes, a five-variable longitudinal generalized eigenproblem including advected entropy, and adaptive velocity root finding. The rest-frame spectrum was approximately \((-0.4729746455, 0, 0.4710840894, 0.4710840894, 0.9033359204)\). The lab-frame speeds agreed with relativistic velocity addition to \(1.4\times10^{-11}\). The native pinch state had \(Q=-0.1955894753\), \(C=541.4993158\), and a fixed-current equilibrium amplitude of \(0.4649728232\). Its homogeneous flat-space initial scalar acceleration was \(+5.8798105713\) for zero scalar momentum.

The global longitudinal map used a 37-by-61 constitutive grid. The local map and eigenvalue-splitting figure use the adaptive arbitrary-precision boundaries and gap centers. [Global longitudinal map](01_longitudinal_phase_diagram.png) · [Pinch geometry](02_pinch_geometry.png) · [Acoustic pair splitting](04_eigenvalue_splitting.png) · [Convexity and causality map](07_convexity_and_causality.png)

The pinch passed the native cold material recovery solve with a normalized residual of \(1.39\times10^{-8}\). The native inner loop accepts a residual below \(2\times10^{-8}\) at the default settings. The minimum Legendre and thermodynamic eigenvalues were \(1.5382559094\) and \(0.3412733298\). The broad scan includes 17 diagnostic states beyond the default production counterflow limit. The reference JSON records the counterflow gate, material inversion result, and convexity margins for each state. These measurements use a local flat metric for constitutive analysis.

## Longitudinal multiplicity and gap scaling

The arbitrary-precision calculation differentiated the master action in ordinary and carrier current components. Imposing current conservation gives a symmetric \(2\times2\) acoustic pencil. At the root, all three independent entries vanish, leaving two independent acoustic amplitudes. The second speed derivative of its determinant is \(-13.5566410772\), giving algebraic multiplicity two. The solution converged at 30, 50, 80, and 100 decimal digits.

The native longitudinal SVD at the standard differentiation step gave singular values \((1.23202, 0.57428, 0.47729, 2.74\times10^{-11}, 3.96\times10^{-12})\). The last two right singular vectors had residuals \(2.74\times10^{-11}\) and \(3.96\times10^{-12}\). The numerical nullity was two across the five differentiation steps at a \(10^{-9}\) threshold. Tighter thresholds started resolving differentiation noise.

Both gap boundaries were traced at positive and negative scalar offsets down to \(10^{-12}\). Their limiting velocity slopes were \(-8.209357418435\) and \(-1.218255976689\). At the smallest tested offsets, the two width ratios were \(6.991101441534\) and \(6.991101441957\). The imaginary characteristic magnitude at the centers was about \(3.22\times10^{-12}\). The gap plot shows the larger-offset curvature on each side. [Gap-width plot](03_gap_width_scaling.png)

## Directional audit

The initial sweep used 1,813 directions in the collinear symmetry plane at each of eight states. It included both reopened gap centers and the points one gap width below and above each gap at \(\Delta\phi=\pm10^{-4}\). It used the native 25-variable frozen-entropy pencil with eight material variables, three scalar variables, and two seven-variable mixed-Proca and cleaning blocks. The five-variable longitudinal calculation includes advected entropy along the flow.

The pinch sweep saw an essentially real spectrum at every sampled direction. A conditioning peak led to an adaptive zero-speed root search near \(64.48^\circ\). The refined root was \(64.476180266^\circ\), with a mirror near \(115.523819734^\circ\). An acoustic speed crosses two stationary material characteristics. The material zero-speed cluster has three eigenvalues and SVD nullity two. The constraint-reduced pencil has six zero-speed eigenvalues and nullity five. Both counts were reproduced using the native centered difference and a fourth-order Richardson stencil at clustering and SVD thresholds from \(10^{-6}\) through \(10^{-8}\). At tighter cluster thresholds, the generic float64 eigensolver splits the defective root numerically. The reduced eigenbasis condition number grew from roughly \(3\times10^4\) to \(3\times10^6\) as the angular offset decreased from \(10^{-3}\) to \(10^{-5}\) degrees. The native frozen-entropy system fails the strong-hyperbolicity check at this oblique direction. [Conditioning plot](05_eigenbasis_conditioning.png)

The two points above the reopened longitudinal gaps had real longitudinal spectra and oblique complex pairs. Their peak imaginary magnitudes were \(4.57\times10^{-4}\) and \(1.86\times10^{-4}\). The two below-gap points had real spectra over the sampled directions. [Directional plot](06_directional_instability.png)

### Entropy-inclusive audit

The second sweep added the advected specific entropy and the thermal force in the ordinary vorticity equation. Its spectrum at the longitudinal pinch matched the five-variable calculation within \(4.1\times10^{-11}\). Setting the entropy perturbation to zero recovered the native symbol to \(3.5\times10^{-18}\) in an oblique matrix cross-check. The 26-variable constraint-reduced symbol covers the material, scalar, and mixed-vector sectors. The passive tracer and the CCZ4 sector were outside this pencil.

| Zero-speed cluster at the oblique pinch collision | Cluster size | SVD nullity |
| --- | ---: | ---: |
| Frozen-entropy material block | 3 | 2 |
| Material block including entropy | 4 | 3 |
| Constraint-reduced scalar, vector, and material symbol including entropy | 7 | 6 |

The native and Richardson stencils agree on the collision angle within \(1.6\times10^{-9}\) degrees. Repeating the angular scan with entropy preserved the oblique complex pairs above the gaps and the high-flow transverse instability. Adaptive refinement also found a defective zero-speed collision in both below-gap samples, even though their spectra remained real. All eight states in the sweep passed the cold material inversion and the native convexity checks. Each failed admissibility through either a complex pair or an incomplete eigenbasis. [Entropy-inclusive directional arrays](hadamard_pinch_entropy_directional_scan.npz)

## Constitutive continuation and transit preflight

The longitudinal pinch continued at all 39 sampled thermodynamic states. These comprise the reference, both signs of five fractional perturbation sizes for each of \(n\), \(d\), and \(\epsilon\), and the eight simultaneous one-percent corners. The Jacobian of the three pinch equations has rank three, with singular values \(37.27517\), \(2.76140\), and \(1.18732\) in the recorded coordinates. The fixed-background submatrix in \((\phi,v_\mathrm{rel},\lambda_*)\) is invertible, so the local pinch is a three-dimensional manifold parameterized by the thermodynamic variables. The largest acoustic residual in the 50-digit solves was \(4.0\times10^{-50}\).

Changing the thermodynamic state steered the pinch. The oblique collision moved with it and persisted at every continued pinch. The sampled collision angles along the manifold ranged from \(64.38564^\circ\) to \(64.56576^\circ\). The neighborhood audit used a \(5\times5\) grid of scalar and relative-velocity offsets at the reference and the eight corners. The offsets reached \(|\Delta\phi|=10^{-3}\) and \(|\Delta v_\mathrm{rel}|=0.01\) from each local pinch. All 225 points retained the seven-root, six-vector oblique defect. This rules out those sampled states as transit candidates and gives evidence of a local obstruction. It leaves open the possibility of routes outside the sampled neighborhood. [Continuation plot](08_pinch_continuation.png) · [Continuation arrays](hadamard_pinch_continuation.npz)

## Homogeneous evolution

The E1 frozen control initialized the homogeneous reference with the native solver and zero scalar momentum and gradient. Native full recovery returned a residual of \(1.14\times10^{-16}\), and the initial scalar acceleration was \(+5.8798105713\). On the flat slice with those sources, the maximum Hamiltonian and momentum constraint residuals were \(107.5580\) and \(13.1526\) at the default gravitational coupling. This initialization was a local control, and zero time steps were taken.

The campaign driver reused the native production time step. Fixed-Minkowski cases froze the geometry at each stage. Both retained the charged mixed-Proca equations and their native Gauss initialization. The drag-free case used the infinite-relaxation limit. For the two coupled CCZ4 cases, a center-of-momentum boost removed the initial momentum density, and an expanding isotropic extrinsic curvature solved the homogeneous Hamiltonian constraint. Initial CCZ4 and Gauss residuals were below \(1.5\times10^{-14}\). The weak-gravity case used \(\kappa=10^{-3}\), and the default case used \(8\pi\).

| Configuration at \(t=0.05\), \(\Delta t=0.0005\) | Final \(\phi\) | Final \(v_\mathrm{rel}\) | Oblique defect angle |
| --- | ---: | ---: | ---: |
| Minkowski, no drag | 0.0184120 | 0.6223493 | \(56.05^\circ\) |
| Minkowski, native drag | 0.0182334 | 0.6063305 | \(54.28^\circ\) |
| CCZ4, weak gravity and native drag | 0.0182358 | 0.6033825 | \(54.19^\circ\) |
| CCZ4, default gravity and native drag | 0.0208123 | 0.5948278 | \(52.98^\circ\) |

The two fixed-geometry cases show substantial scalar growth and relative-velocity reduction even when drag is absent. Native drag adds a further reduction and positive integrated entropy. The oblique defect was present at all 593 accepted states across the two timestep runs. The default-gravity case also developed an early near-longitudinal complex pair. Its measured peak was \(8.70\times10^{-5}\) at \(t=0.0025\) and \(2.44^\circ\) on the finer timestep, versus \(8.33\times10^{-5}\) at \(t=0.002\) and \(2.28^\circ\) on the coarser timestep. The coarse fixed no-drag case was rejected by the production entropy gate on the attempted step to \(t=0.036\). The refined run reached \(t=0.05\). Both fixed-geometry configurations conserved the integrated energy to the precision of the stored values. The finest CCZ4 cases had absolute energy-work residuals below \(2.6\times10^{-5}\) using trapezoidal work integration. These trajectories describe departure from the defective pinch. [Homogeneous arrays](hadamard_pinch_dynamics.npz) · [Trajectories](09_homogeneous_trajectories.png) · [Admissibility history](11_admissibility_history.png) · [Conservation accounting](14_conservation_accounting.png)

## Autonomous transit attempts

Eleven native fixed-Minkowski shots used native drag. Seven varied the initial scalar amplitude, scalar momentum, and relative velocity on the reference thermodynamic background. Four also varied ordinary density, carrier density, and internal energy by five percent. A directional screen ran on every accepted time level and located the tracked oblique collision adaptively. One shot began inside the obstructed region. Seven encountered the zero-speed defect between \(t=0.006\) and \(0.047\). The other three finished at \(t=0.06\). The nearest sampled state passing this screen was 0.2822 from the reference in the unweighted \((\phi,v_\mathrm{rel})\) coordinates. None of the shots reached the pinch neighborhood. These results cover the chosen seeds and directional screen. A controlled transit would need a path outside the sampled obstruction and an explicitly accounted actuator. [Shooting and rejection records](hadamard_pinch_summary.json)

### Controlled transit and thermodynamic detours

Eight controlled shots applied carrier and scalar impulses between native fixed-Minkowski updates with drag. The carrier impulse changed proper velocity at fixed lab-frame particle count and entropy. The scalar impulse changed canonical momentum and supplied the corresponding change in scalar kinetic energy. The energy budget included the carrier and scalar work, the actuator supplied the measured net momentum, and carrier injection was zero. Across the runs, the largest absolute energy budget residual was \(2.91\times10^{-16}\). The integrated momentum and carrier number budgets closed to the stored precision.

Every controlled shot encountered the oblique zero-speed defect by \(t=0.010\), including both shots repeated with half the timestep. The closest screened state was 0.30494 from the pinch in unweighted \((\phi,v_\mathrm{rel})\) coordinates. A wider detour screen covered 1,296 constitutive states across 108 slices. It varied \(n\) from 0.8 to 1.2, \(d\) from 0.2 to 0.3, and \(\epsilon\) from 0.20 to 0.45. All 324 states tested within 0.01 of the reference relative velocity had the oblique defect. The highest velocity to pass the screen on that grid was 0.44. This constrains local transit with the tested controls and thermodynamic ranges. Routes through other parts of state space remain untested. [Controlled trajectories](hadamard_pinch_controlled_transit.json) · [Detour screen](hadamard_pinch_thermodynamic_detours.json) · [Trajectory and mode-transfer plots](controlled_and_longitudinal_followup.png)

## Spatial perturbations and nonnormal growth

The native solver evolved the optimized oblique Fourier seed on a periodic Cartesian axis. The background currents were rotated relative to the grid, and all three velocity and field components were evolved. At \(k=256\), the native-drag gain at \(t=0.016\) was 1.610, 2.139, and 2.280 with 16, 32, and 40 cells. The evolving principal-symbol estimate was 2.542. On the 32-cell grid, halving the timestep changed the gain by 0.00017. Increasing the initial amplitude from \(10^{-5}\) to \(10^{-3}\) changed it by approximately \(1.6\times10^{-6}\). Spatial resolution dominated these two effects in the tested range. The 16-cell, \(k=128\), no-drag run hit the native entropy gate at \(t=0.002\). The corresponding native-drag run finished. [Spatial growth plot](12_spatial_amplification.png)

A separate transverse test seeded the growing eigenvector at \(\phi=0.15\), \(v_\mathrm{rel}=0.85\). At \(k=256\), the initial frozen growth rate was \(75.386\). At \(t=0.0025\), native gains were 1.0403, 1.1081, and 1.1433 with 16, 32, and 64 cells. The evolving principal-symbol gain was 1.1881, and the frozen estimate was 1.2074. The corresponding native average logarithmic growth rates were 15.8, 41.1, and 53.6. Over these three resolutions they follow an approximate linear trend in inverse cell count, extrapolating to about 66.2 versus the evolving principal-symbol rate of 68.9. The longer 64-cell run reached a gain of 1.2823 at \(t=0.0075\), then fell to 0.9169 by \(t=0.016\). At that endpoint the evolving principal-only estimate was 2.8518. The late discrepancy includes effects from the discretization and the evolving sourced system and has not been attributed to a single term. [Transverse growth plot](12_transverse_hadamard.png) · [Early grid refinement](hadamard_pinch_spatial_growth_drag_hadamard_early-growth.npz) · [Full 64-cell trajectory](hadamard_pinch_spatial_growth_drag_hadamard_long-transverse.npz)

The frozen material propagator was computed directly from the entropy-inclusive principal symbol. At the oblique collision and \(kt=256\), the largest singular value was 38.80 in primitive coordinates and 51.86 in the fractional-current and entropy norm. The latter scales current perturbations by each unperturbed proper density and represents the independent entropy perturbation as its contribution to fractional pressure. One degree off the collision, where the sampled spectrum was real, the peak gains over the same window were about 25.8 and 35.0. The low-flow control stayed below 2.52 in both norms. The frozen figures describe the principal symbol at fixed coefficients. For the spatial comparison, the time-dependent principal propagator samples the simulated mean thermodynamics and relative speed in a collinear ordinary-rest-frame model. It omits changes in frame and propagation direction as well as the lower-order source Jacobians. The nonlinear run includes the native sources and numerical fluxes. [Frozen propagator data](hadamard_pinch_transient_growth.npz) · [Propagator plot](13_frozen_propagator.png)

### Longitudinal doublet and broadband perturbations

The longitudinal tests reconstructed the two-dimensional acoustic eigenspace by SVD. The response to a small change in relative velocity resolved two repeatable directions in that space. The replays seeded them separately, in phase, and in opposite phase. A deterministic broadband case populated three harmonics across material, scalar, and entropy variables. The runs covered 16 and 32 cells, fundamental wavenumbers 64, 128, and 256, and amplitudes from \(10^{-9}\) to \(10^{-3}\).

For the isolated seeds at \(k=128\), the main mode retained about 0.90 of its amplitude at 16 cells and 0.95 at 32 cells by \(t=0.004\). The other mode's amplitude stayed below 0.0022 in units of the initial seed amplitude at the endpoints. Extending the 32-cell runs to \(t=0.016\) raised that amplitude to about one percent of the initial seed. The two superpositions agreed with the corresponding linear combinations of the individual replays to within \(2.8\times10^{-6}\) in seed-normalized fundamental amplitude over the longer window. The evolving background can rotate the modes relative to the initial basis. [Short-window data](hadamard_pinch_longitudinal_pairs.npz) · [Longer replay data](hadamard_pinch_longitudinal_extended.npz)

The extended broadband runs exposed a stronger scalar-to-material response. At 32 cells with the smaller seed, the material gains at the first three harmonics were \(13.72\times\), \(21.48\times\), and \(26.53\times\). A 100-fold change in seed amplitude and refinement to 40 cells preserved the growth. The material-only seed produced gains of 1.18, 0.57, and 0.19 in the same normalization. The scalar-only seed reproduced nearly all of the full response, and the two separate material signals added to the full signal within \(4.9\times10^{-4}\) in seed-normalized Fourier norm. The strong broadband growth therefore comes primarily from the seeded scalar perturbation driving the material fields in this experiment. [Broadband growth and seed comparison](longitudinal_broadband_growth.png) · [Follow-up summary](hadamard_pinch_followup_summary.json)

## Transverse growth across frames and resolutions

Paired perturbations measured the first-harmonic response at \(k=256\) in fixed Minkowski space with scalar, mixed-vector, and drag sources active. The follow-up compared axial-only and full carrier momentum coupling. A 2D run uniform in the transverse direction independently reproduced the full-coupling result.

The source-free principal-symbol estimate at \(t=0.016\) was 2.852 in the collinear rest-frame model and 1.359 using the measured grid-frame background. On finer grids, the endpoint log gain was nearly linear in inverse cell count.

| Spatial resolution | Axial-only coupling | Full coupling |
| --- | ---: | ---: |
| 16 cells | 0.4702 | 0.6182 |
| 64-cell-equivalent spacing | 0.921 | 1.233 |
| Fine-grid extrapolation | 1.16 | 1.55 |

Both cases showed net amplification in the fine-grid extrapolation. Full carrier momentum coupling increased the extrapolated gain by about 33 percent. At the finest tested spacing, halving the timestep changed each endpoint gain by less than 0.006. These gains include the combined scalar, vector, and drag effects. [Tangent replay plot](transverse_tangent_replay.png) · [Tangent summary](hadamard_pinch_transverse_tangent_summary.json) · [Comparison curves](transverse_tangent_comparison.npz)

## Sourced variational growth

The sourced follow-up linearized the native timestep on 23 constraint-compatible material, scalar, and mixed-vector fields. It measured the homogeneous source matrix at \(k=0\) and advanced the time-dependent variational system at \(k=256\). The spatial comparison used a Fourier probe at 64-cell-equivalent spacing. Scalar and vector comparisons remove the relevant off-diagonal couplings from the tangent operator. The drag comparison subtracts the independently linearized native drag update. These are sensitivities on the same background, with interactions between sectors retained in the full model.

At high transverse flow, the growing-mode seed reached \(1.233\times\) in the native run and \(1.219\times\) in the model built from nine time samples. At five samples, subtracting the drag contribution reduced the predicted endpoint gain from \(1.204\times\) to \(1.028\times\). Removing vector couplings raised it to \(1.247\times\). Removing scalar couplings changed it to \(1.199\times\). These comparisons show that source effects depend on the seed, and they include the mesh and background used in the replay.

The optimal high-flow seed reached \(76.78\times\) and \(76.80\times\) in two nonlinear replays with a tenfold amplitude difference. The five-sample model predicted \(77.77\times\). Increasing the number of time samples from five to nine gave \(76.82\times\), and doubling the differencing amplitude at five samples changed the gain by less than \(10^{-5}\). This high-flow trajectory includes unstable directions. On the low-flow trajectory, the nonlinear replay instead reached \(1.886\times\) with all 33 background checkpoints and 72 perturbed cell checkpoints passing the directional screen. The low-flow result shows substantial finite-time amplification on the sampled spectrally admissible trajectory. [Sourced growth and nonlinear replay](sourced_optimal_replay.png) · [Follow-up summary](hadamard_pinch_followup_summary.json)

## Reproduction

Run from the `causal_repair_solver` root with Python, NumPy, SciPy, mpmath, and Matplotlib installed.

```powershell
python -B experiments/hadamard_pinch/validate_pinch.py
python -B experiments/hadamard_pinch/refine_directional_collision.py
python -B experiments/hadamard_pinch/audit_transit_corridor.py
python -B experiments/hadamard_pinch/plot_static_validation.py
python -B experiments/hadamard_pinch/plot_constitutive_maps.py
python -B experiments/hadamard_pinch/run_homogeneous_departure.py
python -B experiments/hadamard_pinch/run_homogeneous_departure.py --reaudit
python -B experiments/hadamard_pinch/run_transit_shooting.py
python -B experiments/hadamard_pinch/run_transit_shooting.py --thermodynamic-sweep
python -B experiments/hadamard_pinch/measure_oblique_propagator.py
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --refine time
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --refine space
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --refine amplitude
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --hadamard
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --hadamard --refine early-growth
python -B experiments/hadamard_pinch/run_oblique_spatial_growth.py --native-drag --hadamard --refine long-transverse
python -B experiments/hadamard_pinch/plot_dynamics.py
```

The transverse comparison data records the grid sizes, seed amplitudes, timesteps, and endpoint times. [Comparison data and run settings](hadamard_pinch_transverse_tangent_summary.json)

```powershell
python -B experiments/hadamard_pinch/run_controlled_transit.py
python -B experiments/hadamard_pinch/scan_controlled_detours.py
foreach ($suite in 'pairs','refine','small','extended','broadband_weak','broadband_material','broadband_scalar','broadband_fine') {
    python -B experiments/hadamard_pinch/run_longitudinal_modes.py $suite
}
python -B experiments/hadamard_pinch/run_sourced_variation.py transverse --nodes 5
python -B experiments/hadamard_pinch/run_sourced_variation.py transverse --nodes 5 --epsilon .0004
python -B experiments/hadamard_pinch/run_sourced_variation.py transverse --nodes 9 --epsilon .0004
python -B experiments/hadamard_pinch/run_sourced_variation.py low_flow --nodes 5
foreach ($case in 'low_flow','transverse') {
    python -B experiments/hadamard_pinch/replay_sourced_optimal.py $case
    python -B experiments/hadamard_pinch/replay_sourced_optimal.py $case --amplitude .00002
}
python -B experiments/hadamard_pinch/analyze_followup.py
```

The reference data includes effective solver parameters, package versions, source hashes, native recovery diagnostics, and precision sweeps. The summary includes both directional audits, continuation, homogeneous and spatial runs, the autonomous shots, and propagator gains. [Reference JSON](hadamard_pinch_reference.json) · [Summary JSON](hadamard_pinch_summary.json) · [Frozen-entropy directional arrays](hadamard_pinch_directional_scan.npz)
