# KKT/MOTS Experimental Horizon Observer

## Status

The repository now contains an opt-in, read-only constrained surface observer
for marginally outer trapped surfaces (MOTSs). It does **not** modify the CCZ4,
GRHD, mixed-Proca, or Theory 3.3 evolution equations.

Primary implementation:
[`src/causal_repair/kkt_horizon.py`](../src/causal_repair/kkt_horizon.py).

First binary constraint-slice probe:
[`run_kkt_mots_brill_lindquist.py`](run_kkt_mots_brill_lindquist.py).

## Mathematical contract

On an ADM slice \((\Sigma,\gamma_{ij},K_{ij})\), a star-shaped candidate surface
is represented by

\[
r=h(\theta,\phi;c)
=\exp\!\left(\sum_A c_A Y_A(\theta,\phi)\right).
\]

The monopole basis function is one, so \(c_0\) is the logarithm of the
reference coordinate radius. Real spherical-harmonic deformation modes are
orthonormal. Dipoles are disabled by default because they mostly duplicate a
change of the supplied coordinate center.

The outgoing expansion uses the code's ADM sign convention:

\[
\Theta_+=D_i s^i+K_{ij}s^is^j-K.
\]

The surface is a MOTS when \(\Theta_+=0\) everywhere. The implementation
projects the collocation residual into the represented spectral space,

\[
R_A=M^{-1}_{AB}\int Y_B\Theta_+\,d\Omega,
\qquad
M_{AB}=\int Y_AY_B\,d\Omega,
\]

and minimizes the resolved merit

\[
F(c)=\frac{1}{8\pi}R^TMR.
\]

The solve uses a damped Gauss--Newton direction, a trust radius, backtracking,
and box-constrained KKT active sets. The radius remains positive through the
exponential representation. Monopole and deformation bounds prevent the
surface from leaving the interpolation patch or entering an unresolved
puncture region.

The solver reports two distinct residuals:

- `spectral_residual_norm` is the norm resolved by the selected modes and is
  the convergence gate;
- `rms_expansion` and `maximum_expansion` retain all sampled angular error,
  including unresolved Cartesian-grid anisotropy.

This distinction prevents high-order grid noise from being silently fitted by
a low-order surface representation.

## What “KKT” means here

KKT iteration is numerical pseudo-time. It is not coordinate time, proper
time, a covariant force, surface gravity, or a new stress-energy source.

For coefficient bounds \(l_A\le c_A\le u_A\), the result records

- active lower and upper indices;
- non-negative lower and upper multipliers;
- projected-gradient norm;
- stationarity/complementarity residual;
- accepted step and line-search history.

Minkowski space provides a useful negative control: its spherical outgoing
expansion cannot vanish. The optimizer reaches its maximum-radius constraint,
reports a positive upper multiplier, and returns `found=false` rather than
inventing a horizon.

## Shape-response spectrum

At the final surface, finite differences construct

\[
J_{pA}=\frac{\partial\Theta_+(\Omega_p)}{\partial c_A}.
\]

Projecting \(J\) into the represented basis gives a finite-dimensional
shape-response matrix. Its complex eigenvalues and condition number are
reported.

This raw matrix is a discretized response of the chosen radial
parameterization and is retained as a comparison channel.

`NormalMOTSStabilityOperator` now supplies a separately normalized spectrum.
It converts coefficient perturbations to fixed physical displacement along the
spatial unit normal, uses the induced area measure for both domain and range,
and repeats the central difference at two physical step sizes. It reports the
normalizer condition and round-trip error, step-convergence error, operator
condition, complex spectrum, and area-adjoint defect.

The normalized operator removes raw log-radius scaling, but it remains a
finite-dimensional star-shaped discretization. The high-resolution audit found
angular convergence at fixed grid and failed Cartesian convergence; see
[`KKT_MOTS_NORMAL_STABILITY_REPORT.md`](KKT_MOTS_NORMAL_STABILITY_REPORT.md).

## Independent branch tracking

`MOTSBranchTracker` accepts named seeds, such as

- `individual_plus`;
- `individual_minus`;
- `common_inner`;
- `common_outer`.

Every branch is solved independently. A failed branch is retained and may be
reacquired later. The tracker reports newly found and newly lost transitions,
success/failure counts, and first-found time. Its coefficient continuation and
branch counters have a validated JSON-compatible checkpoint format, so an
observer can accompany restart-and-append production runs.

For a Theory 3.3 state, the legacy observation path is intentionally explicit:

```python
adm = solver.ccz4.to_adm(state.geometry)
observations = tracker.observe(adm)
```

Nothing is added to `Theory33ProductionSolver.diagnostics()` by default because
a multi-branch distorted-surface solve is substantially more expensive than a
local evolution diagnostic.

### Corrected Cartesian and CCZ4 snapshot observer

`ParametricAxisymmetricMOTSFinder` replaces volume differentiation of the
level-set normal with direct induced-surface geometry while still sampling a
general Cartesian ADM slice. `CCZ4PhysicalNormalMOTSTracker` converts a CCZ4
snapshot to physical ADM fields, continues named branches, and appends the
induced-area physical-normal spectrum for every surface that is found. Its
state is restartable and never feeds back into the evolution.

The derivation gates, direct-oracle comparison, resolution limits, and complete
attachment example are in
[`KKT_MOTS_CARTESIAN_CALIBRATION_REPORT.md`](KKT_MOTS_CARTESIAN_CALIBRATION_REPORT.md).
The subsequent local-refinement and grid-phase gate, including the calibrated
CCZ4 single-snapshot policy, is in
[`KKT_MOTS_PHASE_QUALIFICATION_REPORT.md`](KKT_MOTS_PHASE_QUALIFICATION_REPORT.md).

The first dynamical moving-puncture campaign, including the corrected lapse
gate, stability-sign branch split, two-resolution pilot, and qualified launch
contract, is in
[`CCZ4_PUNCTURE_MOTS_CAMPAIGN_REPORT.md`](CCZ4_PUNCTURE_MOTS_CAMPAIGN_REPORT.md).

## Brill--Lindquist common-horizon probe

Run the first controlled binary scan with:

```powershell
python experiments/run_kkt_mots_brill_lindquist.py
```

Add `--common-only --normal-stability` for focused bifurcation scans with the
physical-normal operator.

The output defaults to `runs/kkt_mots_brill_lindquist.json`. It contains the
complete KKT, surface, area, mass, response-spectrum, and transition telemetry
for two individual seeds and one common seed.

Brill--Lindquist data form a sequence of time-symmetric constraint slices. The
separation sequence is not a dynamical inspiral, and its sequence index is not
physical time. The scan only brackets the sampled separation at which this
resolution and spectral truncation cease to find a common MOTS.

Useful convergence attacks are:

```powershell
python experiments/run_kkt_mots_brill_lindquist.py --resolution 40 --maximum-degree 4
python experiments/run_kkt_mots_brill_lindquist.py --resolution 56 --maximum-degree 4
python experiments/run_kkt_mots_brill_lindquist.py --resolution 56 --maximum-degree 6
```

A transition is credible only if its separation converges under both Cartesian
resolution and harmonic degree, while the full sampled expansion decreases.

### Direct conformal benchmark

The independent axisymmetric runner
[`run_kkt_mots_brill_lindquist_direct.py`](run_kkt_mots_brill_lindquist_direct.py)
evaluates the conformal minimal-surface equation directly on the harmonic
surface. It contains no Cartesian volume derivative or trilinear sampling.
The exact Schwarzschild limit and high-order angular convergence predict

\[
d_\star/M_{\rm bare,total}=0.766197\pm0.000005.
\]

See
[`KKT_MOTS_DIRECT_SURFACE_REPORT.md`](KKT_MOTS_DIRECT_SURFACE_REPORT.md) for the
derivation, convergence table, and reproduction commands.

## Falsification program

The observer is designed to test, rather than assume, the proposed black-hole
analogy.

1. **Common-surface formation:** test whether active-set or response-spectrum
   changes precede the independently detected common MOTS.
2. **Branch survival:** retain the individual and common-inner branches after
   an outer common branch appears and measure their actual persistence.
3. **Stability correspondence:** compare the shape-response eigenvalues with a
   separately discretized normal MOTS stability operator.
4. **Ringdown:** compare excess horizon multipoles against finite-radius
   `Psi4` modes without forcing either decay law.
5. **Area law:** measure area changes on the selected outer dynamical branch;
   never project the area to be non-decreasing.
6. **Smarr audit:** independently measure \(M,J,Q,A,\kappa,\Omega,\Phi\) and
   evaluate the Smarr residual. Do not identify a raw KKT multiplier with
   surface gravity by notation alone.

## Validation completed

- real spherical-harmonic quadrature orthonormality;
- negative-control rejection in Minkowski space with a valid active-bound
  multiplier;
- convergence of a distorted \(\ell=2\) seed to the isotropic Schwarzschild
  MOTS;
- finite response eigenvalues on that solved surface;
- independent branch continuation;
- branch checkpoint/restore equivalence;
- Brill--Lindquist smoke scan with both a detected and a rejected common seed.

## Qualification boundaries

- The surface must be star-shaped about its supplied center.
- The current level-set geometry uses Cartesian finite differences and
  trilinear sampling; horizon accuracy cannot exceed the underlying slice.
- Spectral convergence does not imply that unresolved full expansion is small.
- The reported response spectrum is parameterization-dependent.
- No angular momentum, quasi-local surface gravity, or invariant horizon
  multipole extractor has yet been connected to this observer.
- The repository still needs puncture/excision initial data and a qualified
  long black-hole evolution before the complete dynamical-merger experiment.
- Feeding a slice-wise MOTS solve back into local spacetime or matter equations
  would introduce nonlocal elliptic dependence. Such feedback is prohibited
  until a separate causal covariant action and principal-symbol audit exist.

The present module is therefore an instrumented hypothesis tester, not a new
law of black-hole dynamics.
