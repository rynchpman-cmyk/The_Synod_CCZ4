# Boundary-instrumented N=16 extension through t=0.068

## Verdict

This campaign answers the carrier-balance question cleanly: the observed
few-parts-per-million carrier-number decrease is outward transport through the
finite radiative boundary, not an interior continuity error.

It also finds another decrease in the delayed Hawking-mass response exponent:

| interval | fitted power \(p\) in \(\Delta M_H\propto t^p\) | \(R^2_{\log}\) |
|---|---:|---:|
| 0.004--0.016 | 3.3123 | 0.99817 |
| 0.016--0.024 | 2.8676 | 0.99994 |
| 0.024--0.036 | 2.7071 | 0.99992 |
| 0.036--0.052 | 2.5349 | 0.99990 |
| 0.052--0.068 | 2.3765 | 0.99993 |
| late 0.056--0.068 | 2.3573 | 0.99996 |
| trailing 0.060--0.068 | 2.3388 | 0.99998 |

Thus the local exponent still moves toward two, with no plateau or reversal.
This remains evidence for an approach toward quadratic growth, not proof of an
exact asymptotic exponent.

One preregistered control now marginally fails: the outer-to-inner mass-split
ratio is 1.0046%, just above its 1% cutoff. The result is therefore qualified,
not an all-gates-pass production claim. The response is beginning to reach the
outer extraction sphere, making a larger box the correct next geometry.

## Discrete boundary-flux accounting

The Theory 3.3 production solver now exposes the carrier-number flux through
every physical face using the exact outgoing-only numerical flux used in the
carrier continuity RHS. During RK4, the stage-weighted boundary flux is
accumulated with the same quadrature as the evolved charge.

For each step the closure diagnostic is

\[
\mathcal R_N = \Delta N_D + \int \Phi_{\partial\Omega}\,dt.
\]

Instrumentation begins explicitly at the \(t=0.052\) restart join; older
samples are marked uninstrumented rather than assigned inferred fluxes.

Before the extension, direct face integration and volume integration of the
actual 3-D checkpoint RHS agreed within \(6\times10^{-18}\). Over the completed
instrumented interval:

| branch | measured \(\Delta N_D\) | integrated outward flux | closure residual |
|---|---:|---:|---:|
| fast, \(\tau_D=0.03\) | -1.26624953e-7 | 1.26624953e-7 | 7.96e-18 |
| slow, \(\tau_D=0.30\) | -2.16103100e-7 | 2.16103100e-7 | -2.95e-17 |

The maximum relative closure residual is \(2.19\times10^{-16}\). The slow
branch's larger raw carrier change is completely accounted for by its larger
outward flux. There is no detected interior carrier-number source or sink at
this precision.

The final outward rates are \(8.57\times10^{-6}\) and
\(1.58\times10^{-5}\) carrier number per unit time in the fast and slow
branches, respectively. Both rates are increasing, reinforcing the case for a
larger domain before another long extension.

## Restart integrity

The histories now contain 35 samples at uniform \(\Delta t=0.002\). Both
branches have exactly one sample at each restart join,
\(t=0.024,0.036,0.052\), and retain the original \(t=0\) baselines. The nine
samples from \(0.052\) through \(0.068\) are explicitly flux-instrumented.

## Constitutive domain

The widened characteristic audit remains unchanged and passes throughout:

| domain diagnostic | worst audited value |
|---|---:|
| maximum characteristic speed | 0.95953 |
| maximum eigenvector condition number | 3.8803 |
| minimum Legendre eigenvalue | 3.1275 |
| minimum thermodynamic eigenvalue | 1.6065 |

The maximum relative Lorentz factor reached is 1.82072, still well below the
qualified cap of 4.

## Final response at t=0.068

| quantity | slow - fast |
|---|---:|
| Hawking mass, radius 1.5 | 1.57077e-7 |
| Hawking mass, radius 2.5 | 1.57899e-8 |
| Hawking mass, radius 3.0 | 1.57802e-9 |
| inner carrier number | 3.62450e-4 |
| inner baryon mass | -1.00812e-5 |
| maximum relative Lorentz factor | 0.71129 |

The inner split is \(2.78\times10^{-6}\) of the fast branch's enclosed inner
Hawking mass. The outer split remains two orders of magnitude smaller than the
inner response, but its 1.0046% ratio narrowly crosses the fixed 1% control.

## Numerical gates

Both branches finish with zero recovery failures and finite histories. Maximum
relative baryon change is \(3.52\times10^{-7}\). The raw carrier changes reach
\(2.39\times10^{-6}\) and \(3.62\times10^{-6}\), but the new flux-corrected
balance closes to \(2.19\times10^{-16}\) relatively. The largest final core
Hamiltonian L2 residual is \(2.8115\times10^{-3}\).

Constitutive, restart-integrity, resolution-overlap, baryon-balance,
carrier-flux-closure, and recovery gates pass. The legacy 1% outer-mass control
fails marginally, so the aggregate campaign flag is false. The repository test
suite now contains 112 passing tests, including direct face/RHS agreement and
RK4 flux-closure regressions.

## Reproduction

Starting from the completed \(t=0.052\) bundles:

```powershell
$env:PYTHONPATH='src'

python experiments/run_theory33_spherical_collision.py `
  --output runs/theory33_spheres_long/n016_fast `
  --restart-from runs/theory33_spheres_long/n016_fast `
  --points 16 --final-time 0.068 --dt 0.002 `
  --relaxation-time 0.03 --maximum-relative-lorentz-factor 4.0

python experiments/run_theory33_spherical_collision.py `
  --output runs/theory33_spheres_long/n016_slow `
  --restart-from runs/theory33_spheres_long/n016_slow `
  --points 16 --final-time 0.068 --dt 0.002 `
  --relaxation-time 0.30 --maximum-relative-lorentz-factor 4.0

python experiments/analyze_theory33_spherical_long.py
```

The analysis writes the main growth figure, final slices, JSON result, and the
new `runs/theory33_spherical_boundary_flux.png` closure figure.

## Next target

Do not simply append farther in the same box. The carrier flux is physically
accounted for, but it is increasing, and the outer mass control has begun to
respond. The clean next campaign is a fresh, larger-domain matched pair with
the flux diagnostic enabled from \(t=0\). That will restore a causally quiet
outer extraction zone and provide an independent test of the exponent trend.
