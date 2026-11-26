# Theory 3.3 multi-resolution coefficient sweep

## Result

The coefficient law survives continuum extrapolation.

The campaign contains 120 matched evolutions over

\[
N=(16,24,32),\quad K_D=(0.05,0.10,0.20,0.40),\quad
f_D=(0.1,0.2,0.5,1,2),\quad \tau_D=(0.03,0.30).
\]

The time step scales with grid spacing:
\(\Delta t=(0.002,0.001\overline{3},0.001)\). The 40 existing coarse
trajectories were reused and 80 new trajectories were generated. Each
resolution independently solves its CMC initial data.

For each \((K_D,f_D)\), the fast/slow branch split was extrapolated with the
preselected nominal second-order model

\[
\Delta(N)=\Delta_\infty+aN^{-2}.
\]

A joint fit of all 20 extrapolated points gives

\[
\Delta C_\infty
=5.9511\times10^{-3}
K_D^{0.97874}f_D^{1.46591},
\qquad R^2_{\log}=0.999758,
\]

and

\[
\Delta\langle\alpha\rangle_\infty
=1.9336\times10^{-5}
K_D^{0.98601}f_D^{1.49837},
\qquad R^2_{\log}=0.999914.
\]

The target carrier-fraction power is \(1+c_D^2=1.49\), and the target
coefficient power is one. The continuum mean-lapse response lands within
0.0084 of both predicted powers. Compression retains a small nonlinear
fraction-power deficit while its coefficient power moves closer to unity.

## Coefficient-by-coefficient continuum powers

| \(K_D\) | compression \(f_D\) power | mean-lapse \(f_D\) power |
|---:|---:|---:|
| 0.05 | 1.48375 | 1.51011 |
| 0.10 | 1.47885 | 1.50580 |
| 0.20 | 1.46426 | 1.49741 |
| 0.40 | 1.43676 | 1.48018 |

Every individual continuum loading fit has logarithmic
\(R^2>0.99958\). Fitting their amplitudes across \(K_D\) gives

\[
A_C\propto K_D^{0.96118},\qquad
A_{\langle\alpha\rangle}\propto K_D^{0.97492}.
\]

These nested amplitude powers differ slightly from the joint two-variable
powers because the fitted \(f_D\) exponent drifts weakly with \(K_D\).

## Convergence

The compression sequences have a maximum normalized second-order-fit residual
of 0.422%; their \(N=32\) to continuum corrections range from 7.55% to 8.85%.
The final mean-lapse sequences have a maximum residual of 0.287%, with
continuum corrections of only 0.76% to 1.60%.

As a sensitivity check, one convergence order was fitted across all sequences
of each observable. The observed orders are 2.19 for compression and 4.29 for
the periodic spatial mean of the lapse. Repeating the continuum analysis with
those orders gives

| observable | nominal-order \(K_D\) power | shared-order \(K_D\) power |
|---|---:|---:|
| compression amplitude | 0.96118 | 0.96076 |
| mean-lapse amplitude | 0.97492 | 0.97599 |

The individual fraction powers move by less than 0.0011. The physical result
is therefore insensitive to using the nominal or observed shared order.

## Revision of the minimum-lapse diagnostic

The original coarse campaign used the difference between the pointwise
minimum lapse in each branch. The three-level study shows that this is not a
continuum-smooth observable at these resolutions: 8 of 20 sequences are
nonmonotone and the maximum normalized extrapolation residual is 38%.

The cause is the discrete spatial minimum switching between grid samples near
a smooth extremum. A local quadratic reconstruction does not reliably remove
the switching in the strongest cases. The continuum lapse claim therefore
uses the final-time spatial mean,

\[
\Delta\langle\alpha\rangle
=\langle\alpha\rangle_{\tau_D=0.30}
-\langle\alpha\rangle_{\tau_D=0.03},
\]

which has clean convergence. The previous point-minimum result remains a
useful coarse visualization but should not be quoted as continuum-validated.

## Numerical gates

All 120 trajectories pass the constitutive, causal-cone, convexity, and CMC
gates. There are zero primitive-recovery failures. Both currents are conserved
with a maximum relative error of \(8.38\times10^{-16}\). The minimum cumulative
drag, projection, and total step entropy changes across the campaign are all
positive.

The executable operational verdict requires positive extrapolated splits,
normalized continuum residuals below 1%, individual power-law
\(R^2>0.995\), fraction powers within 0.1 of 1.49, and amplitude powers within
0.1 of unity. Both accepted observables pass every condition.

## Reproduction

The coarse inputs are the existing `theory33_kd_005`, `theory33_kd_010`,
`theory33_eos_cs070`, and `theory33_kd_040` scans. The two new levels can be
regenerated for each coefficient with:

```powershell
$env:PYTHONPATH='src'

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 --carrier-K 0.05 `
  --points 24 --final-time 0.12 --dt 0.0013333333333333333 `
  --output runs/theory33_kd_continuum/n024/kd005

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 --carrier-K 0.05 `
  --points 32 --final-time 0.12 --dt 0.001 `
  --output runs/theory33_kd_continuum/n032/kd005
```

Repeat with \((0.10,0.20,0.40)\) and output tags
`kd010`, `kd020`, and `kd040`, then run:

```powershell
python experiments/analyze_theory33_kd_continuum.py
```

The analysis writes `runs/theory33_kd_continuum.json`,
`runs/theory33_kd_continuum.png`, and
`runs/theory33_kd_continuum_convergence.png`.

## Interpretation boundary

This establishes the factorized law in the continuum limit of the present
periodic planar CMC reference problem. It does not establish gauge-independent
observability or transfer the law automatically to spherical, isolated, or
three-dimensional collapse. The most valuable next test is an isolated
spherical two-current collision using an invariant curvature or mass-aspect
response in place of the lapse.
