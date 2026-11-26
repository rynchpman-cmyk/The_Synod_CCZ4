# Theory 3.3 carrier-loading prediction

## Prediction

The fast-drag/slow-drag gravitational branch separation follows the carrier
master-function energy exponent.

For the production33 compact-collision analogue evolved to (t=0.12), a fit
over equilibrium carrier fractions (f_D=0.1\ldots2.0) gives

\[
\Delta C_{\rm peak}
= (3.31\times10^{-4}) f_D^{1.787},
\qquad R^2_{\log}=0.99993,
\]

\[
\Delta\alpha_{\min}
= (1.22\times10^{-5}) f_D^{1.837},
\qquad R^2_{\log}=0.99988.
\]

Here Δ compares the fast-drag branch, τ_D=0.03, with the slow-drag
branch, τ_D=0.30. The two fitted powers bracket

\[
1+c_D^2=1+0.9^2=1.81,
\]

which is exactly the density exponent of the implemented carrier energy

\[
\rho_D=K_D n_D^{1+c_D^2}.
\]

This strongly suggests that the leading gravitational sensitivity to carrier
loading is controlled by the carrier equation of state, rather than scaling
linearly with carrier number. It is a numerical prediction, not yet an
analytic theorem or a multi-resolution determination of the exponent.

## Qualified loading domain

Every evolved fraction passed cellwise Legendre, thermodynamic-convexity,
strong-hyperbolicity, and causal-speed gates before evolution.

| carrier fraction | maximum sampled speed | minimum Legendre eigenvalue | minimum thermodynamic eigenvalue |
|---:|---:|---:|---:|
| 0.05 | 0.98040 | 1.13006 | 1.01534 |
| 0.10 | 0.98290 | 0.99061 | 0.90238 |
| 0.20 | 0.98572 | 0.86838 | 0.80330 |
| 0.35 | 0.98826 | 0.78078 | 0.73218 |
| 0.50 | 0.99002 | 0.72963 | 0.69049 |
| 1.00 | 0.99382 | 0.63959 | 0.61609 |
| 2.00 | 0.99843 | 0.56067 | 0.54675 |

The next tested loading, (f_D=4), remains strongly hyperbolic but fails the
causal cone with a maximum sampled characteristic speed of 1.03779. It is
therefore excluded rather than evolved. The regression suite now locks this
boundary in explicitly.

The parameter (f_D) is an equilibrium carrier-number ratio, not a carrier
mass fraction. Values above one are meaningful for a constituent whose
per-particle energy differs from the baryonic constituent.

## Fourteen-case long evolution

All cases use peak baryon density 0.025, initial baryon collision speed 0.18,
additional carrier counterflow 0.28, and a periodic contracting CMC slice.

| (f_D) | compression fast | compression slow | Δ compression | Δ minimum lapse | final γ_D−1 fast | final γ_D−1 slow |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.146155 | 1.146153 | 1.52e-6 | 5.14e-8 | 0.00689 | 0.03609 |
| 0.10 | 1.146191 | 1.146186 | 5.33e-6 | 1.80e-7 | 0.00689 | 0.03618 |
| 0.20 | 1.146330 | 1.146311 | 1.87e-5 | 6.32e-7 | 0.00692 | 0.03642 |
| 0.35 | 1.146694 | 1.146643 | 5.13e-5 | 1.74e-6 | 0.00691 | 0.03702 |
| 0.50 | 1.147241 | 1.147143 | 9.74e-5 | 3.34e-6 | 0.00688 | 0.03793 |
| 1.00 | 1.150327 | 1.149991 | 3.36e-4 | 1.19e-5 | 0.00674 | 0.04338 |
| 2.00 | 1.161951 | 1.160832 | 1.12e-3 | 4.48e-5 | 0.00626 | 0.09288 |

The absolute gravitational response grows strongly with loading: the minimum
lapse changes from about 0.7195 at (f_D=0.05) to about 0.6888 at (f_D=2).
Fast drag consistently produces slightly greater compression and lapse
collapse than slow drag because it thermalizes relative-flow energy early.

All fourteen runs conserve both number currents to roundoff, have positive
integrated entropy production, and finish with zero primitive-recovery
failures.

## Doubled-resolution endpoint

The causal-edge pair at (f_D=2) was repeated with 32 rather than 16 points
and a timestep reduced from 0.002 to 0.001.

| quantity | fast drag | slow drag |
|---|---:|---:|
| peak compression | 1.187316 | 1.185796 |
| minimum lapse | 0.6900465 | 0.6900924 |
| final max γ_D−1 | 0.004762 | 0.036276 |
| final Hamiltonian L2 | 0.080016 | 0.080024 |
| recovery failures | 0 | 0 |

The compression split is (1.52\times10^{-3}), and the lapse split is
(4.59\times10^{-5}), with the same sign and scale as the coarse result.
The Hamiltonian diagnostic improves from about 0.177 to 0.080. The absolute
compression and slow-branch counterflow remain resolution-sensitive, so the
fitted power law should be treated as a precise hypothesis for the next
convergence campaign, not as a final continuum exponent.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python experiments/run_theory33_collision.py `
  --fraction-sweep --final-time 0.12 --dt 0.002 `
  --output runs/theory33_fraction_sweep

python experiments/render_theory33_fraction_sweep.py `
  runs/theory33_fraction_sweep/scan_summary.json `
  --output runs/theory33_fraction_sweep.png
```

The doubled-resolution trajectories and restart checkpoints are stored under
`runs/theory33_fraction_validation_fast` and
`runs/theory33_fraction_validation_slow`.

## Interpretation boundary

This result belongs to the periodic planar CMC collision analogue. It does not
yet establish the same exponent for isolated spherical stars, nonzero
entrainment, other carrier sound speeds, or the continuum limit. A decisive
follow-up is to vary (c_D): if the mechanism is correct, the measured
loading exponent should track (1+c_D^2).

That falsification campaign is complete in
[`THEORY33_EOS_FALSIFICATION_REPORT.md`](THEORY33_EOS_FALSIFICATION_REPORT.md),
and the exponent tracks the prediction across all three tested equations of
state.
