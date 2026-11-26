# Theory 3.3 carrier-coefficient attack

## Verdict

The independent coefficient attack supports the factorized prediction

\[
\Delta\mathcal G\propto K_D f_D^{1+c_D^2}.
\]

At fixed carrier sound speed \(c_D=0.7\), the campaign varied
\(K_D=(0.05,0.10,0.20,0.40)\), carrier fraction
\(f_D=(0.1,0.2,0.5,1,2)\), and the fast/slow drag times
\(\tau_D=(0.03,0.30)\). This gives 40 trajectories, of which 30 were new;
the previously generated \(K_D=0.20\) branch was reused without retuning.

| \(K_D\) | compression power in \(f_D\) | lapse power in \(f_D\) |
|---:|---:|---:|
| 0.05 | 1.47613 | 1.52259 |
| 0.10 | 1.46767 | 1.51919 |
| 0.20 | 1.44997 | 1.51308 |
| 0.40 | 1.41552 | 1.49998 |

The predicted loading power is \(1+c_D^2=1.49\). The lapse exponent spans
only 0.0226 across an eightfold change in \(K_D\). The compression exponent
spans 0.0606 and drifts downward at the strongest coefficient, consistent
with a modest nonlinear strong-gravity correction. Every individual power-law
fit has logarithmic \(R^2>0.9991\).

The fitted amplitudes obey

\[
A_C=0.003960\,K_D^{0.94855},\qquad R^2=0.999719,
\]

and

\[
A_\alpha=8.797\times10^{-5}\,K_D^{0.98092},\qquad R^2=0.999970.
\]

Thus \(K_D\) changes the response amplitude almost linearly while leaving the
carrier-fraction power nearly fixed. The lapse is the cleanest realization of
the factorized law.

## Executable verdict

The analysis requires every loading curve to have \(R^2>0.995\), every
measured fraction exponent to lie within 0.1 of 1.49, each observable's
exponent span to remain below 0.1, and both amplitude powers in \(K_D\) to lie
within 0.15 of unity with \(R^2>0.995\). All conditions pass. The criteria and
full-precision results are stored in `runs/theory33_kd_attack.json`; the
four-panel comparison is `runs/theory33_kd_attack.png`.

## Unexpected low-stiffness boundary

The coefficient scan also exposes a useful admissibility constraint. At the
standard counterflow audit state and \(f_D=2\), every evolved coefficient is
strongly hyperbolic and causal:

| \(K_D\) | maximum characteristic speed |
|---:|---:|
| 0.05 | 0.98325 |
| 0.10 | 0.93683 |
| 0.20 | 0.90925 |
| 0.40 | 0.89397 |

But \(K_D=0.025\) remains strongly hyperbolic while becoming acausal:
the maximum speed is 1.102 at \(f_D=1\) and 1.473 at \(f_D=2\). Here reducing
the carrier self-stiffness lets the chemical cross-coupling dominate the
mixed characteristic cone. This boundary is now covered by a regression test.

## Doubled-resolution endpoint

The hardest endpoint, \(K_D=0.40, f_D=2\), was repeated at 32 points with
\(dt=0.001\), compared with the 16-point, \(dt=0.002\) scan.

| quantity | coarse | doubled resolution |
|---|---:|---:|
| compression split | 0.004111 | 0.005606 |
| lapse split | 1.0190e-4 | 1.0394e-4 |
| fast final \(\gamma_D-1\) | 0.000928 | 0.001000 |
| slow final \(\gamma_D-1\) | 0.07638 | 0.05426 |
| fast Hamiltonian L2 | 0.2460 | 0.1416 |
| slow Hamiltonian L2 | 0.2461 | 0.1416 |

The lapse split changes by about 2.0%. Compression retains its sign and grows
by about 36%, so its existence is robust but its coarse amplitude is not yet a
continuum estimate. Both refined runs conserve each current to roundoff,
produce positive cumulative entropy, and finish with zero recovery failures.

## Reproduction

```powershell
$env:PYTHONPATH='src'

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 --carrier-K 0.05 `
  --final-time 0.12 --dt 0.002 --output runs/theory33_kd_005

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 --carrier-K 0.10 `
  --final-time 0.12 --dt 0.002 --output runs/theory33_kd_010

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 --carrier-K 0.40 `
  --final-time 0.12 --dt 0.002 --output runs/theory33_kd_040

python experiments/analyze_theory33_kd_attack.py `
  runs/theory33_kd_005/scan_summary.json `
  runs/theory33_kd_010/scan_summary.json `
  runs/theory33_eos_cs070/scan_summary.json `
  runs/theory33_kd_040/scan_summary.json
```

## Interpretation boundary

This is an internal numerical prediction of the reference model, not an
observational gravity claim or a continuum proof. It is still a periodic
planar CMC collision. The next stringent numerical test would perform a
multi-resolution coefficient sweep and fit the lapse and compression powers
after continuum extrapolation; a physically broader test would repeat the
factorization in spherical symmetry with asymptotically isolated data.

The multi-resolution test is now complete. See
`THEORY33_KD_CONTINUUM_REPORT.md`; it also documents why the pointwise minimum
lapse was replaced by a continuum-smooth spatial-mean lapse response.
