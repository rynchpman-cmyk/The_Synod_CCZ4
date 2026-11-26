# Theory 3.3 carrier-EOS falsification test

## Verdict

The prediction survived.

The preceding carrier-loading campaign suggested that the fast-drag/slow-drag
gravitational branch separation scales with carrier fraction as

\[
\Delta\mathcal G\propto f_D^{1+c_D^2},
\]

where the implemented carrier energy is

\[
\rho_D=K_D n_D^{1+c_D^2}.
\]

The falsification campaign varied (c_D) independently while keeping the
collision, loading grid, master-function coefficient (K_D), chemical
susceptibility, drag times, resolution, and evolution duration fixed.

| (c_D) | predicted (1+c_D^2) | measured compression power | measured lapse power |
|---:|---:|---:|---:|
| 0.5 | 1.25 | 1.18937 | 1.26839 |
| 0.7 | 1.49 | 1.44997 | 1.51308 |
| 0.9 | 1.81 | 1.78745 | 1.83687 |

Every individual loading fit over
(f_D=(0.1,0.2,0.5,1,2)) has logarithmic (R^2>0.9993). The executable
operational criterion required (R^2>0.995) and an absolute exponent error
below 0.1 for both observables at every sound speed. All six fits pass.

Across the three equations of state, the measured lapse exponent obeys

\[
p_\alpha
=1.01496(1+c_D^2)+0.00009,
\qquad R^2=0.999995.
\]

The compression exponent obeys

\[
p_C
=1.06728(1+c_D^2)-0.14311,
\qquad R^2=0.999933.
\]

The lapse result is essentially identity tracking. Compression shows a small
systematic finite-resolution offset but follows the predicted EOS dependence
with extremely high correlation.

## What would have falsified it

The hypothesis would have failed the campaign if either geometric observable
kept the original (c_D=0.9) exponent as (c_D) changed, if its exponent
moved in the wrong direction, if the branch splits ceased to be power laws,
or if any measured exponent missed (1+c_D^2) by 0.1 or more. None occurred.

No coefficient was retuned between sound speeds. Lower (c_D) produces a
larger split amplitude at the simulated subunit densities, as expected because
reducing the power of (n_D<1) increases the carrier energy density.

## Dynamical and constitutive gates

All 20 new (c_D=0.5,0.7) runs pass their characteristic, causal-cone,
Legendre, and thermodynamic-convexity gates before evolution. Both number
currents remain conserved to roundoff, total entropy increases, and every run
finishes with zero primitive-recovery failures.

At (f_D=2), the maximum sampled characteristic speeds are 0.81566 for
(c_D=0.5), 0.90925 for (c_D=0.7), and 0.99843 for (c_D=0.9). Thus the
tracking result is not produced by all three scans sitting at the same causal
boundary.

## Doubled-resolution low-sound-speed endpoint

The strongest branch split, (c_D=0.5, f_D=2), was repeated with 32 points
and (dt=0.001), rather than 16 points and (dt=0.002).

| quantity | coarse | doubled resolution |
|---|---:|---:|
| compression split | 0.003733 | 0.004998 |
| lapse split | 3.914e-5 | 3.957e-5 |
| fast final γ_D−1 | 0.000955 | 0.000991 |
| slow final γ_D−1 | 0.072711 | 0.063488 |
| Hamiltonian L2 | 0.2356 | 0.1315 |

The lapse split changes by only about 1.1%, while the compression split retains
its sign and grows by about 34%. This supports the physical reality of the
branch separation, but a multi-resolution fit across every sound speed is
still needed before assigning continuum error bars to the exponents.

## Reproduction

```powershell
$env:PYTHONPATH='src'

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.5 `
  --final-time 0.12 --dt 0.002 --output runs/theory33_eos_cs050

python experiments/run_theory33_collision.py `
  --eos-loading-sweep --carrier-sound-speed 0.7 `
  --final-time 0.12 --dt 0.002 --output runs/theory33_eos_cs070

python experiments/analyze_theory33_eos_falsification.py `
  runs/theory33_eos_cs050/scan_summary.json `
  runs/theory33_eos_cs070/scan_summary.json `
  runs/theory33_fraction_sweep/scan_summary.json
```

The analysis writes a machine-readable verdict and a comparison figure to
`runs/theory33_eos_falsification.json` and
`runs/theory33_eos_falsification.png`.

## Interpretation boundary

This is compelling internal numerical evidence for the proposed scaling law,
not an observational prediction or a continuum proof. The experiment remains
a periodic planar CMC collision. A stronger independent test would change the
carrier coefficient (K_D): the EOS interpretation predicts that (K_D)
should change the split amplitude while leaving the loading exponent fixed.

That independent test is now complete; see `THEORY33_KD_ATTACK_REPORT.md`.
