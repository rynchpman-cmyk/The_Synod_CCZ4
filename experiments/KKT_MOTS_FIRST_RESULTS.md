# First KKT/MOTS Result: Common-Horizon Response Crossing

> **Superseded qualification:** the later physical-normal \(N=64/80\) audit
> found excellent angular convergence but failed Cartesian convergence. See
> [`KKT_MOTS_NORMAL_STABILITY_REPORT.md`](KKT_MOTS_NORMAL_STABILITY_REPORT.md).

## Outcome

The new constrained spectral observer passes its first black-hole geometry
experiment. On equal-mass, time-symmetric Brill--Lindquist binary slices it
finds a common MOTS at small puncture separation and ceases to find that branch
as separation is increased.

In the strictest completed scan, \(N=48\), axisymmetric modes through
\(\ell=6\), and resolved-expansion tolerance \(5\times10^{-3}\):

\[
d=0.78:\quad \text{common MOTS found},
\]

\[
d=0.79:\quad \text{common MOTS not converged}.
\]

The finite-resolution detection bracket is therefore

\[
\boxed{0.78\le d_{\rm transition}<0.79}.
\]

This is an algorithmic detection bracket, not yet a continuum proof that no
common MOTS exists above it.

## Strict-scan data

| Separation | Found | Resolved residual | Full RMS expansion | Minimum response real part | Response condition number |
|---:|:---:|---:|---:|---:|---:|
| 0.75 | yes | 2.077e-3 | 6.522e-2 | +0.3440 | 75.8 |
| 0.76 | yes | 3.312e-4 | 7.320e-2 | +0.3477 | 77.1 |
| 0.77 | yes | 9.860e-4 | 8.271e-2 | +0.2564 | 111.0 |
| 0.78 | yes | 3.145e-3 | 9.459e-2 | +0.2035 | 143.7 |
| 0.79 | no | 9.279e-3 | 1.026e-1 | -0.03544 | 911.9 |
| 0.80 | no | 2.251e-2 | 1.047e-1 | -0.02333 | 1373.8 |

No accepted surface touched a configured coefficient bound. Thus the detected
branch was not created by an active radius or deformation constraint.

## KKT response prediction

The smallest real part of the discrete shape-response spectrum changes sign
between \(d=0.78\) and \(d=0.79\). Linear interpolation gives

\[
\boxed{d_\star^{(N=48,\ell_{\max}=6)}\simeq0.788516477}.
\]

At the same point, the response condition number rises by more than a factor of
six. This is the behavior expected near a nearly singular surface-response
map, and it coincides with loss of the converged common branch.

This is the first useful prediction from the imported KKT idea. It remains
explicitly qualified:

- the response matrix uses log-radius harmonic coefficients;
- its eigenvalues are parameterization-dependent;
- it has not yet been matched to an independent normal-deformation MOTS
  stability operator;
- the zero has not been continuum-extrapolated.

It should therefore be called a **discrete response crossing**, not a measured
surface gravity or invariant MOTS eigenvalue.

## Resolution and angular-order attacks

The completed campaigns gave:

| Grid and modes | Tolerance | Last detected | First not detected |
|---|---:|---:|---:|
| \(N=40,\ell_{\max}=4\) | 2e-2 | 0.80 | 0.90 |
| \(N=48,\ell_{\max}=4\) | 2e-2 | 0.75 | 0.80 |
| \(N=48,\ell_{\max}=6\) | 5e-3 | 0.78 | 0.79 |

The coarse scans establish the transition region; the strict scan resolves it
more finely. These are not sufficient for continuum extrapolation because
only two Cartesian resolutions are present and the sampled brackets differ.

## Reproduction

Strict focused campaign:

```powershell
python experiments/run_kkt_mots_brill_lindquist.py `
  --resolution 48 `
  --maximum-degree 6 `
  --separations 0.75,0.76,0.77,0.78,0.79,0.80 `
  --expansion-tolerance 0.005 `
  --maximum-iterations 45 `
  --common-only `
  --output runs/kkt_mots_brill_lindquist_n48_l6_strict.json
```

Rebuild the machine-readable comparison:

```powershell
python experiments/analyze_kkt_mots_brill_lindquist.py `
  runs/kkt_mots_brill_lindquist.json `
  runs/kkt_mots_brill_lindquist_n48.json `
  runs/kkt_mots_brill_lindquist_n48_l6_strict.json
```

Primary artifacts:

- [`../runs/kkt_mots_brill_lindquist_n48_l6_strict.json`](../runs/kkt_mots_brill_lindquist_n48_l6_strict.json)
- [`../runs/kkt_mots_brill_lindquist_analysis.json`](../runs/kkt_mots_brill_lindquist_analysis.json)
- [`KKT_MOTS_EXPERIMENT.md`](KKT_MOTS_EXPERIMENT.md)

## Next physical gate

Before a dynamical black-hole run, the clean next task is an \(N=64\) or higher
common-only scan with \(\ell_{\max}=6\) and \(8\), followed by a separately
normalized MOTS stability operator. If the zero converges, it becomes a precise
prediction for where the common branch bifurcates in this initial-data family.

Only after that gate should the observer be attached to puncture CCZ4 snapshots
and compared against horizon multipoles and `Psi4` during an actual merger.
