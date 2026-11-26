# Radiation–trapping boundary prediction

## Result

The active-repair field has a reproducible finite-time transition from freely
radiating to matter-trapped behavior.  For the regular branch, the 256-point
results are described by

\[
q_{\rm crit} = 3.9252 - 1.2448 m_A^2
               - 0.0251\ln\tau_*
               - 0.0993(\ln\tau_*)^2,
\qquad q \equiv g\rho_{\rm peak}.
\]

The fit error on its eight calibration regimes is 0.0366 in loading (0.0561
maximum).  Four off-grid 256-point cases, withheld from the calibration, have
a critical-loading MAE of 0.1123 and maximum error of 0.2558.  At the fixed
scan density `rho_peak=0.08`, convert the prediction to coupling with
`g_crit=q_crit/0.08`.

This is close to an effective-mass threshold: increasing bare mass squared
reduces the matter loading required for trapping.  The coefficient is not
exactly unity, so `m_A^2 + q = constant` is a useful first approximation, not
the final measured law.  Relaxation time has a small effect over most of the
surface.

## Operational definition

A run is placed on the boundary when 20% of the total repair-sector energy is
still inside `|x|<3.5` at `T=8`.  The domain is `[-12,12]`.  Consequently this
is a finite-time transport boundary for the specified pulse and matter
profile, not a proof of an eternal bound state or a universal phase boundary.

## High-resolution calibration

| `m_A` | `tau_*` | measured `q_crit` (N=256) |
|---:|---:|---:|
| 0.35 | 0.15 | 3.4239 |
| 0.35 | 0.60 | 3.8058 |
| 0.35 | 2.40 | **5.8788** |
| 0.80 | 0.15 | 2.8150 |
| 0.80 | 0.60 | 3.0593 |
| 0.80 | 2.40 | 3.0783 |
| 1.25 | 0.15 | 1.7125 |
| 1.25 | 0.60 | 1.9768 |
| 1.25 | 2.40 | 1.8343 |

The lightest, slowest case is a distinct **slow-light branch** and is excluded
from the regular-branch formula above.  It needs far more loading than the
smooth surface predicts and changes from the usual five-node trapped profile
to a two-node profile near threshold.  A 512-point check gives retention
0.19916 at `q=5.8788`, agreeing with the target to 0.00084 and confirming that
the branch is not a coarse-grid artifact.  The branch classification was
discovered after the calibration scan and is therefore explicitly post-hoc;
its full boundary should be mapped separately before assigning a universal
analytic criterion.

## Withheld tests

| case (`m_A`, `tau_*`) | predicted | measured N=256 | absolute error |
|---|---:|---:|---:|
| light/fast (0.62, 0.30) | 3.3328 | 3.1885 | 0.1443 |
| middle/slow (0.93, 1.20) | 2.8407 | 2.8347 | 0.0059 |
| heavy/fast (1.32, 0.30) | 1.6424 | 1.3866 | 0.2558 |
| heavy/slow (1.18, 1.20) | 2.1841 | 2.2271 | 0.0430 |

All runs remained stable and the reconstructed Gauss residual stayed at
roundoff scale.

## Secondary prediction: a transparency trough

The boundary is not simply monotone from zero loading.  For moderate and heavy
fields, weak matter coupling can *decrease* core retention before stronger
loading traps the field.  In the pilot at `tau_*=0.6`, the effect is strongest
around `m_A=1.1`: retention falls from about 0.126 uncoupled to about 0.073 near
`q=1`, then rises past the 20% trapping threshold.  This resembles an
impedance-matching or transparency window and is a separate feature from the
high-loading trapping contour.

## Resolution study

The 64-point exploratory map predicts topology well but not production-level
coordinates.  Across the four withheld cases, the mean boundary shift is
0.607 from N=64 to N=128, then falls to 0.150 from N=128 to N=256 (0.302
maximum).  The N=256 calibration is therefore the result to use; another
resolution level is advisable for any claimed percent-level coefficient.

## Reproduction

From the repository root with `PYTHONPATH=src`:

```powershell
python experiments\scan_trapping_boundary.py pilot --points 64
python experiments\scan_trapping_boundary.py map --points 64
python experiments\predict_trapping_boundary.py lock
python experiments\predict_trapping_boundary.py resolve --points 64
python experiments\predict_trapping_boundary.py convergence `
  --forecasts runs\trapping_boundary\high_resolution_results.json `
  --output runs\trapping_boundary\convergence_256.json `
  --cache runs\trapping_boundary\convergence_256_cache.json `
  --points 256 --half-width 0.9 --iterations 5
python experiments\calibrate_trapping_boundary.py
python experiments\plot_trapping_boundary.py
```

All scans are checkpointed after each evolution in their JSON cache files.
The consolidated figure is
`runs/trapping_boundary/trapping_boundary_summary.png`.
