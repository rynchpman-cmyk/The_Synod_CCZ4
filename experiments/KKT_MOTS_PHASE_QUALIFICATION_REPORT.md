# Cartesian MOTS Grid-Phase Qualification

## Verdict

The corrected physical-normal observer is qualified for attachment to
dynamical CCZ4 snapshots with a conservative single-snapshot sign gate:

- at least **16 Cartesian cells per mean coordinate radius**;
- measured \(|\operatorname{Re}\lambda_0|\ge5\times10^{-3}\);
- a valid two-step physical-normal finite-difference audit.

Passing those conditions qualifies the *sign* of the principal mode on one
snapshot. It does not qualify a continuum zero. A claimed zero or bifurcation
still requires multiple resolutions and grid phases.

## Campaign Design

The calibration used the direct \(\ell_{\max}=16\) Brill--Lindquist surface as
a fixed oracle. Corrected Cartesian measurements were performed on a local
length-2 patch at \(N=48,64,80,96\), corresponding to about 9--19 cells per
mean radius. Four relative puncture/grid placements were tested:

- centered;
- a quarter-cell shift along the symmetry axis;
- a half-cell shift along the symmetry axis;
- a half-cell shift in all three Cartesian directions.

The surface was never re-optimized during observer calibration. This isolates
Cartesian geometry and interpolation error from branch-finding error.

## Resolved Surface Gate

Raising the oracle from \(\ell_{\max}=8\) to \(16\) removed a false RMS floor
caused by unresolved angular expansion. For \(d=0.75\):

| Cells per mean radius | Maximum phase RMS expansion | Maximum \(\lambda_0\) error | Maximum \(\lambda_1\) error |
|---:|---:|---:|---:|
| 9.58 | 4.685e-3 | 1.112e-2 | 3.612e-2 |
| 12.78 | 2.092e-3 | 4.175e-3 | 2.787e-2 |
| 15.97 | 1.478e-3 | 8.566e-3 | 2.391e-2 |
| 19.16 | 1.009e-3 | 3.792e-3 | 1.538e-2 |

At \(d=0.765\), the second mode is already stable to about
\(5\times10^{-3}\) at 14.7 cells per radius. The principal mode is more
sensitive because the physical eigenvalue itself is approaching zero.

## Fold-Edge Attack

The direct fold prediction remains

\[
d_\star/M_{\rm bare,total}=0.766197\pm0.000005.
\]

At \(N=96\), or about 17 cells per mean radius:

| Separation | Direct \(\lambda_0\) | Cartesian phase range | Sign agreement |
|---:|---:|---:|:---|
| 0.76610 | 9.247e-3 | 6.014e-3 to 1.431e-2 | 4/4 positive |
| 0.76615 | 6.364e-3 | 5.126e-3 to 1.385e-2 | 4/4 positive |
| 0.76618 | 3.608e-3 | 3.010e-3 to 8.120e-3 | 4/4 positive |
| 0.76619 | 1.921e-3 | -2.877e-3 to 4.061e-3 | mixed |

Thus the phase ensemble preserves the physical sign through \(d=0.76618\)
but becomes ambiguous by \(d=0.76619\). The conservative
\(5\times10^{-3}\) software floor deliberately stops single-snapshot sign
claims slightly before that empirical edge.

At roughly 14 cells per mean radius, one tested phase already becomes slightly
negative at \(d=0.76615\). This is the direct evidence behind the 16-cell
minimum rather than an arbitrary safety factor.

## CCZ4 Tracker Behavior

`CCZ4PhysicalNormalMOTSTracker` now emits a `cartesian_qualification` object for
every named branch. It records:

- cells per mean radius;
- configured resolution and mode floors;
- whether the resolution gate passed;
- whether the principal-mode sign gate passed;
- the measured real and imaginary principal mode;
- a reason string.

`continuum_zero_qualified` is always `false` for a single snapshot. The two
gate thresholds are stored in the observer checkpoint and must match on
restart, preventing an append run from silently changing its qualification
policy.

## Reproduction Artifacts

- [`run_kkt_mots_cartesian_phase_calibration.py`](run_kkt_mots_cartesian_phase_calibration.py)
- [`kkt_mots_cartesian_phase_calibration.json`](../runs/kkt_mots_cartesian_phase_calibration.json)
- [`kkt_mots_cartesian_phase_near_fold.json`](../runs/kkt_mots_cartesian_phase_near_fold.json)
- [`kkt_mots_cartesian_phase_fold_edge.json`](../runs/kkt_mots_cartesian_phase_fold_edge.json)
- [`kkt_mots_cartesian_phase_last_positive.json`](../runs/kkt_mots_cartesian_phase_last_positive.json)

The next physical campaign can now attach the qualified tracker to a cadence of
CCZ4 puncture snapshots. Near a candidate zero it should branch into at least
two resolutions or translated local extraction patches before promoting the
event to a continuum statement.
