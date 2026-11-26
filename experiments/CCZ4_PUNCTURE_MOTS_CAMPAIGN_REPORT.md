# Dynamic CCZ4 Puncture/MOTS Campaign

## Outcome

The dynamic campaign infrastructure is running and the pilot produced the
expected common-MOTS saddle-node signature on evolved CCZ4 slices:

1. no common surface on the initial Brill--Lindquist slice at \(d=0.78\);
2. a first common surface in the sampled interval \(0.375<t\le0.400\);
3. separation into a negative-mode inner branch and positive-mode outer
   branch;
4. subsequent contraction of the inner coordinate radius and expansion of the
   outer coordinate radius.

This is a successful dynamical demonstration, not yet a qualified physical
formation-time measurement. The pilot surfaces have only about 5--7 cells per
mean radius, below the calibrated 16-cell gate, and the short box admits causal
boundary influence near the formation window.

## Campaign Runner

[`run_ccz4_puncture_mots_campaign.py`](run_ccz4_puncture_mots_campaign.py)
provides:

- equal-mass, time-symmetric Brill--Lindquist puncture data;
- pre-collapsed \(1+\log\) lapse and Gamma-driver shift;
- finite Cartesian CCZ4 evolution with radiative boundaries;
- moving puncture-center estimates from conformal-factor minima;
- separately continued individual and common surface seeds;
- stability-sign classification of `common_outer` and `common_inner`;
- automatic partner seeding when only one member of a new common pair is
  found;
- full, puncture-masked, and puncture-plus-boundary-masked constraints;
- calibrated cells-per-radius and principal-mode sign gates;
- restartable evolution and observer checkpoints;
- optional saved CCZ4 snapshots and atomic JSON trajectories.

The analysis and timeline plot are produced by
[`analyze_ccz4_puncture_mots_campaign.py`](analyze_ccz4_puncture_mots_campaign.py).

## Gauge Initialization Gate

The first pilot exposed and corrected an initialization error. This CCZ4
implementation stores

\[
W=\det(\gamma)^{-1/6}=\psi^{-2}
\]

for conformally flat puncture data. Therefore the standard pre-collapsed lapse
is \(\alpha=W=\psi^{-2}\), not \(\sqrt{W}=\psi^{-1}\). The incorrect lapse
created a spurious common-surface detection near \(t=0.18\). That event
disappeared after correction and is excluded from the physical pilot.

## Constraint Domain Gate

The raw and puncture-only Hamiltonian norms are dominated by the nearby
radiative boundary closure on the length-2 pilot box. The runner now reports a
separate interior norm that excludes five fourth-order stencil layers around
both punctures and all physical boundaries.

For the corrected \(N=32\) run, the available interior Hamiltonian norm rises
from \(1.48\times10^{-2}\) to \(3.92\times10^{-2}\) through \(t=0.5\). This is
controlled enough for a discovery pilot, while remaining too large and too
domain-dependent for a continuum claim.

## Two-Resolution Pilot

Both corrected-lapse runs use \(d=0.78\), total bare mass one, box length two,
and CCZ4 time \(t=0.5\).

| Quantity | \(N=24\) | \(N=32\) |
|:--|--:|--:|
| First common-surface bracket | \((0.3733,0.4000]\) | \((0.3750,0.4000]\) |
| Both branches separately recovered | no | by \(t=0.4500\) |
| Final outer mean radius | 0.41484 | 0.41784 |
| Final outer \(\lambda_0\) | +0.12417 | +0.06394 |
| Final outer full RMS expansion | 8.31e-2 | 2.69e-2 |
| Final inner mean radius | not recovered | 0.32390 |
| Final inner \(\lambda_0\) | not recovered | -0.09943 |
| Maximum available interior Hamiltonian L2 | 7.11e-2 | 3.92e-2 |
| Qualified sign observations | 0 | 0 |

The formation bracket agrees at the two pilot resolutions. The outer full RMS
expansion and interior constraint both improve with resolution. The principal
mode magnitude does not yet converge, and only \(N=32\) resolves both members,
so the run is not promoted beyond pilot status.

At \(N=32,t=0.5\), the pair is visibly separated:

\[
\begin{aligned}
r_{\rm outer}&=0.41784,&\lambda_{0,\rm outer}&=+0.06394,\\
r_{\rm inner}&=0.32390,&\lambda_{0,\rm inner}&=-0.09943.
\end{aligned}
\]

The coordinate puncture separation changes very little. This is a useful
reminder that coordinate puncture motion is not the horizon-formation
observable; the evolving spatial geometry and null expansion are.

## Boundary and Resolution Qualification

The final outer surface reaches coordinate radius about 0.66 on a box ending
at radius one. Boundary information can therefore reach it during the
\(t\simeq0.4\) formation window. A boundary-safe production replay should use
at least a length-3 box.

With mean outer radius about 0.418, the 16-cell observer gate requires
\(\Delta x\lesssim0.026\). A practical uniform-grid target is therefore
\(N=120\), length 3, giving \(\Delta x=0.025\) and roughly 16.7 cells per mean
outer radius.

Measured local CPU timings are about 4.1 seconds per CCZ4 RK4 step at \(N=40\)
and 7.0 seconds at \(N=48\). Cubic extrapolation predicts roughly 109 seconds
per step at \(N=120\), or about 15 CPU-hours for 500 steps before surface
observations. The qualified uniform run is consequently packaged and
restartable.

## Qualified Production Command

```powershell
$env:PYTHONPATH = "src"
python experiments/run_ccz4_puncture_mots_campaign.py `
  --resolution 120 `
  --box-length 3 `
  --separation 0.78 `
  --steps 500 `
  --cfl 0.04 `
  --observation-cadence 10 `
  --checkpoint-cadence 50 `
  --maximum-degree 16 `
  --individual-maximum-degree 8 `
  --maximum-iterations 60 `
  --expansion-tolerance 5e-4 `
  --save-snapshots `
  --output-directory runs/ccz4_puncture_mots_n120
```

The continuation command uses the same options plus:

```powershell
--restart-from runs/ccz4_puncture_mots_n120/final_checkpoint.npz
```

## Artifacts

- [\(N=24\) analysis](../runs/ccz4_puncture_mots_lapsefix_n24/analysis.json)
- [\(N=24\) timeline](../runs/ccz4_puncture_mots_lapsefix_n24/timeline.png)
- [\(N=32\) analysis](../runs/ccz4_puncture_mots_lapsefix_n32_early/analysis.json)
- [\(N=32\) timeline](../runs/ccz4_puncture_mots_lapsefix_n32_early/timeline.png)
- [\(N=32\) restart checkpoint](../runs/ccz4_puncture_mots_lapsefix_n32_early/final_checkpoint.npz)

The decisive next result is the \(N=120\), length-3 replay or an equivalent
AMR run that places the horizon at the same local resolution while keeping the
boundary causally separated.
