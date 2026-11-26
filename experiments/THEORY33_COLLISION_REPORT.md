# Theory 3.3 counterflow compact-collision campaign

## Result

The first production33 strong-gravity scenario is running successfully.  It
is a one-dimensional periodic planar analogue of two compact fluid bodies
colliding during constant-mean-curvature contraction.  The baryon pulses move
inward at (0.18c); the carrier current can receive an additional inward
counterflow of (0.28c).

The campaign cleanly separates two-current locking from penetrating flow:

| high-compactness champion | τ_D | initial max γ_D-1 | final max γ_D-1 | compression | minimum lapse |
|---|---:|---:|---:|---:|---:|
| drag locked | 0.03 | 0.09488 | 0.00674 | 1.05611 | 0.89984 |
| penetrating | 0.30 | 0.09488 | 0.06574 | 1.05611 | 0.89984 |

Thus fast drag removes about 93% of the peak relative-Lorentz excess in
(0.04) code time, while slow drag retains about 69%.  The baryon-density and
lapse histories remain nearly identical at the current five-percent carrier
fraction.  This is a useful early physical result: relaxation strongly changes
the internal two-current state before it produces measurable gravitational
backreaction.

Both 32-point validation runs conserve baryon and carrier number to roundoff,
increase total entropy, have zero primitive-recovery failures, and finish with
recovery residuals near (1.1\times10^{-9}).  Doubling the scan resolution
from 16 to 32 points changes the compression by about 0.5% and reduces the
Hamiltonian diagnostic from about 0.123 to 0.0388.

## Coarse scan

The eight cases varied peak density, imposed counterflow, and drag time.

| peak density | counterflow | τ_D | compression | minimum lapse | final max γ_D-1 | outcome |
|---:|---:|---:|---:|---:|---:|---|
| 0.012 | 0.00 | 0.03 | 1.03412 | 0.93997 | 0.01150 | gravity compression |
| 0.012 | 0.00 | 0.30 | 1.03412 | 0.93997 | 0.03401 | gravity compression |
| 0.012 | 0.28 | 0.03 | 1.03412 | 0.93997 | 0.00363 | drag-locked compression |
| 0.012 | 0.28 | 0.30 | 1.03412 | 0.93997 | 0.04107 | counterflow compression |
| 0.025 | 0.00 | 0.03 | 1.05085 | 0.89941 | 0.00721 | gravity compression |
| 0.025 | 0.00 | 0.30 | 1.05085 | 0.89941 | 0.01836 | gravity compression |
| 0.025 | 0.28 | 0.03 | 1.05085 | 0.89940 | 0.00609 | drag-locked compression |
| 0.025 | 0.28 | 0.30 | 1.05085 | 0.89940 | 0.06181 | counterflow compression |

The zero-imposed-counterflow controls develop some relative motion because
the chemically coupled constituents respond differently during compression.
That effect is stronger for slow relaxation and is physical within this model,
not initial counterflow left in the data.

## Numerical discovery and repair

The first nonlinear collision exposed a finite-resolution consistency defect:
independently transported entropy and total energy left the overdetermined
master-function state manifold at practical CFL steps.  production33 now
performs a square nine-variable energy inversion and projects only the
redundant entropy representation at every Runge--Kutta stage.  It leaves both
number currents, total energy-momentum, and carrier canonical momentum
unchanged.  Complete periodic steps are rejected if their integrated entropy
decreases.

The campaign also adds a periodic conformally-flat CMC constraint solver.  Its
mean curvature is determined by the integrated Lichnerowicz equation, avoiding
the impossible maximal-slice zero mode for positive energy in a periodic box.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python experiments/run_theory33_collision.py --sweep --output runs/theory33_collision

python experiments/run_theory33_collision.py `
  --points 32 --peak-density 0.025 --counterflow 0.28 `
  --relaxation-time 0.03 --output runs/theory33_collision_validation_fast

python experiments/run_theory33_collision.py `
  --points 32 --peak-density 0.025 --counterflow 0.28 `
  --relaxation-time 0.30 --output runs/theory33_collision_validation_slow
```

The static comparison figure is generated with
`experiments/render_theory33_collision.py`.  Every case directory contains a
trajectory, JSON summary, and restartable final checkpoint.

## Qualification boundary

This is a periodic planar strong-gravity analogue, not yet an isolated
spherical compact-star collision.  The short early-time window demonstrates
compression, lapse response, dynamic generation and relaxation of counterflow,
and entropy-positive recovery.  It does not establish horizon formation or a
collapse threshold.  The clearest next scan is longer evolution with carrier
fractions above five percent, where the locked and penetrating branches should
begin to separate gravitationally.

That follow-up is complete in
[`THEORY33_CARRIER_LOADING_REPORT.md`](THEORY33_CARRIER_LOADING_REPORT.md).
