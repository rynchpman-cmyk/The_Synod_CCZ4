# Isolated Theory 3.3 two-current sphere collision

## Why Hawking mass

The primary observable is the Hawking quasi-local mass on nested closed
two-surfaces rather than a lapse extremum or pointwise curvature maximum.
For surface area \(A\), tangential extrinsic-curvature trace \(p\), and spatial
mean curvature \(k\), the implemented convention is

\[
M_H=\sqrt{\frac{A}{16\pi}}
\left[1+\frac{1}{16\pi}\oint (p^2-k^2)\,dA\right].
\]

This vanishes for round spheres in Minkowski space and recovers the
Schwarzschild mass. It is a scalar property of each selected surface. The
coordinate-sphere foliation used to select those surfaces remains an explicit
extraction choice, so the result is quasi-local rather than a claim of unique
binary mass partition.

## Experiment

Two initially spherical Gaussian material bodies begin in contact on a finite
Cartesian box and move head-on along the x axis. Their baryon currents travel
at 0.18 and their carrier currents at 0.46. The carrier parameters are

\[
f_D=2,\qquad K_D=0.4,\qquad c_D=0.7.
\]

The fast and slow branches differ only in
\(\tau_D=0.03\) versus \(0.30\). Each branch independently solves the same
isolated conformal transverse-traceless initial data and uses radiative CCZ4,
outgoing fluid, and outgoing carrier boundaries. The accepted campaign covers
\(N=10\) and \(12\) through \(t=0.016\) with \(\Delta t=0.002\).

This is an early-time already-contacting collision, not a TOV neutron-star
merger or a long approach from separated equilibrium stars.

## Result: matter deforms before quasi-local mass does

The carrier dynamics separate strongly:

| quantity | \(N=10\) | \(N=12\) |
|---|---:|---:|
| fast final \(\gamma_{rel}\) | 1.32360 | 1.15344 |
| slow final \(\gamma_{rel}\) | 1.57773 | 1.26850 |
| fast/slow drag-entropy ratio | 6.722 | 6.724 |
| slow-fast inner carrier number | 2.798e-5 | 3.166e-5 |
| slow-fast peak carrier density | -1.007e-4 | -7.928e-5 |

The slow branch retains more integrated carrier number inside the innermost
extraction sphere, while the fast branch builds a sharper carrier peak. The
baryon response has the opposite integrated sign: the slow-fast inner baryon
mass is \(-8.07\times10^{-7}\) and \(-9.89\times10^{-7}\).

The geometric response is far smaller:

| extraction radius | \(\Delta M_H\), \(N=10\) | \(\Delta M_H\), \(N=12\) |
|---:|---:|---:|
| 1.5 | 3.041e-9 | 3.504e-9 |
| 2.5 | 3.007e-10 | 3.290e-10 |
| 3.0 | 5.666e-11 | -2.079e-11 |

Here \(\Delta M_H=M_H^{slow}-M_H^{fast}\). The inner signal has the same
sign at both completed resolutions and differs by 13.2%. Relative to the
enclosed inner Hawking mass it is only \(3.9\times10^{-8}\) at \(N=10\) and
\(7.0\times10^{-8}\) at \(N=12\). The outer control is consistent with zero
and is below 2% of the already tiny inner split.

Thus fast relaxation strongly changes the current and matter distribution but
does not create an immediate quasi-local mass difference. This is the expected
behavior of an internal, energy-momentum-conserving relaxation mechanism: the
geometry responds only after the altered stresses and fluxes redistribute
through the collision region.

After the initially identical interval, the inner mass split grows empirically
as approximately \(t^{3.22}\) at \(N=10\) and \(t^{3.39}\) at \(N=12\).
This near-cubic law is an early-time numerical observation, not yet a derived
or continuum-extrapolated prediction.

## Gates and qualification boundary

Both accepted resolutions pass the characteristic, causal, convexity, CTT,
and primitive-recovery gates. The maximum characteristic speed is 0.88473.
There are zero recovery failures. Relative baryon loss is below
\(3.6\times10^{-8}\), and relative carrier loss is below
\(1.2\times10^{-6}\). The final inner-core Hamiltonian L2 residual improves
from \(1.42\times10^{-3}\) at \(N=10\) to \(1.06\times10^{-3}\) at \(N=12\).

Two attacks were rejected rather than incorporated:

- \(N=8\) leaves the allowed Theory 3.3 counterflow domain before the full
  interval. Therefore \(N=10\) is the minimum completed resolution.
- Doubling the peak density to 0.01 also drives the projected state outside
  the counterflow domain before \(t=0.008\). That regime needs higher
  resolution, a wider master-function domain, or both.

These failures prevent this campaign from being described as a strong-field
merger calculation. They also identify the next concrete engineering boundary.

## Reproduction

```powershell
$env:PYTHONPATH='src'

python experiments/run_theory33_spherical_collision.py `
  --points 10 --relaxation-time 0.03 `
  --output runs/theory33_spheres/fast

python experiments/run_theory33_spherical_collision.py `
  --points 10 --relaxation-time 0.30 `
  --output runs/theory33_spheres/slow

python experiments/run_theory33_spherical_collision.py `
  --points 12 --relaxation-time 0.03 `
  --output runs/theory33_spheres/n012_fast

python experiments/run_theory33_spherical_collision.py `
  --points 12 --relaxation-time 0.30 `
  --output runs/theory33_spheres/n012_slow

python experiments/analyze_theory33_spherical_collision.py
```

The analysis writes `runs/theory33_spherical_collision.json`,
`runs/theory33_spherical_collision.png`, and
`runs/theory33_spherical_collision_slices.png`.

## Next physical step

The clean next target is not simply more density. It is a longer, better
resolved \(N\geq16\) isolated evolution with a master-function domain wide
enough for the collision-generated counterflow. That would test whether the
near-cubic inner mass response continues, saturates, or reverses, while an
outer Hawking or ADM mass remains the conservation control.

That target is now complete. See `THEORY33_SPHERICAL_LONG_REPORT.md` for the
audited \(\gamma_{rel}=4\) domain, \(N=16\) evolution, and the observed
near-cubic to \(t^{2.88}\) growth-regime transition.
