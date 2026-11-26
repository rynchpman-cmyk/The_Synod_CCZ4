# The Entropy Toll at the Finite Gate

## Synodic account

Campaign `NUL-BND-004` separates three quantities which must not be confused:
entropy carried through the open boundary, entropy retained by the evolved
state, and the finite-volume plus constitutive-projection remainder. The same
co-moving material pulse denied under the strict closure of `NUL-BND-003` is
advanced at 12, 18, and 24 cells with fixed Courant ratio.

For each complete RK step the retained remainder is

\[
  \Delta S_{\rm num}
  = \Delta S_{\rm patch}+S_{\rm outward}.
\]

No drag or vector damping is active in this trial. The remainder must be
nonnegative at every step, must decrease monotonically under refinement both
per step and cumulatively, and must remain distinct from the larger,
nonadditive sum of provisional-stage projection corrections.

## Judgment

A positive convergent remainder is numerical entropy production, compatible
with the discrete second law. It is not exact entropy conservation and does
not retroactively pass the strict absolute-closure gates of `NUL-BND-003`.
Longer trajectories, smoother and steeper pulses, higher-order boundary
closures, and a fresh vector/material qualification remain required.
