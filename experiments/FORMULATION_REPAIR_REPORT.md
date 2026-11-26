# Hyperbolic production-formulation repair

## Implemented

Production now evolves the densitized longitudinal charge

\[
\mathcal Q_A=\sqrt\gamma(M_{\rm eff}^2\Phi-\rho_*)
\]

using the conservative identity

\[
\nabla_\mu(M_{\rm eff}^2A^\mu-J_*^\mu)=0.
\]

The normal field is reconstructed locally as

\[
\Phi=\frac{\mathcal Q_A/\sqrt\gamma+\rho_*}{M_{\rm eff}^2}.
\]

No spatial derivative appears in this numerator. The initial Gauss solve only
translates initial data into `Q_A`; it is not used during production evolution.

The production state now contains `a`, `pi`, `astar`, `longitudinal`, and
`gauss_cleaning`. The cleaner monitors

\[
C_A=D_iE^i+\mathcal Q_A/\sqrt\gamma
\]

and propagates/damps violations. Compatible derivatives in the electric and
longitudinal equations preserve satisfied Gauss data to roundoff.

## Coupled shock flux

Total fluid-plus-repair energy and momentum now pass through one conservative
Rusanov interface flux. Previously the fluid went through HLL while repair
stress was appended outside its Riemann solve.

The new bound includes Valencia acoustic speeds, coordinate metric light speed,
and a conservative mismatch factor proportional to
`sqrt(1+chi*|A-Astar|_fluid^2/h)`. The CFL timestep uses the same bound. This is
deliberately diffusive and empirical, not a proof of the exact spectral radius.

## Compatibility and verification

Production checkpoints are now `causal_repair.production.v2`; v1 lacks the
independent longitudinal state and is rejected. The fixed-background solver
retains its legacy constrained path so existing prediction artifacts remain
reproducible.

Verified behavior includes exact initial reconstruction, roundoff Gauss
residual, cleaning damping, mismatch-sensitive speed bounds, periodic total
conservation, physical recovery across discontinuous coupled data, baryon
conservation, exact v2 checkpoint round-trips, and a clean production CLI run.

## Remaining limitation

This removes the avoidable formulation defect without altering the theory. The
quadratic interaction still has a nonhyperbolic island for cold, boosted,
longitudinally mismatched states. An analytic convexity bound and runtime
admissibility guard remain necessary. Global hyperbolicity beyond that domain
would require changing the interaction potential, for example to a covariant
saturation whose Hessian is proven positive.
