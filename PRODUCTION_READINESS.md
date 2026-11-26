# Production-readiness report

## Status

The package is a complete numerical-relativity **reference backend** for the
specified local theory:

```text
CCZ4 geometry + Valencia GRHD + Proca repair + finite-time target
```

It also contains finite physical boundaries, isolated constraint construction,
conservative AMR machinery, optional parallel/device interfaces, horizons, and
wave extraction. “Reference” remains intentional: no claim is made that this
NumPy implementation is yet comparable in scale or campaign validation to a
mature HPC numerical-relativity framework.

## Locked formulation

- Signature `(-,+,+,+)` and `K_ij = -L_n gamma_ij / 2`.
- Covariant CCZ4 with configurable `kappa1`, `kappa2`, and `kappa3`.
- Advective 1+log or harmonic lapse and Gamma-driver shift.
- Algebraic enforcement of `det(gamma_tilde)=1` and `tr(A_tilde)=0`.
- Densitized Valencia variables, MC reconstruction, relativistic HLL fluxes,
  atmosphere recovery, and SSP-RK3 standalone hydrodynamics.
- Theory 2.0 target, effective current, interaction stress, and algebraic Proca
  Gauss reconstruction.
- Total matter energy/momentum evolution, with stage-local fixed-point recovery
  of fluid and repair contributions.

## Operational infrastructure now implemented

1. Cell-centered finite Cartesian patches with fourth/second-order one-sided
   derivatives and explicit, component-leading ghost-zone storage.
2. No-inflow GRHD face fluxes, outflow target data, radiative Proca fields, and
   a linearized incoming Z4 constraint-characteristic projection.
3. A 3D maximal conformally-flat CTT solve for the Hamiltonian and momentum
   constraints, including Picard coupling to the full two-lobe stress tensor and
   controlled nonconvergence for supercritical freely specified data.
4. Ratio-2 limited conservative prolongation, volume restriction,
   Berger–Colella subcycling orchestration, and flux registers/reflux correction.
5. Serial-safe MPI slab partition/halo exchange, deterministic reductions, and
   optional NumPy/CuPy array backends.
6. A coordinate-sphere apparent-horizon expansion root finder with area and
   irreducible mass, plus vacuum electric/magnetic Weyl `Psi4` extraction and
   spin-weighted `l=2` mode quadrature.

## Verification gates passing

The 58-test suite covers prior field-equation, CCZ4, GRHD, Theory 2.0, TOV, and
checkpoint gates, plus:

- exact finite-domain Minkowski fixed point;
- fourth-order one-sided polynomial derivative and explicit ghost fills;
- no wraparound and outward fluid mass loss without opposite-edge injection;
- incoming Z4 constraint-mode damping;
- vacuum and localized-matter CTT convergence;
- coupled weak-field two-lobe CTT construction;
- conservative 3D prolong/restrict and reflux balance;
- serial backend/decomposition fallbacks;
- zero Minkowski `Psi4`;
- isotropic-Schwarzschild horizon radius and irreducible mass.

An 8-cell isolated CLI smoke run also advances the full coupled solver with no
primitive-recovery failures. At that deliberately coarse resolution its
Hamiltonian norm grows to about `3.5e-4` by `t=0.001`; this is a smoke result,
not a convergence claim.

## Remaining campaign-qualification work

These are validation and scaling gates rather than absent package components:

- Wire the generic AMR callbacks to a distributed, load-balanced hierarchy for
  the entire `ProductionState`, then test reflux across moving refinement boxes.
- Port/profile the tensor kernels themselves on CuPy or a compiled backend; the
  current device boundary does not make NumPy-only kernels GPU-native.
- Validate the linearized Z4 boundary against nonlinear characteristic tests and
  compare reflection coefficients across outer-boundary radii.
- Extend the spherical horizon search to a fully distorted surface-flow solver.
- Add higher `l` spin-weighted modes and Cauchy-characteristic extraction.
- Run robust-stability, gauge-wave, shock/TOV evolutions at multiple resolutions,
  spinning/puncture black holes, and long binary-scale campaigns; compare with
  an independent NR code and publish reproducible error budgets.

Until those gates pass, results should be described as produced by a tested
CCZ4+GRHD reference solver, not a production HPC numerical-relativity code.
