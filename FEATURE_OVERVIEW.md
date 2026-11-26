# Causal-Repair Gravity Solver: Complete Feature Overview

This document is the high-level guide to the complete solver repository: its
physical formulations, numerical-relativity stack, multifluid engine,
constraint and boundary machinery, operational tooling, experiments,
visualization suite, verification evidence, and known qualification limits.

Status snapshot: **version 0.1.0, 126 passing tests**.

## 1. What this repository is

The project is a transparent Python/NumPy numerical reference implementation
for covariant causal-repair gravity. It contains a progression of formulations,
from fixed-background Proca and Maxwell--Cattaneo experiments through a coupled
CCZ4+GRHD system and the current Theory 3.3 two-current master-action backend.

The preferred path for current physical work is:

```text
CCZ4 spacetime
  + Valencia relativistic hydrodynamics
  + two covariantly mixed Proca fields
  + baryon/carrier multifluid master function
  + entropy-positive internal drag
  + evolved longitudinal Gauss charges and cleaners
```

This is a tested **production-reference solver**, not yet a mature distributed
HPC numerical-relativity code. Its equations and local numerical mechanisms
are executable and heavily tested; multi-node scaling, long adaptive binary
campaigns, and independent-code comparison remain qualification work.

## 2. Status legend

| Label | Meaning |
|---|---|
| Preferred | Current solver path for new Theory 3.3 experiments |
| Implemented and tested | Executable feature with direct regression coverage |
| Experimentally exercised | Used in one or more saved numerical campaigns |
| Infrastructure-ready | Local primitive/interface exists, but is not wired into a complete scaled production campaign |
| Legacy/reproducibility | Retained to reproduce earlier theory stages and saved results |
| Qualification boundary | Feature is present, but broader physical or HPC claims require more evidence |

## 3. Solver families

### 3.1 Conservative fixed-background Proca branch

Status: **implemented and tested; legacy core**.

The `proca` command evolves a reduced-Hamiltonian massive vector field on a
fixed Minkowski background. It includes:

- all three spatial polarizations;
- configurable kinetic coefficient and mass;
- algebraic reconstruction of the normal component in the legacy constrained
  formulation;
- external charge/current source callbacks;
- positive Hamiltonian decomposition;
- Lorenz and Gauss diagnostics;
- spectral, second-order, or fourth-order spatial derivatives;
- RK4 evolution and compressed trajectory output.

Primary module: [`src/causal_repair/proca.py`](src/causal_repair/proca.py).

### 3.2 Maxwell--Cattaneo relaxation branch

Status: **implemented and tested; legacy core**.

The `relax` command evolves a finite-time rapidity/target relaxation field. It
supports:

- deterministic advective relaxation;
- stochastic thermal forcing;
- reproducible seeded noise;
- cell-volume-aware fluctuation--dissipation normalization;
- configurable mobility, susceptibility, temperature, target, and mode;
- equilibrium-variance and relaxation diagnostics.

Primary module:
[`src/causal_repair/relaxation.py`](src/causal_repair/relaxation.py).

### 3.3 Covariant causal-current map

Status: **implemented and tested**.

The standalone causal-map utilities provide the covariant algebra used by the
rapidity construction:

- timelike four-velocity normalization;
- metric lowering and invariant inner products;
- projection into the matter-frame spatial subspace;
- stable `tanh(x)/x` evaluation at the origin;
- bounded rapidity-to-current mapping;
- automatic reprojection to prevent floating-point drift from violating
  matter-frame orthogonality;
- numerical timelike/causal-margin checks.

Primary module:
[`src/causal_repair/causal_map.py`](src/causal_repair/causal_map.py).

### 3.4 Raw ADM--Proca branch

Status: **experimental/reproducibility**.

The `coupled` command provides a direct raw-ADM coupling for equation and sign
checks. It includes spatial curvature, ADM constraints, prescribed lapse and
shift, Proca stress-energy, and coupled RK4 evolution.

Raw ADM is intentionally not the preferred long-duration evolution system;
its free-evolution hyperbolicity is inferior to the CCZ4 path.

Primary modules:

- [`src/causal_repair/adm.py`](src/causal_repair/adm.py)
- [`src/causal_repair/coupled.py`](src/causal_repair/coupled.py)
- [`src/causal_repair/curved_proca.py`](src/causal_repair/curved_proca.py)

### 3.5 Theory 2.0 branches

Status: **legacy/reproducibility with a documented admissibility boundary**.

Theory 2.0 adds ordinary matter, an equilibrium target, a density-dependent
effective Proca mass, interaction stress, and a generalized source current.
The repository contains:

- a fixed-background two-lobe driver (`theory2`);
- a prescribed-matter raw-ADM bridge;
- a coupled CCZ4+GRHD production-reference command (`production`);
- evolved longitudinal charge and Gauss cleaning in checkpoint format v2;
- one total conservative matter/repair Rusanov flux;
- a mismatch-aware empirical characteristic bound.

The avoidable stagewise Gauss-reconstruction defect was removed. The unchanged
quadratic interaction still has a nonhyperbolic island for cold, boosted,
longitudinally mismatched states, so this branch is not an unrestricted shock
theory.

Primary references:

- [`THEORY_2_AUDIT.md`](THEORY_2_AUDIT.md)
- [`experiments/HYPERBOLICITY_REPORT.md`](experiments/HYPERBOLICITY_REPORT.md)
- [`experiments/FORMULATION_REPAIR_REPORT.md`](experiments/FORMULATION_REPAIR_REPORT.md)

### 3.6 Theory 3.2 mixed-Proca production branch

Status: **implemented and tested; retained production reference**.

The `production3` command implements the Theory 3.2 structural overhaul:

- two mixed Proca fields, conventionally A and B;
- a positive/invertible constant Gauss matrix;
- independent evolved longitudinal charges for both fields;
- one hyperbolic Gauss cleaner per field;
- derivative-free temporal-potential reconstruction;
- conservative target charge and target-current variables;
- a strongly hyperbolic replacement for the defective Theory 3.1 target
  current block;
- corrected Lorentz-force sign with equal/opposite matter/vector exchange;
- an explicit reversible pressureless material reservoir;
- exact finite-step electric damping;
- complete before/after vector-stress deposition;
- entropy-based matter recovery;
- periodic and finite Cartesian evolution;
- Theory 3.2 checkpoint format
  `causal_repair.production.theory3.v1`.

Theory 3.2 remains valuable as the controlled predecessor to Theory 3.3. Its
phenomenological reservoir is not used by the current preferred closure.

Primary references:

- [`experiments/THEORY_3_2_CLOSURE.md`](experiments/THEORY_3_2_CLOSURE.md)
- [`experiments/THEORY3_OVERHAUL_REPORT.md`](experiments/THEORY3_OVERHAUL_REPORT.md)

### 3.7 Theory 3.3 two-current production branch

Status: **preferred, implemented, tested, and experimentally exercised**.

The `production33` command replaces the Theory 3.2 reservoir and prescribed
target direction with a covariant two-current matter model. The physical
currents are a baryon current and a charged carrier current:

\[
N^\mu=n_bu^\mu,
\qquad
C_D^\mu=n_Dv_D^\mu,
\qquad
J_D^\mu=g_DC_D^\mu.
\]

The backend evolves the complete CCZ4, mixed-Proca, and multifluid state while
closing energy, momentum, both number currents, canonical carrier momentum,
and entropy without assigning a residual to an artificial reservoir.

Primary modules:

- [`src/causal_repair/theory33.py`](src/causal_repair/theory33.py)
- [`src/causal_repair/production33.py`](src/causal_repair/production33.py)
- [`experiments/THEORY_3_3_MASTER_ACTION.md`](experiments/THEORY_3_3_MASTER_ACTION.md)
- [`experiments/THEORY33_IMPLEMENTATION_REPORT.md`](experiments/THEORY33_IMPLEMENTATION_REPORT.md)

## 4. Theory 3.3 physics and constitutive features

### 4.1 Invariant M1 master function

The material closure is derived from a scalar master function built from the
baryon density, carrier density, entropy per baryon, and invariant relative
flow. The implemented family includes:

- ideal-gas baryonic thermal energy;
- a constant-sound-speed carrier energy
  \(\rho_D=K_D n_D^{1+c_D^2}\);
- a matter-dependent chemical anchoring fraction;
- positive chemical susceptibility;
- optional linear and quadratic entrainment terms;
- invariant relative-flow energy;
- conjugate baryon and carrier momenta;
- generalized pressure;
- Carter/Hilbert material stress-energy.

The carrier current that sources the B field is derived from the evolved
carrier constituent; it is not an independently prescribed directional
current.

### 4.2 Independent metric-variation check

The analytic material stress has an executable finite metric-variation gate
that holds the densitized pullback currents fixed. This independently checks
the Hilbert stress and guards against double-counting an interaction tensor.

The implementation also verifies the co-moving, zero-carrier, and single-fluid
limits.

### 4.3 Coupled multifluid primitive recovery

Recovery solves cell by cell for:

- baryon proper density;
- carrier proper density;
- baryon spatial four-velocity;
- carrier spatial four-velocity;
- entropy/internal energy consistent with the total state.

The nonlinear inversion closes both Eulerian number densities, total material
energy and momentum, and all three carrier canonical momenta. It uses damped
Newton/least-squares recovery with residual, atmosphere, convexity, and failure
reports.

### 4.4 Stage-local constitutive projection

Independent finite-volume entropy and total-energy fluxes make the represented
state redundant at nonlinear RK stages. Theory 3.3 therefore applies a square
nine-variable energy inversion at every RK stage and projects only the
redundant entropy representation back onto the master-function manifold.

The projection does not alter:

- baryon number;
- carrier number or target charge;
- total energy-momentum;
- carrier canonical momentum.

Periodic steps are rejected if integrated entropy decreases beyond tolerance.

### 4.5 Conservative/path-conservative carrier evolution

The carrier sector uses:

- conservative densitized carrier-number flux;
- densitized canonical-momentum evolution;
- a midpoint straight-path discretization of the genuine nonconservative
  principal product;
- outgoing-only carrier flux on finite boundaries;
- an explicit carrier-vacuum floor and admissibility checks.

### 4.6 Entropy-positive stiff drag

Internal baryon/carrier relaxation is handled by a local implicit collision
map. It:

- exponentially damps relative spatial four-velocity;
- solves simultaneously for post-collision velocities and proper densities;
- preserves baryon and carrier number;
- preserves total material energy and momentum;
- produces nonnegative entropy;
- remains stable even when one step spans many relaxation times;
- removes the explicit source-CFL limitation of the Theory 3.2 relaxation law.

### 4.7 Mixed-Proca force and damping

Theory 3.3 retains the structurally repaired Theory 3.2 vector sector:

- constant-mass mixed A/B Proca system;
- locked `-B_mu J_D^mu` source convention;
- matter force `+H^nu_lambda J_D^lambda`;
- evolved longitudinal A and B charges;
- independent Gauss cleaners;
- derivative-free temporal-potential reconstruction;
- exact local 2x2 electric damping exponential;
- Gauss-preserving longitudinal correction;
- complete finite-step vector energy/momentum deposition;
- thermal entropy deposition from only the exact positive electric loss.

No stagewise Gauss solve is used during production evolution.

### 4.8 Causality, convexity, and hyperbolicity gates

The master-function audit evaluates:

- current-to-momentum Legendre invertibility;
- thermodynamic Hessian positivity;
- real characteristic eigenvalues;
- complete eigenvectors and condition numbers;
- causal characteristic speeds;
- co-moving and counterflow states;
- two-stream/relative-flow limits;
- carrier loading and low-stiffness boundaries.

The default Theory 3.3 domain is audited through relative Lorentz factor 2.
The isolated spherical campaign separately qualified its chosen zero-
entrainment parameters through relative Lorentz factor 4 and carrier/baryon
density ratios from 0.5 to 4. Nondefault entrainment coefficients require a
fresh audit rather than inheriting either domain.

## 5. Spacetime evolution

### 5.1 CCZ4 geometry

The CCZ4 implementation provides:

- conformal metric and trace-free conformal extrinsic curvature;
- conformal factor, trace of extrinsic curvature, Z4 scalar, and connection
  variables;
- configurable `kappa1`, `kappa2`, and `kappa3` damping;
- advective 1+log or harmonic lapse;
- Gamma-driver shift;
- algebraic projection to `det(gamma_tilde)=1`;
- algebraic projection to `tr(A_tilde)=0`;
- Hamiltonian, momentum, Theta, and Z diagnostics;
- conversion to and from ADM variables;
- radiative finite-boundary support with a linearized incoming Z4 projection.

Primary module: [`src/causal_repair/ccz4.py`](src/causal_repair/ccz4.py).

### 5.2 ADM geometry utilities

The lower-level geometry layer supplies:

- inverse metric, determinant, and volume element;
- Christoffel symbols and Ricci tensor;
- covariant Hessians and metric Lie derivatives;
- raw ADM right-hand sides;
- Hamiltonian and momentum constraints;
- flat-metric constructors and ADM state containers.

### 5.3 Gauge support

Moving-puncture gauge helpers include advective 1+log lapse and Gamma-driver
shift state/RHS interfaces. Harmonic lapse is available through CCZ4.

## 6. Relativistic hydrodynamics

The Valencia GRHD solver includes:

- densitized rest mass, momentum, energy, and tracer variables;
- ideal-gas equation of state;
- physical primitive validation;
- atmosphere density and internal-energy floors;
- nonlinear primitive recovery with failure reporting;
- monotonized-central reconstruction;
- relativistic HLL interface fluxes;
- relativistic acoustic characteristic speeds;
- geometric lapse, shift, metric, and extrinsic-curvature sources;
- periodic and no-inflow/outflow finite boundaries;
- standalone SSP-RK3 stepping;
- shock-tube and smooth-wave regression coverage.

The coupled production paths evolve total energy and momentum so vector/matter
exchange remains internal to the combined system.

Primary modules:

- [`src/causal_repair/grhd.py`](src/causal_repair/grhd.py)
- [`src/causal_repair/matter.py`](src/causal_repair/matter.py)

## 7. Grids, derivatives, and time integration

### 7.1 Grid types

- One-, two-, and three-dimensional periodic Cartesian grids.
- One-, two-, and three-dimensional finite cell-centered Cartesian patches.
- Explicit component-leading ghost-zone storage.
- Configurable physical lengths and origins.
- Volume integration and coordinate generation.

### 7.2 Spatial operators

- Fourier spectral first and second derivatives on periodic grids.
- Second-order finite differences.
- Fourth-order finite differences.
- Stable one-sided finite-boundary closures.
- Gradient, divergence, curl, Laplacian, and shifted-neighbor operations.

### 7.3 Time stepping and operator splitting

- Generic array RK4 helper.
- SSP-RK3 standalone GRHD evolution.
- Classical RK4 coupled production evolution.
- Stage-local CCZ4 algebraic projection.
- Stage-local Theory 3.3 constitutive projection.
- Exact split vector damping.
- Implicit split carrier-drag collision map.
- CFL recommendations from metric light speed, acoustic speed, target modes,
  and configured relaxation limits.

## 8. Physical boundary conditions

Finite Cartesian runs support:

- constant-extrapolation ghost filling;
- fluid no-inflow/outflow faces;
- outgoing-only carrier-number and carrier-momentum fluxes;
- Sommerfeld/radiative vector-field boundaries;
- zero-incoming linear target characteristics;
- separate radiative mixed-Proca A/B data;
- Gauss-compatible mixed-Proca boundary correction;
- linearized incoming Z4 constraint-mode suppression.

These boundaries are tested for no periodic wraparound, outward mass/charge
loss, finite-domain Minkowski preservation, and compatible Gauss evolution.
Nonlinear reflection studies at multiple boundary radii remain a qualification
task.

## 9. Initial-data systems

### 9.1 Analytic and benchmark data

- Flat/vacuum geometry.
- Gaussian Proca pulses.
- Rotating and colliding two-lobe matter/target profiles.
- Exact initial Proca constraint setup.
- Homogeneous and localized weak-field states.

### 9.2 Isolated CTT constraint solve

The conformal transverse-traceless solver provides:

- conformally flat isolated Hamiltonian and momentum solves;
- vector-potential momentum correction;
- Picard coupling to the full matter/vector stress;
- residual, iteration, and minimum-conformal-factor reports;
- controlled failure rather than emission of invalid supercritical data.

### 9.3 Periodic CMC constraint solve

The periodic conformally flat solver determines a compatible constant mean
curvature from the integrated Lichnerowicz equation. It is used by the compact
periodic Theory 3.3 collision campaigns, where positive energy makes a maximal
periodic zero mode impossible.

### 9.4 TOV generator

The `tov` command generates spherical relativistic hydrostatic equilibrium for
a polytropic equation of state and reports mass, radius, pressure, density, and
metric profiles.

Primary modules:

- [`src/causal_repair/initial_constraints.py`](src/causal_repair/initial_constraints.py)
- [`src/causal_repair/initial_data.py`](src/causal_repair/initial_data.py)
- [`src/causal_repair/tov.py`](src/causal_repair/tov.py)

## 10. Constraint handling

The solver tracks and controls several distinct constraint systems:

- CCZ4 Hamiltonian and momentum constraints;
- Z4 scalar/vector constraint diagnostics and damping;
- conformal determinant and trace-free algebraic constraints;
- one Gauss constraint for each mixed Proca field;
- one cleaning field per mixed Proca Gauss law;
- evolved longitudinal vector charges;
- initial-data Hamiltonian/momentum elliptic residuals;
- Theory 3.3 constitutive-manifold residuals;
- carrier-current Legendre and thermodynamic admissibility.

Gauss-compatible derivatives, damping correction, finite boundaries, and AMR
transfer projection are designed around the same discrete charge convention.

## 11. Conservation, entropy, and boundary-flux accounting

### 11.1 Conserved groups

Depending on formulation and boundary choice, the code monitors:

- baryon number;
- carrier number and target source charge;
- total material-plus-vector energy and momentum;
- thermal entropy;
- longitudinal vector charges;
- AMR coarse/fine flux balance.

Periodic Theory 3.3 evolution preserves both number currents to roundoff and
enforces nondecreasing integrated entropy.

### 11.2 Exact finite-boundary carrier accounting

The Theory 3.3 solver exposes the same outgoing-only carrier face flux used by
its continuity RHS. The RK4 step accumulates the stage-weighted boundary flux
and reports

\[
\mathcal R_N
=\Delta N_D+\int\Phi_{\partial\Omega}\,dt.
\]

Available diagnostics include:

- instantaneous outward carrier-number rate;
- per-step carrier-number change;
- cumulative boundary outflow;
- absolute and relative balance residuals;
- an instrumentation flag and start time for appended trajectories.

In the completed spherical extension, the maximum relative closure residual is
`2.19e-16`, demonstrating that the measured ppm-scale carrier loss is boundary
transport rather than an interior source.

## 12. Checkpoints, restart, and data products

### 12.1 Checkpoint infrastructure

The portable checkpoint layer stores compressed NPZ arrays plus JSON metadata.
Loaders validate explicit formulation identifiers:

| Backend | Format identifier |
|---|---|
| Theory 2 coupled production | `causal_repair.production.v2` |
| Theory 3.2 | `causal_repair.production.theory3.v1` |
| Theory 3.3 | `causal_repair.production.theory33.v1` |

Incompatible or stale layouts are rejected rather than guessed.

### 12.2 Spherical restart-and-append workflow

The isolated Theory 3.3 collision runner supports in-place continuation from a
run directory or final checkpoint. It:

- reconstructs the solver without rerunning the CTT solve;
- validates the checkpoint, summary, and trajectory as one bundle;
- checks grid, case parameters, timestep, extraction radii, and join time;
- requires the new endpoint to lie on the existing timestep lattice;
- preserves original `t=0` conservation and constraint baselines;
- carries cumulative entropy and boundary-flux diagnostics forward;
- appends without duplicating the join sample;
- stages new files before replacing live artifacts;
- publishes the summary last as the bundle commit marker;
- records restart count and all join times.

A regression test verifies that a checkpointed next step matches the
uninterrupted path across every evolved field.

### 12.3 Output formats

The repository uses:

- JSON-line CLI diagnostics;
- JSON summaries and audit results;
- compressed NPZ checkpoints and trajectories;
- PNG diagnostic figures and field slices;
- dashboard JSON APIs generated directly from saved trajectories.

## 13. Numerical-relativity diagnostics

### 13.1 Local and volume diagnostics

- Hamiltonian and momentum L2 norms.
- Z4 and Theta norms.
- A/B Gauss norms.
- primitive-recovery failure counts and maximum residuals.
- density, internal-energy, lapse, conformal-factor, and relative-flow extrema.
- total number, energy, entropy, drag heat, and damping reports.
- constitutive eigenvalues, characteristic speeds, and condition numbers.

### 13.2 Hawking quasi-local mass

Nested coordinate-sphere extraction computes:

- surface area and areal radius;
- outgoing and ingoing null expansions;
- expansion-product integral;
- Hawking mass.

The diagnostic vanishes for round Minkowski spheres and recovers the
Schwarzschild mass on spherical vacuum slices. The selected coordinate-sphere
foliation is an explicit extraction choice.

### 13.3 Apparent horizons

The inexpensive baseline finder performs a coordinate-sphere mean-expansion
root search and reports:

- horizon found/not-found status;
- coordinate radius;
- area;
- irreducible mass;
- final expansion residual and iteration count.

An opt-in distorted-surface observer represents a positive star-shaped radius
with real spherical harmonics and minimizes the resolved outgoing-expansion
residual using a bound-constrained KKT/Gauss--Newton pseudo-flow. It adds:

- independent monopole and deformation modes;
- trust-region and backtracking safeguards;
- explicit active bounds, multipliers, projected-gradient norm, and
  complementarity residual;
- separate resolved-spectral and full sampled expansion errors;
- finite-dimensional shape-response eigenvalues and conditioning;
- an optional induced-area, physical-unit-normal stability operator with
  perturbation-step and normalization-closure diagnostics;
- a corrected axisymmetric Cartesian observer that constructs the induced
  surface geometry directly and samples physical ADM fields from the grid;
- a direct axisymmetric Brill--Lindquist conformal surface evaluator with
  analytic harmonic derivatives and no Cartesian sampling;
- independent named-branch continuation without winner deletion;
- newly found/lost transition telemetry;
- JSON-compatible branch checkpoint and restore;
- a read-only, restartable CCZ4 snapshot adapter that records the corrected
  physical-normal spectrum for each successfully located branch;
- calibrated per-snapshot resolution and principal-mode sign gates, with an
  explicit prohibition on promoting a single-grid zero to a continuum event;
- a restartable head-on CCZ4 puncture campaign with moving branch centers,
  stability-sign common-pair classification, boundary-aware constraints, and
  saved snapshot/trajectory analysis.

The observer consumes an ADM or reconstructed CCZ4 slice after it has been
evolved. Its pseudo-time never enters the CCZ4 right-hand side. The raw
shape-response spectrum remains parameterization-dependent. The corrected
physical-normal observer passes exact flat and spherical convergence controls,
but the distorted-puncture calibration still requires resolution ensembles
near a mode zero.

### 13.4 Gravitational-wave extraction

Finite-radius vacuum extraction supplies:

- electric and magnetic Weyl tensors;
- a local orthonormal spatial dyad;
- pointwise `Psi4` on a spherical quadrature;
- spin-weighted `l=2`, `m=-2..2` mode integrals.

Higher multipoles and Cauchy-characteristic extraction are not yet included.

Primary module:
[`src/causal_repair/nr_diagnostics.py`](src/causal_repair/nr_diagnostics.py).
Distorted-surface module and contract:
[`src/causal_repair/kkt_horizon.py`](src/causal_repair/kkt_horizon.py) and
[`experiments/KKT_MOTS_EXPERIMENT.md`](experiments/KKT_MOTS_EXPERIMENT.md).

## 14. AMR infrastructure

Status: **infrastructure-ready, locally tested**.

The AMR layer includes:

- ratio-2 limited piecewise-linear conservative prolongation in 1D--3D;
- volume-average restriction;
- Berger--Colella time-subcycling orchestration;
- coarse/fine flux registers and reflux correction;
- Theory 3 grouped registers for A/B electric densities, both longitudinal
  charges, target charge, and target current;
- Gauss reprojection after Theory 3 prolongation/restriction/reflux;
- exact parent/child conservation tests.

The generic pieces are not yet wired into a distributed, moving, load-balanced
hierarchy for the complete production state.

Primary module: [`src/causal_repair/amr.py`](src/causal_repair/amr.py).

## 15. Parallel and accelerator interfaces

Status: **infrastructure-ready; hardware qualification incomplete**.

The backend layer provides:

- a NumPy/CuPy array compatibility boundary;
- automatic NumPy fallback;
- an accelerator-ready closed-form 2x2 damping exponential;
- optional `mpi4py` context with a single-rank fallback;
- slab partition metadata;
- deterministic pairwise reductions;
- explicit halo exchange;
- one complete Theory 3 halo contract covering vector, longitudinal, cleaner,
  target-charge, and target-current fields.

The full tensor evolution is still predominantly NumPy code. Exposing a CuPy
boundary is not the same as demonstrating GPU-native end-to-end performance,
and no multi-node scaling claim is made.

Primary module: [`src/causal_repair/backend.py`](src/causal_repair/backend.py).

## 16. Experiment and prediction suite

### 16.1 Fixed-background active-repair exploration

| Workflow | Capability | Main entry point/report |
|---|---|---|
| Parameter scan | Ranks visually and dynamically interesting repair regimes | [`experiments/scan_active_repair.py`](experiments/scan_active_repair.py) |
| Showcase reproduction | Recreates radiative control, fast trap, ringing trap, slow overshoot, and interaction-dominant trajectories | [`experiments/repair_showcases.py`](experiments/repair_showcases.py), [`experiments/SHOWCASE_RUNS.md`](experiments/SHOWCASE_RUNS.md) |
| Static rendering | Creates overview figures from saved trajectories | [`experiments/render_repair_showcases.py`](experiments/render_repair_showcases.py) |
| Blind forecasting | NumPy radial-basis surrogate with locked out-of-sample predictions | [`experiments/PREDICTION_REPORT.md`](experiments/PREDICTION_REPORT.md) |
| Trapping boundary | Adaptive finite-time radiation/trapping calibration and withheld tests | [`experiments/TRAPPING_BOUNDARY_REPORT.md`](experiments/TRAPPING_BOUNDARY_REPORT.md) |
| Euler--Proca audit | Shock-state characteristic mixing, superluminality, and nonhyperbolic-island mapping | [`experiments/HYPERBOLICITY_REPORT.md`](experiments/HYPERBOLICITY_REPORT.md) |

The blind prediction workflow separates training, locked forecasts, unseen
evolutions, and higher-resolution checks. The trapping workflow includes a
regular boundary fit, withheld cases, resolution studies, a transparency
trough, and a separately identified slow-light branch.

### 16.2 Theory 3.2 derivation and eigenspace audits

- Target-current principal-symbol and Euler-contact coincidence audit.
- Lorentz-force sign check.
- Gauss-matrix positivity/invertibility.
- Evolved-longitudinal-charge closure.
- Complete finite-step damping deposit tests.

Entry point:
[`experiments/audit_theory32_target_eigenspace.py`](experiments/audit_theory32_target_eigenspace.py).

### 16.3 Theory 3.3 derivation gates

The executable master-function audit checks metric variation, Legendre
invertibility, thermodynamic convexity, characteristic causality,
hyperbolicity, and counterflow domains.

Entry point:
[`experiments/audit_theory33_master.py`](experiments/audit_theory33_master.py).

### 16.4 Theory 3.3 dynamical campaigns

| Campaign | Main result | Report |
|---|---|---|
| Periodic counterflow collision | Separates drag-locked from penetrating carrier flow | [`experiments/THEORY33_COLLISION_REPORT.md`](experiments/THEORY33_COLLISION_REPORT.md) |
| Carrier-loading scan | Predicts gravitational branch splitting controlled by the carrier EOS exponent | [`experiments/THEORY33_CARRIER_LOADING_REPORT.md`](experiments/THEORY33_CARRIER_LOADING_REPORT.md) |
| EOS falsification | Varies carrier sound speed and verifies exponent tracking | [`experiments/THEORY33_EOS_FALSIFICATION_REPORT.md`](experiments/THEORY33_EOS_FALSIFICATION_REPORT.md) |
| Carrier-coefficient attack | Finds nearly linear response amplitude in `K_D` and a low-stiffness causal boundary | [`experiments/THEORY33_KD_ATTACK_REPORT.md`](experiments/THEORY33_KD_ATTACK_REPORT.md) |
| Three-resolution continuum campaign | 120 matched evolutions and nominal second-order continuum extrapolation | [`experiments/THEORY33_KD_CONTINUUM_REPORT.md`](experiments/THEORY33_KD_CONTINUUM_REPORT.md) |
| Isolated 3-D sphere collision | Uses nested Hawking masses instead of a gauge-dependent lapse extremum | [`experiments/THEORY33_SPHERICAL_COLLISION_REPORT.md`](experiments/THEORY33_SPHERICAL_COLLISION_REPORT.md) |
| Restarted long spherical campaign | Tracks the delayed mass-response exponent and exact finite-boundary carrier closure | [`experiments/THEORY33_SPHERICAL_LONG_REPORT.md`](experiments/THEORY33_SPHERICAL_LONG_REPORT.md) |

The latest spherical history reaches `t=0.068`. Its fitted inner Hawking-mass
split exponent falls through approximately `3.31, 2.87, 2.71, 2.53, 2.38`,
with a trailing-window estimate near `2.34`. Carrier loss closes against
measured boundary flux at roundoff. The original 1% outer-mass control is
marginally exceeded at the endpoint, so the next clean campaign requires a
larger box rather than another same-box append.

## 17. Dashboard suite

### 17.1 Local trajectory dashboard

The **Causal Repair Observatory** is a dependency-light local web dashboard
for the saved `Interaction Dominant` and `Ringing Trap` trajectories.

Features include:

- scenario navigation;
- synchronized play/pause and timeline scrubbing;
- click-to-select spacetime slices;
- heatmap switching among `A_z`, active mismatch, repair energy, and `Phi`;
- keyboard playback and frame stepping;
- trajectory summaries and timeseries panels;
- a threaded local HTTP server;
- no external application framework.

API endpoints:

- `/api/health`
- `/api/catalog`
- `/api/showcase/<scenario>`

Launch with:

```powershell
python dashboard/server.py
```

or:

```powershell
.\launch_dashboard.ps1
```

See [`dashboard/README.md`](dashboard/README.md).

### 17.2 Standalone binary-neutron-star ShaderToy

The repository also produces a self-contained, shareable ShaderToy `Image`
program. Its offline reduced model supplies leading quadrupole inspiral,
compactness-based radii, Love-number tides, and a smooth visualization bridge
through merger and a mass-scaled post-merger `f2` mode. The shader adds procedural stars, neutron-star
surface raymarching, weak-field lensing, transverse wave distortion, visible
chirping wave ribbons, a waveform panel, and an explicitly illustrative
causal-repair field layer.

A separate prompt-collapse build adds a formation-gated analytic horizon
wireframe. General real `m != 0` parametric MOTS surfaces and a compact
`causal_repair.mots_shader_bake.v1` interface can replace that analytic surface
and color the wireframe/repair glow from a signed offset-expansion response.
The repository does not yet claim a solver-derived BNS horizon packet.

The generated GLSL needs no Python process, checkpoint, texture, or network
connection after it is pasted into ShaderToy. A JSON manifest records the
parameters and scientific qualification boundary, and a local WebGL preview
compiles the exact publication source.

Generate it with:

```powershell
python shadertoy/build_bns_shader.py
```

See [`shadertoy/SHADERTOY_GUIDE.md`](shadertoy/SHADERTOY_GUIDE.md).

## 18. Command-line interface

Install the editable package:

```powershell
python -m pip install -e .
```

The declared runtime is Python 3.10 or newer with NumPy 1.24 or newer. CuPy
and `mpi4py` are optional and loaded only when their backend features are
requested.

The console entry point is `causal-repair`; without installation, use
`python -m causal_repair` after setting `PYTHONPATH=src`.

| Command | Purpose |
|---|---|
| `proca` | Fixed-background conservative Proca pulse |
| `relax` | Deterministic or stochastic relaxation |
| `coupled` | Experimental raw ADM--Proca evolution |
| `theory2` | Fixed-background Theory 2.0 two-lobe benchmark |
| `production` | Coupled CCZ4+GRHD+Theory 2.0 reference |
| `production3` | Theory 3.2 mixed-Proca/reservoir reference |
| `production33` | Preferred Theory 3.3 two-current reference |
| `tov` | Spherical TOV equilibrium generator |
| `initial-data` | Isolated CTT two-lobe constraint solve |

Examples:

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m causal_repair production33 --help

python -m causal_repair production33 `
  --points 24 --derivatives fd2 --final-time 0.002 `
  --carrier-velocity 0.2 --electric-A 0.02 --electric-B=-0.01 `
  --gamma-W 0.2 --gamma-B 0.1 `
  --output runs/theory33_coupled_smoke.npz

python experiments/run_theory33_spherical_collision.py --help
```

Small script-level examples are also available in [`examples`](examples):

- `proca_pulse.py` for the conservative fixed-background field;
- `relaxation.py` for the dissipative branch;
- `coupled_adm_proca.py` for raw ADM coupling;
- `production.py` for the coupled CCZ4+GRHD reference.

## 19. Verification suite

The current repository suite contains **126 passing tests**. Coverage includes:

- periodic and finite grid calculus;
- raw ADM and CCZ4 fixed points and constraints;
- gauge-wave and smooth-wave convergence;
- GRHD primitive recovery, conservation, and shocks;
- Proca energy, Gauss, and source conventions;
- causal current projection;
- Theory 2.0 source/target/stress contracts;
- production checkpoint round trips;
- isolated CTT and periodic CMC initial data;
- TOV equilibrium;
- physical boundary behavior and no wraparound;
- AMR prolongation, restriction, subcycling, and reflux;
- serial MPI/backend fallbacks and accelerator kernel agreement;
- horizon, Hawking-mass, and `Psi4` controls;
- Theory 3.2 Gauss, damping, force, eigenspace, boundary, and AMR contracts;
- Theory 3.3 metric variation, recovery, convexity, causality, drag, damping,
  entropy, finite boundaries, checkpoints, restart equivalence, and coupled
  stepping;
- direct carrier face-flux/RHS agreement and RK4 boundary-balance closure;
- dashboard catalog and trajectory contracts.
- reduced BNS scale/continuity/strain checks and deterministic standalone GLSL
  generation.
- non-axisymmetric parametric MOTS geometry, post-merger `f2` scaling,
  formation-gated wireframes, and MOTS bake serialization/embedding.

Run everything with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

## 20. Package map

| Module | Responsibility |
|---|---|
| `adm.py` | Raw ADM geometry, curvature, evolution, constraints |
| `amr.py` | Conservative transfer, subcycling, reflux, Theory 3 Gauss projection |
| `backend.py` | NumPy/CuPy boundary, MPI slabs, reductions, halo contracts |
| `boundary.py` | Outflow, radiative, target-characteristic, and Z4 boundaries |
| `bns_shadertoy.py` | Reduced BNS dynamics and standalone GLSL export |
| `causal_map.py` | Covariant spatial projection and bounded causal current map |
| `ccz4.py` | CCZ4 evolution, gauge, projections, and diagnostics |
| `cli.py` | Console subcommands and JSON diagnostic emission |
| `coupled.py` | Raw ADM--Proca driver |
| `coupled_theory2.py` | Prescribed-matter Theory 2.0 ADM bridge |
| `curved_proca.py` | Proca equations and stress on ADM slices |
| `fixed_theory2.py` | Closed fixed-background Theory 2.0 driver |
| `gauge.py` | Moving-puncture gauge helpers |
| `grhd.py` | Valencia finite-volume relativistic hydrodynamics |
| `grid.py` | Periodic/finite grids, derivatives, ghost patches |
| `initial_constraints.py` | Isolated CTT and periodic CMC solvers |
| `initial_data.py` | Two-lobe benchmark initial data |
| `integrators.py` | Generic RK4 helper |
| `io.py` | Compressed NPZ/JSON checkpoint layer |
| `kkt_horizon.py` | Constrained spectral MOTS and branch tracking |
| `matter.py` | Ideal-gas fluid primitives and stress-energy |
| `nr_diagnostics.py` | Hawking mass, horizons, Weyl/`Psi4` extraction |
| `proca.py` | Fixed-background conservative Proca branch |
| `production.py` | Coupled Theory 2.0 production reference |
| `production3.py` | Theory 3.2 coupled production reference |
| `production33.py` | Theory 3.3 coupled multifluid production reference |
| `relaxation.py` | Deterministic/stochastic Maxwell--Cattaneo branch |
| `theory2.py` | Theory 2.0 source, target, and interaction contracts |
| `theory3.py` | Mixed-Proca/Gauss/damping formulation |
| `theory33.py` | Two-current master function and characteristic audit |
| `tov.py` | Spherical polytropic equilibrium generator |

## 21. Known limits and honest qualification boundary

### 21.1 Numerical-relativity scale

- The code is a NumPy reference backend, not a mature HPC framework.
- Full moving-box AMR is not wired into the complete production state.
- Multi-rank MPI scaling has not been demonstrated.
- Most tensor kernels are not GPU-native despite the CuPy-compatible boundary.
- Long binary, puncture, spinning-black-hole, and robust-stability campaigns
  remain to be run.
- Independent comparison with another NR code has not been completed.

### 21.2 Boundary and extraction limits

- The Z4 outer-boundary projection is linearized.
- Nonlinear reflection coefficients have not been mapped across box sizes.
- Both horizon finders require a supplied center; the spectral observer still
  assumes a star-shaped surface and is not a general-topology MOTS finder.
- The raw response spectrum remains parameterization-dependent. The corrected
  physical-normal spectrum removes log-radius scaling, but distorted puncture
  surfaces are not yet qualified from a single Cartesian resolution.
- Wave extraction currently stops at finite-radius vacuum `l=2` modes.
- The latest same-box spherical extension has begun to reach the outer mass
  control; a larger-domain rerun is required before extending it farther.

### 21.3 Theory 3.3 physical scope

The current M1 model does not yet include:

- target pair creation or charge-changing reactions;
- simultaneous positive and negative carrier species;
- an independent heat-flow velocity;
- anisotropic directors or elastic target media;
- stochastic Theory 3.3 fluctuation terms or their full Schwinger--Keldysh
  completion;
- superfluid vortex quantization;
- a proof of global hyperbolicity outside the explicitly audited state domain.

Near-carrier-vacuum states, nondefault entrainment, and stronger counterflow
must be separately qualified. Passing coefficient positivity alone is not a
causality or hyperbolicity proof.

### 21.4 Interpretation limits

- Saved predictions are predictions of the implemented numerical theory, not
  observational forecasts for nature.
- The trapping surface is a finite-time operational boundary for a specified
  pulse, matter profile, and box.
- The spherical collisions use already-contacting Gaussian bodies, not
  equilibrium neutron stars.
- Coordinate-sphere Hawking masses are geometrically meaningful for each chosen
  surface, but the foliation is part of the extraction prescription.
- Empirical power laws and continuum fits are numerical findings, not analytic
  theorems.

## 22. Recommended entry points

For a new user:

1. Read [`README.md`](README.md) for installation and short commands.
2. Run the full test suite.
3. Run `python -m causal_repair production33 --help`.
4. Read the Theory 3.3 implementation and master-action reports.
5. Inspect the saved spherical and continuum campaign reports.
6. Launch the dashboard for an interactive view of the earlier active-repair
   regimes.

For formulation work:

1. Start with the Theory 3.2 closure report to see the repaired Gauss, force,
   eigenspace, and damping structure.
2. Continue to the Theory 3.3 master action and executable derivation audit.
3. Rerun the characteristic audit whenever changing master-function
   coefficients or the declared counterflow domain.

For new 3-D campaigns:

1. Solve constraints independently for every resolution/domain.
2. Lock constitutive and characteristic gates before evolution.
3. Use periodic controls or explicit finite-boundary flux accounting.
4. Track recovery, entropy, Gauss, CCZ4 constraints, and outer extraction
   controls together.
5. Treat a failed preregistered control as a qualification result, not a reason
   to move the threshold after the fact.

## 23. Bottom line

The repository now combines a covariant two-current matter closure, mixed
massive-vector dynamics, CCZ4 spacetime evolution, relativistic hydrodynamics,
constraint-solved data, shock-capable finite-volume machinery, exact stiff
source maps, causal/convexity audits, restartable 3-D experiments, boundary
flux closure, quasi-local mass extraction, prediction/falsification workflows,
and an interactive dashboard in one transparent reference codebase.

Its strongest current claim is not that every astrophysical or HPC validation
problem is finished. It is that the local Theory 3.3 formulation and its
reference numerical mechanisms are explicit, executable, regression-tested,
and instrumented well enough to make failures and qualification boundaries
visible rather than silently hiding them.
