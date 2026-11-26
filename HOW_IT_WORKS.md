# How Theory 3.3 / `production33` Works

This is the straightforward guide to the current Theory 3.3 solver. It explains
what the major pieces do, why they are there, and how they work together without
assuming that the reader is a numerical-relativity specialist.

For the exhaustive technical inventory, see
[`FEATURE_OVERVIEW.md`](FEATURE_OVERVIEW.md).

## The short version

Theory 3.3 simulates gravity, ordinary matter, and an additional active-repair
sector at the same time.

The model contains:

- spacetime, which bends in response to energy and momentum;
- an ordinary relativistic fluid, representing baryonic matter;
- a second carrier fluid, which transports the active-repair charge;
- two interacting massive vector fields;
- internal drag, which allows the two fluids to move differently and then
  relax toward one another;
- exact bookkeeping for energy, momentum, charge, entropy, and boundary flow.

A useful mental picture is two interpenetrating fluids moving through the same
curved spacetime. One is ordinary matter. The other carries the repair-sector
charge. They can move together, stream through one another, exchange momentum,
and influence the vector fields and geometry.

The solver's job is to evolve all of that without accidentally creating energy,
losing charge inside the grid, violating causality, or letting the equations
leave their physically allowed state space.

## Mechanical feature checklist

This section lists the concrete mechanisms built into `production33` and the
supporting solver stack. Each item describes what the feature physically or
numerically does.

### Grid and spatial-calculus features

- **One-, two-, and three-dimensional Cartesian grids** — The same field and
  matter systems can be exercised in cheap 1-D tests, intermediate 2-D tests,
  or full 3-D evolutions.

- **Periodic grids** — Opposite sides of the domain connect to one another.
  These are useful for conservation tests and compact periodic collision
  experiments with no physical outer boundary.

- **Finite isolated grids** — The domain has real outer faces through which
  matter, carriers, and radiation can leave.

- **Cell-centered storage** — Conserved matter, charge, field, and geometry
  variables live at the centers of numerical cells, giving all coupled sectors
  one consistent layout.

- **Explicit ghost-zone patches** — Extra cells outside a finite patch hold
  boundary or neighboring-domain data for finite-difference and halo
  operations.

- **Spectral derivatives on periodic grids** — Fourier differentiation gives
  very accurate derivatives for smooth, resolved periodic fields.

- **Second-order finite differences** — A robust lower-order option used by
  shock-capable and finite-boundary campaigns.

- **Fourth-order finite differences** — A higher-order option for smooth-field
  tests and convergence studies.

- **One-sided boundary stencils** — Finite grids differentiate near an outer
  edge without wrapping around to the opposite side.

- **Vector-calculus operators** — Gradient, divergence, curl, and Laplacian
  operations share the same grid conventions used by the evolution and
  constraints.

- **Volume integration** — Global charge, mass, energy, entropy, and residuals
  are computed with the grid's physical cell volume.

### Spacetime and gauge features

- **Full CCZ4 spacetime evolution** — The spatial metric and extrinsic
  curvature evolve dynamically in response to the complete matter and vector
  stress-energy.

- **Z4 constraint variables** — Extra geometry variables turn gravitational
  constraint violations into modes that can propagate and damp instead of
  remaining frozen in place.

- **Configurable CCZ4 damping** — `kappa1`, `kappa2`, and `kappa3` control how
  the formulation handles gravitational constraint errors.

- **Advective 1+log lapse** — A standard moving-puncture time-slicing choice
  that helps the coordinate time adapt to strong gravitational fields.

- **Harmonic lapse option** — An alternate slicing condition useful for clean
  wave and formulation tests.

- **Gamma-driver shift** — Dynamically moves the spatial coordinates to reduce
  coordinate distortion during evolution.

- **Algebraic conformal projection** — Every stage restores unit determinant of
  the conformal metric and zero trace of its conformal trace-free curvature.

- **ADM conversion utilities** — CCZ4 states can be converted to ordinary ADM
  metric/extrinsic-curvature variables for constraints and geometric
  diagnostics.

- **Hamiltonian and momentum constraints** — The solver measures whether each
  evolved spatial slice remains compatible with Einstein's equations.

- **Linearized incoming Z4 boundary control** — The finite boundary suppresses
  incoming gravitational constraint modes in the current reference
  approximation.

### Ordinary-fluid features

- **Valencia GRHD formulation** — Ordinary matter is evolved through conserved
  relativistic mass, momentum, and energy variables.

- **Ideal-gas equation of state** — Pressure and internal energy are related by
  a configurable adiabatic index.

- **Monotonized-central reconstruction** — Cell data are reconstructed at
  interfaces while limiting oscillations near steep gradients.

- **Relativistic HLL fluxes** — A shock-capturing Riemann flux transports
  baryonic matter across cell faces.

- **Relativistic acoustic speeds** — The numerical flux and timestep account
  for both fluid motion and sound waves.

- **Atmosphere floors** — Very low-density cells receive controlled density and
  internal-energy floors so vacuum-adjacent recovery remains numerically
  defined.

- **No-inflow finite faces** — Ordinary matter may leave an isolated grid, but
  the boundary does not inject unrequested material.

- **Tracer support** — A conserved tracer can be advected with the fluid for
  tagging material populations.

### Two-current matter features

- **Independent baryon and carrier currents** — Ordinary matter and repair
  carriers have separate conserved densities and velocities.

- **Invariant M1 master function** — One covariant energy model determines the
  fluids' conjugate momenta, chemical forces, relative-flow energy, and total
  material stress.

- **Constant-sound-speed carrier equation of state** — `carrier_K` and
  `carrier_sound_speed` control the carrier constituent's density-dependent
  energy and stiffness.

- **Chemical anchoring** — The master function can energetically prefer a
  carrier-to-baryon ratio set by `target_fraction` without forcing that ratio
  instantaneously.

- **Chemical susceptibility** — Controls the energetic cost of moving away
  from the preferred carrier loading.

- **Optional invariant entrainment** — Linear and quadratic coefficients can
  couple each current's momentum to their relative flow. Nonzero values require
  a fresh characteristic audit.

- **Hilbert/Carter material stress** — The stress-energy that gravitates is
  derived from the master function, so there is no separately invented
  interaction tensor.

- **Independent metric-variation audit** — A numerical variation of the metric
  checks that the implemented analytic stress has the correct signs and
  components.

- **Conservative baryon-number evolution** — Baryon number changes only through
  numerical boundary flux on a finite domain.

- **Conservative carrier-number evolution** — Carrier charge is transported as
  an evolved densitized conserved quantity.

- **Canonical carrier-momentum evolution** — Three independent carrier
  canonical momenta carry the true relative-flow dynamics.

- **Path-conservative principal product** — The carrier-vorticity term that
  cannot be written as an ordinary flux is discretized along a defined
  straight path instead of being mislabeled as a harmless source.

- **Coupled nine-variable primitive recovery** — A nonlinear cell solve
  reconstructs both proper densities and both velocities while closing total
  energy, momentum, entropy, and carrier momentum.

- **Recovery line search and residual report** — Nonlinear updates are damped
  when necessary, and every recovery exposes its accuracy and failed-cell
  count.

- **Stage-local constitutive projection** — Intermediate RK states are returned
  to the master-function manifold by changing only the redundant entropy
  representation.

- **Entropy admissibility guard** — Periodic complete steps are rejected if
  their integrated entropy decreases beyond numerical tolerance.

- **Implicit stiff drag** — Relative fluid motion can relax safely even when a
  timestep spans many microscopic relaxation times.

- **Conservative drag map** — The collision step preserves both numbers and
  total material energy-momentum while converting relative free energy into
  entropy.

### Mixed-vector-field features

- **Two massive vector fields** — A and B each carry spatial potentials and
  electric/canonical momenta.

- **Configurable field masses and kinetic coefficients** — Each vector's range
  and energy normalization can be adjusted independently.

- **Mass mixing** — A constant off-diagonal mass term allows the two vector
  fields to exchange and share modes.

- **Carrier-derived B source** — The B field is sourced by the physical evolved
  carrier current, not by a prescribed direction callback.

- **Correct equal/opposite Lorentz exchange** — The carrier force and vector
  stress exchange use locked signs so momentum gained by one sector is lost by
  the other.

- **Independent longitudinal A/B charges** — The information normally hidden
  inside a Gauss reconstruction is evolved as part of the hyperbolic state.

- **Derivative-free temporal-potential reconstruction** — Temporal vector
  components are recovered locally without differentiating electric-field
  divergence near shocks.

- **One Gauss cleaner per field** — Small A and B Gauss errors propagate and
  damp independently.

- **Compatible discrete Gauss evolution** — The electric and longitudinal
  equations use matching derivative conventions so satisfied constraints stay
  satisfied to roundoff in periodic tests.

- **Radiative A/B boundaries** — Vector disturbances can leave a finite domain
  through outgoing boundary conditions.

- **Gauss-compatible finite boundaries** — Boundary updates are corrected so
  radiation handling does not arbitrarily break either Gauss law.

- **Exact 2x2 damping exponential** — Coupled electric damping is solved with a
  closed local matrix exponential instead of an explicit approximation.

- **Complete finite damping deposit** — The solver compares vector stress
  before and after damping and transfers the full finite energy/momentum change
  into the multifluid state.

- **Irreversible-heat isolation** — Only the exact positive electric loss is
  added to thermal entropy; reversible chemical or kinetic rearrangement is not
  falsely counted as heat.

### Coupled evolution features

- **Complete stress-energy sourcing** — CCZ4 receives the combined baryon,
  carrier, chemical, entrainment, and vector stress rather than independently
  incomplete pieces.

- **Classical RK4 production stepping** — The hyperbolic coupled system advances
  through four stage evaluations per timestep.

- **Projection at every internal stage** — Geometry and material constraints
  are repaired before the next stage computes fluxes and sources.

- **Split exact/implicit local physics** — Vector damping and carrier drag are
  applied with methods designed for their stiff local timescales.

- **CFL timestep recommendation** — Grid spacing, metric light speed,
  dimensionality, and relaxation limits contribute to a conservative suggested
  timestep.

- **Finite-state guards** — Invalid parameter values, nonphysical recovered
  states, failed constitutive gates, and nonconverged initial data produce
  explicit errors instead of silent output.

### Initial-data features

- **Flat/vacuum constructors** — Exact control states are available for fixed-
  point and boundary tests.

- **Gaussian and two-lobe matter profiles** — Reproducible pulses and collision
  profiles support controlled experiments.

- **Isolated CTT constraint solver** — Solves the Hamiltonian and momentum
  constraints for conformally flat finite-domain data.

- **Matter/geometry Picard iteration** — The initial geometry is repeatedly
  updated against the full stress so the data are mutually consistent.

- **Periodic CMC constraint solver** — Finds a compatible constant mean
  curvature for positive-energy periodic collision data.

- **Controlled elliptic nonconvergence** — Supercritical or incompatible data
  are reported as failed solves rather than being evolved as valid initial
  conditions.

- **TOV equilibrium generator** — Produces spherical polytropic relativistic
  stars for hydrostatic and future strong-gravity benchmarks.

### Boundary and balance features

- **Outgoing carrier boundary flux** — Carrier number and momentum may leave an
  isolated grid without opposite-edge reinjection.

- **Direct physical-face flux integration** — The exact carrier flux used by
  the finite-volume RHS is summed over every exterior face.

- **RK4-weighted cumulative outflow** — Boundary transport uses the same stage
  weights as the charge evolution, avoiding a lower-order diagnostic mismatch.

- **Carrier balance residual** — The code reports interior number change plus
  integrated outward flux, directly testing for unexplained sources or sinks.

- **Instrumentation start flags** — Restarted histories identify which samples
  actually contain measured flux rather than pretending older data were
  instrumented retroactively.

### Diagnostics and analysis features

- **CCZ4 constraint norms** — Hamiltonian, momentum, Z4, and Theta measures show
  whether the spacetime evolution is drifting away from Einstein's equations.

- **A/B Gauss norms** — Each vector constraint is monitored independently.

- **Recovery diagnostics** — Failed cells, atmosphere cells, and maximum
  inversion residuals expose matter-solver health.

- **Constitutive margins** — Minimum Legendre and thermodynamic eigenvalues show
  how close the material model is to losing invertibility or convexity.

- **Characteristic audit** — Wave speeds, eigenvector completeness, condition
  numbers, causal-cone violations, and counterflow limits can be scanned before
  evolution.

- **Relative Lorentz factor** — Measures the invariant strength of baryon/
  carrier counterflow.

- **Entropy and heat reports** — Drag heat, damping heat, projection entropy,
  and total step entropy are separated rather than collapsed into one number.

- **Hawking quasi-local mass** — Measures the geometric mass associated with
  selected nested closed spheres.

- **Null-expansion diagnostics** — Outgoing and ingoing surface expansions help
  interpret trapping and quasi-local mass.

- **Spherical apparent-horizon search** — Locates a coordinate-sphere expansion
  root and reports its area and irreducible mass.

- **Experimental spectral KKT/MOTS observer** — Fits a positive distorted
  star-shaped surface to zero outgoing expansion, records active constraints
  and response modes, and tracks multiple branches without altering the
  spacetime evolution.

- **Corrected Cartesian surface observer** — Builds the physical normal,
  area, and curvature directly on an axisymmetric angular surface instead of
  differentiating an artificial three-dimensional level-set normal.

- **CCZ4 physical-normal mode tracker** — Reads saved or live CCZ4 snapshots,
  continues separately named horizon branches, measures their unit-normal
  stability modes, applies calibrated resolution/sign gates, and checkpoints
  its own observer state for append runs.

- **Dynamic puncture/MOTS campaign** — Evolves head-on Brill–Lindquist
  punctures with moving-puncture CCZ4 gauge, relocates individual search
  centers, separates the common inner/outer pair by stability sign, and
  records restartable horizon timelines with boundary-aware constraints.

- **Direct conformal horizon benchmark** — Evaluates axisymmetric
  Brill--Lindquist minimal surfaces entirely on the angular grid, providing an
  interpolation-free oracle for the Cartesian horizon observer.

- **Finite-radius `Psi4` extraction** — Computes quadrupolar gravitational-wave
  information from the electric and magnetic Weyl tensors.

- **Equatorial field slices** — Saved baryon, carrier, and conformal-factor
  slices make the internal collision structure directly visualizable.

- **Machine-readable summaries** — JSON results preserve gates, parameters,
  extrema, conservation errors, and fitted campaign outcomes.

### Checkpoint and campaign features

- **Compressed Theory 3.3 checkpoints** — The complete geometry, fluids,
  vectors, longitudinal charges, cleaners, and carrier momentum are stored in
  a portable NPZ bundle.

- **Explicit format identifier** — A checkpoint must identify itself as
  `causal_repair.production.theory33.v1`; incompatible layouts are rejected.

- **Restart equivalence** — Regression tests verify that the next evolved step
  agrees with the uninterrupted solver path.

- **Restart-and-append support** — The spherical runner continues a completed
  history without rerunning the original CTT solve or early evolution.

- **Bundle cross-validation** — Checkpoint time, summary, trajectory, grid,
  parameters, timestep, and extraction radii must all agree before continuation.

- **Duplicate-free joins** — Each restart time appears exactly once in the
  appended trajectory.

- **Transactional staging** — New summary, trajectory, and checkpoint files are
  written to pending paths before replacing the live artifacts.

- **Original-baseline preservation** — Conservation, constraint, and entropy
  comparisons continue to reference the true `t=0` state after restarts.

- **Static renderers and fitted analyses** — Campaign scripts generate figures,
  power-law fits, continuum extrapolations, falsification verdicts, and explicit
  pass/fail gates from saved runs.

### AMR, parallel, and accelerator infrastructure

These mechanisms are implemented and locally tested, but they have not yet
been assembled into a campaign-qualified distributed Theory 3.3 evolution.

- **Conservative AMR prolongation** — A limited ratio-2 interpolation creates
  fine cells whose average exactly reproduces the parent cell.

- **Volume restriction** — Fine data return to a coarse level through physical
  cell averaging.

- **Berger--Colella subcycling** — Fine levels can take multiple smaller steps
  during one coarse-level step.

- **Flux registers and refluxing** — Coarse/fine interface flux differences are
  accumulated and corrected to restore conservation.

- **Grouped Theory 3 field reflux** — Both electric densities, longitudinal
  charges, target charge, and target current have coordinated registers.

- **Gauss projection after AMR transfer** — Longitudinal charges are restored to
  compatibility after cell-centered interpolation or restriction.

- **NumPy/CuPy backend boundary** — Selected local kernels can use NumPy or
  CuPy arrays through one interface.

- **Accelerator-ready damping kernel** — The exact local 2x2 exponential uses
  elementwise operations compatible with both array backends.

- **Optional MPI slab partitioning** — The grid can be divided into axis-aligned
  slabs when `mpi4py` is available.

- **Deterministic reductions** — Pairwise summation makes global totals less
  dependent on rank ordering.

- **Complete Theory 3 halo contract** — One exchange description covers vector,
  longitudinal, cleaner, target-charge, and target-current fields.

## The main pieces

### 1. Dynamical spacetime

The geometry is not a fixed background. Matter and fields tell spacetime how to
curve, and the curved spacetime tells matter and fields how to move.

The code evolves the geometry with CCZ4, a formulation of Einstein's equations
designed to track and damp numerical violations of the gravitational
constraints. It also evolves the lapse and shift, which control how the
simulation slices spacetime into successive moments and labels points on each
slice.

In plain language: the solver evolves gravity itself, while continuously
checking whether the numerical spacetime remains consistent with Einstein's
constraint equations.

### 2. Ordinary relativistic matter

The baryonic component is treated as a relativistic fluid. It has:

- density;
- pressure and internal energy;
- velocity;
- entropy;
- energy and momentum.

The fluid is evolved with a shock-capable Valencia GRHD method. This lets it
handle smooth flows as well as steep compression and shock-like features.

In plain language: ordinary matter can collide, compress, heat, and move near
relativistic speeds without being treated as a collection of test particles.

### 3. The carrier fluid

The repair sector is carried by a second conserved current. This carrier fluid
has its own density and velocity, so it does not have to move exactly with the
ordinary matter.

That is important because many of the interesting effects come from
counterflow: ordinary matter moves one way while the carriers move differently
or pass through it.

The carrier component has a configurable equation of state. Its stiffness is
controlled mainly by `carrier_K` and `carrier_sound_speed`. These parameters
determine how much carrier energy is stored at a given carrier density and how
quickly carrier disturbances propagate.

In plain language: the repair charge has a real transport medium with its own
dynamics, rather than being a target value that is simply imposed by hand.

### 4. The two-current master function

The two fluids are tied together by a covariant master function. This is the
central physical closure of Theory 3.3.

The master function determines:

- the energy stored in each fluid;
- the chemical preference for a particular carrier-to-baryon ratio;
- how relative motion contributes to the energy;
- the momenta conjugate to both currents;
- the complete material stress-energy tensor that gravitates.

Because the stress-energy comes from one master function, the code does not
need a fictional leftover-energy reservoir. Chemical, kinetic, thermal, and
relative-flow energies are all represented inside the same physical material
state.

In plain language: one consistent energy model decides how the two fluids push
on each other and on spacetime.

### 5. Chemical anchoring

The model can prefer an equilibrium carrier density proportional to the baryon
density. The `target_fraction` parameter sets that preferred ratio.

The fluids are not forced to remain at that ratio. Compression, counterflow,
field forces, and finite relaxation time can push them away from it. The
chemical term then contributes to the energy and restoring forces.

In plain language: matter provides a preferred amount of carrier loading, but
the carrier fluid can lag, overshoot, pile up, or flow away dynamically.

### 6. Relative flow and entrainment

Theory 3.3 can include invariant couplings that make the momentum of one fluid
depend partly on the motion of the other. This is called entrainment.

The default production slice uses zero explicit entrainment coefficients while
retaining chemical coupling. Nonzero `entrainment_linear` or
`entrainment_quadratic` values are available, but they change the
characteristic structure and must be re-audited before being trusted.

In plain language: the model can represent two fluids whose momenta are partly
intertwined, but the current qualified experiments use the cleaner
zero-entrainment case.

### 7. Two mixed massive vector fields

The active sector contains two Proca-like vector fields, called A and B. Proca
fields are similar to electromagnetic fields, except they are massive and
therefore can support localized or finite-range behavior.

The two fields can mix through a mass matrix. The carrier current sources the
B field, and the fields contribute their own energy, momentum, stresses, and
forces.

In plain language: the carrier fluid talks to a pair of massive vector fields,
and those fields can store, transport, and redistribute repair-sector energy.

### 8. Evolved Gauss charges

Vector-field theories contain Gauss constraints. A naive solver might repeatedly
solve those constraints from spatial derivatives during every timestep. That
approach can create dangerous derivative mixing near shocks.

`production33` instead evolves the longitudinal charges as independent
variables. The temporal field components can then be reconstructed locally,
without differentiating a noisy electric divergence.

Each vector field also has its own cleaning variable that propagates and damps
small numerical Gauss violations.

In plain language: the solver carries the constraint information forward in
time instead of repeatedly rebuilding it from potentially rough data.

### 9. Internal drag and relaxation

The two fluids exchange momentum through internal drag. The
`target_relaxation_time` controls how quickly their relative motion decays.

- A short relaxation time strongly locks the carriers to the baryonic matter.
- A long relaxation time allows more persistent counterflow and penetration.

The drag step is implicit and entropy-positive. It can safely handle a timestep
that spans many microscopic relaxation times. It preserves both number
currents and total material energy-momentum while converting relative free
energy into heat and entropy.

In plain language: friction slows the motion of one fluid through the other,
but the lost organized motion becomes heat instead of disappearing.

### 10. Exact vector damping

The vector fields can also contain damped electric modes. The solver applies
this damping with an exact local matrix exponential rather than a fragile
explicit approximation.

During damping, it computes the complete vector stress before and after the
step. The corresponding finite energy and momentum change is deposited into
the physical multifluid state. Only the genuinely irreversible positive loss
is added to thermal entropy.

The correction also preserves both Gauss residuals.

In plain language: when a vector mode damps, its energy and momentum go
somewhere physical and are accounted for exactly.

### 11. Coupled primitive recovery

The evolution equations advance conserved quantities such as total energy,
momentum, and number. To calculate pressure, velocities, stresses, and fluxes,
the solver must recover the underlying physical variables after every stage.

Theory 3.3 recovery solves for both fluids at once. It checks that the recovered
state reproduces:

- baryon and carrier number;
- total energy and momentum;
- carrier canonical momentum;
- entropy and internal energy;
- physically subluminal velocities.

The solver reports failed cells and the maximum recovery residual. Production
campaigns treat recovery failures as a numerical gate, not something to hide.

In plain language: after evolving the bookkeeping variables, the code solves a
nonlinear puzzle to reconstruct the actual densities and velocities and checks
that the answer really fits.

### 12. Constitutive projection

At intermediate Runge--Kutta stages, independently evolved energy and entropy
can drift slightly away from the exact master-function relationship.

The solver projects the redundant entropy representation back onto the
physical material manifold. It does not change either number current, total
energy-momentum, or carrier momentum.

In plain language: the code gently repairs a redundant thermodynamic label
without altering the conserved physical content.

### 13. Causality and hyperbolicity checks

Before a parameter set is trusted, the code can construct and inspect the local
characteristic system. It checks:

- whether all characteristic speeds are real;
- whether there is a complete set of wave modes;
- whether any mode travels outside the metric light cone;
- whether the current-to-momentum map is invertible;
- whether the thermodynamic energy is locally convex;
- whether strong counterflow triggers a two-stream instability.

These checks matter because a simulation can produce finite-looking numbers
even when its underlying differential equations have left a well-posed domain.

In plain language: the solver checks that the chosen matter model still defines
a sensible cause-and-effect evolution before relying on it.

### 14. Constraint-solved initial data

Gravity cannot start from arbitrary matter and geometry. The initial slice has
to satisfy Einstein's constraint equations.

The solver supports:

- isolated conformal transverse-traceless initial data for finite boxes;
- periodic constant-mean-curvature data for compact periodic collisions;
- iterative coupling between the geometry and the full matter/vector stress;
- explicit convergence and residual reports.

In plain language: before starting the clock, the code adjusts the initial
geometry so that it is consistent with the matter and fields placed inside it.

### 15. Finite and periodic boundaries

The solver can use periodic boxes or finite Cartesian domains.

Finite domains include:

- no-inflow/outflow boundaries for ordinary matter;
- outgoing-only carrier fluxes;
- radiative vector-field boundaries;
- Gauss-compatible corrections;
- a linearized incoming gravitational-constraint projection.

In plain language: material and waves can leave a finite box without appearing
at the opposite side, while the code suppresses unphysical incoming modes.

### 16. Exact carrier boundary-flux accounting

When carrier number leaves a finite box, the total carrier number inside the
grid decreases. That decrease should not automatically be called a conservation
error.

The solver now integrates the exact numerical carrier flux through every
physical boundary face using the same Runge--Kutta weights as the evolution. It
compares the measured carrier-number change with the accumulated outward flow:

```text
change inside the box + integrated outward flow = balance residual
```

In the latest spherical run, this residual closes at roughly one part in
`10^16`. That shows the observed carrier decrease is boundary transport rather
than an unexplained source or sink inside the domain.

In plain language: the code can tell the difference between charge leaving
through the door and charge mysteriously vanishing in the room.

### 17. Restart and append

Long 3-D runs can continue from their final checkpoint without repeating the
expensive initial evolution.

The isolated spherical runner validates the checkpoint, summary, trajectory,
grid, parameters, timestep, and extraction radii before restarting. It then:

- preserves the original initial baselines;
- carries entropy and boundary-flux totals forward;
- appends without duplicating the join time;
- stages replacement files before publishing them;
- records every restart time.

In plain language: a long experiment can be extended safely while keeping one
continuous, auditable history.

### 18. Diagnostics and physical observables

The solver records more than just field snapshots. Important diagnostics
include:

- gravitational Hamiltonian and momentum constraints;
- Z4 and Gauss constraint norms;
- baryon and carrier totals;
- entropy production;
- primitive-recovery failures;
- characteristic speeds and convexity margins;
- relative Lorentz factor between the fluids;
- drag and vector-damping energy reports;
- boundary flux and balance closure;
- density and geometry slices.

For 3-D gravity, it also supports:

- Hawking quasi-local mass on nested spheres;
- outgoing and ingoing null expansions;
- spherical apparent-horizon searches;
- opt-in distorted spectral KKT/MOTS branch tracking;
- finite-radius `Psi4` gravitational-wave extraction for quadrupole modes.

In plain language: the code watches the physical result and the health of the
calculation at the same time.

## What happens during one timestep

A simplified timestep looks like this:

1. Recover both fluids' physical densities, velocities, pressure, and entropy
   from the conserved state.
2. Build the complete material and vector stress-energy tensors.
3. Use that stress-energy to evolve the CCZ4 spacetime variables.
4. Compute shock-capable baryon and carrier fluxes.
5. Evolve both mixed vector fields, their longitudinal charges, and their Gauss
   cleaners.
6. Apply the stage-local geometry and constitutive projections.
7. Combine the Runge--Kutta stages into the new conservative state.
8. Apply implicit internal drag between the two fluids.
9. Apply exact vector damping and deposit its finite stress change.
10. Recover the final physical state and update entropy, constraints,
    conservation, and boundary-flux diagnostics.

This sequence is designed so that every exchange has an opposite side: if one
sector loses energy or momentum, another sector receives it.

## The most important controls

| Parameter | Plain-language meaning |
|---|---|
| `target_fraction` | Preferred carrier number relative to baryon number |
| `carrier_K` | Overall carrier energy/stiffness scale |
| `carrier_sound_speed` | How quickly small carrier-pressure disturbances propagate |
| `target_relaxation_time` | How long carrier/baryon counterflow survives |
| `carrier_charge` | Strength and sign with which the carrier current sources the B field |
| `entrainment_linear` | Linear invariant coupling between the two currents |
| `entrainment_quadratic` | Nonlinear relative-flow coupling |
| `maximum_relative_lorentz_factor` | Declared counterflow domain for constitutive checks |
| `mass_A`, `mass_B` | Range/localization scales of the two vector fields |
| `mixing_mass_squared` | Strength of A/B vector-field mixing |
| `gamma_W`, `gamma_B` | Electric damping rates |
| `kappa1`, `kappa2`, `kappa3` | CCZ4 gravitational-constraint damping controls |

Changing these values can change both the physical behavior and the range in
which the equations remain causal. Large changes should be followed by a new
master-function characteristic audit.

## What the solver has demonstrated so far

The saved Theory 3.3 campaigns have shown that:

- fast drag locks the carrier fluid to baryonic matter;
- slow drag allows carrier counterflow to penetrate and persist;
- gravitational branch splitting follows the carrier equation-of-state
  exponent across loading and sound-speed scans;
- the response amplitude is nearly linear in the carrier coefficient `K_D`;
- low carrier stiffness can create an acausal characteristic domain;
- the factorized loading law survives a three-resolution continuum analysis;
- isolated 3-D collisions produce a delayed inner Hawking-mass response after
  the material currents have already separated;
- the fitted growth exponent has fallen from about 3.31 toward 2.38, with a
  latest trailing estimate near 2.34;
- finite-boundary carrier-number loss closes against measured outward flux at
  roundoff.

These are numerical results of the implemented theory. They are not yet
observational predictions or analytic proofs.

## What it does not claim yet

`production33` is a production-reference implementation, not a finished
astrophysical simulation platform.

Current limits include:

- no demonstrated multi-node production scaling;
- no complete moving-box AMR campaign;
- no independent comparison with another numerical-relativity code;
- no long equilibrium neutron-star merger simulation;
- no positive/negative carrier pair creation or charge-changing reactions;
- no independently moving heat fluid;
- no general-topology or campaign-qualified apparent-horizon finder; the new
  distorted KKT/MOTS observer remains an opt-in star-shaped diagnostic;
- no proof of global hyperbolicity outside the audited parameter domain;
- nonzero entrainment domains require separate qualification;
- the current long spherical box is no longer large enough for another clean
  same-domain extension because the outer mass diagnostic has begun to respond.

## Running the basic backend

From the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m causal_repair production33 --help
```

A small coupled smoke run:

```powershell
python -m causal_repair production33 `
  --points 24 --derivatives fd2 --final-time 0.002 `
  --carrier-velocity 0.2 `
  --electric-A 0.02 --electric-B=-0.01 `
  --gamma-W 0.2 --gamma-B 0.1 `
  --output runs/theory33_coupled_smoke.npz
```

The isolated 3-D collision runner has its own options:

```powershell
python experiments/run_theory33_spherical_collision.py --help
```

Run the Theory 3.3 derivation and characteristic audit with:

```powershell
python experiments/audit_theory33_master.py
```

Run the complete test suite with:

```powershell
python -m unittest discover -s tests -v
```

## Where to go next

- [`experiments/THEORY_3_3_MASTER_ACTION.md`](experiments/THEORY_3_3_MASTER_ACTION.md)
  gives the full mathematical specification.
- [`experiments/THEORY33_IMPLEMENTATION_REPORT.md`](experiments/THEORY33_IMPLEMENTATION_REPORT.md)
  describes the derivation gates and implemented numerical closure.
- [`experiments/THEORY33_SPHERICAL_LONG_REPORT.md`](experiments/THEORY33_SPHERICAL_LONG_REPORT.md)
  records the latest 3-D restart, exponent, and boundary-flux campaign.
- [`FEATURE_OVERVIEW.md`](FEATURE_OVERVIEW.md) covers the complete repository,
  including legacy solvers, AMR, backends, dashboards, and prediction tools.

## Bottom line

Theory 3.3 turns the repair sector into a real dynamical constituent rather
than a prescribed target or an energy ledger. Two fluids, two massive vector
fields, and dynamical spacetime evolve together. Internal relaxation creates
entropy without destroying conserved quantities, vector damping deposits its
complete finite change, constraint information is evolved instead of repeatedly
reconstructed, and boundary flow is measured explicitly.

The result is a solver that can ask physically meaningful questions about
locking, counterflow, carrier loading, delayed gravitational response, and
causal admissibility while keeping its numerical bookkeeping visible and
testable.
