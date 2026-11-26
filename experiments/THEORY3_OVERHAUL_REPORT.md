# Theory 3.2 solver overhaul

## Outcome

The repository now contains a separate Theory 3.2 production path. The legacy
Theory 2 solver and its prediction artifacts remain unchanged.

Primary modules:

- `src/causal_repair/theory3.py`: mixed-Proca formulation, stresses, Gauss
  charges, Lorentz exchange, and exact damping.
- `src/causal_repair/production3.py`: CCZ4 coupling, entropy-based material
  recovery, target-current evolution, conservative fluxes, checkpointing, and
  diagnostics.
- `tests/test_theory3.py`: formulation and end-to-end invariant tests.
- `experiments/audit_theory32_target_eigenspace.py`: independent principal-
  symbol audit.

The command-line entry point is `production3`; the old `production` command
continues to select Theory 2.

## Evolved state

The new state contains:

- the complete CCZ4 geometry state;
- densitized baryon number, total momentum, total energy, thermal entropy, and
  passive tracer;
- spatial potentials and canonical electric densities for both A and B;
- independent longitudinal charges `longitudinal_A` and `longitudinal_B`;
- one Gauss cleaner for each vector;
- a conserved densitized target charge and a densitized relative target
  current.

The temporal potentials are never evolved and are never reconstructed from an
electric divergence. They are obtained from the local inverse of the constant
2x2 Gauss matrix.

## Conservative matter/vector coupling

The energy and momentum variables belong to the combined material-plus-vector
system. Their fluxes use the complete material, A, B, and mass-mixing stress.
After every stage, material primitives are recovered by subtracting the
instantaneous complete vector stress.

This has two useful consequences:

1. The B equation can contain the physical target source while matter receives
   the equal-and-opposite exchange without a separately discretized force
   source.
2. Any finite vector change during damping is deposited exactly because total
   conserved energy and momentum are held fixed while the vector stress is
   recomputed.

The explicit diagnostic exchange uses
`Q_drive^mu = +H^mu_nu J_D^nu`. The force-direction test fixes the sign
operationally: positive stationary charge in a positive B electric field
receives positive force.

## Reservoir and thermal closure

Thermal entropy per baryon determines the ideal-gas pressure. Given total
material energy and momentum, recovery then solves for velocity and pressure
and obtains the pressureless reservoir energy as the residual required by
energy conservation.

The runtime admissibility conditions are:

- positive baryon density;
- positive total material enthalpy;
- successful entropy/pressure recovery;
- `target_speed - sound_speed >= sound_speed_margin` in every active cell.

Reservoir energy may be signed, as required by the Theory 3.2 closure, but the
total enthalpy may not cross zero.

## Target-current characteristics

The old conserved-charge plus advected-current Jordan block is gone. The new
longitudinal target flux is built so its 2x2 Jacobian has the exact coordinate
relativistic speeds. In a flat unit-lapse frame these reduce to
`(v_n +/- c_D) / (1 +/- v_n*c_D)`.

The charge flux remains the physical target-current flux. The second flux row
is determined from the required trace and determinant of the characteristic
matrix. Transverse target currents are material modes with relaxation toward
the configured algebraic equilibrium direction.

The parameter constructor rejects a target speed that does not clear the
ideal-gas sound-speed ceiling, and primitive recovery enforces the local gap.

## Gauss evolution

The two evolved longitudinal charges reduce the constraints to
`C_A = div(E_A) + Q_A/sqrt(gamma)` and
`C_B = div(E_B) + Q_B/sqrt(gamma)`.

Their conservative fluxes are generated from the two divergence-free
effective mass currents. The electric and longitudinal equations use the same
grid derivative family, preserving satisfied periodic constraints to
roundoff. Independent damped-wave cleaners handle deliberately inconsistent
data.

## Finite damping

For every spatial component, the A/B fluid-frame electric pair is advanced by
the exact exponential of the constant 2x2 damping matrix. The implementation:

1. transforms Eulerian electric fields into the frozen fluid frame;
2. applies the exact exponential using the real eigendecomposition;
3. transforms back while leaving the slice magnetic field fixed;
4. changes each longitudinal charge by minus the discrete divergence of its
   electric increment;
5. reconstructs both temporal potentials and the complete vector stress;
6. leaves combined conserved energy and momentum unchanged;
7. deposits the exact positive electric quadratic loss into thermal entropy;
8. lets the reservoir absorb only the remaining reversible longitudinal,
   mass, and mixing contribution.

For nonuniform data, the tests explicitly verify that the full vector-energy
change differs from the heat while the complete matter-plus-vector stress
change still cancels cell by cell.

## Verification

The dedicated suite verifies:

- positivity and invertibility of the mixed Gauss matrix;
- derivative-free reconstruction of both temporal potentials;
- the corrected Lorentz-force direction;
- semidiscrete preservation of both Gauss constraints;
- exact relativistic target-current characteristic speeds;
- initial fluid and reservoir recovery;
- exact combined energy/momentum preservation through damping;
- nonnegative finite thermal entropy deposition;
- nonuniform full-stress damping deposition;
- periodic conservation of baryon, entropy, total energy/momentum, and target
  charge at the semidiscrete level;
- nontrivial CCZ4+GRHD+mixed-vector evolution;
- exact Theory 3 checkpoint round trips;
- exact comoving target-current reconstruction and orthogonality;
- projected connection, acceleration, and extrinsic-curvature sources;
- zero-incoming target characteristics at finite boundaries;
- Gauss-compatible radiation for both mixed Proca fields;
- conservative, Gauss-projected AMR transfers and grouped refluxing;
- the accelerator-ready exact damping kernel and complete MPI halo contract.

All 86 tests pass. The verified periodic smoke artifact is
`runs/theory32_overhaul_smoke.npz`; the finite-boundary CLI smoke artifact is
`runs/theory32_finite_boundary_smoke.npz`.

## Running it

From the repository root:

```powershell
$env:PYTHONPATH='src'
python -m causal_repair production3 `
  --points 64 --derivatives fd2 --final-time 0.01 `
  --gamma-W 0.2 --gamma-B 0.1 `
  --electric-A 0.01 --electric-B=-0.005 `
  --output runs/theory3_example.npz
```

Use `python -m causal_repair production3 --help` for the complete parameter
surface.

Finite-grid evolution is enabled with `--isolated`:

```powershell
python -m causal_repair production3 --isolated `
  --points 64 --length 8 --derivatives fd2 --final-time 0.01
```

## Engineering hardening completed

The original engineering boundary list has been addressed at the reference-
solver tier:

- The exact decomposition `J_D=q_D u+R_D` is now used by the B source. The
  lower-order projected current source contains the spatial Christoffel
  transport, lapse acceleration, extrinsic-curvature rotation, and the fluid-
  acceleration projection from pressure plus the repair force. The audited
  relativistic principal block is unchanged.
- Finite Cartesian grids use zero-incoming target characteristics, outgoing
  material fluxes, radiative A/B potential and electric data, two radiative
  cleaners, and a compatible longitudinal-charge correction that preserves
  both semidiscrete Gauss laws.
- Theory 3 AMR transfers conservatively prolong/restrict both canonical
  electric densities, both longitudinal charges, target charge, and target
  current. Both Gauss laws are reprojected after a cell-centered transfer.
  Grouped Berger--Colella registers reflux all six conserved field groups.
- The local damping hot path uses a closed-form 2x2 exponential containing
  only NumPy/CuPy-compatible kernels. Vector stress assembly is vectorized.
  A single MPI halo contract covers every enlarged Theory 3 field, including
  correct scalar/vector component-axis metadata.

## Remaining qualification and theory boundaries

This is a production-reference formulation, not yet a multi-node numerical-
relativity application. Multi-rank MPI and CuPy hardware were not available in
this test environment, so those paths have executable contracts and local
kernels but not hardware qualification or scaling results. Likewise, the AMR
synchronization layer is verified independently; a long, fully adaptive
strong-field campaign remains a validation project rather than another local
formulation change.

The only unresolved closure issue is theory-level: the reservoir remains the
explicit phenomenological balance law. A microscopic multifluid master action
would replace that closure and determine its positive free energy. It cannot
be manufactured as an engineering patch without changing the theory.

The replacement is specified in
[`THEORY_3_3_MASTER_ACTION.md`](THEORY_3_3_MASTER_ACTION.md). It introduces a
conserved charged carrier current, variational material stress, and
entropy-positive internal drag. Its separate `production33` reference backend
and passing default-domain gates are documented in
[`THEORY33_IMPLEMENTATION_REPORT.md`](THEORY33_IMPLEMENTATION_REPORT.md).
The `production3` backend remains the reproducible Theory 3.2 implementation.
