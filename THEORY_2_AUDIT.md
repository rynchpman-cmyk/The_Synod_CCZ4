# Theory 2.0 implementation audit

## Implemented without additional assumptions

- Ideal-gas EOS and perfect-fluid 3+1 stress-energy.
- Fluid normalization and positivity validation.
- The advected-scalar equilibrium construction `Q -> Aeq`.
- Stable `asinh(q)/q` evaluation.
- Evolved spatial target, reconstructed `Phi_star`, and orthogonality monitor.
- Density-dependent effective Proca mass and `J_star` source contract.
- Algebraic generalized Gauss solve for `Phi`.
- Bare Proca and quadratic-interaction 3+1 stress tensors.
- Total 3+1 source aggregation.
- Fixed-background driven Theory 2.0 evolution.
- A raw-ADM back-reaction bridge with prescribed fluid primitives.
- Moving-puncture gauge RHS interface.
- Sommerfeld, zero-incoming-characteristic, and fluid-outflow operators.
- Rotating two-lobe material/target data and exact Proca constraint setup.

## Electric-field convention correction

Theory 1 used `E_i = n^mu F_{mu i}` in its canonical decomposition. Theory 2
uses `E_i = F_{i mu} n^mu`, which is its negative. The canonical momentum is
retained, but physical 3+1 stress-energy now converts with

```text
E^i_Theory2 = -pi^i / (Z sqrt(gamma)).
```

This changes the sign of the returned Proca momentum density relative to the
original package. Energy and spatial stress are unchanged.

## Remaining equations needed for production evolution

Theory 2.0 fixes the algebraic interfaces but does not yet completely fix the
evolutionary PDE system in four places.

### 1. Target-to-fluid energy-momentum exchange

The displayed action has no kinetic or Schwinger--Keldysh term for `Astar`, but
`Astar` is assigned a finite-time relaxation equation. Its changing mismatch
stress carries energy and momentum. A production GRHD solver therefore needs
an explicit exchange four-force

```text
nabla_mu T_pf^{mu nu} = Q^nu,
```

or a closed action/open-system prescription that derives it. Setting total
stress conservation alone identifies the required divergence formally but
does not provide the local GRHD source in a numerically usable form without
time derivatives of the other sectors.

### 2. Full projected target transport

`Delta_mu^nu u^a nabla_a Astar_nu` contains connection and fluid-acceleration
terms. The interface section supplies only local relaxation plus advection.
The implementation provides that explicit Cartesian interface. Curved,
accelerating-fluid production runs require the expanded 3+1 equation.

### 3. CCZ4 system

The document selects CCZ4 and its gauge but does not provide the CCZ4 evolution
equations, conformal-variable definitions, damping parameters, or boundary
characteristic fields. The moving-puncture gauge interface is implemented;
the existing transparent raw-ADM backend remains experimental. No CCZ4
equations were invented.

### 4. Constraint-solved two-lobe geometry

The benchmark specifies matter profiles and says to solve the coupled elliptic
Hamiltonian and momentum constraints, but it does not select an elliptic
decomposition, outer-domain map, boundary discretization, or solver. The
constructor solves the Proca constraint exactly and marks the conformally flat
GR data as `gravitational_constraints_solved=False`.

The document also gives no initial `sigma` profile corresponding to its manual
two-lobe `Astar`. The implementation initializes `sigma=0`, so this is explicitly
an off-equilibrium transient target that relaxes toward zero.

## Current validated scope

The fixed-background Theory 2.0 repair/target system is closed and runnable.
The raw-ADM bridge can test metric response with prescribed matter. A claim of
fully conservative, self-gravitating CCZ4+GRHD evolution should wait until the
four missing specifications above are supplied.

