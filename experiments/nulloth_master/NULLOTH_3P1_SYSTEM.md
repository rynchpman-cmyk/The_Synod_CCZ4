# Nulloth Complete 3+1 Production System

Status: canonical discrete-to-continuum map

This document fixes the complete state and operator ordering used by the
production backend. The detailed per-operation witnesses are generated in
`PRODUCTION_TRACE.md` from `production_trace.json`.

## State

The 27 array groups are:

1. conformal spatial metric;
2. conformal trace-free extrinsic curvature;
3. conformal factor;
4. trace of extrinsic curvature;
5. CCZ4 Theta;
6. conformal connection functions;
7. lapse;
8. shift;
9. shift driver;
10. densitized baryon number;
11. total densitized momentum;
12. total densitized energy;
13. densitized entropy;
14. densitized tracer;
15. spatial Proca potential A;
16. densitized electric momentum A;
17. spatial Proca potential B;
18. densitized electric momentum B;
19. longitudinal charge A;
20. longitudinal charge B;
21. Gauss cleaner A;
22. Gauss cleaner B;
23. densitized carrier source charge;
24. densitized carrier canonical momentum;
25. coherence `varphi`;
26. scalar momentum `Pi`;
27. scalar gradient covector `Phi_i`.

Groups 11 and 12 contain material, both Proca fields, and scalar contributions.
They are not sector-local reservoirs.

## Physical right-hand side

At every right-hand-side evaluation the solver:

1. reconstructs the physical spatial geometry;
2. subtracts vector and scalar stress from total conserved stress;
3. recovers both material currents and the thermodynamic state;
4. rejects failed or inadmissible recovery;
5. supplies total stress to CCZ4;
6. evaluates the conservative matter balance law;
7. evaluates carrier number and canonical-momentum evolution;
8. evaluates both Proca systems and cleaners;
9. evaluates the first-order scalar system.

The scalar equations and stress are derived in
`NULLOTH_ACTION_DERIVATION.md`. The CCZ4 and constant-mass mixed-Proca
principal operators remain their validated inherited implementations; Nulloth
changes their physical sources, not their differential definitions.

## Runge–Kutta stage map

The explicit reversible system is advanced with classical RK4. At internal
and final stages, CCZ4 algebraic constraints and the scalar-gradient reduction
constraint are projected. Redundant entropy is reconciled with the recovered
master-function energy manifold, and the signed correction is recorded.

After the reversible RK map, the solver applies implicit carrier drag and
exact vector damping. These maps alone add irreversible heat. A final recovery
and entropy check may reject the complete step.

The ordered map is therefore

```text
reversible RK4
  -> CCZ4 algebraic projection
  -> scalar-gradient projection
  -> master-manifold reconciliation
  -> implicit carrier drag
  -> exact vector damping
  -> final recovery and admissibility gate
```

Projection residuals are numerical errors. They are never activation sources,
spell energy, material heat, or authorization signals.

## Constraint classes

Physical constraints:

- baryon and carrier number conservation;
- the two Proca Gauss laws;
- the action-derived constitutive manifold.

First-order and formulation constraints:

- Hamiltonian and momentum constraints;
- CCZ4 Theta and Z fields;
- CCZ4 algebraic conformal constraints;
- scalar `Phi_i-D_i varphi` reduction constraint;
- Proca cleaner residuals.

All are monitored, but only physical field and material variables enter the
stress and activation equations.

## Checkpoint and diagnostic boundary

Checkpoint compatibility versions belong to serialization, not to Nulloth's
name. Every group above is stored explicitly. A restart witness must compare
the subsequent evolved branch, not merely check that arrays can be loaded.

Diagnostics separately report physical integrals, constraint norms,
irreversible heat, projection corrections, admissibility margins, and recovery
failures. The categories must not be collapsed into a generic stability score.

