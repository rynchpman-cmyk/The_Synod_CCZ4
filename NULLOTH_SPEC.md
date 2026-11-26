# Nulloth: Carrier-Induced Repair-Coherence Theory

Status: **formulation contract and runnable periodic-box reference backend**

Nulloth extends the validated Theory 3.3 two-current, mixed-Proca,
CCZ4 stack by a canonical scalar order parameter. The scalar condenses only
when the physical carrier state makes the symmetric phase dynamically
unstable. No spell flag, phenomenological energy reservoir, prescribed target
direction, or numerical-constraint source is permitted.

The implementation files governed by this contract are:

- `nulloth.py`: invariant material action, scalarization, conjugate momenta,
  material stress, local admissibility, and characteristic screening;
- `nulloth_scalar.py`: canonical scalar stress and first-order 3+1 evolution;
- `production_nulloth.py`: total conservative state, multifield recovery,
  CCZ4 coupling, RK evolution, drag, damping, diagnostics, and checkpoints;
- `tests/test_nulloth.py`: minimum formulation regressions.

`theory33.py` and `production33.py` remain unchanged regression oracles.

---

## 1. Conventions

Use signature

\[
(-,+,+,+),
\]

and

\[
K_{ij}=-\frac12\mathcal L_n\gamma_{ij}.
\]

The spacetime metric evolved by CCZ4 is the Einstein-frame physical metric for
Nulloth. All matter, vector, and scalar stresses entering the Einstein
equation are varied with respect to that metric.

Nulloth does **not** silently implement a Jordan-frame scalar-tensor theory.
A non-minimal term \(F(\varphi)R\) changes the frame transformation of the two
currents and the Proca mass matrix. That extension requires a separate complete
derivation and cannot be approximated by changing only the CCZ4 source.

---

## 2. Fundamental fields

The geometry is

\[
g_{\mu\nu}.
\]

The baryon current is

\[
N^\mu=n u^\mu,
\qquad u_\mu u^\mu=-1,
\qquad n>0.
\]

The carrier current is

\[
C^\mu=d v^\mu,
\qquad v_\mu v^\mu=-1,
\qquad d\ge0.
\]

The mixed-Proca fields are \(A_\mu\) and \(B_\mu\), with the unchanged
constant-mass Theory 3.3 vector action and carrier source

\[
J_D^\mu=g_D C^\mu.
\]

The Nulloth repair-coherence field is a real scalar

\[
\varphi.
\]

The baryon entropy per baryon is \(\sigma\), with entropy current

\[
S^\mu=n\sigma u^\mu.
\]

---

## 3. Conserved currents

The two number currents obey

\[
\nabla_\mu N^\mu=0,
\qquad
\nabla_\mu C^\mu=0.
\]

Therefore

\[
\nabla_\mu J_D^\mu=0.
\]

Nulloth contains no pair creation, charge-changing reaction, carrier
creation, or carrier destruction. Any future reaction extension must include
its reaction network, affinity, entropy production, and modified Proca Gauss
compatibility explicitly.

---

## 4. Current invariants

Define

\[
n^2=-N_\mu N^\mu,
\qquad
d^2=-C_\mu C^\mu,
\qquad
x^2=-N_\mu C^\mu.
\]

The relative Lorentz factor and nonnegative counterflow invariant are

\[
\gamma_D=\frac{x^2}{nd}=-u_\mu v^\mu\ge1,
\]

\[
\mathcal W=x^2-nd=nd(\gamma_D-1)\ge0.
\]

---

## 5. Covariant action

The Nulloth action is

\[
\boxed{
S_N=\int d^4x\sqrt{-g}
\left[
\frac{R}{2\kappa}
+\Lambda_{\rm vec}
+\Lambda_N(n^2,d^2,x^2,\sigma,\varphi)
-g_D B_\mu C^\mu
-\frac12\nabla_\mu\varphi\nabla^\mu\varphi
-V_0(\varphi)
\right]
+S_{\rm pullback}.
}
\]

The pullback or equivalent matter-space variables enforce current
conservation and retain vortical flows.

The source interaction \(-g_DB_\mu C^\mu\) has no separately added Hilbert
stress when the densitized pullback current is held fixed. Its physical effect
appears in canonical momentum and equal-and-opposite field/material exchange.

---

## 6. Nulloth master function

The Theory 3.3 M1 material master function is extended as

\[
\boxed{
\Lambda_N
=\Lambda_{M1}
+\frac12\mathcal C(n,d,\mathcal W)\varphi^2.
}
\]

The seed coupling is

\[
\boxed{
\mathcal C
=g_n\left(\frac{n}{n_\star}\right)^2
+g_d\left(\frac{d}{d_\star}\right)^2
+g_W\frac{\mathcal W}{W_\star}.
}
\]

The required seed restrictions are

\[
g_n\ge0,
\qquad g_d\ge0,
\qquad g_W\ge0,
\]

\[
n_\star>0,
\qquad d_\star>0,
\qquad W_\star>0.
\]

The scalar self-potential is

\[
\boxed{
V_0(\varphi)
=\frac12m_\varphi^2\varphi^2
+\frac14\lambda_\varphi\varphi^4,
}
\]

with

\[
m_\varphi^2\ge0,
\qquad
\lambda_\varphi>0.
\]

The quartic term is mandatory in this seed family because it bounds the
homogeneous scalar energy when the quadratic mode becomes tachyonic.

### 6.1 Dimensional and nondimensional contract

In continuum geometrized units, choosing dimensionless \(\varphi\),

\[
[\Lambda_N]=[V_0]=[T^{\mu\nu}]=L^{-2},
\qquad
[m_\varphi^2]=[\mathcal C]=[\lambda_\varphi]=L^{-2}.
\]

Number currents have

\[
[n]=[d]=L^{-3},
\qquad
[\mathcal W]=L^{-6},
\]

so

\[
[n_\star]=[d_\star]=L^{-3},
\qquad
[W_\star]=L^{-6},
\qquad
[g_n]=[g_d]=[g_W]=L^{-2}.
\]

The backend evolves nondimensional variables after reference length, density,
energy, and scalar scales are chosen. Numeric scalar couplings,
`scalar_mass**2`, and `scalar_quartic` are therefore expressed in the same
nondimensional inverse-length-squared unit. No physical energy or ship-scale
cost follows until those reference scales are separately declared.

---

## 7. Emergent activation

At a frozen material state the effective scalar mass is

\[
\boxed{
m_{\rm eff}^2=m_\varphi^2-\mathcal C.
}
\]

The symmetric branch \(\varphi=0\) is linearly stable when

\[
m_{\rm eff}^2>0.
\]

It is linearly unstable when

\[
m_{\rm eff}^2<0.
\]

For a homogeneous frozen material state, nonzero equilibria satisfy

\[
\varphi_{\rm eq}^2
=-\frac{m_{\rm eff}^2}{\lambda_\varphi}.
\]

This is the Nulloth activation mechanism. A configuration does not become
magical because it is labeled a spell. It becomes coherent only if the coupled
field equations dynamically enter a nonzero admissible branch.

---

## 8. Material conjugate coefficients

Write the material energy as

\[
\rho_N
=\rho_0
+a_1\mathcal W
+\frac{a_2}{2M_D^4}\mathcal W^2
-\frac12\mathcal C\varphi^2.
\]

The relative-flow derivative is

\[
Q
=\frac{\partial\rho_N}{\partial\mathcal W}
=a_1+\frac{a_2\mathcal W}{M_D^4}
-\frac12\frac{g_W}{W_\star}\varphi^2.
\]

The explicit density derivatives receive

\[
\left.\frac{\partial\rho_N}{\partial n}\right|_{d,\mathcal W,\sigma}
=\rho_{0,n}-g_n\frac{n}{n_\star^2}\varphi^2,
\]

\[
\left.\frac{\partial\rho_N}{\partial d}\right|_{n,\mathcal W,\sigma}
=\rho_{0,d}-g_d\frac{d}{d_\star^2}\varphi^2.
\]

The Carter coefficients are

\[
\mathcal B_N=\frac{\rho_{N,n}-Qd}{n},
\qquad
\mathcal B_D=\frac{\rho_{N,d}-Qn}{d},
\qquad
\mathcal A=Q.
\]

The conjugate momenta are

\[
\mu_\nu=\mathcal B_N N_\nu+\mathcal A C_\nu,
\]

\[
\chi_\nu=\mathcal B_D C_\nu+\mathcal A N_\nu.
\]

The generalized pressure is

\[
\Psi
=\Lambda_N
+\mathcal B_Nn^2
+\mathcal B_Dd^2
+2\mathcal A x^2.
\]

---

## 9. Material stress

The complete material stress is

\[
\boxed{
T_{\rm mf}^{\mu\nu}
=\Psi g^{\mu\nu}
+\mathcal B_NN^\mu N^\nu
+\mathcal B_DC^\mu C^\nu
+\mathcal A(N^\mu C^\nu+C^\mu N^\nu).
}
\]

No additional scalarization interaction stress is added on top of this tensor.
The \(\mathcal C\varphi^2/2\) term is already included through the master
function and its invariant derivatives.

---

## 10. Scalar equation and stress

The scalar equation is

\[
\boxed{
\Box\varphi
=V_{0,\varphi}-\Lambda_{N,\varphi}
=(m_\varphi^2-\mathcal C)\varphi
+\lambda_\varphi\varphi^3.
}
\]

The canonical scalar stress contains the kinetic term and \(V_0\), but not a
second copy of the material coupling:

\[
\boxed{
T_\varphi^{\mu\nu}
=\nabla^\mu\varphi\nabla^\nu\varphi
-g^{\mu\nu}
\left[
\frac12\nabla_\alpha\varphi\nabla^\alpha\varphi+V_0
\right].
}
\]

The coupling contribution to total stress is already contained in
\(T_{\rm mf}^{\mu\nu}\).

---

## 11. Einstein equation and CCZ4 interface

The continuum metric equation is

\[
\boxed{
G_{\mu\nu}+\Lambda_c g_{\mu\nu}
=\kappa
\left(
T_{\mu\nu}^{\rm mf}
+T_{\mu\nu}^{\rm vec}
+T_{\mu\nu}^{\varphi}
\right).
}
\]

The CCZ4 implementation receives only the total Eulerian projections

\[
\rho=n_\mu n_\nu T^{\mu\nu},
\]

\[
S_i=-\gamma_{i\mu}n_\nu T^{\mu\nu},
\]

\[
S_{ij}=\gamma_{i\mu}\gamma_{j\nu}T^{\mu\nu}.
\]

The physical Nulloth scalar is never identified with the CCZ4 fields
\(\Theta\) or \(Z_i\). Hamiltonian, momentum, \(\Theta\), \(Z_i\), Proca
Gauss, and scalar-gradient residuals are numerical diagnostics only.

---

## 12. 3+1 scalar variables

Define

\[
\Pi=-n^\mu\nabla_\mu\varphi,
\qquad
\Phi_i=D_i\varphi.
\]

Then

\[
\partial_t\varphi
=\beta^i\Phi_i-\alpha\Pi,
\]

\[
\partial_t\Pi
=\beta^iD_i\Pi
-\alpha D_i\Phi^i
+\alpha K\Pi
+\Phi^iD_i\alpha
+\alpha\left(V_{0,\varphi}-\Lambda_{N,\varphi}\right),
\]

\[
\partial_t\Phi_i
=\beta^jD_j\Phi_i
+\Phi_jD_i\beta^j
-\alpha D_i\Pi
-\Pi D_i\alpha.
\]

The first-order reduction constraint is

\[
\mathcal C_i^\varphi=\Phi_i-D_i\varphi=0.
\]

The reference RK implementation projects \(\Phi_i\) back to
\(D_i\varphi\) at internal and final stages. Any future continuous cleaning
term must be convergence-tested and must not be interpreted as physical
repair energy.

The scalar Eulerian stress is

\[
\rho_\varphi
=\frac12(\Pi^2+\Phi_i\Phi^i)+V_0,
\]

\[
S_i^\varphi=\Pi\Phi_i,
\]

\[
S_{ij}^\varphi
=\Phi_i\Phi_j
+\gamma_{ij}
\left[
\frac12(\Pi^2-\Phi_k\Phi^k)-V_0
\right].
\]

---

## 13. Total evolved state

The production state contains

\[
U_N=
\left(
U_{\rm CCZ4},
D_N,
S_i^{\rm total},
E^{\rm total},
D_N\sigma,
D_N\,\text{tracer},
U_{\rm Proca},
D_D,
P_{D i},
\varphi,
\Pi,
\Phi_i
\right).
\]

`target_charge` retains the Theory 3.3 meaning

\[
\sqrt\gamma\,g_DdW_D.
\]

`target_current` retains the Theory 3.3 meaning of densitized carrier
canonical momentum.

The evolved total energy and momentum include material, both Proca fields, and
the Nulloth scalar. Sector exchange is therefore internal to one conservative
balance law.

---

## 14. Primitive recovery

At fixed geometry, Proca state, and scalar state, compute

\[
E_{\rm mf}=E_{\rm total}-E_{\rm vec}-E_\varphi,
\]

\[
S_i^{\rm mf}=S_i^{\rm total}-S_i^{\rm vec}-S_i^\varphi.
\]

The coupled primitive solve recovers

\[
(n,\sigma,u^i,d,v^i)
\]

from

1. densitized baryon number;
2. densitized carrier number;
3. material energy and momentum;
4. three carrier canonical momenta;
5. transported entropy or the energy-manifold entropy projection;
6. the Nulloth master-function relations at the local value of \(\varphi\).

No residual energy variable may absorb a failed solve.

---

## 15. Reversible and irreversible exchange

The scalar/material exchange implied by the action is internal. At continuum
level,

\[
\nabla_\mu
\left(T_{\rm mf}^{\mu\nu}+T_\varphi^{\mu\nu}\right)
\]

contains equal-and-opposite terms proportional to

\[
\Lambda_{N,\varphi}\nabla^\nu\varphi.
\]

The mixed-Proca Lorentz exchange and exact vector damping deposit retain the
Theory 3.3 sign and accounting rules.

Carrier drag remains

\[
r_D^\mu=\mathcal R_D(u^\mu-\gamma_Dv^\mu),
\qquad
\mathcal R_D\ge0,
\]

with heating

\[
H_R=\mathcal R_D(\gamma_D^2-1)\ge0.
\]

The entropy inequality is

\[
\boxed{
\nabla_\mu S^\mu
=\frac{H_R+H_{\rm vec}}{T}\ge0.
}
\]

Canonical scalar evolution and reversible scalar/material exchange produce no
entropy by themselves.

---

## 16. Admissible state domain

Every active cell must satisfy

\[
n>0,
\qquad d\ge0,
\qquad \gamma_D\ge1,
\qquad T>0,
\]

\[
\mathcal B_N>0,
\qquad
\mathcal B_D>0,
\qquad
\mathcal B_N\mathcal B_D-\mathcal A^2>0,
\]

and the thermodynamic Hessian must have the required positive eigenvalues.

The run domain also declares

\[
\gamma_D\le\gamma_{D,\max},
\qquad
|\varphi|\le\varphi_{\max},
\qquad
m_{\rm eff}^2\ge m_{\rm eff,min}^2.
\]

The effective-mass lower bound is a declared numerical campaign boundary, not
a fundamental prohibition on tachyonic condensation. It prevents an
unresolved growth time from being hidden inside a nominally stable run.

Crossing an admissibility wall rejects the state or step. It is not converted
into a dramatic in-world effect.

---

## 17. Characteristic and hyperbolicity gate

The canonical scalar principal block has metric-null characteristics. The
material coefficients depend on \(\varphi\), and derivatives of the conjugate
momenta produce scalar/material principal coupling.

Therefore a qualified parameter domain must build the **complete coupled
symbol**, not merely append \(\pm1\) to the Theory 3.3 speeds.

For every state and propagation direction in the declared run domain:

1. construct the exact scalar-multifluid-Proca principal symbol;
2. verify all characteristic roots are real;
3. verify a complete eigenbasis or positive symmetrizer;
4. verify bounded characteristic condition number;
5. verify all physical modes lie within or on the metric null cone;
6. scan finite counterflow for two-stream and coincident-mode failure;
7. locate and record the first failed state rather than hiding it with floors.

`NullothMasterFunction.characteristic_screen` is only a necessary frozen-block
screen. It is not the final coupled qualification audit.

---

## 18. Discrete evolution order

One production step is

1. recover all physical primitives;
2. evaluate material, vector, and scalar stresses;
3. evolve CCZ4 from the total stress;
4. evolve total conservative energy-momentum;
5. evolve baryon and carrier currents;
6. evolve carrier canonical momentum path-conservatively;
7. evolve mixed-Proca fields and cleaners;
8. evolve \((\varphi,\Pi,\Phi_i)\);
9. project CCZ4 algebraic constraints;
10. project \(\Phi_i=D_i\varphi\);
11. reconcile redundant entropy with the constitutive manifold;
12. apply implicit entropy-positive carrier drag;
13. apply exact vector damping and deposit only its positive electric loss;
14. recover again and reject any inadmissible state.

The scalar-gradient and constitutive projections are numerical representation
operations. Their corrections must converge away with resolution and must be
reported separately from physical entropy production.

---

## 19. Checkpoint contract

The checkpoint format is

```text
causal_repair.production.nulloth.v0.1
```

It stores explicit names for

- all CCZ4 fields;
- total energy and momentum;
- entropy and tracer;
- complete mixed-Proca state;
- carrier source charge;
- carrier canonical momentum;
- `nulloth_varphi`;
- `nulloth_scalar_momentum`;
- `nulloth_scalar_gradient`;
- time;
- optional serialized Nulloth parameters.

A restart must reproduce the same diagnostic stream to the expected numerical
tolerance.

---

## 20. Required diagnostics

At minimum record

- Hamiltonian, momentum, \(\Theta\), and \(Z_i\) norms;
- both Proca Gauss norms;
- scalar first-order constraint norms;
- combined energy;
- baryon number;
- carrier number and charge;
- carrier-number balance residual;
- entropy integral and per-step entropy change;
- drag and vector-damping heat;
- projection entropy correction;
- maximum \(|\varphi|\);
- minimum scalar energy density;
- minimum \(m_{\rm eff}^2\);
- maximum \(\mathcal C\);
- maximum relative Lorentz factor;
- minimum Legendre eigenvalue;
- minimum thermodynamic eigenvalue;
- recovery residual and failure count.

---

## 21. Nulloth qualification tests

The backend must not be called qualified until all applicable tests pass.

### 21.1 Constitutive regressions

- At \(\varphi=0\), all Theory 3.3 master quantities agree exactly.
- Finite differentiation of \(\Lambda_N\) with respect to \(\varphi\) agrees
  with \(\mathcal C\varphi\).
- Finite metric variation at fixed densitized currents agrees with the written
  material stress.
- Current-to-momentum inversion round-trips throughout the declared domain.

### 21.2 Homogeneous phase tests

- Below threshold, small \(\varphi\) perturbations oscillate without secular
  growth.
- Above threshold, the measured linear growth rate converges to the continuum
  prediction.
- The nonlinear homogeneous equilibrium approaches
  \(|\varphi|=\sqrt{-m_{\rm eff}^2/\lambda_\varphi}\).
- Total energy is conserved in reversible runs.

### 21.3 Coupled numerical tests

- Smooth scalar-material waves converge at design order.
- Scalar/material energy exchange is equal and opposite.
- Carrier and baryon numbers remain conservative.
- Drag and vector damping remain entropy-positive.
- Scalar-gradient projection corrections converge away.
- Checkpoint/restart reproduces the uninterrupted branch.
- Resolution changes do not move the physical activation threshold beyond the
  measured convergence error.

### 21.4 Characteristic tests

- Full coupled symbol scan over density, temperature, scalar amplitude,
  carrier fraction, propagation direction, and counterflow.
- Explicit coincident-mode conditioning tests near scalar and material speed
  crossings.
- Shock tests near but inside the first loss-of-hyperbolicity boundary.

---

## 22. Explicit limitations

Nulloth is a covariant classical effective field theory and periodic-box
production reference. It does not yet provide

- a derived microscopic particle model for the carrier or scalar;
- quantum tunneling, fluctuations, or vacuum decay;
- pair creation or charge conversion;
- finite outer boundary conditions for the complete coupled symbol;
- AMR transfer qualification for the scalar state;
- a non-minimal Jordan-frame gravity implementation;
- a proof of global hyperbolicity;
- stable localized solitons by assumption;
- teleportation, retrocausality, transmutation, or healing by implication.

Those capabilities exist only if separately specified equations admit them.

---

## 23. Locked Nulloth result

Nulloth is

\[
\boxed{
\text{CCZ4 Einstein gravity}
+\text{Theory 3.3 two-current matter}
+\text{mixed Proca fields}
+\text{carrier-induced scalarization}.
}
\]

Magic is identified with dynamically generated, nonzero repair coherence
\(\varphi\) and its coupled material, vector, and gravitational effects.

The solver determines whether a proposed configuration exists, remains
admissible, is stable, and performs useful work. The worldbuilding does not
overrule the PDEs.
