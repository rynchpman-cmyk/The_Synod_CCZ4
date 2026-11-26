# Theory 3.3: two-current master-action closure

Status: formulation specification; the default-domain reference
implementation and gate results are recorded in
[`THEORY33_IMPLEMENTATION_REPORT.md`](THEORY33_IMPLEMENTATION_REPORT.md).

Theory 3.3 replaces the pressureless phenomenological reservoir of Theory 3.2
with a covariant two-current material action. It retains the mixed-Proca
sector, the corrected Lorentz-force sign, evolved longitudinal Proca charges,
and exact finite-step vector damping deposit established in Theory 3.2.

The specification is deliberately stricter than an equation sketch. It fixes
the fundamental variables, variational stress, conservative exchange,
irreversible closure, admissible state domain, and tests that must pass before
the production backend may be relabeled Theory 3.3.

## 1. Locked physical interpretation

The material sector contains two conserved currents:

1. the baryon-number current

   \[
   N^\mu=n_b u^\mu,
   \qquad u_\mu u^\mu=-1,
   \qquad n_b>0,
   \]

2. a target-carrier number current

   \[
   C_D^\mu=n_D v_D^\mu,
   \qquad v_{D\mu}v_D^\mu=-1,
   \qquad n_D\ge0.
   \]

The source of the target-carrier Proca field is

\[
\boxed{J_D^\mu=g_D C_D^\mu,}
\]

where the microscopic target charge per carrier, \(g_D\), is a constant. Its
sign fixes the sign of the target charge. A single carrier species therefore
cannot change charge sign continuously. A theory with positive and negative
target carriers requires two carrier currents and lies beyond minimal Theory
3.3.

Both currents are conserved:

\[
\boxed{\nabla_\mu N^\mu=0,\qquad \nabla_\mu C_D^\mu=0.}
\]

Consequently \(\nabla_\mu J_D^\mu=0\), as required by the mixed-Proca Gauss
system. Target pair creation, destruction, and chemical conversion are not
part of Theory 3.3.

The baryon entropy per baryon is \(\sigma\). The thermal entropy current is

\[
S^\mu=n_b\sigma u^\mu.
\]

It is conserved in the reversible sector and has nonnegative divergence in
the dissipative sector.

Use metric signature \((- + + +)\).

## 2. Relation to the Theory 3.2 target variables

Define the baryon-frame projector

\[
\Delta^\mu{}_{\nu}=\delta^\mu{}_{\nu}+u^\mu u_\nu
\]

and the relative Lorentz factor

\[
\gamma_D=-u_\mu v_D^\mu\ge1.
\]

The source current decomposes exactly as

\[
J_D^\mu=q_Du^\mu+R_D^\mu,
\qquad u_\mu R_D^\mu=0,
\]

with

\[
\boxed{
q_D=g_Dn_D\gamma_D,
\qquad
R_D^\mu=g_Dn_D\left(v_D^\mu-\gamma_Du^\mu\right).
}
\]

The new microscopic closure therefore imposes

\[
q_D^2-R_{D\mu}R_D^\mu=g_D^2n_D^2\ge0.
\]

This is a new Theory 3.3 admissibility condition. Theory 3.2 allowed an
arbitrary conserved four-current, including spacelike states. Minimal Theory
3.3 describes a material carrier and is restricted to the timelike or vacuum
branch. At \(n_D=0\), both \(q_D\) and \(R_D^\mu\) must vanish together.

The old parameters `target_strength`, `target_direction`, and the prescribed
algebraic \(R_{\rm eq}^\mu\) are not fundamental Theory 3.3 data. Matter drives
the target current through the master-function chemical affinity and its
gradients. A material director can be added later as an anisotropic order
parameter, but it is not part of the minimal two-current theory.

## 3. Invariant master action

Introduce the three current invariants

\[
n^2=-N_\mu N^\mu,
\qquad
d^2=-C_{D\mu}C_D^\mu,
\qquad
x^2=-N_\mu C_D^\mu.
\]

On the physical branch,

\[
n= n_b,
\qquad
d=n_D,
\qquad
x^2=n_bn_D\gamma_D.
\]

The complete action is

\[
\boxed{
S_{3.3}=\int d^4x\sqrt{-g}\left[
\frac{\mathcal R}{2\kappa}
+\Lambda_{\rm vec}
+\Lambda_{\rm mf}(n^2,d^2,x^2,\sigma)
-g_DB_\mu C_D^\mu
\right]+S_{\rm pullback}.
}
\]

Here \(S_{\rm pullback}\) denotes the convective or matter-space constraints
that make the two current variations consistent with their conservation
laws. The implementation may use an equivalent complete Clebsch
parameterization, but it must retain enough labels to represent vortical
flows. A single phase is only the irrotational special case.

The conservative vector Lagrangian is unchanged:

\[
\begin{aligned}
\Lambda_{\rm vec}={}&
-\frac{Z_A}{4}F_{\mu\nu}F^{\mu\nu}
-\frac{Z_B}{4}H_{\mu\nu}H^{\mu\nu}
-\frac{Z_Am_A^2}{2}A_\mu A^\mu\\
&-\frac{Z_Bm_B^2}{2}B_\mu B^\mu
-\frac{\mu^2}{2}(A_\mu-B_\mu)(A^\mu-B^\mu).
\end{aligned}
\]

Use the Theory 3.2 conditions

\[
Z_A>0,\quad Z_B>0,\quad m_A^2>0,\quad m_B^2>0,\quad \mu^2\ge0.
\]

The pullback current density \(\sqrt{-g}\,C_D^\mu\) is the metric-independent
variational object. Therefore

\[
\sqrt{-g}\,g_DB_\mu C_D^\mu
=g_DB_\mu\left(\sqrt{-g}\,C_D^\mu\right)
\]

has no independent Hilbert interaction stress. Its physical effect appears
through the target canonical momentum and the equal-and-opposite vector and
material force. This removes the metric-variation ambiguity that forced the
Theory 3.2 reservoir closure.

## 4. Conjugate momenta and material stress

Define

\[
\mu_\nu=\frac{\partial\Lambda_{\rm mf}}{\partial N^\nu},
\qquad
\chi_\nu=\frac{\partial\Lambda_{\rm mf}}{\partial C_D^\nu}.
\]

They have the two-current form

\[
\boxed{
\mu_\nu=\mathcal B_N N_\nu+\mathcal A C_{D\nu},
\qquad
\chi_\nu=\mathcal B_D C_{D\nu}+\mathcal A N_\nu,
}
\]

where

\[
\mathcal B_N=-2\frac{\partial\Lambda_{\rm mf}}{\partial n^2},
\qquad
\mathcal B_D=-2\frac{\partial\Lambda_{\rm mf}}{\partial d^2},
\qquad
\mathcal A=-\frac{\partial\Lambda_{\rm mf}}{\partial x^2}.
\]

The cross coefficient \(\mathcal A\) is the entrainment. In general, a
constituent momentum is not parallel to its transport current.

Define the generalized pressure

\[
\Psi=\Lambda_{\rm mf}-N^\rho\mu_\rho-C_D^\rho\chi_\rho.
\]

The Hilbert stress of the complete material system is

\[
\boxed{
T_{\rm mf}^{\mu}{}_{\nu}
=\Psi\delta^\mu{}_{\nu}
+N^\mu\mu_\nu
+C_D^\mu\chi_\nu.
}
\]

It is symmetric when \(\Lambda_{\rm mf}\) depends only on the scalar
invariants above. This one tensor replaces

\[
T_{\rm th}^{\mu\nu}+e_Du^\mu u^\nu.
\]

There is no separately evolved reservoir energy and no separately invented
target stress.

The implementation must obtain the Eulerian projections by direct projection
of this tensor:

\[
\rho_{\rm mf}=n_\mu n_\nu T_{\rm mf}^{\mu\nu},
\qquad
S_{{\rm mf},i}=-\gamma_{i\mu}n_\nu T_{\rm mf}^{\mu\nu},
\qquad
S_{{\rm mf},ij}=\gamma_{i\mu}\gamma_{j\nu}T_{\rm mf}^{\mu\nu}.
\]

No perfect-fluid decomposition may be assumed away from the co-moving limit.
Entrainment generally produces heat-flux-like momentum and anisotropic stress
in the baryon frame.

## 5. Minimal master-function family M1

Theory 3.3 fixes the allowed invariant dependence but not a unique
microphysical equation of state. The first implementation and all reference
tests use the following explicit seed family.

Define the nonnegative relative-flow invariant

\[
\mathcal W=x^2-n d= n_bn_D(\gamma_D-1)\ge0.
\]

Use

\[
\boxed{
\Lambda_{\rm M1}
=-\rho_0(n,d,\sigma)
-a_1(n,d,\sigma)\mathcal W
-\frac{a_2(n,d,\sigma)}{2M_D^4}\mathcal W^2.
}
\]

The co-moving energy density is

\[
\boxed{
\rho_0
=\rho_{\rm th}(n,\sigma)
+K_Dd^{1+c_0^2}
+\frac{1}{2\chi_D}
\left[d-y_\star(n,\sigma)n\right]^2.
}
\]

Interpretation:

- \(K_Dd^{1+c_0^2}\) is a simple barotropic carrier energy. When uncoupled,
  its pressure is \(p_D=c_0^2\rho_D\), so its rest-frame sound speed is
  exactly \(c_0\).
- \(y_\star(n,\sigma)\) is a matter-dependent chemical anchoring fraction.
  Compression or heating changes the local carrier affinity and drives
  redistribution through \(C_D^\mu\). Because the barotropic term and global
  carrier-number constraint also contribute, \(d=y_\star n\) is not asserted
  to be the exact equilibrium density.
- \(\chi_D>0\) is a chemical susceptibility.
- \(a_1\) and \(a_2\) control entrainment and nonlinear relative-flow energy.

The seed parameter restrictions are

\[
K_D>0,
\qquad
0<c_0<1,
\qquad
\chi_D>0,
\qquad
a_1\ge0,
\qquad
a_2\ge0,
\qquad
y_\star\ge0.
\]

These inequalities are necessary design restrictions, not a proof of global
stability or causality. The complete Hessian and characteristic tests in
section 10 remain mandatory for every parameter domain.

If the conserved total carrier number is incompatible with the pointwise
minimum \(d=y_\star n\), equilibrium is the constrained minimum at fixed
\(\int C_D^\mu d\Sigma_\mu\). Theory 3.3 never relaxes the total target charge
by silently changing \(q_D\).

## 6. Reversible field equations and force balance

Let

\[
\omega^{(N)}_{\mu\nu}=2\nabla_{[\mu}\mu_{\nu]},
\qquad
\omega^{(D)}_{\mu\nu}=2\nabla_{[\mu}\chi_{\nu]}.
\]

Let the thermal coefficient be

\[
\Theta=-\frac{\partial\Lambda_{\rm mf}}{\partial\sigma}.
\]

On the co-moving equilibrium branch, \(\Theta=n_bT\). Define the constituent
force densities

\[
f^{(N)}_\nu=N^\mu\omega^{(N)}_{\mu\nu}
-\Theta\nabla_\nu\sigma,
\qquad
f^{(D)}_\nu=C_D^\mu\omega^{(D)}_{\mu\nu}.
\]

The conservative equations are fixed operationally by

\[
\boxed{
f^{(N)}_\nu+f^{(D)}_\nu
=H_{\nu\lambda}J_D^\lambda.
}
\]

In the absence of internal resistance, the target constituent receives the
\(B\)-field force and the baryon constituent receives only the compensating
entrainment/pressure response:

\[
f^{(D)}_\nu=H_{\nu\lambda}J_D^\lambda,
\qquad
f^{(N)}_\nu=0.
\]

Equivalent canonical-momentum equations obtained by combining \(\chi_\mu\)
with \(B_\mu\) are permitted only if their expansion reproduces these signs.
The operational sign lock remains

\[
\boxed{
\nabla_\mu T_{\rm mf}^{\mu\nu}
=+H^\nu{}_{\lambda}J_D^\lambda,
\qquad
\nabla_\mu T_{\rm vec}^{\mu\nu}
=-H^\nu{}_{\lambda}J_D^\lambda
}
\]

before explicit damping. A positive stationary carrier in an outward positive
\(B\)-electric field must accelerate outward.

## 7. Solver-friendly irreversible closure

Theory 3.3 uses a local, isotropic, Onsager-positive drag between the carrier
and baryon flows. Define the four-force on the target constituent

\[
\boxed{
r_D^\mu=\mathcal R_D
\left(u^\mu-\gamma_Dv_D^\mu\right),
\qquad \mathcal R_D\ge0.
}
\]

This force obeys

\[
v_{D\mu}r_D^\mu=0
\]

and opposes the carrier velocity in the baryon rest frame. For small relative
velocity its spatial part is \(-\mathcal R_D\boldsymbol v_{\rm rel}\).

The constituent equations become

\[
\boxed{
f^{(D)}_\nu=H_{\nu\lambda}J_D^\lambda+r_{D\nu},
\qquad
f^{(N)}_\nu=-r_{D\nu}.
}
\]

Internal forces cancel in the total material stress equation. The exact drag
heating in the baryon frame is

\[
\boxed{
H_R=u_\mu r_D^\mu
=\mathcal R_D(\gamma_D^2-1)\ge0.
}
\]

The entropy law is

\[
\boxed{
\nabla_\mu S^\mu
=\frac{H_R+H_{\rm vec}}{T}\ge0,
}
\]

where \(H_{\rm vec}\) is the exact positive electric damping loss retained
from Theory 3.2. Conservative Lorentz exchange, mass mixing, Gauss-compatible
longitudinal correction, and reversible chemical readjustment do not produce
entropy.

At continuum balance-law level, explicit vector damping is represented by an
equal-and-opposite exchange \(Q_{\rm damp}^\nu\):

\[
\nabla_\mu T_{\rm mf}^{\mu\nu}
=H^\nu{}_{\lambda}J_D^\lambda+Q_{\rm damp}^\nu,
\qquad
\nabla_\mu T_{\rm vec}^{\mu\nu}
=-H^\nu{}_{\lambda}J_D^\lambda-Q_{\rm damp}^\nu.
\]

Only the positive electric quadratic loss inside this exchange is
\(H_{\rm vec}\). Reversible temporal-potential, mass, and mixing changes are
represented by the recovered multifluid mechanical and chemical energy, not
by entropy.

The scalar resistance is parameterized as

\[
\mathcal R_D=\frac{\mathcal I_D}{\tau_R},
\qquad \tau_R>0,
\]

where \(\mathcal I_D>0\) is the target inertial coefficient obtained from the
master-function Hessian. This makes \(\tau_R\), rather than a coordinate-time
damping rate, the material-frame relaxation time.

The regular vacuum limit requires \(\mathcal I_D\), and hence
\(\mathcal R_D\), to vanish sufficiently rapidly as \(n_D\to0\). No drag
force may depend on the undefined vacuum value of \(v_D^\mu\).

An anisotropic positive-semidefinite resistance tensor may replace
\(\mathcal R_D\Delta_{\mu\nu}\) in a later extension. Negative eigenvalues of
the resistance tensor are forbidden.

## 8. Maxwell--Cattaneo limit

Linearize about a homogeneous co-moving state

\[
u^\mu=v_D^\mu,
\qquad R_D^\mu=0,
\qquad n_b=n_{b0},
\qquad n_D=n_{D0}.
\]

Target-current conservation supplies the charge equation. The spatial target
Euler equation supplies an inertial current equation. With the drag above,
their near-equilibrium combination has the form

\[
\tau_R\Delta^\mu{}_{\nu}u^\alpha\nabla_\alpha R_D^\nu
+R_D^\mu
=-\kappa_D\Delta^{\mu\nu}\nabla_\nu\mu_D
+\tau_R\mathcal F_B^\mu
+O(2),
\]

where \(\mu_D\), \(\kappa_D\), and the projected field force
\(\mathcal F_B^\mu\) are determined by derivatives of \(\Lambda_{\rm mf}\),
not inserted independently.

This is the microscopic replacement for the Theory 3.2 equation

\[
\tau_R\dot R_D^\mu+R_D^\mu
=R_{\rm eq}^\mu-\tau_Rc_D^2\nabla_\perp^\mu q_D.
\]

The differences are deliberate:

- the thermodynamic driving force is a chemical-potential gradient, not an
  arbitrary gradient of charge;
- the field force is derived from \(-B_\mu J_D^\mu\);
- there is no prescribed directional current;
- the target characteristic speeds are eigenvalues of the complete master
  Hessian.

The old `target_speed` is therefore a calibration target and diagnostic, not
a fundamental Theory 3.3 parameter.

## 9. Mixed-Proca and Gauss sector retained from Theory 3.2

The mixed-Proca field equations, Gauss matrix, evolved longitudinal charges,
and cleaner signs are unchanged after replacing the arbitrary source current
by \(J_D^\mu=g_DC_D^\mu\).

In particular, retain

\[
Z_B\nabla_\mu H^{\mu\nu}+\text{constant mass terms}=J_D^\nu
\]

and the two conserved effective currents

\[
I_A^\mu=K_{AA}A^\mu+K_{AB}B^\mu,
\]

\[
I_B^\mu=K_{BA}A^\mu+K_{BB}B^\mu-J_D^\mu/Z_B.
\]

Because \(\nabla_\mu J_D^\mu=0\), both effective-current conservation laws
and the evolved longitudinal-charge construction remain exact.

No Gauss solve may be reintroduced during Runge--Kutta stages or damping.

## 10. Stability, causality, and strong hyperbolicity

The master action does not make an arbitrary parameter choice healthy. Every
declared EOS-and-state domain must pass all of the following.

### 10.1 Legendre and thermodynamic convexity

At co-moving equilibrium, require a nonsingular current-to-momentum Legendre
map. A necessary entrainment condition is

\[
\boxed{
\mathcal B_N>0,
\qquad
\mathcal B_D>0,
\qquad
\mathcal B_N\mathcal B_D-\mathcal A^2>0.
}
\]

The constrained Hessian of the equilibrium energy with respect to
\((n_b,n_D,\sigma)\) must have the signs required by positive heat capacity,
positive compressibility, and positive chemical susceptibility.

These algebraic conditions are screening tests. They do not replace the
principal-symbol audit.

### 10.2 Full characteristic audit

For every state and propagation direction in the run domain:

1. build the coupled baryon/carrier principal symbol from the exact
   master-function derivatives;
2. verify real characteristic speeds;
3. verify a complete eigenbasis or a positive symmetrizer;
4. verify every material characteristic lies inside or on the metric null
   cone;
5. verify bounded condition numbers of the characteristic transformation.

The Theory 3.2 rule \(c_D\ge c_{s,\max}+\delta_c\) is not a fundamental
Theory 3.3 axiom. A symmetric-hyperbolic action system may have a semisimple
coincident pair. Until the numerical characteristic implementation proves
that coincidence is regular, the existing speed gap remains an operational
fallback.

### 10.3 Relative-flow and two-stream audit

Repeat the full symbol scan over

\[
1\le\gamma_D\le\gamma_{D,\max}.
\]

Entrainment and chemical cross-coupling can create two-stream instability at
finite counterflow even when the co-moving state is causal and stable. The
admissible run domain must terminate before the first complex root, loss of
symmetrizer positivity, or unacceptable eigenvector conditioning.

### 10.4 Shock admissibility

The finite-volume system must use the entropy inequality in section 7 to
select dissipative weak solutions. A smooth-field characteristic audit alone
does not qualify shocks. Shock tubes must test both co-moving and counterflow
states, including states near the carrier vacuum boundary.

## 11. Production 3+1 state and primitive recovery

The Theory 3.3 primitive material state is

\[
\boxed{
(n_b,\sigma,u^i,n_D,v_D^i).
}
\]

The source variables \((q_D,R_D^i)\) are derived primitives, not independent
thermodynamic coordinates.

The production conserved state must contain at least

- densitized baryon number;
- densitized carrier number;
- densitized thermal entropy;
- combined material-plus-vector energy and momentum;
- three independent target canonical-momentum or equivalent relative-flow
  variables derived from the target Euler equation;
- the complete existing mixed-Proca, Gauss-charge, cleaner, and CCZ4 state.

A suitable target momentum variable is based on the spatial conjugate
momentum

\[
\chi_i=\gamma_i{}^\mu\chi_\mu.
\]

Using current conservation, the spatial target Euler equation admits the
coordinate balance-law skeleton

\[
\partial_t\left(\sqrt{-g}\,C_D^0\chi_i\right)
+\partial_j\left(\sqrt{-g}\,C_D^j\chi_i\right)
=\sqrt{-g}\left(C_D^\mu\partial_i\chi_\mu+f_i^{(D)}\right).
\]

The derivative term on the right is a genuine principal nonconservative
product unless it is absorbed into a derived conservative flux. It must be
handled by a path-conservative discretization or an analytically equivalent
flux formulation. It must not be labeled a lower-order source.

Primitive recovery simultaneously solves for all nine material primitives
from

1. the two current densities;
2. the three target momentum variables;
3. total energy and momentum after subtracting the complete vector stress;
4. the entropy density;
5. the master-function constitutive relations.

Recovery is rejected unless the Legendre map is nonsingular and every
admissibility condition in section 12 holds. There is no residual variable
that may absorb a failed energy solve.

## 12. Admissible states

Every active cell must satisfy

\[
n_b>0,
\qquad
n_D\ge0,
\qquad
u^2=v_D^2=-1,
\qquad
\gamma_D\ge1,
\]

\[
q_D^2-R_D^2=g_D^2n_D^2\ge0,
\qquad
T>0,
\qquad
\rho_{\rm mf}>0,
\]

together with the convexity, causal-cone, and run-domain restrictions of
section 10.

The carrier atmosphere uses \(n_D=n_{D,\rm atm}\ge0\) and co-moving
\(v_D^\mu=u^\mu\). It must converge away with resolution or atmosphere-floor
reduction. A floor may not create net target charge without an explicitly
recorded conservative compensation.

If \(g_D=0\), the carrier decouples from \(B_\mu\) but remains a material
constituent. If \(n_D=0\), target velocity is undefined and recovery must use
the regular current or momentum variables rather than divide by \(n_D\).

## 13. Initial and boundary data

The default equilibrium initial data are co-moving and have spatially constant
carrier chemical potential:

\[
\left.\frac{\partial\rho_0}{\partial d}\right|_{n_b,\sigma}
=\mu_{D,0},
\qquad
v_D^\mu=u^\mu,
\qquad
R_D^\mu=0.
\]

The constant \(\mu_{D,0}\) is chosen to match the desired conserved total
carrier number. When the quadratic anchoring term dominates, this reduces
approximately to \(n_D\simeq y_\star n_b\).

Non-equilibrium experiments may perturb the carrier fraction, relative
rapidity, or target chemical potential. The total carrier number and target
charge must be recorded in metadata.

After constructing \(J_D^\mu\), initialize the two Proca longitudinal charges
with the unchanged Theory 3.2 Gauss matrix. A Gauss solve is allowed only for
this initial translation.

Finite outer boundaries use the incoming characteristics of the complete
two-current symbol. The old fixed-speed target boundary rule is only a
temporary compatibility condition and must not be claimed as the Theory 3.3
boundary condition.

## 14. Exact damping and energy deposition

Retain the exact Theory 3.2 electric matrix exponential and the complete
before/after vector-stress snapshot. During an electric damping substep:

1. update the two electric fields with the exact local exponential;
2. preserve both Gauss residuals through the longitudinal-charge correction;
3. reconstruct both temporal potentials from the evolved charges;
4. recompute the complete vector stress;
5. hold combined material-plus-vector energy and momentum fixed;
6. add only the exact positive electric loss to thermal entropy;
7. recover the multifluid primitives from the remaining material energy and
   momentum.

The old final step "assign the remaining comoving energy difference to
\(e_D\)" is deleted. The remaining change must be represented by the carrier
chemical, kinetic, and entrainment energy contained in \(T_{\rm mf}^{\mu\nu}\).
If no admissible multifluid primitive state exists, reject and reduce the
step.

During a carrier-drag substep, integrate the relative flow with an exact or
A-stable map when possible. Deposit

\[
\Delta e_{R}=\int H_R\,d\tau\ge0
\]

into thermal entropy while preserving total material energy and momentum and
both number currents. The discrete drag map must satisfy the same entropy
inequality for arbitrarily stiff \(\tau_R\).

## 15. Implementation sequence

Theory 3.3 is introduced behind a new backend. Theory 3.2 remains available as
a regression oracle until the following sequence is complete.

### Phase A: symbolic constitutive layer

- implement \(\Lambda_{\rm M1}\) and automatic or symbolic first and second
  derivatives;
- construct \(\mu_\mu\), \(\chi_\mu\), \(\Psi\), and
  \(T_{\rm mf}^{\mu\nu}\);
- verify the Hilbert stress by independent metric variation;
- verify the co-moving perfect-fluid and zero-carrier limits.

### Phase B: local state and characteristic audit

- implement current/momentum Legendre inversion;
- construct the exact directional principal symbol;
- scan co-moving and counterflow domains for causality, completeness, and
  two-stream instability;
- determine the calibrated M1 domain and whether a numerical speed gap is
  still needed.

### Phase C: finite-volume material backend

- evolve baryon and carrier number conservatively;
- implement the target momentum equation with a conservative or
  path-conservative discretization;
- implement coupled multifluid primitive recovery;
- add entropy-positive stiff drag.

### Phase D: mixed-Proca coupling

- replace the Theory 3.2 arbitrary target current by \(g_DC_D^\mu\);
- retain evolved longitudinal charges and cleaners;
- couple through total stress and verify the locked Lorentz sign;
- replace reservoir damping recovery by multifluid recovery.

### Phase E: qualification

- shock tubes, smooth convergence, counterflow waves, finite boundaries, AMR,
  checkpointing, and long coupled CCZ4 runs;
- comparison against Theory 3.2 in its small-drift Maxwell--Cattaneo regime;
- parameter-domain metadata and restart-safe admissibility checks.

## 16. Required tests

The Theory 3.3 backend is accepted only when all of the following pass:

- independent metric variation reproduces the written material stress;
- the interaction term has no separately double-counted Hilbert stress;
- baryon number, carrier number, and target charge are conserved to the
  expected discrete accuracy;
- a positive stationary target charge feels force along the positive
  \(B\)-electric field;
- material and vector four-momentum exchange are equal and opposite;
- the current-to-momentum Legendre map round-trips throughout the declared
  state domain;
- equilibrium Hessians and entrainment matrices pass their positivity tests;
- the full characteristic symbol is causal and complete for all sampled
  directions and counterflows;
- the first two-stream or loss-of-hyperbolicity boundary is located and
  recorded rather than hidden by floors;
- target-vacuum and co-moving limits are regular;
- carrier drag decreases relative kinetic/free energy and gives exactly
  nonnegative entropy;
- conservative Lorentz exchange gives zero entropy;
- the exact vector damping map retains its Theory 3.2 Gauss, energy,
  momentum, and entropy properties without a reservoir;
- coupled primitive recovery closes total energy and momentum without a
  residual ledger;
- smooth waves converge at design order and multifluid shocks converge to
  stable weak solutions;
- finite boundaries and AMR transfers use the complete Theory 3.3
  characteristic and conservation structure.

## 17. Explicit limitations

Theory 3.3 is a covariant hydrodynamic effective theory. Calling the carrier
"microscopic" means that its number current, charge, energy, and interaction
are represented by an action rather than a bookkeeping reservoir. It does not
identify a fundamental particle species or derive \(\Lambda_{\rm mf}\) from a
quantum field theory.

The following are outside the minimal specification:

- target pair creation or charge-changing reactions;
- simultaneous positive and negative target carriers;
- an independent heat-flow velocity;
- anisotropic directors or elastic target media;
- stochastic fluctuation terms and their Schwinger--Keldysh completion;
- superfluid vortex quantization;
- a proof of global hyperbolicity outside the declared numerical state domain.

Each extension requires new fields or constitutive data and a renewed
principal-symbol audit.

## 18. Locked Theory 3.3 result

Theory 3.3 replaces the phenomenological reservoir with a genuine charged
carrier constituent. Its stress and energy follow from the invariant master
function; its source current follows from a conserved carrier current; its
matter coupling follows from a chemical affinity and entrainment; and its
relaxation follows from an entropy-positive internal drag.

The following Theory 3.2 structures remain locked:

- constant-mass mixed Proca action;
- \(-B_\mu J_D^\mu\) source convention;
- matter force \(+H^\nu{}_{\lambda}J_D^\lambda\);
- evolved longitudinal Proca charges;
- independent Gauss cleaners;
- exact finite-step electric damping;
- complete before/after vector-stress energy and momentum deposit.

The following Theory 3.2 structures are retired:

- pressureless signed reservoir energy \(e_D\);
- prescribed `target_direction` and `target_strength`;
- independently selected target characteristic speed;
- arbitrary independent \((q_D,R_D^\mu)\) states;
- assigning unmatched damping energy to a residual ledger.

This document is the mathematical contract for the Theory 3.3 derivation and
implementation. It does not declare the existing Theory 3.2 code to be a
Theory 3.3 solver.

## References

- L. Gavassino, *Stability and causality of Carter's multifluid theory*,
  [arXiv:2202.06760](https://arxiv.org/abs/2202.06760).
- N. Andersson, G. L. Comer, and I. Hawke, *A variational approach to
  resistive relativistic plasmas*,
  [arXiv:1610.00445](https://arxiv.org/abs/1610.00445).
- L. Gavassino, *Mapping GENERIC hydrodynamics into Carter's multifluid
  theory*, [arXiv:2311.10897](https://arxiv.org/abs/2311.10897).
- L. Samuelsson, C. S. Lopez-Monsalvo, N. Andersson, and G. L. Comer,
  *Relativistic two-stream instability*,
  [arXiv:0906.4002](https://arxiv.org/abs/0906.4002).
- G. L. Comer, N. Andersson, T. Celora, and I. Hawke, *Action based approach
  to dissipative relativistic fluid systems*,
  [arXiv:2606.17686](https://arxiv.org/abs/2606.17686).
