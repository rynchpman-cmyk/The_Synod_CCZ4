# Nulloth Canonical Action Derivation

Status: canonical variational derivation  
Metric signature: `(-,+,+,+)`  
Scalar momentum: `Pi = -n^mu grad_mu varphi`

This document derives the equations used by the Nulloth backend from the
canonical action. The inherited constant-mass mixed-Proca sector is written
explicitly so that the word *complete* does not conceal a reference to an
unspecified force law.

## 1. Canonical action

Let

\[
F_{\mu\nu}=2\nabla_{[\mu}A_{\nu]},\qquad
H_{\mu\nu}=2\nabla_{[\mu}B_{\nu]},\qquad
W_\mu=A_\mu-B_\mu.
\]

The action is

\[
S=\int d^4x\sqrt{-g}\left[
\frac{R-2\Lambda_c}{2\kappa}
+\Lambda_N
-\frac12\nabla_\mu\varphi\nabla^\mu\varphi
-V_0(\varphi)
-\frac{Z_A}{4}F_{\mu\nu}F^{\mu\nu}
-\frac{Z_B}{4}H_{\mu\nu}H^{\mu\nu}
-\frac{Z_A m_A^2}{2}A_\mu A^\mu
-\frac{Z_B m_B^2}{2}B_\mu B^\mu
-\frac{\mu^2}{2}W_\mu W^\mu
-B_\mu J_D^\mu
\right]+S_{\rm pullback}.
\]

Here

\[
J_D^\mu=g_D C^\mu,
\]

and

\[
\Lambda_N=\Lambda_{M1}+\frac12\mathcal C\varphi^2,
\qquad
\mathcal C=
g_n\left(\frac{n}{n_\star}\right)^2
+g_d\left(\frac{d}{d_\star}\right)^2
+g_W\frac{\mathcal W}{W_\star}.
\]

The pullback variables enforce conservation of the two material currents. The
source term uses the same carrier current; it is not an independently
prescribed vector and does not receive a second phenomenological stress.

## 2. Invariant derivatives and conjugate momenta

The material invariants are

\[
n^2=-N_\mu N^\mu,\qquad
d^2=-C_\mu C^\mu,\qquad
x^2=-N_\mu C^\mu,
\]

with

\[
\mathcal W=x^2-nd.
\]

Direct differentiation gives

\[
\frac{\partial\Lambda_N}{\partial N^\nu}
=-2\Lambda_{N,n^2}N_\nu-\Lambda_{N,x^2}C_\nu,
\]

\[
\frac{\partial\Lambda_N}{\partial C^\nu}
=-2\Lambda_{N,d^2}C_\nu-\Lambda_{N,x^2}N_\nu.
\]

Define

\[
\mathcal B_N=-2\Lambda_{N,n^2},\qquad
\mathcal B_D=-2\Lambda_{N,d^2},\qquad
\mathcal A=-\Lambda_{N,x^2}.
\]

Then the material momentum covectors are

\[
\mu_\nu=\mathcal B_NN_\nu+\mathcal A C_\nu,
\qquad
\chi_\nu=\mathcal B_DC_\nu+\mathcal A N_\nu.
\]

Writing \(\rho_N=-\Lambda_N\) and
\(Q=\partial\rho_N/\partial\mathcal W\), the chain rule at fixed
\((d,\mathcal W,\sigma,\varphi)\) and fixed
\((n,\mathcal W,\sigma,\varphi)\) yields

\[
\mathcal B_N=\frac{\rho_{N,n}-Qd}{n},\qquad
\mathcal B_D=\frac{\rho_{N,d}-Qn}{d},\qquad
\mathcal A=Q.
\]

This is the exact relation implemented by `NullothMasterFunction`; it is not a
post-action correction.

## 3. Pullback variations and current equations

The matter-space pullbacks give the kinematic equations

\[
\nabla_\mu N^\mu=0,\qquad
\nabla_\mu C^\mu=0,
\]

and advection of entropy per baryon on the reversible branch,

\[
N^\mu\nabla_\mu\sigma=0.
\]

Define the material vorticities

\[
\omega^{(N)}_{\mu\nu}=2\nabla_{[\mu}\mu_{\nu]},\qquad
\omega^{(C)}_{\mu\nu}=2\nabla_{[\mu}\chi_{\nu]}.
\]

The reversible baryon equation is

\[
N^\mu\omega^{(N)}_{\mu\nu}=nT\nabla_\nu\sigma,
\]

while variation of the carrier pullback in the presence of
\(-g_DB_\mu C^\mu\) gives

\[
C^\mu\,2\nabla_{[\mu}(\chi_{\nu]}-g_DB_{\nu]})=0,
\]

or equivalently

\[
C^\mu\omega^{(C)}_{\mu\nu}
=g_D C^\mu H_{\mu\nu}.
\]

The scalar dependence of \(\Lambda_N\) changes both momentum covectors. It
does not introduce an arbitrary external material force; the scalar exchange
follows from the explicit spacetime dependence of the master function and the
Noether identity derived below.

## 4. Scalar Euler–Lagrange equation

Varying \(\varphi\), integrating the kinetic variation by parts, and holding
the material pullbacks fixed gives

\[
\delta_\varphi S=\int d^4x\sqrt{-g}
\left(\Box\varphi-V_{0,\varphi}+\Lambda_{N,\varphi}\right)
\delta\varphi.
\]

Therefore

\[
\boxed{
\Box\varphi=V_{0,\varphi}-\Lambda_{N,\varphi}
=(m_\varphi^2-\mathcal C)\varphi
+\lambda_\varphi\varphi^3.
}
\]

The sign pair implemented in 3+1 form is consequently

\[
\Pi=-n^\mu\nabla_\mu\varphi,
\qquad
\partial_t\varphi=\beta^i\Phi_i-\alpha\Pi.
\]

## 5. Mixed-Proca Euler–Lagrange equations

Varying \(A_\nu\) gives

\[
Z_A\nabla_\mu F^{\mu\nu}
=Z_A m_A^2A^\nu+\mu^2(A^\nu-B^\nu).
\]

Varying \(B_\nu\) gives

\[
Z_B\nabla_\mu H^{\mu\nu}
=Z_Bm_B^2B^\nu-\mu^2(A^\nu-B^\nu)+J_D^\nu.
\]

After division by the kinetic coefficients, the algebraic mass matrix is

\[
K=\begin{pmatrix}
m_A^2+\mu^2/Z_A & -\mu^2/Z_A\\
-\mu^2/Z_B & m_B^2+\mu^2/Z_B
\end{pmatrix},
\]

which is the matrix used by `Theory3System.gauss_matrix`.

Taking a divergence and using antisymmetry of the field strengths yields the
two longitudinal compatibility laws. The second remains compatible because

\[
\nabla_\mu J_D^\mu=g_D\nabla_\mu C^\mu=0.
\]

## 6. Metric variation and stress partition

At fixed densitized pullback currents, metric variation of
\(\sqrt{-g}\Lambda_N\) gives

\[
T_{\rm mf}^{\mu\nu}
=\Psi g^{\mu\nu}
+\mathcal B_NN^\mu N^\nu
+\mathcal B_DC^\mu C^\nu
+\mathcal A(N^\mu C^\nu+C^\mu N^\nu),
\]

where

\[
\Psi=\Lambda_N+\mathcal B_Nn^2+\mathcal B_Dd^2+2\mathcal A x^2.
\]

Because \(\mathcal C\varphi^2/2\) is already inside \(\Lambda_N\), its
metric response is already present in \(\Psi,\mathcal B_N,\mathcal B_D\), and
\(\mathcal A\). Adding another interaction stress would double count it.

The scalar variation gives

\[
T_\varphi^{\mu\nu}
=\nabla^\mu\varphi\nabla^\nu\varphi
-g^{\mu\nu}\left[
\frac12\nabla_\alpha\varphi\nabla^\alpha\varphi+V_0
\right].
\]

The vector variation gives the two Maxwell stresses, the two Proca mass
stresses, and the mixing stress

\[
\mu^2\left(W^\mu W^\nu-\frac12g^{\mu\nu}W^2\right).
\]

The source term is not appended as an independent stress. Its force is
already exchanged between the carrier canonical momentum equation and the
\(B_\mu\) field equation.

Finally,

\[
G_{\mu\nu}+\Lambda_cg_{\mu\nu}
=\kappa\left(T_{\mu\nu}^{\rm mf}
+T_{\mu\nu}^{\varphi}+T_{\mu\nu}^{\rm vec}\right).
\]

## 7. Noether exchange identities

The canonical scalar identity is

\[
\nabla_\mu T_\varphi^{\mu\nu}
=(\Box\varphi-V_{0,\varphi})\nabla^\nu\varphi
=-\Lambda_{N,\varphi}\nabla^\nu\varphi.
\]

The material identity on the reversible current equations is

\[
\nabla_\mu T_{\rm mf}^{\mu\nu}
=\Lambda_{N,\varphi}\nabla^\nu\varphi
+H^{\nu}{}_{\mu}J_D^\mu.
\]

The vector identity is

\[
\nabla_\mu T_{\rm vec}^{\mu\nu}
=-H^{\nu}{}_{\mu}J_D^\mu.
\]

Thus every reversible exchange cancels:

\[
\nabla_\mu\left(
T_{\rm mf}^{\mu\nu}+T_\varphi^{\mu\nu}+T_{\rm vec}^{\mu\nu}
\right)=0.
\]

This identity is why neither a scalar energy reservoir nor a separately
invented vector interaction stress is permitted.

## 8. Irreversible completion

Carrier drag adds an equal-and-opposite internal four-force with

\[
r_D^\mu=\mathcal R_D(u^\mu-\gamma_Dv^\mu),\qquad
\mathcal R_D\ge0.
\]

Its rest-frame heat is

\[
H_R=\mathcal R_D(\gamma_D^2-1)\ge0.
\]

Exact vector damping contributes the positive electric loss \(H_{\rm vec}\).
The entropy law is therefore

\[
\nabla_\mu(n\sigma u^\mu)
=\frac{H_R+H_{\rm vec}}{T}\ge0.
\]

Scalar-material exchange is absent from this numerator because it is
reversible.

## 9. 3+1 scalar reduction

With

\[
\Phi_i=D_i\varphi,
\]

the scalar equations are

\[
\partial_t\varphi=\beta^i\Phi_i-\alpha\Pi,
\]

\[
\partial_t\Pi
=\beta^iD_i\Pi-\alpha D_i\Phi^i+\alpha K\Pi
+\Phi^iD_i\alpha
+\alpha(V_{0,\varphi}-\Lambda_{N,\varphi}),
\]

\[
\partial_t\Phi_i
=\beta^jD_j\Phi_i+\Phi_jD_i\beta^j
-\alpha D_i\Pi-\Pi D_i\alpha.
\]

The reduction constraint is

\[
\mathcal C_i^\varphi=\Phi_i-D_i\varphi.
\]

Its stage projection is a numerical representation operation. It does not
enter any physical stress, material force, activation rule, or entropy law.

## 10. Dimensional contract

In continuum geometrized units, with dimensionless \(\varphi\),

\[
[\nabla]=L^{-1},\quad [R]=[T^{\mu\nu}]=[\Lambda_N]=[V_0]=L^{-2},
\]

\[
[m_\varphi]=L^{-1},\quad
[m_\varphi^2]=[\mathcal C]=[\lambda_\varphi]=L^{-2}.
\]

For number currents,

\[
[N^\mu]=[C^\mu]=[n]=[d]=L^{-3},\qquad
[\mathcal W]=L^{-6}.
\]

Consequently

\[
[n_\star]=[d_\star]=L^{-3},\qquad
[W_\star]=L^{-6},\qquad
[g_n]=[g_d]=[g_W]=L^{-2}.
\]

Rest masses and the coefficients in \(\rho_0\) supply the remaining powers
needed to convert number density to energy density. The Python backend uses
nondimensional variables after choosing reference length, density, energy,
and scalar scales. Its numeric `scalar_*_coupling`, `scalar_mass**2`, and
`scalar_quartic` therefore represent the corresponding continuum coefficients
in the same chosen inverse-length-squared unit.

No physical SI or ship-scale energy claim may be inferred until those
reference scales are declared.

## 11. Evidence and remaining scope

The fixed-current Hilbert variation is witnessed by campaign `NUL-FND-001`.
All implemented first and second constitutive derivatives are witnessed by
campaign `NUL-FND-002`. The scalar stress components and Pi sign pair are
protected by `tests/test_nulloth_foundation.py`.

This derivation establishes the canonical continuum equations. It does not by
itself prove global hyperbolicity, nonlinear stability, finite-boundary
well-posedness, or convergence of every discrete exchange term. Those remain
separate workplan gates.

