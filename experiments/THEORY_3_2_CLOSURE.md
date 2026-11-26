# Theory 3.2 closure for the production overhaul

Status: formulation lock for implementation. This document replaces the five
open parts of Theory 3.1. It does not claim that the resulting phenomenological
matter model has already been derived from a single microscopic action.

## 1. Conventions and the conservative split

Use signature `(-,+,+,+)`. Let $n^\mu$ be the future Eulerian normal and

\[
A^\mu=a^\mu+\Phi_A n^\mu,
\qquad
B^\mu=b^\mu+\Phi_B n^\mu,
\qquad
n_\mu a^\mu=n_\mu b^\mu=0.
\]

The conservative vector stress is the Hilbert stress obtained from the two
kinetic terms, the two constant Proca masses, and the constant $A-B$ mass
mixing. In compact notation,

\[
T_{\rm vec}^{\mu\nu}
=T_F^{\mu\nu}+T_H^{\mu\nu}
+Z_A m_A^2\!\left(A^\mu A^\nu-\tfrac12g^{\mu\nu}A^2\right)
+Z_B m_B^2\!\left(B^\mu B^\nu-\tfrac12g^{\mu\nu}B^2\right)
+\mu^2\!\left(W^\mu W^\nu-\tfrac12g^{\mu\nu}W^2\right),
\]

where $W^\mu=A^\mu-B^\mu$, and $T_F,T_H$ use the sign convention that
gives positive Eulerian electric-plus-magnetic energy. The implementation must
test the 3+1 projections rather than infer their signs from this shorthand.

The term $-B_\mu J_D^\mu$ supplies the field source, but its metric variation
is not silently counted as a second interaction stress. Until a complete
multifluid master function for $J_D^\mu$ is supplied, the closed system is a
covariant balance-law model:

\[
\nabla_\mu T_{\rm mat}^{\mu\nu}=Q_{\rm drive}^\nu+Q_{\rm damp}^\nu,
\qquad
\nabla_\mu T_{\rm vec}^{\mu\nu}=-Q_{\rm drive}^\nu-Q_{\rm damp}^\nu.
\]

This avoids both an undefined metric variation of the target variables and
double counting the interaction energy.

## 2. Target stress and reservoir energy

The minimal closed target stress is a comoving, pressureless internal
reservoir. Define

\[
e=e_{\rm th}+e_D,
\qquad
T_{\rm mat}^{\mu\nu}=e u^\mu u^\nu+p\Delta^{\mu\nu}
= (e+p)u^\mu u^\nu+p g^{\mu\nu},
\]

with $e_{\rm th}=n_b(1+\epsilon)$,
$\Delta^{\mu\nu}=g^{\mu\nu}+u^\mu u^\nu$, and

\[
T_D^{\mu\nu}=e_D u^\mu u^\nu.
\]

The EOS pressure remains $p=p(n_b,s)$; $e_D/n_b$ is an independent
material contact variable. Consequently the reservoir does not change the
Euler acoustic Hessian at principal order. It does change the inertia through

\[
w_{\rm tot}=e_{\rm th}+e_D+p.
\]

Admissible states must satisfy

\[
n_b>0,qquad w_{\rm tot}>0,qquad 0<c_s^2<1.
\]

The reservoir is a reversible energy ledger and is not required to be
positive pointwise. If positivity is desired later, it requires a specified
convex target free energy, not an after-the-fact floor. A production run must
nevertheless use an admissibility floor on $w_{\rm tot}$ and reject any step
that crosses it.

Baryon conservation and the thermal first law give

\[
\nabla_\mu(n_bu^\mu)=0,
\qquad
u^\mu\nabla_\mu e_{\rm th}+(e_{\rm th}+p)\nabla_\mu u^\mu
=H_{\rm irr}.
\]

The exact reservoir balance is therefore

\[
\boxed{
\nabla_\mu(e_Du^\mu)
=-u_\nu(Q_{\rm drive}^\nu+Q_{\rm damp}^\nu)-H_{\rm irr}
}
\]

and the entropy law is

\[
\nabla_\mu(n_bs u^\mu)=\frac{H_{\rm irr}}{T}\ge0.
\]

During a purely conservative drive step, $H_{\rm irr}=0$. During electric
damping, $H_{\rm irr}$ is the exact integrated positive electric loss given
in section 6. Any additional finite-step energy change caused by the
longitudinal constraint correction is reversible and goes to $e_D$, not to
entropy.

The current-relaxation source may optionally be assigned

\[
H_R=\chi_R\lVert R_D-R_{\rm eq}\rVert^2/\tau_R,
\qquad \chi_R\ge0,
\]

in which case $H_R$ is included in $H_{\rm irr}$ and removed from $e_D$.
If $\chi_R=0$, the target current is an internal source label with no separate
thermodynamic cost. This choice must be stored in run metadata.

## 3. Correct Lorentz-force sign

Keep the Theory 3.1 field-source and Gauss conventions:

\[
Z_B\nabla_\mu H^{\mu\nu}+\text{constant mass terms}=J_D^\nu,
\qquad
D_iE_B^i+\text{mass terms}-\rho_D/Z_B=0.
\]

With those conventions, the vector stress identity is

\[
\nabla_\mu T_{\rm vec}^{\mu\nu}ig|_{J_D}
=-H^\nu{}_{\lambda}J_D^\lambda.
\]

Therefore the force received by matter is

\[
\boxed{Q_{\rm drive}^\nu=+H^\nu{}_{\lambda}J_D^\lambda.}
\]

The minus sign in Theory 3.1 section 15 is reversed.

There is a direct flat-space sign check. For a positive stationary target
charge, the written B Gauss law gives an outward $E_B^i$. Since
$H^i{}_0J_D^0=\rho_D E_B^i$, the boxed sign accelerates positive matter along
the outward electric field. The old sign accelerates it inward. The production
test suite must include this one-cell force-direction check and a paired
vector/matter four-momentum balance check.

## 4. Strongly hyperbolic target current

### 4.1 Why the Theory 3.1 current fails

Replace $r_Dd^\mu$ by the regular spatial current vector

\[
R_D^\mu\equiv\Delta^\mu{}_{\nu}J_D^\nu,
\qquad
J_D^\mu=q_Du^\mu+R_D^\mu,
\qquad
u_\mu R_D^\mu=0.
\]

This remains regular when the target amplitude vanishes. A unit director may
still define $R_{\rm eq}^\mu$, but it is not evolved as the principal current
variable.

In one frozen spatial direction, Theory 3.1 gives the target principal block

\[
\partial_t
\begin{pmatrix}\delta q_D\\\delta R_D^n\end{pmatrix}
+
\begin{pmatrix}v_n&1\\0&v_n\end{pmatrix}
\partial_n
\begin{pmatrix}\delta q_D\\\delta R_D^n\end{pmatrix}=0.
\]

The repeated eigenvalue $v_n$ has only one eigenvector. The subsystem is a
Jordan block and is only weakly hyperbolic.

### 4.2 Replacement current law

Use target conservation plus a causal Maxwell--Cattaneo current equation:

\[
\nabla_\mu J_D^\mu=0,
\]

\[
\boxed{
\tau_R\Delta^\mu{}_{\nu}u^\alpha\nabla_\alpha R_D^\nu+R_D^\mu
=R_{\rm eq}^\mu
-\tau_Rc_D^2\Delta^{\mu\nu}\nabla_\nu q_D
}
\]

with $0<c_D<1$, $\tau_R>0$, and algebraic $R_{\rm eq}^\mu$. Terms needed
to preserve $u_\mu R_D^\mu=0$ are included as algebraic/geometric sources;
they do not alter the frozen principal symbol.

In the fluid rest frame the longitudinal target block is

\[
\begin{pmatrix}0&1\\c_D^2&0\end{pmatrix},
\]

with speeds $\pm c_D$, eigenvectors $(1,\pm c_D)$, and positive symmetrizer
$\operatorname{diag}(c_D^2,1)$. In a moving frame,

\[
\lambda_D^\pm=\frac{v_n\pm c_D}{1\pm v_nc_D}.
\]

### 4.3 Euler-contact/current coincidence audit

For rest-frame variables
$(\delta n_b,\delta v_n,\delta p,\delta q_D,\delta R_D^n)$, a representative
longitudinal principal matrix is

\[
P=
\begin{pmatrix}
0&n_b&0&0&0\\
0&0&1/w_{\rm tot}&0&0\\
0&w_{\rm tot}c_s^2&0&0&0\\
0&q_D&0&0&1\\
0&0&0&c_D^2&0
\end{pmatrix}.
\]

Its eigenvalues are $0,\pm c_s,\pm c_D$. The zero-speed Euler contact is
semisimple with the target pair moved away from it. However, if
$c_D=c_s$ and $q_D\ne0$, each coincident acoustic/current branch loses an
eigenvector: the acoustic velocity perturbation forces the target conservation
equation at exactly its own characteristic speed.

The production admissibility rule is therefore

\[
\boxed{c_D\ge c_{s,\max}+\delta_c,qquad c_D<1,qquad \delta_c>0.}
\]

Here $c_{s,\max}$ is an EOS-and-run-domain bound, not merely the sound speed
of the initial data. Runtime primitive recovery must monitor
$c_D-c_s\ge\delta_c$. If an EOS permits $c_s\to1$ with no stricter domain
bound, a fixed subluminal $c_D$ cannot provide a global gap and the run domain
must be restricted or the target model redesigned.

Transverse components of $R_D^i$ are advected/relaxed material modes. They
must not feed derivatives into the Euler contact block. The equilibrium
director and any target composition scalars are likewise passive contacts at
principal order.

The executable audit is
`experiments/audit_theory32_target_eigenspace.py`; it records both the original
Jordan defect and the repaired resonance behavior.

## 5. Evolved longitudinal charges for the mixed Proca pair

Define the constant Gauss matrix

\[
K=
\begin{pmatrix}
m_A^2+\mu^2/Z_A&-\mu^2/Z_A\\
-\mu^2/Z_B&m_B^2+\mu^2/Z_B
\end{pmatrix},
\]

with

\[
\det K=m_A^2m_B^2
+\mu^2\left(m_A^2/Z_B+m_B^2/Z_A\right)>0.
\]

Evolve the two densitized longitudinal charges

\[
\boxed{
\begin{aligned}
\mathcal Q_A&=\sqrt\gamma(K_{AA}\Phi_A+K_{AB}\Phi_B),\\
\mathcal Q_B&=\sqrt\gamma(K_{BA}\Phi_A+K_{BB}\Phi_B-\rho_D/Z_B).
\end{aligned}}
\]

The constraints become

\[
C_A=D_iE_A^i+\mathcal Q_A/\sqrt\gamma,
\qquad
C_B=D_iE_B^i+\mathcal Q_B/\sqrt\gamma.
\]

At every reconstruction, obtain the temporal potentials from the local solve

\[
K
\begin{pmatrix}\Phi_A\\\Phi_B\end{pmatrix}
=
\begin{pmatrix}
\mathcal Q_A/\sqrt\gamma\\
\mathcal Q_B/\sqrt\gamma+\rho_D/Z_B
\end{pmatrix}.
\]

No divergence of an electric field appears in this reconstruction.

The two covariantly conserved effective currents are

\[
I_A^\mu=K_{AA}A^\mu+K_{AB}B^\mu,
\]

\[
I_B^\mu=K_{BA}A^\mu+K_{BB}B^\mu-J_D^\mu/Z_B.
\]

The antisymmetry of $F,H$, together with
$\nabla_\mu J_D^\mu=0$, gives
$\nabla_\mu I_A^\mu=\nabla_\mu I_B^\mu=0$. If
$I_X^\mu=q_Xn^\mu+j_X^\mu$, evolve

\[
\boxed{
\partial_t\mathcal Q_X
+\partial_i\!\left[\sqrt\gamma
(\alpha j_X^i-\beta^iq_X)\right]=0,
\qquad X\in\{A,B\}.
}
\]

This replaces all stagewise Gauss solves. A Gauss solve is permitted only to
translate initial data into $\mathcal Q_A,\mathcal Q_B$.

Use one cleaner $Z_X$ per constraint. One consistent principal sign pair is

\[
(\partial_t-\mathcal L_\beta)E_X^i\supset-\alpha D^iZ_X,
\qquad
(\partial_t-\mathcal L_\beta)Z_X=-\alpha C_X-\alpha\kappa_XZ_X.
\]

Changing either sign requires changing the other. The charge flux and electric
divergence must use the same discrete derivative operator so satisfied Gauss
data remain satisfied to roundoff on periodic uniform grids.

## 6. Complete finite-step damping deposit

In a frozen fluid orthonormal frame, damp each spatial component with

\[
\frac{d}{d\tau}
\begin{pmatrix}E_A\\E_B\end{pmatrix}
=M
\begin{pmatrix}E_A\\E_B\end{pmatrix},
\qquad
M=
\begin{pmatrix}
-\Gamma_W/Z_A&\Gamma_W/Z_A\\
\Gamma_W/Z_B&-(\Gamma_W+\Gamma_B)/Z_B
\end{pmatrix}.
\]

Use the exact $2\times2$ exponential

\[
\mathbf E^+=\exp(\Delta\tau M)\mathbf E^-
\]

or an A-stable solve whose energy inequality is tested. Forward Euler is not a
production damping map.

The exact irreversible heat is

\[
\boxed{
\Delta e_{\rm irr}
=\int_0^{\Delta\tau}
\left[\Gamma_W\lVert E_A-E_B\rVert^2
+\Gamma_B\lVert E_B\rVert^2\right]d\tau\ge0.
}
\]

For the exponential update this equals the decrease of the positive electric
quadratic form

\[
\tfrac12 Z_A\lVert E_A\rVert^2+\tfrac12 Z_B\lVert E_B\rVert^2.
\]

The complete constraint-compatible damping substep is:

1. Snapshot $T_{\rm vec}^{\mu\nu,-}$, including kinetic, both mass terms,
   mixing, temporal potentials, and all 3+1 cross terms.
2. Apply the exact electric matrix exponential in the frozen fluid frame and
   transform back to Eulerian variables.
3. Preserve the pre-step Gauss residuals by applying

   \[
   \Delta\mathcal Q_A=-\sqrt\gamma D_i\Delta E_A^i,
   \qquad
   \Delta\mathcal Q_B=-\sqrt\gamma D_i\Delta E_B^i.
   \]

   This is a discrete constraint-compatible source correction, not a strictly
   point-local operation.
4. Reconstruct $\Phi_A,\Phi_B$ from $K^{-1}$ and the updated charges. Never
   reconstruct them from $D_iE^i$.
5. Recompute the complete $T_{\rm vec}^{\mu\nu,+}$ and its Eulerian
   projections

   \[
   \rho_{\rm vec}=n_\mu n_\nu T_{\rm vec}^{\mu\nu},
   \qquad
   S_{{\rm vec},i}=-\gamma_{i\mu}n_\nu T_{\rm vec}^{\mu\nu}.
   \]

6. Deposit the exact finite difference into the matter conserved variables:

   \[
   \boxed{
   \Delta(\sqrt\gamma E_{\rm mat})
   =-\Delta(\sqrt\gamma\rho_{\rm vec}),
   \qquad
   \Delta(\sqrt\gamma S_{{\rm mat},i})
   =-\Delta(\sqrt\gamma S_{{\rm vec},i}).
   }
   \]

7. Add only $\Delta e_{\rm irr}$ to thermal energy/entropy. Assign the
   remaining comoving energy difference to $e_D$. Recover
   $(n_b,\epsilon,u^i,e_D)$ in one coupled primitive solve so the deposited
   Eulerian energy and momentum and the thermal entropy increment are all
   satisfied simultaneously.
8. Store the complete spatial stress
   $S_{{\rm vec},ij}^+$ for the next flux and CCZ4 source evaluation. Spatial
   stress is not itself deposited into a Valencia conserved variable.

The old sequence "damp electric fields, deposit
$\Delta\tau H_{\rm diss}$, then solve Gauss again" is removed. It misses the
finite change in temporal/mass/mixing energy and can change total energy merely
by enforcing a constraint.

## 7. Required implementation guards and tests

The overhaul is accepted only when all of the following pass:

- exact positive-charge Lorentz-force direction and equal/opposite vector and
  matter four-momentum exchange;
- target-current eigenvector completeness over the EOS state domain, with the
  runtime $c_D-c_s$ margin enforced;
- roundoff preservation of both Gauss constraints on a periodic grid using
  evolved $\mathcal Q_A,\mathcal Q_B$;
- convergence of both cleaners for deliberately constraint-violating data;
- damping energy monotonicity for arbitrary $Z_A,Z_B>0$ and
  $\Gamma_W,\Gamma_B\ge0$;
- cellwise finite-step energy/momentum balance after the full-stress deposit;
- nonnegative entropy increment and no entropy contribution from the
  conservative drive;
- admissible primitive recovery with $w_{\rm tot}>0$ after every split
  substep;
- shock-tube convergence with the target speed separated from the maximum EOS
  sound speed.

## Locked result

Theory 3.1 is not production-ready as written because its target-current block
is defective, its Lorentz-force sign is reversed under its own Gauss
convention, its target stress is undefined, and its damping/Gauss sequence does
not deposit the full finite-step vector energy.

The definitions in this document close those points. They are suitable as the
mathematical contract for the code overhaul, subject to the explicit
phenomenological-reservoir caveat and the speed-gap admissibility domain.
