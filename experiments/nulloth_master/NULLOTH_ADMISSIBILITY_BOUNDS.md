# Nulloth analytic admissibility bounds

## Scope

These are sufficient local constitutive bounds for the registered discovery
box. They are not global necessary conditions. They use the canonical
discovery parameters, including zero ordinary entrainment coefficients,
scalar baryon coupling zero, carrier scalar coupling twelve, and counterflow
scalar coupling one fifth.

The box is

\[
0.4\le n\le0.9,\quad 0.1\le\epsilon\le0.35,\quad
0.12\le d\le0.24,\quad |v_D|\le0.3,\quad |\varphi|\le0.16.
\]

## Thermodynamic Hessian

Write the density Hessian as

\[
H_\rho=\begin{pmatrix}\rho_{nn}&\rho_{nd}\\
\rho_{nd}&\rho_{dd}\end{pmatrix}.
\]

For the canonical parameters,

\[
\rho_{nn}=\gamma(\gamma-1)\frac{\epsilon}{n}
 +\frac{f^2}{\chi},\qquad
\rho_{nd}=-\frac f\chi,
\]

and

\[
\rho_{dd}=K_D(1+c_D^2)c_D^2d^{c_D^2-1}
 +\frac1\chi-g_D\varphi^2.
\]

Because \(c_D^2-1<0\), the lower diagonal bounds occur at low internal
energy, high baryon density, high carrier density, and maximum coherence
amplitude. Gershgorin's inequality gives the box-wide sufficient condition

\[
\lambda_{\min}(H_\rho)\ge
\min(\rho_{nn}^{\rm low},\rho_{dd}^{\rm low})-|\rho_{nd}|>0.
\]

The executable campaign evaluates this lower bound at the production cap and
solves the associated guaranteed-amplitude inequality at 80-digit precision.

## Legendre map

The current-momentum Legendre matrix is

\[
H_L=\begin{pmatrix}B_N&Q\\Q&B_D\end{pmatrix},\qquad
Q=-\frac12 g_W\varphi^2.
\]

The scalar counterflow term makes \(Q\le0\). Consequently the terms
\(-Qd/n\) in \(B_N\) and \(-Qn/d\) in \(B_D\) are nonnegative and may be
dropped to obtain conservative lower bounds. Over the entire box,

\[
B_N\ge B_N^{(0)}-g_N\varphi^2,qquad
B_D\ge B_D^{(0)}-g_D\varphi^2.
\]

Another Gershgorin bound then gives

\[
\lambda_{\min}(H_L)\ge
\min(B_N^{\rm low},B_D^{\rm low})-\frac12g_W\varphi^2>0.
\]

This bound is valid for every signed counterflow in the box because ordinary
linear and quadratic entrainment vanish for the discovery parameter set.

## Ordering of denial surfaces

On the central homogeneous state, the high-precision audit separately solves

\[
\det H_\rho=0,\qquad \det H_L=0.
\]

Those exact convexity zeros are compared with the measured characteristic
condition-number denial and the stricter production amplitude cap. The
resulting order is

\[
|\varphi|_{\rm production}
<|\varphi|_{\rm condition}
<|\varphi|_{\det H_\rho=0}
<|\varphi|_{\det H_L=0}.
\]

Thus production denial has a deliberate margin before loss of strong
hyperbolicity, and characteristic denial precedes the exact thermodynamic
zero. The numerical values are generated into the Evidence Ledger by campaign
NUL-CHR-006 rather than copied by hand into this derivation.
