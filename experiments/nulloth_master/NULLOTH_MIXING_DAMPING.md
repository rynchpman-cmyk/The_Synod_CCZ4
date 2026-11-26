# Nulloth Mixing and Relaxation Map

## Vector-mode crossing

The mixed-Proca normal modes are eigenvectors of the Gauss mass matrix (K).
When the mixing mass vanishes, sweeping (m_B) through (m_A) produces an
exact eigenmass crossing.  With nonzero mixing, the same sweep produces an
avoided crossing: the eigenvalues remain separated and the field identities
rotate continuously through an equal (A/B) mixture.  Campaign `NUL-LIN-003`
measures both controls on 81 mass samples.

For the registered (\mu^2=0.05) sweep, the minimum squared-mass separation
is (0.1), while the uncoupled control closes to floating-point zero.  At the
center of the avoided crossing, each eigenmode carries one-half of its norm in
each vector field.  This is normal-mode conversion, not energy creation.

## Cleaning decay

For Fourier wavenumber (k) and cleaning damping (\kappa), the cleaning
exponents are

\[
  s=-\frac{\kappa}{2}\pm\frac{1}{2}\sqrt{\kappa^2-4k^2}.
\]

The underdamped and overdamped regimes therefore meet at critical damping
(\kappa=2|k|).  The production RHS reproduces this relation over the
registered damping sweep.

## Electric damping

The exact frozen-fluid electric step exponentiates

\[
  D=
  \begin{pmatrix}
  -\gamma_W/Z_A & \gamma_W/Z_A\\
  \gamma_W/Z_B & -(\gamma_W+\gamma_B)/Z_B
  \end{pmatrix}.
\]

Each measured normal mode decays by \(\exp(s_D\Delta t)\), where (s_D) is an
eigenvalue of (D).  The lost vector quadratic energy is deposited as heat.
An undamped common mode exists in the registered \(\gamma_B=0\) control, so
zero heat is permitted; negative heat is not.

## Carrier drag

The production collision map targets the relative spatial four-velocity

\[
  q_D-q_N\longmapsto
  e^{-\Delta t/\tau}(q_D-q_N)
\]

while solving simultaneously for material energy, momentum, both number
densities, and entropy.  Across the registered relaxation times, the measured
decay is monotone with inverse (\tau), total conserved energy and momentum
are unchanged by the split, and entropy increases.

These lower-order processes alter finite-frequency decay and mode identity.
They do not alter the principal causal cones qualified by the characteristic
atlas.

