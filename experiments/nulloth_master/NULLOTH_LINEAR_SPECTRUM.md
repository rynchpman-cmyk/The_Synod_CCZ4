# Canonical Linear Spectrum of Nulloth

## Scope

This article gives the frozen-background dispersion relations used by campaign
`NUL-LIN-001`.  It covers three spatially homogeneous, zero-flow equilibria:

1. the stable symmetric state at carrier density (d=0.10) and
   \(\varphi_0=0\);
2. the activated symmetric state at (d=0.18) and \(\varphi_0=0\); and
3. either condensed state at (d=0.18) and
   \(\varphi_0=\pm\sqrt{-m_{\rm eff}^2/\lambda_\varphi}\).

The audit keeps the constitutive, scalar, mixed-Proca, and CCZ4 equations used
by the production solver.  The branchwise direct sum is the frozen linear
operator.  Nonlinear backreaction, nonuniform backgrounds, finite damping
steps, and finite boundaries belong to later campaigns.

## Material branches

Let (T_{ab}) and (X_{ab}(s)) be the eight-variable time and directional
space matrices obtained by differentiating the two current-conservation laws
and six Carter-vorticity equations with respect to the material primitives.
For a Fourier perturbation \(\exp(i k s-i\omega t)\),

\[
  \det\!\left(X-vT\right)=0,
  \qquad \omega_a(k)=k v_a.
\]

At each registered equilibrium the polynomial supplies two bidirectional
acoustic pairs and four zero-speed contact or transverse-vorticity modes.  The
campaign diagonalizes the same (8\times8) block embedded in the complete
physical characteristic audit rather than substituting an effective
single-fluid sound speed.

## Coherence branch

Linearizing \(\Box\varphi=m_{\rm eff}^2\varphi+lambda_\varphi\varphi^3\)
about \(\varphi_0\) gives

\[
  \omega_\varphi^2=k^2+M_\varphi^2,
  \qquad
  M_\varphi^2=m_{\rm eff}^2+3\lambda_\varphi\varphi_0^2.
\]

Thus the stable symmetric state has (M_\varphi^2=0.04), the activated
symmetric state has (M_\varphi^2=-0.2288), and the condensed state has
(M_\varphi^2=-2m_{\rm eff}^2=0.4576).  Activated modes grow when

\[
  k^2<-M_\varphi^2,
  \qquad
  \Gamma(k)=\sqrt{-M_\varphi^2-k^2}.
\]

The independently evolved three-component scalar gradient also carries three
zero-frequency reduction-constraint modes in a one-dimensional Fourier
slice.  These are constraints, not additional coherent particles.

For a periodic length (L), the nonhomogeneous mode numbers satisfy
(k_j=2\pi j/L).  The first nonhomogeneous activated mode exists only for

\[
  L>L_{\rm crit}=\frac{2\pi}{\sqrt{-M_\varphi^2}}.
\]

## Mixed-Proca and cleaning branches

The two vector mass eigenvalues are the eigenvalues \(\mu_\pm^2\) of the
production Gauss matrix

\[
  K=
  \begin{pmatrix}
    m_A^2+\mu^2/Z_A & -\mu^2/Z_A\\
    -\mu^2/Z_B & m_B^2+\mu^2/Z_B
  \end{pmatrix}.
\]

Each eigenmass supplies three physical polarizations and the dispersion

\[
  \omega_{\pm}^2=k^2+\mu_\pm^2.
\]

For cleaning damping \(\kappa\), each of the two vector fields also supplies a
cleaning pair with evolution exponents

\[
  s_{\rm clean}=-\frac{\kappa}{2}
  \mathbin{\pm}\frac{1}{2}\sqrt{\kappa^2-4k^2}.
\]

Mass mixing and target-current mismatch remain lower order: they set the
finite-frequency eigenmasses and sources but not the light-speed principal
cones.

## CCZ4 branches

With harmonic slicing and the registered Gamma driver, the algebraically
reduced 23-variable CCZ4 principal generator contains four transverse-traceless
gravitational tensor modes at speeds \(\pm1\), two Gamma-driver modes at
speeds \(\pm\sqrt{4/3}\), and seventeen residual gauge or constraint modes.
The residual family comprises fourteen light-speed and three zero-speed
modes.  The superluminal coordinate gauge pair is not a physical causal cone.

## Numerical audit

The campaign constructs Fourier generators directly from the scalar,
mixed-Proca, and CCZ4 right-hand sides on spectral `PeriodicGrid` instances at
16, 24, and 32 points.  It compares their eigenvalues with the equations above
at multiple mode numbers.  The material block is obtained from the canonical
constitutive Jacobian.  Agreement across this campaign qualifies the named
linear branches only within the stated homogeneous, periodic, frozen-coefficient
domain.

