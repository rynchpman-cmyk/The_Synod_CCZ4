# The Coordinate Energy and Geometric Work Account of Nulloth

## Synodic identity

Campaign `NUL-GRV-008` closes the periodic balance of the evolved densitized
material-plus-vector-plus-scalar energy.  The integrated finite-volume flux
divergence cancels on the periodic grid, leaving

\[
  \frac{dE_{\rm coord}}{dt}
  =\int\sqrt{h}\left(
    \alpha K_{ij}S^{ij}-j^iD_i\alpha
  \right)d^3x.
\]

The right-hand side is evaluated at the actual projected RK stages with the
same classical RK4 weights used by production evolution.  The retained
coordinate-energy change is then compared with this integrated geometric
work.  The post-RK constitutive projection, carrier drag, and vector damping
must preserve the total evolved energy to the declared tolerance.

## Convergence and vetoes

The identity is tested in the weak-field fine state and across the 4, 8, and
16 point elevated-curvature ladder.  The elevated geometric work must approach
its fine-grid value independently of the algebraic balance closure.  CCZ4 and
field constraints, number conservation, entropy, convexity, and recovery
remain separate vetoes.

## Proscription

`E_coord` is a coordinate-slice integral of non-gravitational stress energy.
Its change measures work exchanged with the evolving geometry.  Neither side
is an ADM, Bondi, Hawking, or other global gravitational energy.  No Synodic
Article may cite this identity as global energy conservation, radiated energy,
or a gravitational-wave luminosity.
