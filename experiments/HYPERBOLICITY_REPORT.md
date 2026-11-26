# Euler–Proca characteristic-mixing audit

## Bottom line

The density-dependent effective mass

\[
M_{\rm eff}^2(n_b)=m_A^2+\frac{\chi n_b}{Z}
\]

does **not by itself alter the principal Proca light cone**.  It multiplies
undifferentiated fields in the covariant repair equation, so for a frozen
physical state it is a lower-order mass/source coefficient.  The high-frequency
Proca characteristics remain metric-null, while the uncoupled ideal-fluid
characteristics are the usual contact and relativistic acoustic modes.

The implemented conservative system nevertheless develops characteristic
mixing when the repair field lags its target.  The interaction stress depends
on `chi*n_b*(A-Astar)^2`, and the production recovery reconstructs `Phi` using
`M_eff^2(n_b)`.  Consequently the total-matter flux Jacobian changes with
density, velocity, and mismatch.  The numerical audit finds:

- the standard evolved shock remains locally real and diagonalizable;
- sufficiently large mismatch pushes acoustic modes outside the metric light
  cone even while they remain real;
- cold, highly boosted mismatch states develop a complex-conjugate eigenvalue
  pair, so the unrestricted system cannot currently be called strongly
  hyperbolic.

## What was tested

The audit evolves a 256-cell relativistic shock tube to `T=0.12` with the GRHD
solver, then evaluates the generalized directional Jacobian

\[
J_x=\frac{\partial F_x}{\partial P}
    \left(\frac{\partial Q}{\partial P}\right)^{-1}
\]

of the total fluid plus repair energy-momentum block at sampled shock states.
The primitive coordinates use log density, log internal energy, and spatial
four-velocity, avoiding finite-difference excursions outside the physical
state domain.  A state is flagged when an eigenvalue has imaginary part above
`1e-7` or the eigenvector matrix becomes numerically singular.

This is a frozen-Proca, local necessary-condition audit.  It is not a complete
first-order reduction or symmetrizer proof for the full CCZ4–GRHD–target–Proca
system.

## Standard shock result

The evolved shock reached:

- density range: `0.125` to `1.0`;
- maximum density gradient: `14.1753`;
- maximum fluid speed: `0.43998`;
- no primitive-recovery failures.

For `m_A=0.5`, the effective-mass-squared ranges were:

| coupling | range of `M_eff^2` across shock |
|---:|---:|
| 10 | 1.50 – 10.25 |
| 30 | 4.00 – 30.25 |
| 100 | 12.75 – 100.25 |

Despite this large density-driven variation, all sampled standard-shock
Jacobians remained real and diagonalizable for target amplitudes through 3.0
with `A=Astar/2`.  Thus high gradient magnitude alone did not destroy local
hyperbolicity.

The fastest eigenvalue did exceed the metric light speed at the following
first sampled amplitudes:

| coupling | first `|Astar|` with `max|lambda|>1` |
|---:|---:|
| 10 | 1.35 |
| 30 | 0.75 |
| 100 | 0.40 |

Superluminality is distinct from loss of strong hyperbolicity: these modes are
still real, but the GRHD HLL light/sound-speed bound is no longer sufficient.

## Cold boosted upstream result

The more dangerous regime is a cold upstream state with density `0.125`,
specific internal energy `0.10`, a longitudinal target, and a finite mismatch
`A=Astar/2`.  At `|Astar|=0.10`, the first sampled velocities with a non-real
pair were approximately:

| coupling | onset velocity |
|---:|---:|
| 10 | 0.675 |
| 30 | 0.350 |
| 100 | 0.175 |

For example, at `g=30`, `|Astar|=0.10`, and `v=0.70`, the acoustic pair has an
imaginary part about `0.078`.  The result is stable against changing the
finite-difference step by more than three orders of magnitude.  The loss forms
a bounded island in mismatch/velocity space: at very large amplitudes the
tested eigenvalues can become real again, generally with superluminal speeds.

## The key control

When `A=Astar`, the same maximum-velocity shock state remains real and well
conditioned even at `g=100` and `|Astar|=2`.  Its fastest local mode is only
about `0.880`.  This isolates the mechanism:

1. `M_eff^2(n_b)` alone is not changing the principal cone.
2. Target lag creates interaction energy, momentum, and anisotropic stress.
3. Longitudinal fluid motion mixes those contributions into the acoustic
   Jacobian.
4. Beyond a state-dependent threshold, the acoustic roots coalesce and become
   complex.

## Original formulation caveat and implemented repair

The original production path eliminated `Phi` using

\[
\Phi=\frac{\rho_* - D_iE^i}{M_{\rm eff}^2(n_b)}.
\]

The subsequent evolution differentiated this reconstructed quantity, creating
avoidable density–longitudinal-field derivative mixing at shocks.

Production now evolves

\[
\mathcal Q_A=\sqrt\gamma(M_{\rm eff}^2\Phi-\rho_*)
\]

using `nabla_mu(M_eff^2 A^mu-J_star^mu)=0`.  It reconstructs
`Phi=(Q_A/sqrt(gamma)+rho_star)/M_eff^2` locally, without `div(E)`, and evolves
a damped Gauss-cleaning field.  Compatible discrete divergences preserve the
generalized Gauss constraint to roundoff in the discontinuity regression.

The original split flux also used uncoupled Valencia HLL speeds and appended
repair flux afterward.  Production now uses one conservative Rusanov interface
flux for the complete matter state.  Its bound includes Valencia acoustic
speeds, metric light speed, and an overestimated mismatch deformation; the CFL
timestep uses the same bound.

Therefore the accurate production statement is:

> The avoidable Gauss-elimination and split-flux defects are repaired.  Strong
> hyperbolicity is still not global: cold boosted states with longitudinal
> mismatch are a genuine admissibility boundary of the unchanged quadratic
> interaction.

## Recommended formulation work

Before using this sector for unrestricted relativistic shocks:

1. The independent longitudinal variable and Gauss cleaner are now implemented;
   retain their compatible-divergence regression as the system evolves.
2. Derive the full first-order Euler–Proca–target principal symbol and an
   admissible-state/convexity condition in terms of mismatch, enthalpy, and
   coupling.
3. Replace the implemented conservative empirical Rusanov bound with a proven
   global bound if one can be established.
4. Add a runtime characteristic guard that rejects complex eigenvalues and
   reports approaching eigenvector ill-conditioning.
5. Repeat shock convergence tests with the dynamically coupled Riemann flux.

## Artifacts

- `experiments/audit_euler_proca_hyperbolicity.py`
- `runs/hyperbolicity_audit/characteristic_audit.json`
- `runs/hyperbolicity_audit/characteristic_audit.png`

Reproduce with:

```powershell
$env:PYTHONPATH='src'
python experiments\audit_euler_proca_hyperbolicity.py
```
