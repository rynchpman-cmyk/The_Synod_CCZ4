# Nulloth integration and qualification report

## Integration status

Nulloth is integrated into `src/causal_repair` without changes to
`theory33.py` or `production33.py`. The package exports the constitutive,
scalar, production, checkpoint, discovery-preset, and characteristic-audit
interfaces.

The delivered compile command passes. The original six handoff tests and nine
repository-specific Nulloth tests pass.

## Theory 3.3 production regression

On the smallest real periodic grid (`PeriodicGrid((4,), (2.0,),
method="fd2")`):

- `initialize -> recover -> rhs` succeeds at `varphi = 0`;
- all first 24 Nulloth RHS groups agree with `production33` when the three
  scalarization couplings are zero;
- after one RK4 step, the maximum discrepancy in the first 24 groups decreases
  quadratically as the uncoupled scalar amplitude is halved.

The scalar contributes stress at nonzero amplitude even when its material
couplings are disabled, so convergence to Theory 3.3, not equality at finite
amplitude, is the correct one-step test.

## Homogeneous scalarization experiment

The frozen-material experiment obtains:

```text
m_eff^2              = -0.32000000000000006
predicted growth     = 0.5656854249492381
measured growth      = 0.5656854249492044
relative error       = 5.97e-14
```

## Full coupled principal symbol

`NullothMasterFunction.coupled_characteristic_audit` constructs a directional
25-variable generalized eigenproblem containing:

- eight material primitives and the two current-conservation plus six Carter
  vorticity equations;
- `varphi`, `Pi`, and the directional scalar gradient;
- two explicit seven-variable Maxwell-Proca cleaning blocks.

The material matrices retain the nonzero scalar-material derivative columns.
The audit also constructs the first-order constraint-reduced symbol obtained
by substituting `varphi_t = -Pi` and `D_s varphi = Phi_s`. It requires the raw
and reduced systems to have complete bases and matching spectra.

## Resolution of the apparent Jordan failure

The first audit used the raw vectors returned by `numpy.linalg.eig`. For exact
sevenfold `-1`, zero, and `+1` eigenvalues, that routine returned a duplicate
vector and falsely reported rank 24.

Direct null-space analysis shows algebraic and geometric multiplicity seven in
all three repeated sectors. The SVD-constructed basis at the formerly reported
failure has:

```text
rank                  = 25 / 25
eigensystem residual  = 6.60e-16
condition number      = 3.30
```

The corrected original scan reports:

```text
failed states                       = 0 / 216
maximum absolute characteristic     = 1.0000000000000013
maximum imaginary characteristic    = 8.01e-17
maximum eigenvector condition        = 3.8938
minimum Legendre eigenvalue          = 0.5689503486
minimum thermodynamic eigenvalue     = 0.1265654221
maximum scalar-material column norm  = 0.0675442239
```

The raw and constraint-reduced spectra agree exactly. The apparent Jordan
coupling was an eigensolver degeneracy artifact, not evidence that the seed
action needs modification.

## Activation-capable discovery domain

The locked discovery preset uses carrier coupling `12`, scalar mass `0.4`, and
quartic coupling `50`. The stronger quartic bounds the condensed amplitude
while retaining a carrier-driven tachyonic phase.

The declared sampled box is:

```text
density                   0.4 .. 0.9
specific internal energy  0.1 .. 0.35
carrier density           0.12 .. 0.24
fluid speed               0.0 .. 0.2
carrier speed             0.0 .. 0.3
varphi                    -0.16 .. 0.16
```

The campaign scanned 216 boundary states plus 1,024 seeded three-dimensional
states:

```text
failures                         = 0 / 1240
activated samples                = 1240
maximum eigenvector condition    = 54.9619
maximum reduced condition        = 3.4377
maximum eigensystem residual     = 9.71e-15
minimum Legendre eigenvalue      = 0.25840
minimum thermodynamic eigenvalue = 0.12324
minimum effective mass squared   = -0.53329
maximum equilibrium amplitude    = 0.10327
```

This qualifies the explicitly sampled box for candidate discovery. It is not
a proof for unsampled states or a qualification of any nonlinear spell.

## First discovery candidates

`discover_nulloth_linear_modes.py` tested 20 carrier-density/Fourier-mode
combinations on a real spectral `PeriodicGrid`. Ten modes were unstable and
all ten passed their growth-rate, scalar-constraint, and equilibrium-symbol
checks.

The fastest candidate has carrier density `0.24`, mode number zero, measured
growth `0.7288346863301846`, and predicted growth `0.7288346863315439`.

The leading localized candidate subsequently passed nonlinear turnover,
active-versus-control, resolution, conservation, constraint, admissibility,
and restart gates. It is promoted as Nulloth Spell 001, the Carrier Coherence
Brake; see `NULLOTH_SPELL_001_REPORT.md`.

## Verification

```text
Nulloth tests: 15 passed
Repository suite: 143 passed
```
