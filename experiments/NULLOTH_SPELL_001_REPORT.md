# Nulloth Spell 001: Carrier Coherence Brake

Status: **promoted inside the declared periodic-box discovery domain**

## Operational definition

The Carrier Coherence Brake is a dynamically growing, mode-one Nulloth
coherence field supported by a modulated carrier distribution. Relative to an
otherwise identical scalar-free control, it suppresses the carrier kinetic
response while remaining conservative, admissible, causal in the sampled
domain, and reproducible across restart.

This is a brake or shielding effect. It is not propulsion and does not create
energy.

## Initial configuration

```text
box length                 32
fluid density              0.6
specific internal energy   0.2
carrier density            0.18 + 0.04 cos(2 pi x / 32)
initial varphi              0.02 cos(2 pi x / 32)
initial Pi                  growing-mode value
gravity coupling            1e-3
geometry                    constraint-solved periodic CMC
```

The material, scalar, both mixed-Proca fields, cleaners, and CCZ4 geometry are
evolved by `NullothProductionSolver`.

## Nonlinear branch test

At eight points through `t = 6`:

```text
initial scalar amplitude        0.0200000
peak scalar amplitude           0.1424057
final scalar amplitude          0.1272720
peak-to-final drop              10.63 percent
late fitted growth              0.05252
declared amplitude limit        0.17
```

The field therefore reaches the nonlinear quartic branch and turns over. It
does not merely continue the fitted linear exponential.

At the end of the nonlinear run:

```text
baryon relative drift           3.70e-16
carrier relative drift          4.63e-16
scalar constraint L2            0
Gauss maximum L2                1.36e-18
Hamiltonian L2                  7.80e-6
momentum-constraint L2          7.52e-6
minimum Legendre eigenvalue     0.3650
minimum thermodynamic eigenvalue 0.2858
restart maximum error           2.46e-14
```

## Active-versus-control effect

The effect was compared against independently constraint-solved scalar-free
controls at the same resolutions. At `t = 2`:

| Points | Scalar amplitude | Carrier velocity effect L2 | Kinetic effect | Relative kinetic effect |
|---:|---:|---:|---:|---:|
| 4 | 0.064692 | 2.619e-4 | -8.204e-5 | -2.94% |
| 8 | 0.064933 | 1.105e-3 | -3.195e-4 | -7.10% |
| 16 | 0.065264 | 1.586e-3 | -1.792e-4 | -3.83% |
| 32 | 0.065451 | 1.598e-3 | -1.634e-4 | -3.39% |

From 16 to 32 points:

- scalar amplitude changes by 0.29%;
- carrier velocity effect changes by 0.73%;
- the kinetic effect retains its suppressive sign and changes by about 9%;
- Hamiltonian L2 decreases from `2.53e-6` to `1.29e-6`.

The control-subtracted carrier-velocity response is resolved. The carrier
density difference converges more slowly and is not used as the primary spell
observable.

## Promotion gates

- carrier-driven emergence: pass;
- full production evolution: pass;
- nonlinear turnover: pass;
- active-versus-control effect: pass;
- resolution stability of primary observable: pass;
- baryon and carrier conservation: pass;
- entropy nondecrease: pass;
- scalar and Proca constraints: pass;
- gravitational constraint convergence: pass;
- constitutive admissibility: pass;
- checkpoint/restart reproducibility: pass;
- sampled coupled-symbol qualification: pass.

## Scope

Promotion applies to the declared one-dimensional periodic-box campaign with
weak but evolving gravity. It does not establish stability in finite domains,
three dimensions, shocks, strong gravity, or arbitrarily long evolutions.
Those are expansion campaigns for this spell, not assumptions built into its
definition.
