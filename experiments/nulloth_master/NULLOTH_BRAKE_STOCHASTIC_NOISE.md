# The Brake Is Not Guaranteed Against Shape Noise

Status: **worst-case ten-percent RMS shape-noise guarantee denied**

## Ensemble

Eight predeclared multimode profiles perturbed scalar modes two through five
with independent signed Gaussian coefficients. Every realization had exactly
ten-percent RMS noise relative to the fundamental and was renormalized to the
same peak seed amplitude `0.02`.

```text
realizations passing two percent    7 of 8
empirical pass fraction             0.875
minimum suppression                 1.31965 percent
mean suppression                    2.91626 percent
maximum suppression                 3.79316 percent
sample standard deviation           0.87648 percentage points
95 percent Wilson interval          0.529 to 0.978
```

The individual answers were:

| Seed | Relative suppression | Decision |
|---:|---:|:---|
| 1103 | 3.68636% | pass |
| 2207 | 2.62105% | pass |
| 3313 | 3.79316% | pass |
| 4421 | 3.37363% | pass |
| 5527 | 2.01697% | pass, narrow |
| 6637 | 3.00916% | pass |
| 7753 | 3.51010% | pass |
| 8861 | 1.31965% | fail |

One failure is sufficient to deny a worst-case guarantee. Seven successes are
not sufficient to license a population pass probability. The Wilson interval
is reported to show how little eight trajectories determine about that rate.

## Safety

Every realization remained admissible. Maximum number drift was `3.70e-16`,
minimum entropy increase was `2.37e-2`, maximum gravitational-constraint L2
was `1.29e-6`, and maximum recovery residual was `3.41e-11`. Seed `8861` is a
safe miss of the requested objective.

## Synodic use

Open-loop initial-shape uncertainty of ten-percent RMS is not an approved
guarantee for a two-percent brake at amplitude `0.02`. The Synod may increase
reserve only through a separately qualified robust optimization, or it may
measure the realized profile and weaken or deny the request. It shall not quote
`87.5%` as a deployment probability.

## Reservation

This ensemble samples eight mode-2-through-5 initial profiles. It does not
represent live sensor noise, feedback error, other spectra, temporal noise,
finite boundaries, or damaged hardware. Population qualification requires
more samples or a reduced model first validated against these sealed
production trajectories.
