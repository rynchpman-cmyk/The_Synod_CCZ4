# The Qualified Economy of the Brake

Status: **qualified for the declared periodic production domain**

## Writ

The Carrier Coherence Brake no longer requires the original `0.02` seed to
meet a two-percent kinetic-suppression request in its declared periodic test
geometry. The qualified thirty-two-point threshold is:

```text
last failing seed              0.0145
first passing seed             0.0150
passing suppression            2.01398 percent
bracket width                  0.0005
sixteen-point passing bound    0.0140
sixteen-to-thirty-two shift    7.14286 percent
allowed shift                  10 percent
```

The sixteen-point bracket was `[0.01325, 0.01400]`. The eight-point bracket
was rejected after shifting upward by `36.59%` at sixteen points. The final
sixteen-to-thirty-two comparison passes the tolerance fixed before execution.
Thus `0.015` is admitted as the least passing seed resolved by this search,
while `0.0145` is a demonstrated failure for the same request.

## Cause

The active trajectory evolves canonical Nulloth. The no-command control is
independently constraint-solved and receives no scalar seed. A separate
coupling-free control receives the selected scalar profile and momentum but
removes the three scalarization couplings. In that control the response is
approximately zero and has the wrong sign for a brake. The suppression is
therefore assigned to the evolved canonical couplings, not to scalar stress
alone, altered geometry alone, or an intention-dependent source.

## Accounts

All five new thirty-two-point trajectories remained admissible. Their worst
number drift was `1.85e-16`, minimum entropy increase was `2.37e-2`, maximum
Gauss L2 was `1.01e-18`, scalar constraint L2 was zero, maximum gravitational
constraint L2 was `1.29e-6`, and maximum recovery residual was `3.62e-11`.
The minimum Legendre and thermodynamic eigenvalues remained `0.5283` and
`0.3590` respectively.

## The correction in the margin

The raw campaign represented the intended `0.0005` bracket as
`0.0005000000000000004` and therefore failed its literal comparison by four
ulps. The raw denial remains preserved. A separate assessment recognized
equality within eight ulps while leaving the scientific width and all physical
tolerances unchanged.

## Limit of the writ

This economy belongs to a mode-one, one-dimensional periodic box through time
two, with the declared carrier profile, weak evolving gravity, harmonic gauge,
and ideal initial-data seeding. It does not yet specify a realizable
time-dependent actuator trajectory, tolerate targeting or timing error,
survive damaged infrastructure, or operate through a qualified finite
boundary. The working spell may use `0.015` only inside this domain; deployed
practice retains larger reserve until those trials are passed.
