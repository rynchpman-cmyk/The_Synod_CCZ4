# The Brake Requires Alignment

Status: **the registered full robustness envelope is denied**

## Judgment

The conservative `0.02` seed possesses useful reserve, but not enough to
tolerate a `pi/8` alignment error while guaranteeing two-percent suppression.

```text
sealed nominal                       3.23571 percent
amplitude minus 10 percent           2.77168 percent   pass
amplitude plus 10 percent            3.79801 percent   pass
phase minus pi/8                     1.98618 percent   fail
phase plus pi/8                      1.98618 percent   fail
second harmonic at 10 percent        4.18251 percent   pass
growing momentum underdrive 20 pct   2.84084 percent   pass
```

Both phase errors fail by approximately `0.0138` percentage points. Their
relative asymmetry is `9.0e-13`. We therefore assign the loss to scalar/carrier
misalignment, not to grid direction.

## What remains licensed

Within this sampled periodic domain, the working seed tolerates independent
amplitude error of ten percent, deterministic second-harmonic shape error of
ten percent at fixed peak amplitude, and twenty-percent underdrive of its
growing-mode momentum. These are observed robustness points, not yet complete
tolerance radii.

The second-harmonic perturbation increases suppression. It is not promoted as
an improved command because no inverse search, resolution comparison, or
combined-error test has yet established that advantage.

## Safety classification

Every trajectory remained admissible. Maximum number drift was `1.85e-16`,
minimum entropy increase was `2.36e-2`, maximum gravitational-constraint L2
was `1.29e-6`, and maximum recovery residual was `3.79e-11`. The phase-error
branches are safe failures of the requested objective, not unsafe states.

The Synod shall therefore weaken, realign, or deny a two-percent brake request
when predicted phase error approaches `pi/8`; it shall not classify the same
trajectory as a runtime emergency merely because the requested suppression is
missed.

## Next writ

The phase boundary must be bracketed below `pi/8`. After that come combined
worst-case targeting errors, stochastic sensor and feedback noise, real causal
delay through finite incoming data, damaged routes, and a shutdown trajectory.
