# Active-repair prediction report

## Question

Can the completed runs predict quantitative dynamics at parameter combinations
that the solver has not yet evaluated?

## Protocol

1. The 54-case exploration sweep was reduced to 45 unique physical regimes so
   duplicate zero-coupling controls could not leak across validation folds.
2. Predictor inputs were `log(m_A)`, `log(m_A tau_star)`, and
   `log(1 + g rho_peak / m_A^2)`.
3. A small NumPy radial-basis kernel surrogate was selected using grouped
   leave-one-regime-out validation.
4. Forecasts for eight unseen, interpolating parameter combinations were written
   to `runs/repair_predictions/locked_forecasts.json` before any of those cases
   were evolved.
5. The solver then resolved all eight cases at `N=80`. Three representative
   cases were subsequently checked at `N=160`.

## Blind forecast performance

| Observable | Blind MAE | Correlation |
|---|---:|---:|
| Peak mismatch / initial field | 0.0265 | 0.970 |
| Peak interaction / bare Proca energy | 0.0172 | 0.994 |
| Maximum exterior energy fraction | 0.0255 | 0.989 |
| Late field fraction | 0.0225 | 0.944 |
| Phase-reversal count | 0.25 crossings | 0.908 |

The integer phase-reversal count was predicted exactly in six of eight blind
cases and missed by one in the other two. All blind evolutions remained stable;
the final reconstructed Gauss residual was `1e-18` or smaller.

Notable successful forecasts:

- `heavy_uncoupled`: mismatch overshoot `1.114` predicted, `1.098` measured;
- `heavy_fast_loaded`: interaction peak `0.274` predicted, `0.284` measured,
  with five reversals predicted and measured;
- `adiabatic_dense`: mismatch peak `1.0093` predicted, `1.0070` measured,
  with four reversals predicted and measured;
- `moderate_wake`: interaction peak `0.1026` predicted, `0.1060` measured.

The weakest observable is outward transport near the trapping transition. The
largest exterior-fraction error was `0.080`, so the current surrogate should not
be used for precision forecasts of trapping boundaries.

## Interpretable reversal law

The field equations contain the local effective mass

```text
m_eff^2 = m_A^2 + g rho / Z,
```

with `Z=1` in these runs. A simple phase-count estimate follows:

```text
N_flip ~= floor(T sqrt(m_A^2 + g rho_peak) / pi) + 1.
```

Without a fitted coefficient, this predicts six of eight blind reversal counts
exactly and misses the remaining two by one. Across the 54 training cases plus
eight blind cases it is exact for about 85% of regimes, with mean absolute error
`0.177` crossings. This supports the interpretation that localized matter
loading changes the ringing clock primarily through the theory's effective
Proca mass.

## Resolution audit

For `heavy_fast_loaded`, `near_trap_edge`, and `adiabatic_dense`, increasing
resolution from 80 to 160 cells preserved every reversal count. Peak mismatch
changed by at most `0.0031`, interaction ratio by at most `0.0040`, and late
field fraction by at most `0.0028`. The largest change was `0.0125` in exterior
energy for the adiabatic case.

## Scope

These are numerical predictions of the specified fixed-background theory, not
validated predictions about nature. The training and blind runs share the same
two-lobe family, target amplitude, width, and final time. Extrapolation to other
initial-data families, nonlinear gravity, or parameter ranges outside the scan
requires new validation.
