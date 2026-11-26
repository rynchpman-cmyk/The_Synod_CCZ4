# Campaign classification NUL-CTL-007-CR-001

Status: recorded invalid-control and physical-denial classification  
Raw campaign: `NUL-CTL-007`  
Replacement assessment: `NUL-CTL-007-ASSESS-001`

## Result

The raw campaign predicted that a negative ten-percent second harmonic would
remain a valid isolated control before being combined with phase, amplitude,
and momentum errors. It did not. The negative-sign branch produced `1.53251%`
suppression and missed the fixed two-percent objective.

The previously measured positive-sign branch produced `4.18251%`. The
`2.65`-percentage-point sign difference forbids treating a single deterministic
second-harmonic realization as evidence of generic noise robustness.

Two phase-error pairs failed. Phase plus momentum underdrive narrowly passed
at `2.07279%`; phase plus low amplitude failed at `1.90447%`; phase plus
negative shape failed at `0.64399%`. The all-error corner failed at `0.64362%`.
All trajectories remained admissible and passed every independent safety gate.

## Classification

The replacement assessment does not rescue the raw combined-control premise.
It records the shape-sign denial, the safe joint failures, and the one passing
phase-plus-momentum pair. Because the negative-shape input already failed in
isolation, the all-error loss cannot be assigned solely to nonlinear error
interaction.

No objective, tolerance, input, or trajectory is changed.

## Preservation

The raw rejected record remains unaltered at:

`runs/nulloth_master/NUL-CTL-007-1d74e7cf52075281/attempt-001/run_record.json`
