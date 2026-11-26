# Campaign correction NUL-SPL-001-CR-001

Status: recorded methodological supersession  
Raw campaign: `NUL-SPL-001`  
Replacement assessment: `NUL-SPL-001-ASSESS-001`

## Error

The raw campaign required `promoted_resolutions == 4` for four short runs at
`t = 2`. The underlying promotion function correctly withholds full promotion
from those runs because they have not yet reached the nonlinear condensed
branch or demonstrated turnover.

The short runs exist to test resolution stability of the active-versus-control
effect. Nonlinear turnover is tested by the separate `t = 6` run. Requiring
each short run to pass the long-run gates conflated two stages of the declared
promotion protocol.

## Correction

The replacement assessment requires all four short runs to pass every gate
applicable before nonlinear turnover. Their only permitted rejection reasons
are the two gates explicitly deferred to the long run:

- coherence did not reach the nonlinear condensed branch;
- coherence did not demonstrate nonlinear turnover.

The long run must still pass full promotion. Every numerical tolerance for
resolution stability, conservation, constraints, admissibility, restart,
amplitude, and turnover is unchanged.

## Preservation

The raw solver execution and its rejected scientific assessment remain
unaltered at:

`runs/nulloth_master/NUL-SPL-001-3aec69fdbb40b1de/attempt-001/run_record.json`

The earlier import-path failure also remains preserved under its separate run
identity. No raw trajectory was rerun or discarded for this correction.

