# Campaign correction NUL-CTL-004-CR-001

Status: recorded numerical-representation supersession  
Raw campaign: `NUL-CTL-004`  
Replacement assessment: `NUL-CTL-004-ASSESS-001`

## Error

The raw campaign required the final bracket width to be no greater than
`0.0005`. Four exact bisections of the declared decimal interval
`[0.012, 0.020]` produce that width. Binary floating arithmetic recorded the
subtraction as `0.0005000000000000004`, an excess of approximately
`4.34e-19`, or four ulps of the declared bound.

Every other registered criterion passed. The physical crossing was
`[0.0145, 0.0150]`; the upper bound differed from the sixteen-point reference
by `7.14%`, within the predeclared ten-percent allowance.

## Correction

The replacement assessment recognizes representational equality when the
width exceeds the unchanged `0.0005` bound by no more than eight ulps of that
bound. It requires the width criterion to be the sole raw failure and every
non-width criterion to have passed.

This is not an enlarged scientific bracket. The declared width remains
`0.0005`; no physical, convergence, conservation, entropy, constraint,
recovery, or admissibility tolerance changes.

## Preservation

The raw production execution and its rejected literal floating comparison
remain unaltered at:

`runs/nulloth_master/NUL-CTL-004-4ceb45997c80a0b3/attempt-001/run_record.json`
