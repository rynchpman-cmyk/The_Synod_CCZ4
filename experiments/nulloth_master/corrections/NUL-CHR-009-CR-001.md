# NUL-CHR-009 correction record

The first execution terminated before producing scientific observables because
one of four intended parameter controls assigned zero Proca masses. The
canonical `Theory3Parameters` contract requires both masses to be strictly
positive.

That invalid control was replaced with equal masses of `0.05`. Masses remain
algebraic and therefore absent from the tested principal block. No question,
parameter-set count, field, source orientation, physical prediction,
acceptance criterion, or tolerance changed.

The failed setup record remains at
`runs/nulloth_master/NUL-CHR-009-e75d77e6a4b52647/attempt-001/run_record.json`.
The accepted record is
`runs/nulloth_master/NUL-CHR-009-b8b84f93aca7842d/attempt-001/run_record.json`.
