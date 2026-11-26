# NUL-CHR-002 correction record

The first registered execution of NUL-CHR-002 completed normally but was
scientifically rejected. The rejection did not expose a deficient CCZ4 or
composed eigenspace: both requested ranks were already complete and the
maximum eigenspace residual was below `2e-10`.

Two diagnostic implementation defects caused four gates to fail:

1. `numpy.sort_complex` ordered nearly imaginary repeated spectra using
   roundoff-scale real parts, pairing opposite-speed modes when spectra from
   different Fourier orders were compared.
2. The numerical SVD collector admitted every singular vector below a global
   cutoff. At a repeated CCZ4 speed this could admit more vectors than the
   known algebraic multiplicity, producing a rectangular full-rank basis for
   which the audit deliberately reported infinite condition.

The corrected audit orders characteristic frequencies by imaginary part and
takes exactly the algebraic multiplicity of right-singular vectors from each
cluster. No equation, state, parameter, physical prediction, acceptance
tolerance, or raw result was changed. The rejected execution remains archived
at
`runs/nulloth_master/NUL-CHR-002-4942f745d05abc3f/attempt-001/run_record.json`.
The accepted corrected execution is
`runs/nulloth_master/NUL-CHR-002-ff54097f25f47011/attempt-001/run_record.json`.
