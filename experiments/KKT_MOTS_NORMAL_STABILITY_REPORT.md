# Normalized MOTS Stability and High-Resolution Convergence Report

> **Resolved by an independent benchmark:** direct angular evaluation removes
> the Cartesian drift and predicts
> \(d_\star/M_{\rm bare,total}=0.766197\pm0.000005\). See
> [`KKT_MOTS_DIRECT_SURFACE_REPORT.md`](KKT_MOTS_DIRECT_SURFACE_REPORT.md).
>
> **Cartesian replacement:** the parametric-surface observer and its CCZ4
> snapshot adapter are calibrated in
> [`KKT_MOTS_CARTESIAN_CALIBRATION_REPORT.md`](KKT_MOTS_CARTESIAN_CALIBRATION_REPORT.md).

## Verdict

The requested \(N\ge64\) campaign and separately normalized MOTS stability
operator are complete.

The result has two parts:

1. **Angular convergence passed at \(N=64\).** The last two converged surfaces
   predict

   \[
   d_\star^{(64,6)}=0.763231094622,
   \qquad
   d_\star^{(64,8)}=0.763342257906.
   \]

   Their difference is only \(1.11\times10^{-4}\).

2. **Cartesian convergence did not pass.** The \(N=80,\ell_{\max}=6\) attack
   moves the corresponding response crossing to approximately

   \[
   d_\star^{(80,6)}\simeq0.739224794.
   \]

The current code therefore does **not** yet have a continuum-qualified precise
common-MOTS bifurcation prediction. The earlier \(d\simeq0.7885\) raw-response
number and the new fixed-\(N\) values must remain finite-resolution results.

## Physical-normal operator

The original response matrix differentiates expansion with respect to
log-radius coefficients. The new operator instead differentiates with respect
to physical displacement \(f\) along the spatial unit normal.

For the level set

\[
F=\frac{r}{h(\theta,\phi)}-1,
\]

a log-radius perturbation has normal displacement

\[
f=\frac{\delta\log h}{|dF|_\gamma}.
\]

The implementation constructs the coefficient-to-normal-displacement map with
the induced area inner product,

\[
M_{AB}=\int_S Y_A Y_B\,dA,
\]

inverts that map, and then takes central finite differences of
\(\Theta_+\) at fixed physical normal amplitude. The resulting matrix is

\[
L_{AB}=M^{-1}_{AC}\int_S Y_C\,\delta_{f_B}\Theta_+\,dA.
\]

This removes the arbitrary log-radius amplitude from the spectrum. It is still
a finite-dimensional star-shaped discretization and is not claimed to be a
general-topology invariant operator.

## Operator verification

On the Schwarzschild regression surface:

- the principal mode is real and positive;
- the normalizer condition number is approximately one;
- normalizer round-trip error is at machine precision;
- halving the physical perturbation changes the operator by less than
  \(10^{-9}\).

On the last accepted binary surfaces at \(N=64\):

- normalizer condition numbers remain about 3.7--3.8;
- round-trip errors remain at floating-point precision;
- finite-difference relative errors remain around \(10^{-9}\);
- no accepted surface touches a KKT coefficient bound.

The stability normalization itself is therefore not the source of the
resolution drift.

## N=64 angular-order comparison

Both refined scans used a resolved-expansion tolerance of \(3\times10^{-3}\).

| \(d\) | \(\lambda_0^{(\ell_{max}=6)}\) | Found | \(\lambda_0^{(\ell_{max}=8)}\) | Found |
|---:|---:|:---:|---:|:---:|
| 0.7600 | +9.2723e-2 | yes | +9.4875e-2 | yes |
| 0.7625 | +2.0980e-2 | yes | +2.3909e-2 | yes |
| 0.7650 | +2.0336e-2 | no | -8.9731e-3 | no |
| 0.7675 | +1.3201e-2 | no | +1.0652e-3 | no |
| 0.7700 | -2.6467e-2 | no | +9.1997e-3 | no |

Only `found=yes` surfaces are used for the on-branch continuation. Linear
continuation of the final two accepted eigenvalues gives the two values quoted
above. Sign changes among failed residual minima are recorded but excluded
from the physical fit because they do not reliably remain on one MOTS branch.

The two accepted \(N=64\) estimates agree to about \(1.5\times10^{-4}\) in
relative separation. Increasing angular degree beyond eight at this grid
would not address the dominant error.

## N=80 Cartesian attack

The first \(N=80\) run started too close to the fold from a generic spherical
seed and found only off-branch minima. It was rejected as a continuation test.

A second run rebuilt continuation from \(d=0.72\):

| \(d\) | Found | Resolved residual | Full RMS expansion | Normal \(\lambda_0\) |
|---:|:---:|---:|---:|---:|
| 0.72 | yes | 3.189e-4 | 2.447e-2 | +8.7812e-2 |
| 0.74 | no | 2.025e-3 | 3.369e-2 | -3.5409e-3 |
| 0.75 | no | 1.184e-2 | 3.576e-2 | +1.305e-2 |

The first sign interpolation gives \(d\simeq0.739224794\), but because its
upper endpoint narrowly misses the acceptance threshold this is a bracketed
finite-resolution indicator, not a final branch eigenvalue.

At the same lower separations, \(N=64\) remains comfortably on branch and has
substantially different eigenvalues. This discrepancy is much larger than the
\(\ell=6\) versus \(\ell=8\) split.

An \(N=80,\ell_{max}=8\) repetition was deliberately not promoted as the next
useful calculation: angular convergence already passed at \(N=64\), while the
\(N=80,\ell=6\) run failed the Cartesian gate. More harmonics cannot repair a
grid-sampling error.

## Likely numerical boundary

The current surface evaluator:

1. constructs a level-set field on the Cartesian volume;
2. differentiates its normal with finite differences;
3. samples the resulting expansion back onto the spectral surface with
   trilinear interpolation.

The full RMS expansion decreases with higher Cartesian resolution on safely
resolved surfaces, but the small principal eigenvalue is a derivative of that
already differentiated and interpolated quantity. Near a fold, this weak mode
is much more sensitive than the expansion residual or area.

The evidence points to Cartesian level-set differentiation/interpolation as
the dominant remaining error. Physical-normal finite-difference step error is
orders of magnitude smaller.

## Reproduction artifacts

- [`../runs/kkt_mots_brill_lindquist_n64_l6_refined.json`](../runs/kkt_mots_brill_lindquist_n64_l6_refined.json)
- [`../runs/kkt_mots_brill_lindquist_n64_l8_refined.json`](../runs/kkt_mots_brill_lindquist_n64_l8_refined.json)
- [`../runs/kkt_mots_brill_lindquist_n80_l6_seeded.json`](../runs/kkt_mots_brill_lindquist_n80_l6_seeded.json)
- [`../runs/kkt_mots_normal_stability_analysis.json`](../runs/kkt_mots_normal_stability_analysis.json)

The runner accepts `--normal-stability` and
`--stability-relative-step`. The analyzer separately reports raw-response sign
changes, physical-normal sign changes, and on-branch linear zero estimates.

## Next derivation gate

The next task should no longer be a larger repetition of the same Cartesian
surface operator. It should be an independent Brill--Lindquist benchmark that
evaluates the conformal minimal-surface equation directly on the angular
surface:

\[
\Theta_+
=\psi^{-2}\left(H_{\rm flat}+4n^i\partial_i\log\psi\right),
\qquad K_{ij}=0.
\]

That removes volume-grid differentiation and trilinear sampling from the
reference problem. Once the direct-surface zero agrees across harmonic order,
it can be used to measure the Cartesian observer's convergence rate and decide
whether higher-order surface interpolation or a direct spectral geometry
kernel belongs in production.

This is a successful falsification gate: the normalized operator works, the
fixed-grid angular zero converges, and the claimed physical prediction is being
withheld because the independent Cartesian convergence requirement failed.
