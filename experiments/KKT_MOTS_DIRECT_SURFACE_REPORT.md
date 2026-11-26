# Direct-Surface Brill--Lindquist Bifurcation Prediction

## Result

The common-MOTS bifurcation in the equal-mass, time-symmetric
Brill--Lindquist family is predicted at

\[
\boxed{
\frac{d_\star}{M_{\rm bare,total}}
=0.766197\pm0.000005
}
\]

for punctures placed at \(z=\pm d/2\).

The highest completed harmonic orders give

\[
d_\star^{(\ell_{\max}=14)}=0.766198642,
\qquad
d_\star^{(\ell_{\max}=16)}=0.766195004.
\]

Their difference is \(3.64\times10^{-6}\). Direct branch finding independently
brackets the transition between \(d=0.76615\) and \(d=0.76620\).

This is a prediction for a one-parameter initial-data family, not a merger time
or a claim that apparent horizons are slicing invariant.

## Independent surface equation

The reference evaluator never constructs a volume level-set field. For

\[
\gamma_{ij}=\psi^4\delta_{ij},
\qquad K_{ij}=0,
\]

the outgoing expansion is evaluated directly as

\[
\Theta_+
=\psi^{-2}\left(H_{\rm flat}
+4n^i\partial_i\log\psi\right).
\]

The equal-mass Brill--Lindquist conformal factor is

\[
\psi
=1+\frac{M_{\rm bare,total}/2}{2r_+}
  +\frac{M_{\rm bare,total}/2}{2r_-}.
\]

For the axisymmetric radial graph \(r=h(\theta)\), define

\[
q=\sqrt{h^2+h_\theta^2}.
\]

The outward flat unit normal and flat mean curvature are

\[
n=\frac{h\,e_r-h_\theta e_\theta}{q},
\]

\[
H_{\rm flat}
=\frac{
2h^2+3h_\theta^2-hh_{\theta\theta}
-\frac{h_\theta}{h}(h^2+h_\theta^2)\cot\theta
}{q^3}.
\]

The induced physical area measure is

\[
dA=\psi^4 hq\,d\Omega.
\]

All angular derivatives are analytic derivatives of the Legendre/harmonic
basis. No Cartesian finite difference or trilinear interpolation enters this
calculation.

Primary implementation:
[`src/causal_repair/kkt_horizon.py`](../src/causal_repair/kkt_horizon.py).

## Exact analytic gate

At \(d=0\), the two punctures coincide and the data reduce to isotropic
Schwarzschild with total mass \(M\). The direct evaluator recovers

\[
r_H=M/2,
\qquad
A_H=16\pi M^2,
\qquad
M_{\rm irr}=M.
\]

For the even axisymmetric modes \(\ell=0,2,4,\ldots\), the normalized operator
recovers the exact discrete spectrum

\[
\lambda_\ell=\frac{\ell(\ell+1)+1}{4M^2},
\]

including \(0.25\), \(1.75\), and \(5.25\) for \(M=1\). The regression test
checks the horizon, area, mass, expansion, normalization closure, and these
first three eigenvalues.

## Stability normalization

The same physical-normal operator used in the Cartesian audit is applied to
the direct surface. Coefficient perturbations are converted to fixed physical
normal displacement and projected with the induced area measure.

Near a generic fold, the soft principal mode obeys

\[
\lambda_0\propto\sqrt{d_\star-d}.
\]

Therefore the bifurcation estimator fits \(\lambda_0^2\) linearly against
separation using the final two converged on-branch surfaces. Linear fits of
\(\lambda_0\) itself remain in the analyzer as diagnostics but are not used as
the preferred fold estimate.

Failed residual minima beyond the branch endpoint are also excluded from the
fit, even when their response matrix happens to contain a small eigenvalue.

## Angular convergence

| Maximum degree | Fold-scaled estimate |
|---:|---:|
| 6 | 0.766411328 |
| 8 | 0.765219551 |
| 10 | 0.766203337 |
| 12 | 0.766139567 |
| 14 | 0.766198642 |
| 16 | 0.766195004 |

Degree eight is a non-monotonic truncation outlier near the sharply pinched
inner portion of the common surface. The degree-10 through degree-16 sequence
returns to a stable high-order limit. The degree-14/16 spread sets the quoted
angular uncertainty; the final value is rounded conservatively to
\(\pm5\times10^{-6}\).

For both degrees 14 and 16:

- \(d=0.76615\) converges with resolved residual below \(10^{-9}\);
- \(d=0.76620\) does not satisfy the \(10^{-7}\) acceptance gate;
- no accepted surface reaches a KKT coefficient bound;
- the normalized principal eigenvalue remains positive on every accepted
  surface and approaches zero toward the bracket.

## Relationship to the Cartesian audit

The earlier Cartesian observer gave grid-dependent fold locations because it
differentiated a volume level set and sampled the result back onto the surface.
The direct benchmark removes both operations and converges rapidly in harmonic
order.

That does not make the Cartesian observer useless. It now has a precise oracle:

\[
d_\star/M_{\rm bare,total}=0.766197\pm0.000005.
\]

Future Cartesian work can measure its error against this value and determine
whether direct surface geometry or higher-order interpolation should replace
the existing diagnostic kernel in dynamical CCZ4 runs.

## Reproduction

Run the high-order scans with:

```powershell
$separations = "0.7655,0.7660,0.7661,0.76615,0.7662,0.76625,0.7663"

python experiments/run_kkt_mots_brill_lindquist_direct.py `
  --maximum-degree 14 --n-theta 64 `
  --separations $separations `
  --output runs/kkt_mots_brill_lindquist_direct_l14_refined.json

python experiments/run_kkt_mots_brill_lindquist_direct.py `
  --maximum-degree 16 --n-theta 72 `
  --separations $separations `
  --output runs/kkt_mots_brill_lindquist_direct_l16_refined.json
```

Analyze them with:

```powershell
python experiments/analyze_kkt_mots_brill_lindquist.py `
  runs/kkt_mots_brill_lindquist_direct_l14_refined.json `
  runs/kkt_mots_brill_lindquist_direct_l16_refined.json `
  --output runs/kkt_mots_brill_lindquist_direct_refined_analysis.json
```

Primary artifacts:

- [`../runs/kkt_mots_brill_lindquist_direct_l14_refined.json`](../runs/kkt_mots_brill_lindquist_direct_l14_refined.json)
- [`../runs/kkt_mots_brill_lindquist_direct_l16_refined.json`](../runs/kkt_mots_brill_lindquist_direct_l16_refined.json)
- [`../runs/kkt_mots_brill_lindquist_direct_refined_analysis.json`](../runs/kkt_mots_brill_lindquist_direct_refined_analysis.json)

## Next physical use

The direct benchmark closes the static derivation gate. The next useful step is
to use it as a calibration oracle for the Cartesian observer and then attach
the corrected observer to actual CCZ4 puncture snapshots. In a dynamical run,
the prediction to test is not the static number \(0.766197\) itself, but whether
the physical-normal principal mode approaches zero at common-MOTS branch
creation and whether the independently tracked inner and outer branches behave
as the stability spectrum predicts.
