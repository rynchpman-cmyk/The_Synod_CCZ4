# BNS Shader Physics and MOTS Upgrade

## Implemented

- Explicit mass-scaled post-merger `(2,2)` fluid frequency.
- Default `M_total=2.7 M_sun`, `f2=3.0 kHz`, and
  `omega_22 M=0.250676956`.
- Continuous inspiral/contact/post-merger wave phase.
- Four milliseconds and 12 displayed post-merger cycles with a 2.5 ms decay
  time.
- Matter-bar rotation locked to half of the `m=2` wave phase.
- Default hypermassive-neutron-star ending with no claimed horizon.
- Separate analytic prompt-collapse shader with a formation-gated wireframe.
- General parametric Cartesian MOTS geometry for real `m != 0` harmonics.
- `causal_repair.mots_shader_bake.v2` packet and direct GLSL embedding.
- Direct-surface and CCZ4 campaign JSON adapters.
- General real-harmonic GLSL evaluation through `ell=16`.
- Independent principal-stability (`lambda_0`) wireframe coloring.
- Signed offset-expansion-response coloring for the baked repair glow.

## Validation

The default, analytic prompt-collapse, direct BL `ell=16`, and dynamical CCZ4
pilot variants compile through the local WebGL 2 preview. A fixed-time render
at `u=0.9` verifies that the high-order BL surface is raymarched as the common
horizon and carries its stability-colored wire mesh.

The parametric surface evaluator preserves the existing axisymmetric control,
recovers `Theta_+=2/R` and `A=4 pi R^2` for a round surface in flat space, and
gives matching area and expansion norms for rotated real `(2,2)` cosine and
sine deformations.

## Withheld physical claim

No solver-derived BNS horizon or expansion-color packet is published. The
current solver lacks quasi-circular BNS initial data and a qualified orbital
collapse campaign. The implemented bake interface is ready for that data but
does not substitute an analytic surface for it.

The published BL overlay is a qualified direct vacuum initial-data surface,
not a dynamical BNS horizon. The N=32 CCZ4 overlay is real trajectory output
but remains an under-resolved pilot; its manifest and bake packet say so.

See [`../shadertoy/MOTS_BAKE_SPEC.md`](../shadertoy/MOTS_BAKE_SPEC.md) for the
data contract and campaign gates.
