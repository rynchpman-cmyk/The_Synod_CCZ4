# MOTS Shader Bake Contract

## Purpose

`causal_repair.mots_shader_bake.v2` carries a compact, time-dependent apparent
horizon from an offline CCZ4 campaign into one standalone ShaderToy program.
It supports real spherical harmonics with all `m` values through `ell=16`, so a
bar-like or otherwise non-axisymmetric common horizon is not forced into an
axisymmetric radial embedding.

The packet separately carries the physical-normal principal stability
eigenvalue `lambda_0` and a signed outgoing-expansion **response**. The response must be
measured on prescribed inward/outward offset surfaces, or supplied as an
equivalently normalized physical-normal response. It must not be described as
the expansion on the MOTS itself: a converged MOTS has `Theta_+=0`.

## JSON format

```json
{
  "format": "causal_repair.mots_shader_bake.v2",
  "normalized_times": [0.86, 0.90, 0.94],
  "centers": [[0, 0, 0], [0.01, 0, 0], [0.02, -0.01, 0]],
  "modes": [[0, 0], [2, -2], [2, 0], [2, 2]],
  "log_radius_coefficients": [
    [0.69, 0.02, -0.03, 0.08],
    [0.66, 0.01, -0.04, 0.06],
    [0.62, -0.01, -0.05, 0.03]
  ],
  "expansion_response_coefficients": [
    [0, -0.4, 0.1, 0.5],
    [0, -0.3, 0.08, 0.4],
    [0, -0.2, 0.05, 0.25]
  ],
  "principal_eigenvalues": [0.08, 0.03, 0.004],
  "display_scale": 1.0,
  "source_kind": "ccz4_puncture_campaign",
  "source_label": "campaign/trajectory.json:common_outer",
  "qualified": true
}
```

The shape convention is

```text
log R(theta,phi,t) = sum_lm c_lm(t) Y_lm(theta,phi).
```

Positive `m` is the real cosine harmonic and negative `m` is the real sine
harmonic, matching `causal_repair.kkt_horizon.real_spherical_harmonic`.
Normalized times must be strictly increasing in `[0,1]`. Packets contain 2 to
16 frames. Centers are in the same total-mass coordinate units as the shader.
`display_scale` is an explicit uniform presentation transform; it changes
neither the stored modal coefficients nor `lambda_0`. Version-1 packets remain
readable and receive zero principal eigenvalues and unit display scale.

## Producing a packet

For a sequence of qualified `SpectralMOTSResult` objects with identical mode
sets:

```python
from causal_repair.bns_shadertoy import MOTSShaderBake

bake = MOTSShaderBake.from_results(
    normalized_times,
    mots_results,
    offset_expansion_response_coefficients,
    principal_eigenvalues,
)
bake.write_json("runs/bns_collapse/mots_shader_bake.json")
```

Each result must have `found=True`. Campaign-level resolution, phase, and mode
qualification must be checked before calling this a physical horizon bake.

Generate the final standalone shader with:

```powershell
python shadertoy/build_bns_shader.py `
  --mots-bake runs/bns_collapse/mots_shader_bake.json `
  --output shadertoy/BNS_SOLVER_BAKED.glsl
```

Supplying a bake automatically enables the prompt-collapse branch. The shader
interpolates centers and coefficients, uses the radial graph as the raymarched
horizon, and colors the wireframe from `lambda_0`: cyan is positive/stable,
magenta is negative/unstable, and gold is the approach to zero/the fold. The
separate response field splits repair glow into cyan outward/untrapped and
magenta inward/trapped components when offset-surface data actually exists.

## Campaign adapters and checked-in overlays

The preprocessing step accepts both horizon schemas already produced by this
repository:

```powershell
# Converged direct Brill-Lindquist common surfaces, including ell=16
python shadertoy/bake_mots_campaign.py direct `
  runs/kkt_mots_brill_lindquist_direct_l16_refined.json `
  shadertoy/data/BL_DIRECT_L16_BAKE.json `
  --start 0.82 --end 0.94 --display-scale 6

# Dynamical common-outer CCZ4 pilot; the override is required because this
# run fails the existing Cartesian resolution gate.
python shadertoy/bake_mots_campaign.py ccz4 `
  runs/ccz4_puncture_mots_lapsefix_n32_early/trajectory.json `
  shadertoy/data/CCZ4_N32_OUTER_PILOT_BAKE.json `
  --branch common_outer --allow-unqualified --display-scale 6
```

Failed direct-surface points beyond the fold are never baked. By default the
CCZ4 adapter also rejects any surface whose principal-mode sign gate failed;
`--allow-unqualified` is a visualization-only escape hatch and puts
`UNDER-RESOLVED PILOT` in the packet source label.

`BNS_BL_BIFURCATION_OVERLAY.glsl` contains the four converged common surfaces
at separations `0.7655`, `0.766`, `0.7661`, and `0.76615`, with their unmodified
`ell=0,2,...,16` log-radius coefficients and principal eigenvalues. Its
sixfold uniform display scale makes the coordinate-small BL surface legible
in the BNS camera; use `--display-scale 1` for literal source coordinates.
`BNS_CCZ4_PUNCTURE_PILOT_OVERLAY.glsl` is the separately labeled dynamical,
under-resolved demonstration.

## Current qualification boundary

The general parametric Cartesian observer now supports `m != 0` surfaces and
passes flat-sphere and rotated `(2,+/-2)` controls. The shader packet, campaign
adapters, GLSL arrays, `ell<=16` harmonic evaluation, interpolation, stability
coloring, and WebGL path are implemented and tested.

The repository does **not** yet contain quasi-circular, constraint-solved
binary-neutron-star initial data or a resolution-qualified BNS collapse
campaign. Existing strong-gravity matter campaigns are head-on collision
experiments, and the dynamical puncture campaign is vacuum. Therefore no file
in the repository is promoted as a solver-derived BNS MOTS bake. The direct BL
overlay is a qualified vacuum initial-data calibration; the dynamical CCZ4
overlay is explicitly an under-resolved vacuum pilot.

The next physical campaign requires:

1. quasi-equilibrium irrotational BNS initial data for a specified EOS;
2. orbital CCZ4+GRHD evolution with sufficient refinement through merger;
3. common-MOTS detection with the general parametric observer;
4. at least one resolution and grid-phase attack on the surface coefficients;
5. inward/outward offset evaluations of `Theta_+` or a qualified stability
   response;
6. extraction-radius checks for the accompanying `Psi4(2,2)` waveform;
7. only then, construction of `mots_shader_bake.json`.
