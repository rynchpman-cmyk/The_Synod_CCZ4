"""Generate the paste-ready Binary Neutron Star ShaderToy program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from causal_repair.bns_shadertoy import (  # noqa: E402
    BNSReducedModel,
    BNSShaderParameters,
    MOTSShaderBake,
    write_shadertoy_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bake a reduced binary-neutron-star model into one standalone "
            "ShaderToy Image shader."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "shadertoy" / "BNS_CAUSAL_REPAIR.glsl",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mass-ratio", type=float, default=1.0)
    parser.add_argument("--compactness-1", type=float, default=0.17)
    parser.add_argument("--compactness-2", type=float, default=0.17)
    parser.add_argument("--love-k2-1", type=float, default=0.09)
    parser.add_argument("--love-k2-2", type=float, default=0.09)
    parser.add_argument("--initial-separation", type=float, default=13.5)
    parser.add_argument("--animation-seconds", type=float, default=32.0)
    parser.add_argument("--total-mass-solar", type=float, default=2.7)
    parser.add_argument("--postmerger-f2-khz", type=float, default=3.0)
    parser.add_argument("--postmerger-duration-ms", type=float, default=4.0)
    parser.add_argument("--postmerger-damping-ms", type=float, default=2.5)
    parser.add_argument("--prompt-collapse", action="store_true")
    parser.add_argument("--horizon-formation-fraction", type=float, default=0.86)
    parser.add_argument("--hide-horizon-wireframe", action="store_true")
    parser.add_argument(
        "--mots-bake",
        type=Path,
        help="causal_repair.mots_shader_bake.v1/v2 JSON packet",
    )
    parser.add_argument("--wave-gain", type=float, default=1.0)
    parser.add_argument("--lensing-gain", type=float, default=0.82)
    parser.add_argument(
        "--repair-strength",
        type=float,
        default=0.22,
        help="illustrative auxiliary-field brightness; use 0 for a GR-only render",
    )
    parser.add_argument("--quality", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mots_bake = MOTSShaderBake.from_json(args.mots_bake) if args.mots_bake else None
    parameters = BNSShaderParameters(
        mass_ratio=args.mass_ratio,
        compactness_1=args.compactness_1,
        compactness_2=args.compactness_2,
        love_k2_1=args.love_k2_1,
        love_k2_2=args.love_k2_2,
        initial_separation=args.initial_separation,
        animation_seconds=args.animation_seconds,
        total_mass_solar=args.total_mass_solar,
        postmerger_f2_khz=args.postmerger_f2_khz,
        postmerger_duration_ms=args.postmerger_duration_ms,
        postmerger_damping_ms=args.postmerger_damping_ms,
        prompt_collapse=args.prompt_collapse or mots_bake is not None,
        show_horizon_wireframe=not args.hide_horizon_wireframe,
        horizon_formation_fraction=args.horizon_formation_fraction,
        wave_visual_gain=args.wave_gain,
        lensing_gain=args.lensing_gain,
        repair_strength=args.repair_strength,
        quality=args.quality,
    )
    shader, manifest = write_shadertoy_package(
        args.output,
        parameters,
        manifest_output=args.manifest,
        mots_bake=mots_bake,
    )
    model = BNSReducedModel(parameters)
    print(
        json.dumps(
            {
                "shader": str(shader.resolve()),
                "manifest": str(manifest.resolve()),
                "contact_separation_M": parameters.contact_separation,
                "inspiral_orbits": parameters.inspiral_orbits,
                "postmerger_f2_khz": parameters.postmerger_f2_khz,
                "postmerger_f2_omega_M": parameters.postmerger_f2_omega_M,
                "shader_bytes": shader.stat().st_size,
                "repair_layer": (
                    "disabled" if parameters.repair_strength == 0.0 else "illustrative"
                ),
                "model": model.manifest()["model"],
                "mots_bake_frames": (
                    0 if mots_bake is None else len(mots_bake.normalized_times)
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
