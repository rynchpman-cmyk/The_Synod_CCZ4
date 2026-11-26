"""Convert direct-surface or CCZ4 horizon campaign JSON into a shader bake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from causal_repair.mots_shader_adapter import (  # noqa: E402
    ccz4_campaign_bake,
    direct_surface_bake,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("direct", "ccz4"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--branch")
    parser.add_argument("--start", type=float, default=0.82)
    parser.add_argument("--end", type=float, default=0.94)
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--maximum-frames", type=int, default=16)
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="permit under-resolved CCZ4 pilot surfaces and label them unqualified",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = dict(
        normalized_start=args.start,
        normalized_end=args.end,
        display_scale=args.display_scale,
        maximum_frames=args.maximum_frames,
    )
    if args.kind == "direct":
        bake = direct_surface_bake(
            args.input, branch=args.branch or "common", **common
        )
    else:
        bake = ccz4_campaign_bake(
            args.input,
            branch=args.branch or "common_outer",
            require_qualified=not args.allow_unqualified,
            **common,
        )
    output = bake.write_json(args.output)
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "frames": len(bake.normalized_times),
                "modes": [list(mode) for mode in bake.modes],
                "lambda0_range": [
                    float(bake.principal_eigenvalues.min()),
                    float(bake.principal_eigenvalues.max()),
                ],
                "display_scale": bake.display_scale,
                "source": bake.source_label,
                "qualified": bake.qualified,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
