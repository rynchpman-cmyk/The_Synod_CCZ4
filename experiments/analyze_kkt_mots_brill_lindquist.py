"""Analyze common-MOTS detection and shape-response crossings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_campaign(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "causal_repair.kkt_mots_brill_lindquist.v1",
        "causal_repair.kkt_mots_brill_lindquist_direct.v1",
    }:
        raise ValueError(f"{path} is not a KKT/MOTS Brill--Lindquist campaign")
    return payload


def linear_zero(x0: float, y0: float, x1: float, y1: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def analyze(path: Path) -> dict[str, object]:
    payload = load_campaign(path)
    rows: list[dict[str, object]] = []
    for record in sorted(payload["records"], key=lambda item: item["separation"]):
        branch = record["branches"]["common"]
        eigenvalues = np.asarray(branch["response_eigenvalues_real"], dtype=float)
        normal = branch.get("normal_stability")
        normal_valid = bool(normal and normal.get("valid"))
        rows.append(
            {
                "separation": float(record["separation"]),
                "found": bool(branch["found"]),
                "reason": branch["reason"],
                "spectral_residual_norm": float(branch["spectral_residual_norm"]),
                "full_rms_expansion": float(branch["rms_expansion"]),
                "area": float(branch["area"]),
                "irreducible_mass": float(branch["irreducible_mass"]),
                "minimum_response_real_part": (
                    float(np.min(eigenvalues)) if eigenvalues.size else None
                ),
                "response_condition_number": float(branch["response_condition_number"]),
                "normal_stability_valid": normal_valid,
                "normal_principal_eigenvalue_real": (
                    float(normal["principal_eigenvalue_real"])
                    if normal_valid
                    else None
                ),
                "normal_principal_eigenvalue_imag": (
                    float(normal["principal_eigenvalue_imag"])
                    if normal_valid
                    else None
                ),
                "normal_operator_condition_number": (
                    float(normal["operator_condition_number"])
                    if normal_valid
                    else None
                ),
                "normalizer_condition_number": (
                    float(normal["normalizer_condition_number"])
                    if normal_valid
                    else None
                ),
                "normalizer_roundtrip_error": (
                    float(normal["normalizer_roundtrip_error"])
                    if normal_valid
                    else None
                ),
                "normal_finite_difference_relative_error": (
                    float(normal["finite_difference_relative_error"])
                    if normal_valid
                    else None
                ),
                "active_constraint_count": len(branch["active_lower"])
                + len(branch["active_upper"]),
            }
        )

    detection_brackets: list[dict[str, float]] = []
    response_crossings: list[dict[str, float]] = []
    normal_stability_crossings: list[dict[str, float]] = []
    for lower, upper in zip(rows[:-1], rows[1:]):
        if lower["found"] and not upper["found"]:
            detection_brackets.append(
                {
                    "last_detected": lower["separation"],
                    "first_not_detected": upper["separation"],
                }
            )
        y0 = lower["minimum_response_real_part"]
        y1 = upper["minimum_response_real_part"]
        if y0 is not None and y1 is not None and y0 * y1 < 0.0:
            response_crossings.append(
                {
                    "lower_separation": lower["separation"],
                    "upper_separation": upper["separation"],
                    "lower_eigenvalue": y0,
                    "upper_eigenvalue": y1,
                    "linear_zero_estimate": linear_zero(
                        lower["separation"], y0, upper["separation"], y1
                    ),
                }
            )
        normal_y0 = lower["normal_principal_eigenvalue_real"]
        normal_y1 = upper["normal_principal_eigenvalue_real"]
        if (
            normal_y0 is not None
            and normal_y1 is not None
            and normal_y0 * normal_y1 < 0.0
        ):
            normal_stability_crossings.append(
                {
                    "lower_separation": lower["separation"],
                    "upper_separation": upper["separation"],
                    "lower_eigenvalue": normal_y0,
                    "upper_eigenvalue": normal_y1,
                    "linear_zero_estimate": linear_zero(
                        lower["separation"],
                        normal_y0,
                        upper["separation"],
                        normal_y1,
                    ),
                }
            )
    converged_normal_rows = [
        row
        for row in rows
        if row["found"] and row["normal_principal_eigenvalue_real"] is not None
    ]
    on_branch_zero = None
    on_branch_squared_zero = None
    if len(converged_normal_rows) >= 2:
        lower, upper = converged_normal_rows[-2:]
        y0 = lower["normal_principal_eigenvalue_real"]
        y1 = upper["normal_principal_eigenvalue_real"]
        if y1 < y0:
            estimate = linear_zero(
                lower["separation"], y0, upper["separation"], y1
            )
            on_branch_zero = {
                "lower_separation": lower["separation"],
                "upper_separation": upper["separation"],
                "lower_eigenvalue": y0,
                "upper_eigenvalue": y1,
                "linear_zero_estimate": estimate,
                "is_extrapolation": estimate > upper["separation"],
            }
            squared_estimate = linear_zero(
                lower["separation"],
                y0 * y0,
                upper["separation"],
                y1 * y1,
            )
            on_branch_squared_zero = {
                "lower_separation": lower["separation"],
                "upper_separation": upper["separation"],
                "lower_eigenvalue": y0,
                "upper_eigenvalue": y1,
                "linear_lambda_squared_zero_estimate": squared_estimate,
                "fold_scaling_assumption": "lambda_0 proportional to sqrt(d_star-d)",
                "is_extrapolation": squared_estimate > upper["separation"],
            }
    return {
        "source": str(path),
        "configuration": payload["configuration"],
        "detection_brackets": detection_brackets,
        "response_crossings": response_crossings,
        "normal_stability_crossings": normal_stability_crossings,
        "normal_on_branch_linear_zero": on_branch_zero,
        "normal_on_branch_squared_zero": on_branch_squared_zero,
        "maximum_condition_number": max(
            row["response_condition_number"] for row in rows
        ),
        "all_detected_surfaces_inactive_at_box_bounds": all(
            row["active_constraint_count"] == 0 for row in rows if row["found"]
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/kkt_mots_brill_lindquist_analysis.json"),
    )
    args = parser.parse_args()
    analyses = [analyze(path) for path in args.inputs]
    output = {
        "format": "causal_repair.kkt_mots_brill_lindquist_analysis.v1",
        "warning": (
            "Response crossings are finite-resolution, parameterization-dependent "
            "KKT/MOTS predictions, not invariant stability eigenvalues."
        ),
        "campaigns": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix + ".pending")
    pending.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    pending.replace(args.output)
    for campaign in analyses:
        print(campaign["source"])
        for bracket in campaign["detection_brackets"]:
            print(
                "  detection bracket "
                f"[{bracket['last_detected']:.9g}, "
                f"{bracket['first_not_detected']:.9g}]"
            )
        for crossing in campaign["response_crossings"]:
            print(
                "  response zero "
                f"{crossing['linear_zero_estimate']:.9g} within "
                f"[{crossing['lower_separation']:.9g}, "
                f"{crossing['upper_separation']:.9g}]"
            )
        for crossing in campaign["normal_stability_crossings"]:
            print(
                "  physical-normal stability zero "
                f"{crossing['linear_zero_estimate']:.9g} within "
                f"[{crossing['lower_separation']:.9g}, "
                f"{crossing['upper_separation']:.9g}]"
            )
        prediction = campaign["normal_on_branch_linear_zero"]
        if prediction is not None:
            print(
                "  converged-branch normal zero "
                f"{prediction['linear_zero_estimate']:.9g} from "
                f"[{prediction['lower_separation']:.9g}, "
                f"{prediction['upper_separation']:.9g}]"
            )
        fold_prediction = campaign["normal_on_branch_squared_zero"]
        if fold_prediction is not None:
            print(
                "  fold-scaled normal zero "
                f"{fold_prediction['linear_lambda_squared_zero_estimate']:.9g} "
                "from lambda^2 on "
                f"[{fold_prediction['lower_separation']:.9g}, "
                f"{fold_prediction['upper_separation']:.9g}]"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
