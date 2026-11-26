"""Adapters from horizon-finder campaign JSON to standalone shader packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bns_shadertoy import MOTSShaderBake


def _read_records(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("campaign JSON must be a record list or contain records")
    return source, records


def _normalized_source_times(
    values: Iterable[float], start: float, end: float
) -> np.ndarray:
    source = np.asarray(list(values), dtype=float)
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("shader time window must satisfy 0 <= start < end <= 1")
    if len(source) < 2:
        raise ValueError("at least two accepted horizon observations are required")
    span = float(source[-1] - source[0])
    if span <= 0.0:
        return np.linspace(start, end, len(source))
    return start + (end - start) * (source - source[0]) / span


def _uniformly_limit(records: list[Any], maximum: int) -> list[Any]:
    if maximum < 2:
        raise ValueError("maximum frame count must be at least two")
    if len(records) <= maximum:
        return records
    indices = np.unique(np.rint(np.linspace(0, len(records) - 1, maximum)).astype(int))
    return [records[int(index)] for index in indices]


def _result_arrays(
    observations: list[tuple[dict[str, Any], dict[str, Any]]]
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray, np.ndarray]:
    modes = tuple(tuple(int(value) for value in mode) for mode in observations[0][1]["modes"])
    for _, result in observations:
        current = tuple(tuple(int(value) for value in mode) for mode in result["modes"])
        if current != modes:
            raise ValueError("campaign changes harmonic modes between accepted frames")
    centers = np.asarray([result["center"] for _, result in observations], dtype=float)
    coefficients = np.asarray(
        [result["coefficients"] for _, result in observations], dtype=float
    )
    eigenvalues = np.asarray(
        [
            result["normal_stability"]["principal_eigenvalue_real"]
            for _, result in observations
        ],
        dtype=float,
    )
    return modes, centers, coefficients, eigenvalues


def direct_surface_bake(
    path: str | Path,
    *,
    branch: str = "common",
    normalized_start: float = 0.82,
    normalized_end: float = 0.94,
    display_scale: float = 1.0,
    maximum_frames: int = 16,
) -> MOTSShaderBake:
    """Bake converged direct-surface observations from a KKT/BL scan.

    Failed/non-surface points beyond the fold are intentionally excluded: their
    near-zero diagnostic eigenvalues do not describe accepted MOTS surfaces.
    """

    source, records = _read_records(path)
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        result = record.get("branches", {}).get(branch)
        stability = result.get("normal_stability", {}) if result else {}
        if result and result.get("found") and stability.get("valid"):
            accepted.append((record, result))
    accepted = _uniformly_limit(accepted, maximum_frames)
    if len(accepted) < 2:
        raise ValueError(f"fewer than two converged direct surfaces for branch {branch!r}")
    source_values = [float(record["separation"]) for record, _ in accepted]
    times = _normalized_source_times(source_values, normalized_start, normalized_end)
    modes, centers, coefficients, eigenvalues = _result_arrays(accepted)
    label = (
        f"{source.name}:{branch}; separation "
        f"{source_values[0]:.9g}..{source_values[-1]:.9g}"
    )
    return MOTSShaderBake(
        normalized_times=times,
        centers=centers,
        modes=modes,
        log_radius_coefficients=coefficients,
        expansion_response_coefficients=np.zeros_like(coefficients),
        principal_eigenvalues=eigenvalues,
        display_scale=display_scale,
        source_kind="direct_surface_scan",
        source_label=label,
        qualified=True,
        source_parameter_name="Brill-Lindquist separation d",
        source_parameter_values=np.asarray(source_values),
    )


def ccz4_campaign_bake(
    path: str | Path,
    *,
    branch: str = "common_outer",
    normalized_start: float = 0.82,
    normalized_end: float = 0.96,
    display_scale: float = 1.0,
    maximum_frames: int = 16,
    require_qualified: bool = True,
) -> MOTSShaderBake:
    """Bake classified common-horizon observations from a CCZ4 trajectory."""

    source, records = _read_records(path)
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in records:
        result = record.get("common_classified", {}).get(branch)
        stability = result.get("normal_stability", {}) if result else {}
        if not (result and result.get("found") and stability.get("valid")):
            continue
        sign_gate = bool(
            result.get("cartesian_qualification", {}).get(
                "principal_mode_sign_gate_passed", False
            )
        )
        if require_qualified and not sign_gate:
            continue
        accepted.append((record, result))
    accepted = _uniformly_limit(accepted, maximum_frames)
    if len(accepted) < 2:
        qualifier = "qualified " if require_qualified else ""
        raise ValueError(
            f"fewer than two {qualifier}CCZ4 observations for branch {branch!r}; "
            "use require_qualified=False only for an explicitly labeled pilot render"
        )
    source_values = [float(record["time"]) for record, _ in accepted]
    times = _normalized_source_times(source_values, normalized_start, normalized_end)
    modes, centers, coefficients, eigenvalues = _result_arrays(accepted)
    gates = [
        bool(
            result.get("cartesian_qualification", {}).get(
                "principal_mode_sign_gate_passed", False
            )
        )
        for _, result in accepted
    ]
    qualified = all(gates)
    label = (
        f"{source.parent.name}/{source.name}:{branch}; coordinate time "
        f"{source_values[0]:.9g}..{source_values[-1]:.9g}"
    )
    if not qualified:
        label += "; UNDER-RESOLVED PILOT"
    return MOTSShaderBake(
        normalized_times=times,
        centers=centers,
        modes=modes,
        log_radius_coefficients=coefficients,
        expansion_response_coefficients=np.zeros_like(coefficients),
        principal_eigenvalues=eigenvalues,
        display_scale=display_scale,
        source_kind="ccz4_puncture_campaign",
        source_label=label,
        qualified=qualified,
        source_parameter_name="CCZ4 coordinate time",
        source_parameter_values=np.asarray(source_values),
    )


__all__ = ["ccz4_campaign_bake", "direct_surface_bake"]
