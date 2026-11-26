"""Validate the bidirectional equation-to-code trace for Nulloth."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "specification",
    "declaration",
    "implementation",
    "verification",
    "status",
    "gap",
}


def validate_equation_trace(trace_path: Path, root: Path) -> list[str]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    statuses = set(trace.get("status_vocabulary", []))
    seen: set[str] = set()
    for entry in trace.get("entries", []):
        label = entry.get("id", "unnamed equation")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        if label in seen:
            errors.append(f"{label}: duplicate equation id")
        seen.add(label)
        if entry.get("status") not in statuses:
            errors.append(f"{label}: unknown trace status")
        if entry.get("status") in {"PARTIAL", "UNIMPLEMENTED"} and not entry.get(
            "gap"
        ):
            errors.append(f"{label}: incomplete entry must declare its gap")
        if entry.get("status") == "TESTED" and not entry.get("verification"):
            errors.append(f"{label}: tested entry has no verification witness")
        for kind in ("implementation", "verification"):
            for witness in entry.get(kind, []):
                path = root / witness.get("path", "")
                symbol = witness.get("symbol", "")
                if not path.is_file():
                    errors.append(f"{label}: missing {kind} path {witness.get('path')}")
                elif symbol not in path.read_text(encoding="utf-8"):
                    errors.append(
                        f"{label}: missing {kind} symbol {symbol!r} in {witness.get('path')}"
                    )
        if "Nulloth 0.1" in json.dumps(entry):
            errors.append(f"{label}: prohibited versioned prose name")
    return errors

