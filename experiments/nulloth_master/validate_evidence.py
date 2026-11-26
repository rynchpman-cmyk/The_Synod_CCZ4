"""Validate the machine-readable evidence behind the Nulloth Master Spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CLAIM_FIELDS = {
    "id",
    "title",
    "status",
    "statement",
    "domain",
    "evidence",
    "metrics",
    "limitations",
    "campaigns",
}
PROHIBITED_PROSE_NAMES = ("Nulloth 0.1",)


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def validate_ledger(ledger_path: Path, repository_root: Path) -> list[str]:
    """Return all ledger validation errors without stopping at the first one."""
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    statuses = set(data.get("status_vocabulary", []))
    claims = data.get("claims", [])

    if data.get("canonical_name") != "Nulloth":
        errors.append("canonical_name must be Nulloth")
    if not statuses:
        errors.append("status_vocabulary must not be empty")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a nonempty list")
        return errors

    seen_ids: set[str] = set()
    for index, claim in enumerate(claims):
        label = claim.get("id", f"claim[{index}]")
        missing = REQUIRED_CLAIM_FIELDS - set(claim)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
        claim_id = claim.get("id")
        if claim_id in seen_ids:
            errors.append(f"{label}: duplicate claim id")
        elif isinstance(claim_id, str):
            seen_ids.add(claim_id)
        if claim.get("status") not in statuses:
            errors.append(f"{label}: unknown status {claim.get('status')!r}")
        for evidence_path in claim.get("evidence", []):
            if not (repository_root / evidence_path).is_file():
                errors.append(f"{label}: missing evidence {evidence_path}")
        for prose in _strings(claim):
            for prohibited in PROHIBITED_PROSE_NAMES:
                if prohibited in prose:
                    errors.append(
                        f"{label}: prohibited prose name {prohibited!r}"
                    )

    return errors


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    ledger_path = Path(__file__).with_name("evidence.json")
    errors = validate_ledger(ledger_path, repository_root)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"validated {ledger_path.relative_to(repository_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
