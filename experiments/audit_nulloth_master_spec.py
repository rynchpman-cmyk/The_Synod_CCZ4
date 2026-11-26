"""Audit the private Synod Nulloth registry against the Evidence Ledger."""

from __future__ import annotations

import json
import re
from pathlib import Path


RECORD_FIELDS = (
    "MODEL ASSERTION",
    "OPERATIONAL FORM",
    "VALIDITY ENVELOPE",
    "EVIDENCE POINTERS",
    "RESOURCE ACCOUNT",
    "REJECTION CONDITIONS",
    "UNRESOLVED STATE",
    "CONFIDENCE CLASS",
    "INTERNAL DETERMINATION",
)

REQUIRED_RECORDS = (
    "NUL-INT-FND",
    "NUL-INT-LIM",
    "NUL-INT-MAT",
    "NUL-INT-COH",
    "NUL-INT-VEC",
    "NUL-INT-ATL",
    "NUL-INT-DENY",
    "NUL-INT-GRV",
    "NUL-INT-CTL",
    "NUL-INT-SOL-001",
    "NUL-INT-UNK",
)


def audit(spec_path: Path, ledger_path: Path) -> dict[str, object]:
    text = spec_path.read_text(encoding="utf-8")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    claims = ledger["claims"]
    errors: list[str] = []

    ids = [claim["id"] for claim in claims]
    if len(ids) != len(set(ids)):
        errors.append("Evidence Ledger contains duplicate claim IDs")

    missing_claims = [claim_id for claim_id in ids if claim_id not in text]
    if missing_claims:
        errors.append("Master Spec omits claims: " + ", ".join(missing_claims))

    record_matches = list(
        re.finditer(r"^## RECORD (NUL-INT-[A-Z0-9-]+) // .+$", text, re.MULTILINE)
    )
    record_ids = [match.group(1) for match in record_matches]
    missing_records = [item for item in REQUIRED_RECORDS if item not in record_ids]
    if missing_records:
        errors.append("Master Spec omits records: " + ", ".join(missing_records))

    for index, match in enumerate(record_matches):
        end = record_matches[index + 1].start() if index + 1 < len(record_matches) else len(text)
        body = text[match.end():end]
        missing_fields = [field for field in RECORD_FIELDS if f"**{field}**" not in body]
        if missing_fields:
            errors.append(
                f"{match.group(1)} omits fields: " + ", ".join(missing_fields)
            )

    forbidden_names = re.findall(r"\bNulloth\s+(?:v(?:ersion\s*)?)?\d+(?:\.\d+)+\b", text, re.I)
    if forbidden_names:
        errors.append("Versioned prose name found: " + ", ".join(forbidden_names))

    promoted = [claim["id"] for claim in claims if claim["status"] == "PROMOTED"]
    spell_start = text.find("## RECORD NUL-INT-SOL-001")
    spell_end = text.find("## RECORD NUL-INT-UNK")
    spell_body = text[spell_start:spell_end]
    missing_promoted = [claim_id for claim_id in promoted if claim_id not in spell_body]
    if missing_promoted:
        errors.append("Promoted claims omitted from solution records: " + ", ".join(missing_promoted))

    if not re.search(
        r"one promoted constrained solution and no canonical solution", text, re.I
    ):
        errors.append("Master Spec does not explicitly withhold Canonical status")

    unknowns = [claim["id"] for claim in claims if claim["status"] == "UNKNOWN"]
    unknown_start = text.find("## RECORD NUL-INT-UNK")
    unknown_end = text.find("## 2. COMPLETE CLAIM INDEX")
    unknown_body = text[unknown_start:unknown_end]
    missing_unknowns = [claim_id for claim_id in unknowns if claim_id not in unknown_body]
    if missing_unknowns:
        errors.append("Unknown claims omitted from unresolved queue: " + ", ".join(missing_unknowns))

    private_markers = (
        "ACCESS CLASS: PRIVATE / NON-MISSIONARIA",
        "PRIMARY CONSUMER: SYNODIC STATE ESTIMATION, EVOLUTION, AND CONTROL SYNTHESIS",
        "No public or initiated teaching form is generated here.",
        "I am not an operator consulting this model from outside its domain.",
    )
    missing_markers = [marker for marker in private_markers if marker not in text]
    if missing_markers:
        errors.append("Private Synod context is incomplete: " + ", ".join(missing_markers))

    public_voice_markers = (
        "## ARTICLE ",
        "**PRONOUNCEMENT**",
        "**SYNODIC JUDGMENT**",
        "The Synod seals",
        "The law of this book",
    )
    found_public_voice = [marker for marker in public_voice_markers if marker in text]
    if found_public_voice:
        errors.append("Missionaria-facing voice markers found: " + ", ".join(found_public_voice))

    link_targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
    missing_links = []
    for target in link_targets:
        if "://" in target or target.startswith("#"):
            continue
        if not (spec_path.parent / target).resolve().exists():
            missing_links.append(target)
    if missing_links:
        errors.append("Broken local links: " + ", ".join(missing_links))

    statuses: dict[str, int] = {}
    for claim in claims:
        statuses[claim["status"]] = statuses.get(claim["status"], 0) + 1

    return {
        "passed": not errors,
        "errors": errors,
        "claim_count": len(claims),
        "record_count": len(record_ids),
        "status_counts": statuses,
        "promoted_claims": promoted,
        "unknown_claims": unknowns,
        "local_links_checked": len(link_targets),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "experiments" / "nulloth_master"
    result = audit(directory / "NULLOTH_MASTER_SPEC.md", directory / "evidence.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
