"""Run and validate reproducible campaigns for the Nulloth Master Spec."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .validate_evidence import validate_ledger
    from .validate_equation_trace import validate_equation_trace
    from .validate_production_trace import validate_production_trace
except ImportError:  # Direct script execution.
    from validate_evidence import validate_ledger
    from validate_equation_trace import validate_equation_trace
    from validate_production_trace import validate_production_trace


MANIFEST_FIELDS = {
    "manifest_schema",
    "campaign_id",
    "title",
    "question",
    "prediction",
    "falsifier",
    "decisive_observables",
    "acceptance_criteria",
    "controls",
    "resolution_ladder",
    "seed",
    "seed_rationale",
    "parameters",
    "command",
    "timeout_seconds",
    "stdout_format",
    "inputs",
    "outputs",
    "supports_claims",
}
CAMPAIGN_ID = re.compile(r"^NUL-[A-Z0-9-]+$")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(
    manifest: dict[str, Any], root: Path, claim_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    missing = MANIFEST_FIELDS - set(manifest)
    extra = set(manifest) - MANIFEST_FIELDS
    label = manifest.get("campaign_id", "unnamed campaign")
    if missing:
        errors.append(f"{label}: missing manifest fields {sorted(missing)}")
    if extra:
        errors.append(f"{label}: unknown manifest fields {sorted(extra)}")
    if manifest.get("manifest_schema") != "nulloth.campaign.v1":
        errors.append(f"{label}: unsupported manifest schema")
    if not CAMPAIGN_ID.fullmatch(str(manifest.get("campaign_id", ""))):
        errors.append(f"{label}: invalid campaign id")
    if not manifest.get("prediction") or not manifest.get("falsifier"):
        errors.append(f"{label}: prediction and falsifier must be nonempty")
    if not manifest.get("decisive_observables"):
        errors.append(f"{label}: decisive_observables must be nonempty")
    if not manifest.get("acceptance_criteria"):
        errors.append(f"{label}: acceptance_criteria must be nonempty")
    valid_operators = {"<", "<=", ">", ">=", "=="}
    for criterion in manifest.get("acceptance_criteria", []):
        if criterion.get("operator") not in valid_operators:
            errors.append(f"{label}: invalid acceptance operator")
        if criterion.get("observable") not in manifest.get(
            "decisive_observables", []
        ):
            errors.append(
                f"{label}: acceptance criterion is not a decisive observable"
            )
    seed = manifest.get("seed")
    if seed is not None and not isinstance(seed, int):
        errors.append(f"{label}: seed must be an integer or null")
    if seed is None and not manifest.get("seed_rationale"):
        errors.append(f"{label}: a null seed requires a rationale")
    if manifest.get("stdout_format") not in {"json", "text"}:
        errors.append(f"{label}: stdout_format must be json or text")
    command = manifest.get("command")
    if not isinstance(command, list) or not command:
        errors.append(f"{label}: command must be a nonempty token list")
    if manifest.get("timeout_seconds", 0) <= 0:
        errors.append(f"{label}: timeout_seconds must be positive")
    for relative in manifest.get("inputs", []):
        if not (root / relative).is_file():
            errors.append(f"{label}: missing input {relative}")
    for relative in manifest.get("outputs", []):
        output = Path(relative)
        if output.is_absolute() or ".." in output.parts:
            errors.append(f"{label}: output must stay inside the attempt directory")
    for claim_id in manifest.get("supports_claims", []):
        if claim_id not in claim_ids:
            errors.append(f"{label}: unknown supported claim {claim_id}")
    for value in manifest.values():
        if "Nulloth 0.1" in json.dumps(value):
            errors.append(f"{label}: prohibited versioned prose name")
            break
    return errors


def input_inventory(manifest: dict[str, Any], root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(root / relative)
        for relative in sorted(manifest["inputs"])
    }


def deterministic_run_id(
    manifest: dict[str, Any], inventory: dict[str, str]
) -> str:
    identity = {
        "manifest": manifest,
        "inputs": inventory,
    }
    return f"{manifest['campaign_id']}-{canonical_digest(identity)[:16]}"


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_record(root: Path) -> dict[str, Any]:
    try:
        project_version = importlib.metadata.version("causal-repair-gravity")
    except importlib.metadata.PackageNotFoundError:
        project_version = None
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "numerical_backend": "numpy",
        "default_float_dtype": str(np.asarray(0.0).dtype),
        "float64_epsilon": float(np.finfo(np.float64).eps),
        "project_distribution_version": project_version,
        "git_revision": _git_revision(root),
    }


def _next_attempt(run_directory: Path) -> Path:
    run_directory.mkdir(parents=True, exist_ok=True)
    indices = []
    for child in run_directory.glob("attempt-*"):
        try:
            indices.append(int(child.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = run_directory / f"attempt-{max(indices, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def evaluate_acceptance(
    criteria: list[dict[str, Any]], result: Any
) -> dict[str, Any]:
    """Evaluate registered scalar criteria against parsed JSON output."""
    comparisons = {
        "<": lambda observed, expected: observed < expected,
        "<=": lambda observed, expected: observed <= expected,
        ">": lambda observed, expected: observed > expected,
        ">=": lambda observed, expected: observed >= expected,
        "==": lambda observed, expected: observed == expected,
    }
    evaluations = []
    passed = isinstance(result, dict)
    for criterion in criteria:
        observable = criterion["observable"]
        observed = result
        for component in observable.split("."):
            if not isinstance(observed, dict) or component not in observed:
                observed = None
                break
            observed = observed[component]
        expected = criterion["value"]
        operator = criterion["operator"]
        try:
            criterion_passed = bool(comparisons[operator](observed, expected))
        except (TypeError, ValueError):
            criterion_passed = False
        passed = passed and criterion_passed
        evaluations.append(
            {
                "observable": observable,
                "observed": observed,
                "operator": operator,
                "expected": expected,
                "passed": criterion_passed,
            }
        )
    return {"passed": passed, "criteria": evaluations}


def run_campaign(manifest_path: Path, root: Path) -> Path:
    manifest = load_json(manifest_path)
    ledger = load_json(Path(__file__).with_name("evidence.json"))
    claim_ids = {claim["id"] for claim in ledger["claims"]}
    errors = validate_manifest(manifest, root, claim_ids)
    if errors:
        raise ValueError("\n".join(errors))

    inventory = input_inventory(manifest, root)
    run_id = deterministic_run_id(manifest, inventory)
    attempt = _next_attempt(root / "runs" / "nulloth_master" / run_id)
    command = [
        sys.executable
        if token == "{python}"
        else token.replace("{run_dir}", str(attempt))
        for token in manifest["command"]
    ]
    run_environment = os.environ.copy()
    source_path = str(root / "src")
    previous_pythonpath = run_environment.get("PYTHONPATH")
    run_environment["PYTHONPATH"] = (
        source_path
        if not previous_pythonpath
        else os.pathsep.join((source_path, previous_pythonpath))
    )

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    termination = "completed"
    return_code: int | None
    try:
        process = subprocess.run(
            command,
            cwd=root,
            env=run_environment,
            text=True,
            capture_output=True,
            timeout=float(manifest["timeout_seconds"]),
            check=False,
        )
        stdout = process.stdout
        stderr = process.stderr
        return_code = process.returncode
        if return_code != 0:
            termination = "failed"
    except subprocess.TimeoutExpired as timeout:
        stdout = timeout.stdout or ""
        stderr = timeout.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        return_code = None
        termination = "timeout"
    duration = time.perf_counter() - started

    stdout_path = attempt / "stdout.txt"
    stderr_path = attempt / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    parsed_stdout: Any = None
    parse_error: str | None = None
    if manifest["stdout_format"] == "json" and stdout.strip():
        try:
            parsed_stdout = json.loads(stdout)
        except json.JSONDecodeError as error:
            parse_error = str(error)
            termination = "failed"

    scientific_acceptance = evaluate_acceptance(
        manifest["acceptance_criteria"], parsed_stdout
    )
    captured_outputs = {
        "stdout.txt": sha256_file(stdout_path),
        "stderr.txt": sha256_file(stderr_path),
    }
    missing_outputs = []
    for relative in manifest["outputs"]:
        output = attempt / relative
        if output.is_file():
            captured_outputs[Path(relative).as_posix()] = sha256_file(output)
        else:
            missing_outputs.append(relative)
    if missing_outputs:
        scientific_acceptance["passed"] = False

    record = {
        "run_record_schema": "nulloth.run.v1",
        "run_id": run_id,
        "attempt": attempt.name,
        "campaign_id": manifest["campaign_id"],
        "manifest_path": manifest_path.resolve().relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "source_fingerprint": canonical_digest(inventory),
        "input_artifacts": inventory,
        "environment": environment_record(root),
        "command": command,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "termination": termination,
        "return_code": return_code,
        "stdout_format": manifest["stdout_format"],
        "stdout_parse_error": parse_error,
        "parsed_stdout": parsed_stdout,
        "scientific_acceptance": scientific_acceptance,
        "missing_declared_outputs": missing_outputs,
        "output_artifacts": captured_outputs,
        "supports_claims": manifest["supports_claims"],
    }
    record_path = attempt / "run_record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record_path


def validate_artifact_index(index_path: Path, root: Path) -> list[str]:
    data = load_json(index_path)
    errors: list[str] = []
    seen: set[str] = set()
    for artifact in data.get("artifacts", []):
        relative = artifact.get("path")
        if relative in seen:
            errors.append(f"duplicate indexed artifact {relative}")
            continue
        seen.add(relative)
        path = root / str(relative)
        if not path.is_file():
            errors.append(f"missing indexed artifact {relative}")
        elif sha256_file(path) != artifact.get("sha256"):
            errors.append(f"stale indexed artifact {relative}")
    return errors


def validate_run_record(record_path: Path, root: Path) -> list[str]:
    """Validate an immutable attempt against its own recorded artifacts."""
    record = load_json(record_path)
    errors: list[str] = []
    label = record.get("run_id", record_path.as_posix())
    for relative, expected_hash in record.get("input_artifacts", {}).items():
        path = root / relative
        if not path.is_file():
            errors.append(f"{label}: missing recorded input {relative}")
    for relative, expected_hash in record.get("output_artifacts", {}).items():
        path = record_path.parent / relative
        if not path.is_file():
            errors.append(f"{label}: missing recorded output {relative}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"{label}: corrupted recorded output {relative}")
    return errors


def validate_graph(root: Path) -> list[str]:
    directory = Path(__file__).parent
    ledger_path = directory / "evidence.json"
    ledger = load_json(ledger_path)
    claim_ids = {claim["id"] for claim in ledger["claims"]}
    errors = validate_ledger(ledger_path, root)
    errors.extend(
        validate_equation_trace(directory / "equation_trace.json", root)
    )
    errors.extend(
        validate_production_trace(directory / "production_trace.json", root)
    )
    errors.extend(validate_artifact_index(directory / "artifact_index.json", root))
    campaign_ids: set[str] = set()
    manifests: dict[str, dict[str, Any]] = {}
    supersessions = {
        item["campaign_id"]: item
        for item in ledger.get("campaign_supersessions", [])
    }
    for path in sorted((directory / "campaigns").glob("*.json")):
        manifest = load_json(path)
        campaign_id = manifest.get("campaign_id")
        if campaign_id in campaign_ids:
            errors.append(f"duplicate campaign id {campaign_id}")
        campaign_ids.add(campaign_id)
        manifests[campaign_id] = manifest
        errors.extend(validate_manifest(manifest, root, claim_ids))
    for claim in ledger["claims"]:
        for campaign_id in claim.get("campaigns", []):
            if campaign_id not in campaign_ids:
                errors.append(
                    f"{claim['id']}: unknown supporting campaign {campaign_id}"
                )
            elif claim["id"] not in manifests[campaign_id]["supports_claims"]:
                errors.append(
                    f"{claim['id']}: campaign {campaign_id} lacks reciprocal link"
                )
    for campaign_id, manifest in manifests.items():
        for claim_id in manifest["supports_claims"]:
            claim = next(claim for claim in ledger["claims"] if claim["id"] == claim_id)
            if campaign_id not in claim.get("campaigns", []):
                errors.append(
                    f"{campaign_id}: claim {claim_id} lacks reciprocal link"
                )
        inventory = input_inventory(manifest, root)
        run_id = deterministic_run_id(manifest, inventory)
        run_directory = root / "runs" / "nulloth_master" / run_id
        accepted_current_attempt = False
        for record_path in sorted(run_directory.glob("attempt-*/run_record.json")):
            errors.extend(validate_run_record(record_path, root))
            record = load_json(record_path)
            if (
                record.get("termination") == "completed"
                and record.get("scientific_acceptance", {}).get("passed") is True
                and record.get("manifest_sha256")
                == sha256_file(
                    directory / "campaigns" / f"{campaign_id}.json"
                )
            ):
                accepted_current_attempt = True
        if not accepted_current_attempt and campaign_id not in supersessions:
            errors.append(f"{campaign_id}: no accepted current execution record")
    for campaign_id, supersession in supersessions.items():
        replacement = supersession.get("replacement_campaign_id")
        record = root / str(supersession.get("record"))
        if campaign_id not in manifests:
            errors.append(f"unknown superseded campaign {campaign_id}")
        if replacement not in manifests:
            errors.append(f"{campaign_id}: unknown replacement campaign {replacement}")
        if not record.is_file():
            errors.append(f"{campaign_id}: missing supersession record")
    return errors


def render_evidence(root: Path) -> Path:
    directory = Path(__file__).parent
    ledger = load_json(directory / "evidence.json")
    lines = [
        "# Nulloth Evidence Tables",
        "",
        "This file is generated from `evidence.json`. Do not edit it by hand.",
        "",
        "| Claim | Seal | Declaration | Domain |",
        "|---|---|---|---|",
    ]
    for claim in ledger["claims"]:
        statement = claim["statement"].replace("|", "\\|")
        domain = claim["domain"].replace("|", "\\|")
        lines.append(
            f"| {claim['id']} — {claim['title']} | {claim['status']} | "
            f"{statement} | {domain} |"
        )
    lines.extend(["", "## Decisive metrics", ""])
    for claim in ledger["claims"]:
        if not claim["metrics"]:
            continue
        lines.extend([f"### {claim['id']}", "", "| Metric | Value |", "|---|---:|"])
        for name, value in sorted(claim["metrics"].items()):
            lines.append(f"| `{name}` | {value} |")
        lines.append("")
    output = directory / "EVIDENCE_TABLES.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    render_equation_trace(root)
    render_production_trace(root)
    return output.relative_to(root)


def render_equation_trace(root: Path) -> Path:
    directory = Path(__file__).parent
    trace = load_json(directory / "equation_trace.json")
    lines = [
        "# Nulloth Equation-to-Code Trace",
        "",
        "This file is generated from `equation_trace.json`. Do not edit it by hand.",
        "",
        "| ID | Specification | Status | Implementation | Verification | Gap |",
        "|---|---|---|---|---|---|",
    ]
    for entry in trace["entries"]:
        implementations = "<br>".join(
            f"`{item['path']}::{item['symbol']}`"
            for item in entry["implementation"]
        )
        verifications = "<br>".join(
            f"`{item['path']}::{item['symbol']}`"
            for item in entry["verification"]
        ) or "—"
        gap = entry["gap"].replace("|", "\\|") or "—"
        specification = entry["specification"].replace("|", "\\|")
        lines.append(
            f"| {entry['id']} | {specification} | {entry['status']} | "
            f"{implementations} | {verifications} | {gap} |"
        )
    output = directory / "EQUATION_TRACE.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output.relative_to(root)


def render_production_trace(root: Path) -> Path:
    directory = Path(__file__).parent
    trace = load_json(directory / "production_trace.json")
    lines = [
        "# Nulloth Production Trace",
        "",
        "This file is generated from `production_trace.json`. Do not edit it by hand.",
        "",
        "| ID | Kind | Operation | Continuum basis | Implementation | Verification |",
        "|---|---|---|---|---|---|",
    ]
    for entry in trace["entries"]:
        implementation = entry["implementation"]
        verifications = "<br>".join(
            f"`{item['path']}::{item['symbol']}`"
            for item in entry["verification"]
        )
        lines.append(
            f"| {entry['id']} | {entry['kind']} | {entry['operation']} | "
            f"{entry['continuum_basis']} | "
            f"`{implementation['path']}::{implementation['symbol']}` | "
            f"{verifications} |"
        )
    output = directory / "PRODUCTION_TRACE.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output.relative_to(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the evidence graph")
    run_parser = subparsers.add_parser("run", help="execute a campaign manifest")
    run_parser.add_argument("manifest", type=Path)
    subparsers.add_parser("render", help="regenerate evidence tables")
    args = parser.parse_args()
    root = repository_root()

    if args.command == "validate":
        errors = validate_graph(root)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Nulloth evidence graph valid")
        return 0
    if args.command == "render":
        print(render_evidence(root))
        return 0
    record = run_campaign(args.manifest.resolve(), root)
    print(record.relative_to(root))
    record_data = load_json(record)
    successful = (
        record_data["termination"] == "completed"
        and record_data["scientific_acceptance"]["passed"]
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
