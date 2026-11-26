from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.nulloth_master import observatory
from experiments.nulloth_master import validate_equation_trace
from experiments.nulloth_master import validate_evidence
from experiments.nulloth_master import validate_production_trace
from experiments import audit_nulloth_master_spec


class NullothMasterEvidenceTests(unittest.TestCase):
    def test_seed_ledger_is_valid(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        validator_path = (
            repository_root
            / "experiments"
            / "nulloth_master"
            / "validate_evidence.py"
        )
        ledger_path = validator_path.with_name("evidence.json")

        errors = validate_evidence.validate_ledger(ledger_path, repository_root)

        self.assertEqual(errors, [])

    def test_observatory_graph_is_valid_and_run_id_is_deterministic(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        observatory_path = (
            repository_root
            / "experiments"
            / "nulloth_master"
            / "observatory.py"
        )
        errors = observatory.validate_graph(repository_root)
        manifest_path = (
            observatory_path.parent / "campaigns" / "NUL-MOD-001.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = observatory.input_inventory(manifest, repository_root)

        first = observatory.deterministic_run_id(manifest, inventory)
        second = observatory.deterministic_run_id(manifest, inventory)

        self.assertEqual(errors, [])
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("NUL-MOD-001-"))

    def test_scientific_acceptance_is_distinct_from_process_success(self) -> None:
        criteria = [
            {"observable": "relative_error", "operator": "<=", "value": 1e-8}
        ]

        accepted = observatory.evaluate_acceptance(
            criteria, {"relative_error": 1e-12}
        )
        rejected = observatory.evaluate_acceptance(
            criteria, {"relative_error": 1e-4}
        )

        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])

    def test_scientific_acceptance_reads_nested_observables(self) -> None:
        criteria = [
            {
                "observable": "qualification.failures",
                "operator": "==",
                "value": 0,
            }
        ]

        result = observatory.evaluate_acceptance(
            criteria, {"qualification": {"failures": 0}}
        )

        self.assertTrue(result["passed"])

    def test_equation_trace_is_valid(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        trace_path = (
            repository_root
            / "experiments"
            / "nulloth_master"
            / "equation_trace.json"
        )

        errors = validate_equation_trace.validate_equation_trace(
            trace_path, repository_root
        )

        self.assertEqual(errors, [])

    def test_production_trace_and_operator_order_are_valid(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        trace_path = (
            repository_root
            / "experiments"
            / "nulloth_master"
            / "production_trace.json"
        )

        errors = validate_production_trace.validate_production_trace(
            trace_path, repository_root
        )

        self.assertEqual(errors, [])

    def test_private_synod_registry_is_complete_and_cross_linked(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        directory = repository_root / "experiments" / "nulloth_master"

        result = audit_nulloth_master_spec.audit(
            directory / "NULLOTH_MASTER_SPEC.md",
            directory / "evidence.json",
        )

        self.assertTrue(result["passed"], result["errors"])


if __name__ == "__main__":
    unittest.main()
