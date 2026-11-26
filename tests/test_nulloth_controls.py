from __future__ import annotations

import unittest
from pathlib import Path

from experiments.validate_nulloth_control_registry import validate_registry


class NullothControlRegistryTests(unittest.TestCase):
    def test_registry_has_no_noncanonical_override(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = validate_registry(
            root / "experiments" / "nulloth_master" / "control_registry.json"
        )

        self.assertEqual(result["direct_state_overrides"], 0)
        self.assertEqual(result["unknown_channel_references"], 0)
        self.assertEqual(result["objectives_without_controls"], 0)
        self.assertEqual(result["objectives_without_abort"], 0)


if __name__ == "__main__":
    unittest.main()
