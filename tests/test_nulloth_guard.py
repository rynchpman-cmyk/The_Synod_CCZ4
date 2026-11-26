from __future__ import annotations

import unittest

from experiments.audit_nulloth_runtime_guard import run_audit


class NullothRuntimeGuardTests(unittest.TestCase):
    def test_registered_safe_watch_and_deny_paths(self) -> None:
        result = run_audit()

        self.assertEqual(result.safe_status_correct, 1)
        self.assertEqual(result.watch_status_correct, 1)
        self.assertEqual(result.amplitude_deny_status_correct, 1)
        self.assertEqual(result.superluminal_deny_status_correct, 1)
        self.assertEqual(result.enforce_raised_on_deny, 1)
        self.assertEqual(result.production_recovery_status_correct, 1)
        self.assertEqual(result.frame_status_agreement, 1)
        self.assertLessEqual(result.frame_maximum_metric_speed_error, 1.0e-14)


if __name__ == "__main__":
    unittest.main()
