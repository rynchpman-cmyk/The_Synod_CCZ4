from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.audit_nulloth_brake_robustness import normalized_seed_profile


class NullothBrakeRobustnessTests(unittest.TestCase):
    def test_profile_normalization_preserves_commanded_peak(self) -> None:
        coordinate = np.arange(32, dtype=float)
        profile = normalized_seed_profile(
            coordinate, 2.0 * np.pi / 32.0, np.pi / 8.0, 0.1
        )
        self.assertAlmostEqual(float(np.max(np.abs(profile))), 1.0)

    def test_opposite_phase_errors_have_equal_norm(self) -> None:
        coordinate = np.arange(32, dtype=float)
        left = normalized_seed_profile(
            coordinate, 2.0 * np.pi / 32.0, -np.pi / 8.0, 0.0
        )
        right = normalized_seed_profile(
            coordinate, 2.0 * np.pi / 32.0, np.pi / 8.0, 0.0
        )
        self.assertAlmostEqual(float(np.linalg.norm(left)), float(np.linalg.norm(right)))


if __name__ == "__main__":
    unittest.main()
