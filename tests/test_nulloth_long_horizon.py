from __future__ import annotations

import unittest

from experiments.measure_nulloth_long_horizon import run_long_horizon


class NullothLongHorizonTests(unittest.TestCase):
    def test_zero_duration_wrapper_reports_no_lifetime_promotion(self) -> None:
        result = run_long_horizon(points=4, duration=0.0, dt=0.02)

        self.assertEqual(result["transitioned_from_sustained_to_turnover"], 0)
        self.assertEqual(result["retained_coherence_at_horizon"], 0)
        self.assertEqual(result["coherence_lifetime_lower_bound"], 0.0)


if __name__ == "__main__":
    unittest.main()
