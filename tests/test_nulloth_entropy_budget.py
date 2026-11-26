from __future__ import annotations

import unittest

from experiments.close_nulloth_entropy_budget import _relative_spread


class NullothEntropyBudgetTests(unittest.TestCase):
    def test_relative_spread_is_normalized_to_fine_value(self) -> None:
        values = [1.001, 1.0005, 1.0]

        self.assertAlmostEqual(_relative_spread(values), 0.001)


if __name__ == "__main__":
    unittest.main()
