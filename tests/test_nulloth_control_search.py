from __future__ import annotations

import unittest

from experiments.search_nulloth_brake_controls import (
    BranchResult,
    _relative_suppression,
)


def _branch(kinetic: float) -> BranchResult:
    return BranchResult(
        "test",
        0.0,
        0.0,
        0.0,
        kinetic,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        True,
    )


class NullothControlSearchTests(unittest.TestCase):
    def test_relative_suppression_has_brake_sign(self) -> None:
        self.assertAlmostEqual(_relative_suppression(_branch(0.8), _branch(1.0)), 0.2)
        self.assertLess(_relative_suppression(_branch(1.2), _branch(1.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
