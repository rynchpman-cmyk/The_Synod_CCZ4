from __future__ import annotations

import unittest

from experiments.map_nulloth_wall_collisions import _run


class NullothWallTests(unittest.TestCase):
    def test_boosted_periodic_wall_pair_travels_and_preserves_constraints(self) -> None:
        run, _ = _run(64, 1, 2.0, 0.01, 0.4)

        self.assertEqual(run.initial_zero_crossings, 2)
        self.assertEqual(run.final_zero_crossings, 2)
        self.assertLess(run.incoming_speed_error, 0.005)
        self.assertLess(run.final_constraint_l2, 1.0e-12)
        self.assertEqual(run.bounded, 1)


if __name__ == "__main__":
    unittest.main()
