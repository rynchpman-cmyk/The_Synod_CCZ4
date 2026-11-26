import unittest

import numpy as np

from dashboard.server import catalog, trajectory


class DashboardDataTests(unittest.TestCase):
    def test_catalog_exposes_both_showcases(self):
        payload = catalog()
        scenarios = {item["id"]: item for item in payload["scenarios"]}
        self.assertEqual(set(scenarios), {"interaction_dominant", "ringing_trap"})
        self.assertTrue(all(item["available"] for item in scenarios.values()))

    def test_trajectory_contract_is_finite_and_aligned(self):
        required = {
            "x", "time", "a_z", "astar_z", "phi", "energy", "a_l2",
            "astar_l2", "mismatch_l2", "interaction_energy", "proca_energy",
        }
        for name in ("interaction_dominant", "ringing_trap"):
            payload = trajectory(name)
            arrays = payload["arrays"]
            self.assertTrue(required.issubset(arrays))
            frames, points = len(arrays["time"]), len(arrays["x"])
            self.assertEqual((frames, points), (161, 160))
            self.assertEqual(np.asarray(arrays["a_z"]).shape, (frames, points))
            self.assertTrue(np.all(np.isfinite(np.asarray(arrays["energy"]))))


if __name__ == "__main__":
    unittest.main()
