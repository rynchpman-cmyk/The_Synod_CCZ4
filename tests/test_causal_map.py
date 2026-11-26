import unittest

import numpy as np

from causal_repair.causal_map import causal_current, causal_margin, inner, project_spatial, tanh_over_x


class CausalMapTests(unittest.TestCase):
    def test_origin_is_smooth(self):
        values = tanh_over_x(np.array([0.0, 1.0e-8, 1.0e-3]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values[0], 1.0)

    def test_arbitrary_matter_frame_remains_timelike(self):
        gamma = 1.0 / np.sqrt(1.0 - 0.6**2)
        u = np.array([gamma, gamma * 0.6, 0.0, 0.0])[:, None]
        raw = np.array([[0.2, -0.1], [4.0, 20.0], [3.0, -5.0], [1.0, 2.0]])
        xi = project_spatial(raw, u)
        np.testing.assert_allclose(inner(u, xi, np.diag([-1.0, 1.0, 1.0, 1.0])), 0.0, atol=1e-12)
        current, _, chi = causal_current(np.array([2.0, 3.0]), u, xi)
        expected = np.array([4.0, 9.0]) / np.cosh(chi) ** 2
        np.testing.assert_allclose(causal_margin(current), expected, rtol=1e-11, atol=1e-12)
        self.assertTrue(np.all(causal_margin(current) > 0.0))


if __name__ == "__main__":
    unittest.main()

