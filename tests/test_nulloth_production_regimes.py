from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments.map_nulloth_production_regimes import (
    _classify,
    _high_wavenumber_fraction,
)
from experiments.map_nulloth_stress_edge import run_stress_edge


class NullothProductionRegimeTests(unittest.TestCase):
    def test_outcome_classifier_distinguishes_registered_regimes(self) -> None:
        self.assertEqual(_classify(0.0, 0.0, 0.0, True), "scalar-free")
        self.assertEqual(_classify(0.02, 0.14, 0.118, True), "nonlinear-turnover")
        self.assertEqual(_classify(0.005, 0.12, 0.12, True), "sustained-coherence")
        self.assertEqual(_classify(0.06, 0.065, 0.02, True), "dispersed")
        self.assertEqual(_classify(0.02, 0.023, 0.018, True), "bounded-oscillatory")
        self.assertEqual(_classify(0.02, 0.03, 0.025, False), "denied")

    def test_high_wavenumber_fraction_separates_low_and_high_modes(self) -> None:
        points = 24
        indices = np.arange(points)
        low_mode = np.cos(2.0 * np.pi * indices / points)
        high_mode = np.cos(2.0 * np.pi * 10.0 * indices / points)

        self.assertLess(_high_wavenumber_fraction(low_mode), 1.0e-28)
        self.assertGreater(_high_wavenumber_fraction(high_mode), 1.0 - 1.0e-12)

    @patch("experiments.map_nulloth_stress_edge._run_configuration")
    def test_stress_edge_uses_the_registered_seed(self, run_configuration) -> None:
        marker = object()
        run_configuration.return_value = marker

        result = run_stress_edge(points=8, duration=1.5, dt=0.02)

        self.assertIs(result, marker)
        run_configuration.assert_called_once_with(
            8,
            1.5,
            0.02,
            ("high-amplitude-stress", 0.18, 0.12, "rest"),
        )


if __name__ == "__main__":
    unittest.main()
