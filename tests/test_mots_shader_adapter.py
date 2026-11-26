from pathlib import Path
import unittest

import numpy as np

from causal_repair.bns_shadertoy import render_shadertoy_glsl
from causal_repair.mots_shader_adapter import (
    ccz4_campaign_bake,
    direct_surface_bake,
)


ROOT = Path(__file__).resolve().parents[1]


class MOTSShaderAdapterTests(unittest.TestCase):
    def test_direct_l16_scan_bakes_only_accepted_surfaces(self):
        bake = direct_surface_bake(
            ROOT / "runs" / "kkt_mots_brill_lindquist_direct_l16_refined.json"
        )
        self.assertEqual(len(bake.normalized_times), 4)
        self.assertIn((16, 0), bake.modes)
        self.assertTrue(bake.qualified)
        self.assertGreater(float(np.min(bake.principal_eigenvalues)), 0.0)
        self.assertTrue(np.all(bake.expansion_response_coefficients == 0.0))
        shader = render_shadertoy_glsl(mots_bake=bake)
        self.assertIn("ivec2(16, 0)", shader)
        self.assertIn("associatedLegendre", shader)
        self.assertIn("MOTS_LAMBDA0", shader)

    def test_ccz4_pilot_requires_explicit_unqualified_override(self):
        path = (
            ROOT
            / "runs"
            / "ccz4_puncture_mots_lapsefix_n32_early"
            / "trajectory.json"
        )
        with self.assertRaises(ValueError):
            ccz4_campaign_bake(path)
        bake = ccz4_campaign_bake(path, require_qualified=False)
        self.assertEqual(len(bake.normalized_times), 3)
        self.assertFalse(bake.qualified)
        self.assertIn("UNDER-RESOLVED PILOT", bake.source_label)


if __name__ == "__main__":
    unittest.main()
