import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_repair.bns_shadertoy import (
    BNSReducedModel,
    BNSShaderParameters,
    MOTSShaderBake,
    render_shadertoy_glsl,
    write_shadertoy_package,
)


class BNSShaderToyTests(unittest.TestCase):
    def test_default_binary_has_consistent_derived_scales(self):
        p = BNSShaderParameters()
        p.validate()
        self.assertAlmostEqual(p.mass_1 + p.mass_2, 1.0)
        self.assertAlmostEqual(p.symmetric_mass_ratio, 0.25)
        self.assertAlmostEqual(p.contact_separation, 1.0 / 0.17)
        self.assertGreater(p.inspiral_time, 0.0)
        self.assertGreater(p.inspiral_orbits, 1.0)
        self.assertGreater(p.tidal_bulge_1_at_contact, 0.0)
        self.assertAlmostEqual(p.postmerger_f2_omega_M, 0.2507, delta=5.0e-4)
        self.assertAlmostEqual(
            p.postmerger_f2_khz * p.postmerger_duration_ms,
            12.0,
        )

    def test_inspiral_is_monotone_and_continuous_at_contact(self):
        model = BNSReducedModel()
        p = model.parameters
        u = np.linspace(0.0, p.inspiral_fraction, 1001)
        state = model.orbital_state(u)
        self.assertTrue(np.all(np.diff(state.separation) <= 1.0e-12))
        self.assertTrue(np.all(np.diff(state.orbital_phase) >= -1.0e-12))
        self.assertAlmostEqual(float(state.separation[0]), p.initial_separation)
        self.assertAlmostEqual(float(state.separation[-1]), p.contact_separation)
        self.assertAlmostEqual(float(state.orbital_phase[-1]), p.contact_phase)
        around = model.orbital_state(
            np.array([p.inspiral_fraction - 1.0e-9, p.inspiral_fraction + 1.0e-9])
        )
        self.assertLess(abs(float(np.diff(around.separation)[0])), 1.0e-5)
        self.assertLess(abs(float(np.diff(around.orbital_phase)[0])), 1.0e-5)

    def test_strain_is_finite_and_ringdown_decays(self):
        model = BNSReducedModel()
        p = model.parameters
        u = np.linspace(0.0, 1.0, 1000)
        plus, cross = model.leading_strain(u, distance=100.0, inclination=0.7)
        self.assertTrue(np.all(np.isfinite(plus)))
        self.assertTrue(np.all(np.isfinite(cross)))
        state = model.orbital_state(np.array([p.merger_fraction, 1.0]))
        self.assertGreater(state.ringdown_envelope[0], state.ringdown_envelope[1])

    def test_remnant_rotation_is_continuous_and_does_not_freeze_at_contact(self):
        model = BNSReducedModel()
        p = model.parameters
        contact_to_merger = model.orbital_state(
            np.array([p.inspiral_fraction, p.merger_fraction - 1.0e-6])
        )
        self.assertGreater(
            contact_to_merger.remnant_phase[1],
            contact_to_merger.remnant_phase[0],
        )
        around_merger = model.orbital_state(
            np.array([p.merger_fraction - 1.0e-9, p.merger_fraction + 1.0e-9])
        )
        self.assertLess(abs(float(np.diff(around_merger.remnant_phase)[0])), 1.0e-6)
        ringdown = model.orbital_state(np.array([p.merger_fraction, 1.0]))
        self.assertGreater(ringdown.remnant_phase[1], ringdown.remnant_phase[0])

        epsilon = 1.0e-6
        contact = model.orbital_state(
            np.array(
                [
                    p.inspiral_fraction - epsilon,
                    p.inspiral_fraction,
                    p.inspiral_fraction + epsilon,
                ]
            )
        ).remnant_phase
        left_rate = (contact[1] - contact[0]) / epsilon
        right_rate = (contact[2] - contact[1]) / epsilon
        self.assertAlmostEqual(left_rate / right_rate, 1.0, delta=2.0e-4)

        merger = model.orbital_state(
            np.array(
                [
                    p.merger_fraction - epsilon,
                    p.merger_fraction,
                    p.merger_fraction + epsilon,
                ]
            )
        ).remnant_phase
        left_rate = (merger[1] - merger[0]) / epsilon
        right_rate = (merger[2] - merger[1]) / epsilon
        self.assertAlmostEqual(left_rate / right_rate, 1.0, delta=2.0e-4)

    def test_shader_is_standalone_and_fully_substituted(self):
        shader = render_shadertoy_glsl()
        self.assertIn("void mainImage(out vec4 fragColor, in vec2 fragCoord)", shader)
        self.assertIn("const float ETA = 0.25;", shader)
        self.assertIn("const float F2_KHZ = 3.0;", shader)
        self.assertIn("cos(o.wavePhase22)", shader)
        self.assertIn("#define PROMPT_COLLAPSE_BRANCH 0", shader)
        self.assertIn("#define SHOW_HORIZON_WIREFRAME 1", shader)
        self.assertNotIn("{{", shader)
        self.assertNotIn("sampler2D", shader)
        self.assertNotIn("#include", shader)
        self.assertEqual(shader.count("void mainImage"), 1)

    def test_prompt_collapse_variant_enables_formation_gated_wireframe(self):
        shader = render_shadertoy_glsl(
            BNSShaderParameters(prompt_collapse=True, horizon_formation_fraction=0.85)
        )
        self.assertIn("#define PROMPT_COLLAPSE_BRANCH 1", shader)
        self.assertIn("const float HORIZON_FORMATION_U = 0.85;", shader)
        self.assertIn("horizonWireframe(wirePoint)", shader)

    def test_nonaxisymmetric_mots_bake_round_trip_and_shader_embedding(self):
        bake = MOTSShaderBake(
            np.array([0.86, 0.94]),
            np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.0]]),
            ((0, 0), (2, -2), (2, 0), (2, 2)),
            np.array(
                [
                    [np.log(2.0), 0.03, -0.02, 0.08],
                    [np.log(1.8), -0.02, -0.04, 0.04],
                ]
            ),
            np.array(
                [
                    [0.0, -0.4, 0.1, 0.5],
                    [0.0, -0.2, 0.05, 0.25],
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = bake.write_json(Path(directory) / "mots.json")
            restored = MOTSShaderBake.from_json(path)
            np.testing.assert_allclose(
                restored.log_radius_coefficients,
                bake.log_radius_coefficients,
            )
        shader = render_shadertoy_glsl(
            BNSShaderParameters(prompt_collapse=True),
            mots_bake=bake,
        )
        self.assertIn("#define HAS_MOTS_BAKE 1", shader)
        self.assertIn("const int MOTS_FRAME_COUNT = 2;", shader)
        self.assertIn("ivec2(2, -2)", shader)
        self.assertIn("repairOutGlow", shader)
        self.assertIn("MOTS_LAMBDA0", shader)
        self.assertIn("associatedLegendre", shader)
        self.assertNotIn("{{", shader)

    def test_package_writer_emits_reproducible_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "binary.glsl"
            shader_path, manifest_path = write_shadertoy_package(output)
            self.assertTrue(shader_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["units"], "G=c=M_total=1")
            self.assertEqual(
                manifest["qualification"]["repair_layer"],
                "illustrative unless calibrated to a solver campaign",
            )
            first = shader_path.read_bytes()
            write_shadertoy_package(output)
            self.assertEqual(first, shader_path.read_bytes())

    def test_checked_in_publication_build_matches_generator(self):
        root = Path(__file__).resolve().parents[1]
        shader_path = root / "shadertoy" / "BNS_CAUSAL_REPAIR.glsl"
        manifest_path = root / "shadertoy" / "BNS_CAUSAL_REPAIR.json"
        self.assertEqual(
            shader_path.read_text(encoding="utf-8"),
            render_shadertoy_glsl(),
        )
        expected_manifest = BNSReducedModel().manifest()
        expected_manifest["mots_bake"] = None
        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            expected_manifest,
        )

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            BNSReducedModel(BNSShaderParameters(mass_ratio=1.2))
        with self.assertRaises(ValueError):
            BNSReducedModel(BNSShaderParameters(initial_separation=5.0))


if __name__ == "__main__":
    unittest.main()
