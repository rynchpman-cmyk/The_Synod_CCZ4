"""Reduced binary-neutron-star models and standalone ShaderToy export.

The model uses geometric units with ``G = c = M_total = 1``.  Its inspiral is
the leading, adiabatic, quasi-circular quadrupole solution.  Tidal bulges use
the supplied quadrupolar Love numbers.  Contact, merger, and ringdown are a
smooth visualization bridge rather than a numerical-relativity waveform.

The exporter intentionally emits a single ShaderToy ``Image`` program.  No
checkpoint, texture, Python process, or network connection is required after
the GLSL has been generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SOLAR_MASS_TIME_SECONDS = 4.925490947e-6


@dataclass(frozen=True)
class BNSShaderParameters:
    """Physical and presentation parameters for the reduced BNS model.

    ``mass_ratio`` is ``m2/m1`` and is restricted to ``(0, 1]``.  Distances
    are measured in total-mass units.  ``repair_strength`` controls only the
    auxiliary visual layer; it is not interpreted as a calibrated prediction.
    """

    mass_ratio: float = 1.0
    compactness_1: float = 0.17
    compactness_2: float = 0.17
    love_k2_1: float = 0.09
    love_k2_2: float = 0.09
    initial_separation: float = 13.5
    inspiral_fraction: float = 0.68
    merger_fraction: float = 0.80
    animation_seconds: float = 32.0
    total_mass_solar: float = 2.7
    postmerger_f2_khz: float = 3.0
    postmerger_duration_ms: float = 4.0
    postmerger_damping_ms: float = 2.5
    prompt_collapse: bool = False
    show_horizon_wireframe: bool = True
    horizon_formation_fraction: float = 0.86
    wave_visual_gain: float = 1.0
    lensing_gain: float = 0.82
    repair_strength: float = 0.22
    quality: int = 1

    def validate(self) -> None:
        if not 0.0 < self.mass_ratio <= 1.0:
            raise ValueError("mass_ratio must satisfy 0 < q <= 1")
        for name, value in (
            ("compactness_1", self.compactness_1),
            ("compactness_2", self.compactness_2),
        ):
            if not 0.08 <= value <= 0.35:
                raise ValueError(f"{name} must lie in [0.08, 0.35]")
        for name, value in (("love_k2_1", self.love_k2_1), ("love_k2_2", self.love_k2_2)):
            if not 0.0 <= value <= 0.3:
                raise ValueError(f"{name} must lie in [0, 0.3]")
        if not 0.2 < self.inspiral_fraction < self.merger_fraction < 0.95:
            raise ValueError(
                "timeline must satisfy 0.2 < inspiral_fraction < "
                "merger_fraction < 0.95"
            )
        if self.animation_seconds <= 1.0 or self.total_mass_solar <= 0.0:
            raise ValueError("animation time and total mass must be positive")
        if not 1.5 <= self.postmerger_f2_khz <= 5.0:
            raise ValueError("postmerger_f2_khz must lie in [1.5, 5.0]")
        if self.postmerger_duration_ms <= 0.0 or self.postmerger_damping_ms <= 0.0:
            raise ValueError("post-merger duration and damping time must be positive")
        if not self.merger_fraction < self.horizon_formation_fraction < 0.98:
            raise ValueError(
                "horizon_formation_fraction must follow merger and precede loop end"
            )
        if self.wave_visual_gain < 0.0 or self.lensing_gain < 0.0:
            raise ValueError("visual gains cannot be negative")
        if not 0.0 <= self.repair_strength <= 2.0:
            raise ValueError("repair_strength must lie in [0, 2]")
        if self.quality not in (0, 1, 2):
            raise ValueError("quality must be 0, 1, or 2")
        if self.initial_separation <= 1.03 * self.contact_separation:
            raise ValueError(
                "initial_separation must exceed the contact separation by at least 3%"
            )

    @property
    def mass_1(self) -> float:
        return 1.0 / (1.0 + self.mass_ratio)

    @property
    def mass_2(self) -> float:
        return self.mass_ratio / (1.0 + self.mass_ratio)

    @property
    def symmetric_mass_ratio(self) -> float:
        return self.mass_1 * self.mass_2

    @property
    def chirp_mass(self) -> float:
        return self.symmetric_mass_ratio ** (3.0 / 5.0)

    @property
    def radius_1(self) -> float:
        return self.mass_1 / self.compactness_1

    @property
    def radius_2(self) -> float:
        return self.mass_2 / self.compactness_2

    @property
    def contact_separation(self) -> float:
        return self.radius_1 + self.radius_2

    @property
    def inspiral_time(self) -> float:
        eta = self.symmetric_mass_ratio
        return (5.0 / (256.0 * eta)) * (
            self.initial_separation**4 - self.contact_separation**4
        )

    @property
    def contact_omega(self) -> float:
        return self.contact_separation ** (-1.5)

    @property
    def contact_phase(self) -> float:
        eta = self.symmetric_mass_ratio
        return (
            self.initial_separation**2.5 - self.contact_separation**2.5
        ) / (32.0 * eta)

    @property
    def inspiral_orbits(self) -> float:
        return self.contact_phase / (2.0 * np.pi)

    @property
    def contact_display_angular_speed(self) -> float:
        """Angular speed in radians per displayed second at first contact."""
        return (
            self.contact_omega
            * self.inspiral_time
            / self.inspiral_fraction
            / self.animation_seconds
        )

    @property
    def ringdown_display_angular_speed(self) -> float:
        """Late bar-mode angular speed in radians per displayed second."""
        return (
            np.pi
            * self.postmerger_f2_khz
            * self.postmerger_duration_ms
            / (1.0 - self.merger_fraction)
            / self.animation_seconds
        )

    @property
    def postmerger_f2_omega_M(self) -> float:
        """Dimensionless physical angular frequency ``omega_22 M``."""
        mass_seconds = self.total_mass_solar * SOLAR_MASS_TIME_SECONDS
        return 2.0 * np.pi * self.postmerger_f2_khz * 1.0e3 * mass_seconds

    @property
    def contact_to_merger_orbits(self) -> float:
        contact_rate = self.contact_display_angular_speed * self.animation_seconds
        ring_rate = self.ringdown_display_angular_speed * self.animation_seconds
        phase_change = 0.5 * (contact_rate + ring_rate) * (
            self.merger_fraction - self.inspiral_fraction
        )
        return phase_change / (2.0 * np.pi)

    @property
    def tidal_bulge_1_at_contact(self) -> float:
        return (
            1.5
            * self.love_k2_1
            * (self.mass_2 / self.mass_1)
            * (self.radius_1 / self.contact_separation) ** 3
        )

    @property
    def tidal_bulge_2_at_contact(self) -> float:
        return (
            1.5
            * self.love_k2_2
            * (self.mass_1 / self.mass_2)
            * (self.radius_2 / self.contact_separation) ** 3
        )


@dataclass(frozen=True)
class BNSOrbitalState:
    normalized_time: np.ndarray
    separation: np.ndarray
    orbital_phase: np.ndarray
    orbital_omega: np.ndarray
    merger_blend: np.ndarray
    ringdown_envelope: np.ndarray
    remnant_phase: np.ndarray
    postmerger_time_ms: np.ndarray
    wave_phase_22: np.ndarray


@dataclass(frozen=True)
class MOTSShaderBake:
    """Compact non-axisymmetric horizon packet for a standalone shader.

    Shape coefficients describe ``log R = sum c_lm Y_lm``.  The response
    coefficients are not the MOTS residual (which should vanish); they encode
    a signed outgoing-expansion response measured on prescribed offset
    surfaces or an equivalently normalized physical-normal response.
    """

    normalized_times: np.ndarray
    centers: np.ndarray
    modes: tuple[tuple[int, int], ...]
    log_radius_coefficients: np.ndarray
    expansion_response_coefficients: np.ndarray
    principal_eigenvalues: np.ndarray | None = None
    display_scale: float = 1.0
    source_kind: str = "unspecified"
    source_label: str = ""
    qualified: bool = False
    source_parameter_name: str = ""
    source_parameter_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.normalized_times, dtype=float)
        centers = np.asarray(self.centers, dtype=float)
        shape = np.asarray(self.log_radius_coefficients, dtype=float)
        response = np.asarray(self.expansion_response_coefficients, dtype=float)
        eigenvalues = (
            np.zeros(len(times), dtype=float)
            if self.principal_eigenvalues is None
            else np.asarray(self.principal_eigenvalues, dtype=float)
        )
        source_values = (
            None
            if self.source_parameter_values is None
            else np.asarray(self.source_parameter_values, dtype=float)
        )
        modes = tuple((int(ell), int(m)) for ell, m in self.modes)
        if times.ndim != 1 or not 2 <= len(times) <= 16:
            raise ValueError("MOTS bake requires 2 to 16 time frames")
        if np.any(np.diff(times) <= 0.0) or times[0] < 0.0 or times[-1] > 1.0:
            raise ValueError("MOTS bake times must increase within [0,1]")
        if not modes or modes[0] != (0, 0) or len(set(modes)) != len(modes):
            raise ValueError("MOTS bake modes must be unique and start with (0,0)")
        if any(ell < 0 or ell > 16 or abs(m) > ell for ell, m in modes):
            raise ValueError("Shader bake supports real harmonics through ell=16")
        expected = (len(times), len(modes))
        if centers.shape != (len(times), 3):
            raise ValueError("MOTS bake centers must have shape (frames,3)")
        if shape.shape != expected or response.shape != expected:
            raise ValueError("MOTS coefficient arrays must have shape (frames,modes)")
        if eigenvalues.shape != (len(times),):
            raise ValueError("principal_eigenvalues must have shape (frames,)")
        if source_values is not None and source_values.shape != (len(times),):
            raise ValueError("source_parameter_values must have shape (frames,)")
        if not np.isfinite(self.display_scale) or self.display_scale <= 0.0:
            raise ValueError("display_scale must be positive and finite")
        if not all(
            np.all(np.isfinite(array))
            for array in (
                times,
                centers,
                shape,
                response,
                eigenvalues,
                source_values if source_values is not None else np.zeros(1),
            )
        ):
            raise ValueError("MOTS bake data must be finite")
        object.__setattr__(self, "normalized_times", times.copy())
        object.__setattr__(self, "centers", centers.copy())
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "log_radius_coefficients", shape.copy())
        object.__setattr__(self, "expansion_response_coefficients", response.copy())
        object.__setattr__(self, "principal_eigenvalues", eigenvalues.copy())
        object.__setattr__(self, "display_scale", float(self.display_scale))
        object.__setattr__(self, "source_kind", str(self.source_kind))
        object.__setattr__(self, "source_label", str(self.source_label))
        object.__setattr__(self, "qualified", bool(self.qualified))
        object.__setattr__(self, "source_parameter_name", str(self.source_parameter_name))
        object.__setattr__(
            self,
            "source_parameter_values",
            None if source_values is None else source_values.copy(),
        )

    @classmethod
    def from_results(
        cls,
        normalized_times: Sequence[float],
        results: Sequence[Any],
        expansion_response_coefficients: Any,
        principal_eigenvalues: Any | None = None,
    ) -> "MOTSShaderBake":
        if len(results) != len(normalized_times) or not results:
            raise ValueError("one MOTS result is required per normalized time")
        modes = tuple(results[0].modes)
        if any(tuple(result.modes) != modes for result in results):
            raise ValueError("all MOTS results must use the same modes")
        if any(not bool(result.found) for result in results):
            raise ValueError("only found/qualified MOTS results may be baked")
        return cls(
            np.asarray(normalized_times, dtype=float),
            np.asarray([result.center for result in results], dtype=float),
            modes,
            np.asarray([result.coefficients for result in results], dtype=float),
            np.asarray(expansion_response_coefficients, dtype=float),
            principal_eigenvalues,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "causal_repair.mots_shader_bake.v2",
            "normalized_times": self.normalized_times.tolist(),
            "centers": self.centers.tolist(),
            "modes": [list(mode) for mode in self.modes],
            "log_radius_coefficients": self.log_radius_coefficients.tolist(),
            "expansion_response_coefficients": (
                self.expansion_response_coefficients.tolist()
            ),
            "principal_eigenvalues": self.principal_eigenvalues.tolist(),
            "display_scale": self.display_scale,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "qualified": self.qualified,
            "source_parameter_name": self.source_parameter_name,
            "source_parameter_values": (
                None
                if self.source_parameter_values is None
                else self.source_parameter_values.tolist()
            ),
            "response_semantics": (
                "signed outgoing-expansion response on prescribed offset surfaces; "
                "not the vanishing MOTS residual"
            ),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "MOTSShaderBake":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") not in {
            "causal_repair.mots_shader_bake.v1",
            "causal_repair.mots_shader_bake.v2",
        }:
            raise ValueError("unsupported MOTS shader bake format")
        return cls(
            data["normalized_times"],
            data["centers"],
            tuple(tuple(mode) for mode in data["modes"]),
            data["log_radius_coefficients"],
            data["expansion_response_coefficients"],
            data.get("principal_eigenvalues"),
            data.get("display_scale", 1.0),
            data.get("source_kind", "unspecified"),
            data.get("source_label", ""),
            data.get("qualified", False),
            data.get("source_parameter_name", ""),
            data.get("source_parameter_values"),
        )

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output


class BNSReducedModel:
    """Leading-order inspiral with a continuous merger/ringdown bridge."""

    def __init__(self, parameters: BNSShaderParameters | None = None) -> None:
        self.parameters = parameters or BNSShaderParameters()
        self.parameters.validate()

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def orbital_state(self, normalized_time: Any) -> BNSOrbitalState:
        p = self.parameters
        u = np.clip(np.asarray(normalized_time, dtype=float), 0.0, 1.0)
        inspiral_progress = np.clip(u / p.inspiral_fraction, 0.0, 1.0)
        radius_fourth = (
            p.initial_separation**4
            + (p.contact_separation**4 - p.initial_separation**4)
            * inspiral_progress
        )
        inspiral_separation = np.maximum(radius_fourth, 0.0) ** 0.25
        inspiral_phase = (
            p.initial_separation**2.5 - inspiral_separation**2.5
        ) / (32.0 * p.symmetric_mass_ratio)
        merge_span = p.merger_fraction - p.inspiral_fraction
        merge_progress = np.clip(
            (u - p.inspiral_fraction) / merge_span, 0.0, 1.0
        )
        contact_rate = p.contact_display_angular_speed * p.animation_seconds
        ring_rate = p.ringdown_display_angular_speed * p.animation_seconds
        merger_phase = p.contact_phase + merge_span * (
            contact_rate * merge_progress
            + 0.5 * (ring_rate - contact_rate) * merge_progress**2
        )
        phase_at_merger = p.contact_phase + 0.5 * merge_span * (
            contact_rate + ring_rate
        )
        ring_phase = phase_at_merger + ring_rate * np.maximum(
            u - p.merger_fraction, 0.0
        )
        phase = np.where(
            u <= p.inspiral_fraction,
            inspiral_phase,
            np.where(u <= p.merger_fraction, merger_phase, ring_phase),
        )
        merger = self._smoothstep(p.inspiral_fraction, p.merger_fraction, u)
        separation = (1.0 - merger) * inspiral_separation + merger * (
            0.08 * p.contact_separation
        )
        omega = np.where(
            u <= p.inspiral_fraction,
            inspiral_separation ** (-1.5),
            p.contact_omega,
        )
        postmerger_time_ms = p.postmerger_duration_ms * np.clip(
            (u - p.merger_fraction) / (1.0 - p.merger_fraction), 0.0, 1.0
        )
        # Unity through contact/merger, followed by the post-merger decay.
        ringdown = np.exp(-postmerger_time_ms / p.postmerger_damping_ms)
        remnant_phase = phase.copy()
        wave_phase_22 = np.where(
            u <= p.merger_fraction,
            2.0 * phase,
            2.0 * phase_at_merger
            + 2.0 * np.pi * p.postmerger_f2_khz * postmerger_time_ms,
        )
        return BNSOrbitalState(
            u,
            separation,
            phase,
            omega,
            merger,
            ringdown,
            remnant_phase,
            postmerger_time_ms,
            wave_phase_22,
        )

    def leading_strain(
        self,
        normalized_time: Any,
        distance: float = 100.0,
        inclination: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return leading-order ``(h_plus, h_cross)`` in total-mass units."""
        if distance <= 0.0:
            raise ValueError("distance must be positive")
        state = self.orbital_state(normalized_time)
        cosine = np.cos(float(inclination))
        amplitude = (
            4.0
            * self.parameters.symmetric_mass_ratio
            / (distance * np.maximum(state.separation, 1.0e-12))
        )
        # The post-contact bridge is intentionally damped into the ringdown.
        bridge = (1.0 - state.merger_blend) + state.merger_blend * state.ringdown_envelope
        amplitude *= bridge
        plus = 0.5 * amplitude * (1.0 + cosine**2) * np.cos(state.wave_phase_22)
        cross = amplitude * cosine * np.sin(state.wave_phase_22)
        return plus, cross

    def manifest(self) -> dict[str, Any]:
        p = self.parameters
        return {
            "model": "leading_quadrupole_tidal_bns_with_visual_merger_bridge",
            "units": "G=c=M_total=1",
            "parameters": asdict(p),
            "derived": {
                "mass_1": p.mass_1,
                "mass_2": p.mass_2,
                "symmetric_mass_ratio": p.symmetric_mass_ratio,
                "chirp_mass": p.chirp_mass,
                "radius_1": p.radius_1,
                "radius_2": p.radius_2,
                "contact_separation": p.contact_separation,
                "inspiral_time": p.inspiral_time,
                "contact_omega": p.contact_omega,
                "contact_phase": p.contact_phase,
                "inspiral_orbits": p.inspiral_orbits,
                "contact_display_angular_speed": p.contact_display_angular_speed,
                "ringdown_display_angular_speed": p.ringdown_display_angular_speed,
                "contact_to_merger_orbits": p.contact_to_merger_orbits,
                "postmerger_f2_omega_M": p.postmerger_f2_omega_M,
                "postmerger_f2_cycles": (
                    p.postmerger_f2_khz * p.postmerger_duration_ms
                ),
                "tidal_bulge_1_at_contact": p.tidal_bulge_1_at_contact,
                "tidal_bulge_2_at_contact": p.tidal_bulge_2_at_contact,
            },
            "qualification": {
                "inspiral": "leading adiabatic quasi-circular quadrupole model",
                "tides": "Love-number equilibrium bulge approximation",
                "merger_and_ringdown": "phenomenological visualization bridge",
                "repair_layer": "illustrative unless calibrated to a solver campaign",
            },
        }


def _glsl_float(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "e" not in rendered.lower() and "." not in rendered:
        rendered += ".0"
    return rendered


def _mots_glsl_values(bake: MOTSShaderBake | None) -> dict[str, str]:
    if bake is None:
        times = np.array([0.0])
        centers = np.zeros((1, 3))
        modes = ((0, 0),)
        shape = np.zeros((1, 1))
        response = np.zeros((1, 1))
        eigenvalues = np.zeros(1)
        display_scale = 1.0
        enabled = "0"
    else:
        times = bake.normalized_times
        centers = bake.centers
        modes = bake.modes
        shape = bake.log_radius_coefficients
        response = bake.expansion_response_coefficients
        eigenvalues = bake.principal_eigenvalues
        display_scale = bake.display_scale
        enabled = "1"
    normalizations = []
    for ell, m in modes:
        order = abs(m)
        if ell == 0:
            normalizations.append(1.0)
        else:
            log_norm = 0.5 * (
                math.log((2.0 * ell + 1.0) / (4.0 * math.pi))
                + math.lgamma(ell - order + 1.0)
                - math.lgamma(ell + order + 1.0)
            )
            normalizations.append(math.exp(log_norm))
    return {
        "HAS_MOTS_BAKE": enabled,
        "MOTS_FRAME_COUNT": str(len(times)),
        "MOTS_MODE_COUNT": str(len(modes)),
        "MOTS_TIMES": ", ".join(_glsl_float(value) for value in times),
        "MOTS_CENTERS": ", ".join(
            "vec3(" + ", ".join(_glsl_float(value) for value in center) + ")"
            for center in centers
        ),
        "MOTS_MODES": ", ".join(
            f"ivec2({ell}, {m})" for ell, m in modes
        ),
        "MOTS_NORMALIZATIONS": ", ".join(
            _glsl_float(value) for value in normalizations
        ),
        "MOTS_SHAPE_COEFFICIENTS": ", ".join(
            _glsl_float(value) for value in shape.ravel()
        ),
        "MOTS_RESPONSE_COEFFICIENTS": ", ".join(
            _glsl_float(value) for value in response.ravel()
        ),
        "MOTS_PRINCIPAL_EIGENVALUES": ", ".join(
            _glsl_float(value) for value in eigenvalues
        ),
        "MOTS_DISPLAY_SCALE": _glsl_float(display_scale),
    }


def render_shadertoy_glsl(
    parameters: BNSShaderParameters | None = None,
    *,
    template_path: Path | None = None,
    mots_bake: MOTSShaderBake | None = None,
) -> str:
    """Render the self-contained ShaderToy ``Image`` source."""
    model = BNSReducedModel(parameters)
    p = model.parameters
    if template_path is None:
        template_path = (
            Path(__file__).resolve().parents[2]
            / "shadertoy"
            / "templates"
            / "bns_causal_repair.glsl.in"
        )
    template = template_path.read_text(encoding="utf-8")
    values = {
        "MASS_1": p.mass_1,
        "MASS_2": p.mass_2,
        "ETA": p.symmetric_mass_ratio,
        "RADIUS_1": p.radius_1,
        "RADIUS_2": p.radius_2,
        "INITIAL_SEPARATION": p.initial_separation,
        "CONTACT_SEPARATION": p.contact_separation,
        "INSPIRAL_FRACTION": p.inspiral_fraction,
        "MERGER_FRACTION": p.merger_fraction,
        "ANIMATION_SECONDS": p.animation_seconds,
        "TOTAL_MASS_SOLAR": p.total_mass_solar,
        "POSTMERGER_F2_KHZ": p.postmerger_f2_khz,
        "POSTMERGER_DURATION_MS": p.postmerger_duration_ms,
        "POSTMERGER_DAMPING_MS": p.postmerger_damping_ms,
        "POSTMERGER_F2_OMEGA_M": p.postmerger_f2_omega_M,
        "CONTACT_OMEGA": p.contact_omega,
        "CONTACT_PHASE": p.contact_phase,
        "INSPIRAL_TIME": p.inspiral_time,
        "TIDAL_1": p.tidal_bulge_1_at_contact,
        "TIDAL_2": p.tidal_bulge_2_at_contact,
        "WAVE_GAIN": p.wave_visual_gain,
        "LENSING_GAIN": p.lensing_gain,
        "REPAIR_STRENGTH": p.repair_strength,
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", _glsl_float(value))
    for key, value in _mots_glsl_values(mots_bake).items():
        template = template.replace("{{" + key + "}}", value)
    template = template.replace("{{QUALITY}}", str(p.quality))
    template = template.replace("{{PROMPT_COLLAPSE}}", "1" if p.prompt_collapse else "0")
    template = template.replace(
        "{{SHOW_HORIZON}}", "1" if p.show_horizon_wireframe else "0"
    )
    template = template.replace(
        "{{HORIZON_FORMATION_FRACTION}}",
        _glsl_float(p.horizon_formation_fraction),
    )
    unresolved = [token for token in template.split() if "{{" in token or "}}" in token]
    if unresolved:
        raise ValueError(f"unresolved shader template tokens: {unresolved[:5]}")
    return template


def write_shadertoy_package(
    output: str | Path,
    parameters: BNSShaderParameters | None = None,
    *,
    manifest_output: str | Path | None = None,
    template_path: Path | None = None,
    mots_bake: MOTSShaderBake | None = None,
) -> tuple[Path, Path]:
    """Write a paste-ready GLSL file and its machine-readable manifest."""
    model = BNSReducedModel(parameters)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_shadertoy_glsl(
            model.parameters,
            template_path=template_path,
            mots_bake=mots_bake,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = (
        Path(manifest_output)
        if manifest_output is not None
        else output_path.with_suffix(".json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = model.manifest()
    manifest["mots_bake"] = (
        None
        if mots_bake is None
        else {
            "format": "causal_repair.mots_shader_bake.v2",
            "frames": len(mots_bake.normalized_times),
            "modes": [list(mode) for mode in mots_bake.modes],
            "first_time": float(mots_bake.normalized_times[0]),
            "last_time": float(mots_bake.normalized_times[-1]),
            "response_semantics": "signed offset-surface outgoing-expansion response",
            "stability_semantics": "principal physical-normal MOTS eigenvalue lambda_0",
            "principal_eigenvalue_range": [
                float(np.min(mots_bake.principal_eigenvalues)),
                float(np.max(mots_bake.principal_eigenvalues)),
            ],
            "display_scale": mots_bake.display_scale,
            "source_kind": mots_bake.source_kind,
            "source_label": mots_bake.source_label,
            "qualified": mots_bake.qualified,
            "source_parameter_name": mots_bake.source_parameter_name,
            "source_parameter_values": (
                None
                if mots_bake.source_parameter_values is None
                else mots_bake.source_parameter_values.tolist()
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, manifest_path


__all__ = [
    "BNSOrbitalState",
    "BNSReducedModel",
    "BNSShaderParameters",
    "MOTSShaderBake",
    "render_shadertoy_glsl",
    "write_shadertoy_package",
]
