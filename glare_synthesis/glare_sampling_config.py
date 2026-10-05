"""Sampling distribution for synthetic lens glare: which lights, how many, how strong.

Everything random about a glare sample is drawn from a `GlareSamplingConfig`. The defaults are
the training distribution; review sheets and tests build narrowed copies (one light type, fixed
severity) with `dataclasses.replace`. Parameter ranges and their justification are in
`glare_synthesis/CLAUDE.md`.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class LightSourceType(str, Enum):
    """The light-source families the renderer can reflect in a lens."""

    RING_LIGHT = "ring_light"
    SOFTBOX = "softbox"
    WINDOW = "window"
    SCREEN = "screen"
    POINT_GLINT = "point_glint"
    STRIP_LIGHT = "strip_light"
    CEILING_GRID = "ceiling_grid"
    SKY_WASH = "sky_wash"
    ENVIRONMENT = "environment"


class GlareSeverity(str, Enum):
    """How hard a source hits the sensor: WEAK stays see-through, STRONG blows out its core."""

    WEAK = "weak"
    STRONG = "strong"


def default_light_type_probabilities() -> dict[LightSourceType, float]:
    """Relative frequency of each light type; normalized at sampling time."""
    return {
        LightSourceType.RING_LIGHT: 0.2,
        LightSourceType.SOFTBOX: 0.12,
        LightSourceType.WINDOW: 0.03,
        LightSourceType.SCREEN: 0.2,
        LightSourceType.POINT_GLINT: 0.16,
        LightSourceType.STRIP_LIGHT: 0.14,
        LightSourceType.CEILING_GRID: 0.08,
        LightSourceType.SKY_WASH: 0.04,
        LightSourceType.ENVIRONMENT: 0.03,
    }


def default_hdri_category_probabilities() -> dict[str, float]:
    """Mix of HDRI scene categories for image-based reflections."""
    return {
        "indoor": 0.35,
        "outdoor": 0.35,
        "studio": 0.15,
        "night": 0.15,
    }


def default_source_count_probabilities() -> dict[int, float]:
    """Probability of 1, 2 or 3 light sources in one sample."""
    return {
        1: 0.6,
        2: 0.3,
        3: 0.1,
    }


def default_severity_probabilities() -> dict[GlareSeverity, float]:
    """Severity mix among glared samples (research doc: 45% weak, 40% strong of all samples)."""
    return {
        GlareSeverity.WEAK: 0.53,
        GlareSeverity.STRONG: 0.47,
    }


@dataclass(frozen=True)
class GlareSamplingConfig:
    """All knobs of the glare sampling distribution (defaults = training distribution).

    Ranges are (low, high) tuples. Peak brightness is expressed as the linear light the
    reflection adds at its brightest point, where 1.0 is the sensor clip level.
    """

    no_glare_probability: float = 0.12
    # Share of glared samples whose main reflection is image-based (a mirrored HDRI); the rest are procedural.
    hdri_reflection_share: float = 0.6
    hdri_baked_manifest_path: Path = Path("/Volumes/vega/datasets/glare-off/hdri/polyhaven/hdri_baked_manifest.jsonl")
    hdri_category_probabilities: dict[str, float] = field(default_factory=default_hdri_category_probabilities)
    # Rotate the scene so a bright emitter (sun, window, lamp) lands inside a lens.
    hdri_emitter_aim_probability: float = 0.75
    # Reflectance per front surface: AR-coated vs uncoated lenses.
    coated_reflectance_range: tuple[float, float] = (0.005, 0.015)
    uncoated_reflectance_range: tuple[float, float] = (0.04, 0.08)
    # Exposure of the face relative to the environment (backlit faces see stronger reflections).
    hdri_exposure_jitter_range: tuple[float, float] = (0.5, 12.0)
    # Entrance pupil diameter: phone (~1.5-4 mm) to large-sensor portrait lens (~20 mm).
    camera_aperture_mm_range: tuple[float, float] = (1.5, 12.0)
    hdri_ghost_probability: float = 0.6
    # Chance an HDRI sample also carries one procedural emitter (glint, ring light, strip, screen).
    hdri_extra_emitter_probability: float = 0.25
    face_form_wrap_degrees_range: tuple[float, float] = (0.0, 8.0)
    head_yaw_standard_deviation_degrees: float = 8.0
    # Pantoscopic tilt plus head pitch (positive tips the lens bottom toward the cheek).
    lens_pitch_degrees_range: tuple[float, float] = (-6.0, 14.0)
    # Per-sample rim softness so imperfect masks never print their outline (pixels at 512 wide).
    lens_mask_erosion_pixels_range: tuple[float, float] = (0.0, 2.0)
    lens_mask_feather_sigma_pixels_range: tuple[float, float] = (0.5, 1.5)
    # Reflection stronger toward one side of the lens (incidence angle varies across it).
    edge_gain_probability: float = 0.6
    edge_gain_strength_range: tuple[float, float] = (0.2, 0.7)
    light_type_probabilities: dict[LightSourceType, float] = field(default_factory=default_light_type_probabilities)
    source_count_probabilities: dict[int, float] = field(default_factory=default_source_count_probabilities)
    severity_probabilities: dict[GlareSeverity, float] = field(default_factory=default_severity_probabilities)
    # Peak added linear light per severity; 1.0 = clip. STRONG starts just under clip so some
    # strong sources only clip after the soft knee and tint push one channel over.
    weak_peak_linear_range: tuple[float, float] = (0.06, 0.45)
    strong_peak_linear_range: tuple[float, float] = (0.8, 8.0)
    # Spectacle front-surface base curve in diopters; mirror focal length f = R/2 = 0.265/BC m.
    base_curve_diopter_range: tuple[float, float] = (1.0, 8.0)
    # Rare flat-lens tail (Private Eye measured f up to 2.7 m): large, lens-filling reflections.
    flat_lens_probability: float = 0.1
    flat_lens_focal_length_m_range: tuple[float, float] = (0.3, 2.7)
    camera_distance_m_range: tuple[float, float] = (0.3, 2.0)
    # Anti-reflective coating share; the rest are uncoated (neutral, stronger reflections).
    ar_coating_probability: float = 0.7
    # One lens catches the light, the other does not (head turned, brow or frame occlusion).
    single_lens_probability: float = 0.15
    # Inner-surface reflection: a fainter, rescaled, offset copy.
    ghost_reflection_probability: float = 0.4
    # Low-frequency haze across the lens (sky or room gradient) added on top of the sources.
    veil_probability: float = 0.25
    # Multiplicative low-frequency lumps on the reflection (coating wear, smudges, uneven room light).
    # Always on so no wash or fill is ever a constant tint.
    coating_texture_probability: float = 1.0
    # Residual-colour drift across the lens (incidence angle changes the AR residual hue).
    tint_gradient_probability: float = 0.35
    # Local tone mapping darkening the lens slightly around strong reflections.
    transmission_attenuation_probability: float = 0.15
    transmission_attenuation_range: tuple[float, float] = (0.01, 0.05)
    # Camera: phone soft-knee highlight roll-off vs hard clip; bloom around clipped light.
    soft_knee_probability: float = 0.6
    # Always on: a blown lens must glow softly past the rim, never end as a hard flat shape.
    bloom_probability: float = 1.0
    bloom_sigma_fraction_of_lens_width_range: tuple[float, float] = (0.015, 0.06)
    bloom_strength_range: tuple[float, float] = (0.12, 0.3)
    # Reflection defocus (disc radius, as a fraction of lens width) on top of the photo's own blur.
    defocus_radius_fraction_of_lens_width_range: tuple[float, float] = (0.006, 0.06)
    motion_blur_probability: float = 0.1
    # JPEG applied to the glare delta only (input and target share every other artifact).
    jpeg_probability: float = 0.5
    jpeg_quality_range: tuple[int, int] = (60, 95)
    # Multiplier on the photo's estimated noise level, applied to the glare's own shot noise.
    glare_noise_scale_range: tuple[float, float] = (0.7, 1.4)
