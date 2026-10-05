"""Data classes describing one fully sampled glare scene (every random choice, nothing rendered).

Splitting sampling (`glare_scene_sampling`) from rendering (`render_lens_glare`) lets review
tools re-render one fixed scene at several brightnesses, and keeps rendering itself random-free.
"""

from dataclasses import dataclass, field

import numpy as np

from glare_synthesis.glare_sampling_config import GlareSeverity, LightSourceType


@dataclass
class ReflectionPlacement:
    """Where and how one source's virtual image lands in one lens.

    Normalized lens coordinates: origin at the lens centroid, 1 unit = half the nominal
    (un-foreshortened) lens width, x right, y down.
    """

    lens_index: int
    center_offset_normalized: np.ndarray
    # Source-plane units (canvas spans 2) to normalized lens units.
    scale_normalized_per_source_unit: float
    rotation_degrees: float
    horizontal_foreshortening: float
    vertical_squash: float
    barrel_distortion_k1: float
    perspective_coefficients: np.ndarray
    intensity_multiplier: float


@dataclass
class SampledLightSource:
    """One light source: its coloured canvas, brightness, blur, and per-lens placements."""

    light_type: LightSourceType
    severity: GlareSeverity
    canvas_linear_rgb: np.ndarray
    peak_linear: float
    extends_beyond_canvas: bool
    placements: list[ReflectionPlacement]
    ghost_placements: list[ReflectionPlacement]
    defocus_radius_pixels: float
    motion_blur_length_pixels: float
    motion_blur_angle_degrees: float


@dataclass
class LensVeil:
    """Low-frequency haze over each whole lens: colour, peak strength, gradient direction."""

    veil_linear_rgb: np.ndarray
    peak_linear: float
    gradient_angle_radians: float
    gradient_floor: float


@dataclass
class HdriReflection:
    """An image-based reflection: the lens front surface (and a fainter back surface) mirroring an HDRI.

    Angles are radians. Lens tilts are (yaw about the vertical axis, pitch about the horizontal
    axis) of each lens's surface normal, in camera frame; the environment rotation is applied to
    reflected directions before the equirect lookup.
    """

    hdri_id: str
    environment_yaw: float
    environment_pitch: float
    front_surface_radius_m: float
    lens_width_m: float
    camera_distance_m: float
    lens_normal_tilts: list[np.ndarray]
    lens_intensity_multipliers: list[float]
    # Reflectance x exposure jitter / skin albedo; times the face's linear luminance = light scale.
    relative_exposure: float
    defocus_radius_pixels: float
    ghost_surface_radius_m: float | None = None
    ghost_intensity: float = 0.0
    ghost_normal_tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class GlareScene:
    """Everything needed to render glare onto one crop deterministically."""

    is_glare_free: bool
    light_sources: list[SampledLightSource] = field(default_factory=list)
    hdri_reflection: HdriReflection | None = None
    coating_tint_linear_rgb: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    # Second tint the residual colour drifts toward across the lens, and the drift direction.
    gradient_tint_linear_rgb: np.ndarray | None = None
    tint_gradient_angle_radians: float = 0.0
    veil: LensVeil | None = None
    coating_texture_amplitude: float = 0.0
    coating_texture_grid_cells: int = 4
    coating_texture_seed: int = 0
    rim_falloff_floor: float = 1.0
    # Reflection strength ramp across each lens (incidence angle): gain 1 +- strength along the angle.
    edge_gain_strength: float = 0.0
    edge_gain_angle_radians: float = 0.0
    # Per-sample soft rim: random erosion and feather so mask imperfections never print an outline.
    lens_mask_erosion_pixels: float = 1.0
    lens_mask_feather_sigma_pixels: float = 0.8
    transmission_attenuation: float = 0.0
    soft_knee_start: float | None = None
    bloom_sigma_pixels: float = 0.0
    bloom_strength: float = 0.0
    glare_noise_scale: float = 1.0
    noise_seed: int = 0
    jpeg_quality: int | None = None
