"""Draw a random `GlareScene` for a crop: lens optics, light sources, placements, camera settings.

Mirror model (the size of a reflection):
- The lens front surface is a convex mirror of focal length f = R/2. For a source of size H at
  distance d, the virtual image has size H * f / (d + f) and sits d * f / (d + f) behind the lens.
- Seen from a camera at distance D, its size relative to the lens is
  H * f / (d + f) * D / (D + d_i) / lens_width.
- f comes from spectacle base curves (1-8 D, R = 0.53 / BC m, so f = 3-26 cm), plus a rare flat
  tail (0.3-2.7 m, the Private Eye measurements) that produces lens-filling washes.
"""

import dataclasses

import numpy as np

from glare_synthesis.color_science import color_temperature_to_linear_rgb, sample_ar_coating_tint
from glare_synthesis.glare_sampling_config import GlareSamplingConfig, GlareSeverity, LightSourceType
from glare_synthesis.glare_scene import GlareScene, LensVeil, ReflectionPlacement, SampledLightSource
from glare_synthesis.hdri_environment_library import HdriEnvironmentLibrary
from glare_synthesis.hdri_scene_sampling import sample_hdri_reflection
from glare_synthesis.lens_geometry import REFERENCE_CROP_WIDTH, LensGeometry
from glare_synthesis.light_source_registry import LIGHT_SOURCE_SPECS, draw_colored_source_canvas

# Tooling index used to convert base curve diopters to a surface radius (R = (n - 1) / BC).
LENS_TOOLING_INDEX_MINUS_ONE = 0.53
LENS_PHYSICAL_WIDTH_M_RANGE = (0.045, 0.058)
REFLECTION_SIZE_RATIO_LIMITS = (0.015, 6.0)
CANVAS_SIZE_LIMITS = (8, 160)
BARREL_DISTORTION_K1_RANGE = (0.05, 0.3)
PERSPECTIVE_COEFFICIENT_LIMIT = 0.1
VERTICAL_SQUASH_RANGE = (0.85, 1.0)
# Symmetric (face-form wrap) and common (source direction) parts of the two lenses' offsets.
SYMMETRIC_OFFSET_SHARE = 0.4
SECOND_LENS_OFFSET_JITTER = 0.1
SECOND_LENS_SCALE_RATIO_RANGE = (0.9, 1.1)
SECOND_LENS_INTENSITY_RATIO_RANGE = (0.7, 1.0)
SECOND_LENS_ROTATION_JITTER_DEGREES = 3.0
MINIMUM_FORESHORTENING = 0.35
GHOST_SCALE_RANGE = (0.3, 1.2)
GHOST_INTENSITY_RANGE = (0.1, 0.35)
GHOST_OFFSET_RANGE = 0.25
MOTION_BLUR_MAX_FRACTION_OF_LENS_WIDTH = 0.04
DEFOCUS_CAP_FRACTION_OF_REFLECTION_SIZE = 0.25
VEIL_PEAK_LINEAR_RANGE = (0.005, 0.04)
RIM_FALLOFF_FLOOR_RANGE = (0.6, 1.0)
SOFT_KNEE_START_RANGE = (0.6, 0.85)
COATING_TEXTURE_AMPLITUDE_RANGE = (0.1, 0.4)
COATING_TEXTURE_GRID_CELLS_RANGE = (3, 9)
# Compact emitters that may ride along with an HDRI reflection (rooms rarely contain these in HDRIs).
HDRI_COMPANION_EMITTER_TYPES = (LightSourceType.RING_LIGHT, LightSourceType.POINT_GLINT, LightSourceType.STRIP_LIGHT, LightSourceType.SCREEN)


def sample_from_probability_dict(random_generator: np.random.Generator, probabilities: dict):
    """Pick a key of `probabilities` with probability proportional to its value."""
    keys = list(probabilities)
    weights = np.array([probabilities[key] for key in keys], dtype=np.float64)
    return keys[random_generator.choice(len(keys), p=weights / weights.sum())]


def log_uniform(random_generator: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Sample log-uniformly between two positive bounds."""
    return float(np.exp(random_generator.uniform(np.log(value_range[0]), np.log(value_range[1]))))


def sample_mirror_focal_length_m(random_generator: np.random.Generator, config: GlareSamplingConfig) -> float:
    """Front-surface mirror focal length: base-curve lenses, or the rare flat-lens tail."""
    if random_generator.random() < config.flat_lens_probability:
        return log_uniform(random_generator, config.flat_lens_focal_length_m_range)
    base_curve_diopters = random_generator.uniform(*config.base_curve_diopter_range)
    return LENS_TOOLING_INDEX_MINUS_ONE / base_curve_diopters / 2.0


def reflection_size_relative_to_lens(source_size_m: float, source_distance_m: float, focal_length_m: float,
                                     camera_distance_m: float, lens_width_m: float) -> float:
    """Apparent reflection size over apparent lens size (see module docstring)."""
    magnification = focal_length_m / (source_distance_m + focal_length_m)
    image_depth_behind_lens = source_distance_m * magnification
    apparent_ratio = source_size_m * magnification * camera_distance_m / (camera_distance_m + image_depth_behind_lens) / lens_width_m
    return float(np.clip(apparent_ratio, *REFLECTION_SIZE_RATIO_LIMITS))


def sample_peak_linear(random_generator: np.random.Generator, light_type: LightSourceType, severity: GlareSeverity, config: GlareSamplingConfig) -> float:
    """Peak added linear light (1 = clip) for a source of the given type and severity."""
    spec = LIGHT_SOURCE_SPECS[light_type]
    if severity == GlareSeverity.WEAK:
        peak_linear = log_uniform(random_generator, config.weak_peak_linear_range)
    else:
        peak_linear = log_uniform(random_generator, config.strong_peak_linear_range)
    return min(peak_linear * random_generator.uniform(*spec.small_source_peak_multiplier_range), spec.peak_linear_ceiling)


def build_lens_placements(random_generator: np.random.Generator, light_type: LightSourceType, size_ratio: float,
                          lens_geometries: list[LensGeometry], config: GlareSamplingConfig) -> list[ReflectionPlacement]:
    """Place one source in both lenses: same orientation, mirrored-plus-common offset, own distortion."""
    spec = LIGHT_SOURCE_SPECS[light_type]
    nominal_lens_width = max(lens.width_pixels for lens in lens_geometries)
    # Spread is a fraction of lens WIDTH; normalized units are half-widths, hence the factor 2.
    symmetric_offset_x = random_generator.uniform(-1, 1) * spec.placement_spread_fraction * 2.0 * SYMMETRIC_OFFSET_SHARE
    common_offset_x = random_generator.uniform(-1, 1) * spec.placement_spread_fraction * 2.0 * (1.0 - SYMMETRIC_OFFSET_SHARE)
    offset_y = random_generator.uniform(-1, 1) * spec.placement_spread_fraction * 1.6
    rotation_degrees = random_generator.uniform(*spec.rotation_degrees_range)
    base_k1 = random_generator.uniform(*BARREL_DISTORTION_K1_RANGE)
    vertical_squash = random_generator.uniform(*VERTICAL_SQUASH_RANGE)
    lens_indices = list(range(len(lens_geometries)))
    if len(lens_indices) == 2 and random_generator.random() < config.single_lens_probability:
        lens_indices = [int(random_generator.integers(2))]
    placements = []
    for lens_index in lens_indices:
        lens = lens_geometries[lens_index]
        is_image_left_lens = lens.lens_label == 1
        is_second_lens = lens_index == 1
        jitter = random_generator.uniform(-SECOND_LENS_OFFSET_JITTER, SECOND_LENS_OFFSET_JITTER, size=2) if is_second_lens else np.zeros(2)
        # Mirror the wrap part about the face midline: toward the nose on one lens is toward the nose on the other.
        signed_symmetric_x = -symmetric_offset_x if is_image_left_lens else symmetric_offset_x
        placements.append(
            ReflectionPlacement(
                lens_index=lens_index,
                center_offset_normalized=np.array([signed_symmetric_x + common_offset_x, offset_y]) + jitter,
                scale_normalized_per_source_unit=size_ratio * (random_generator.uniform(*SECOND_LENS_SCALE_RATIO_RANGE) if is_second_lens else 1.0),
                rotation_degrees=rotation_degrees + (random_generator.uniform(-1, 1) * SECOND_LENS_ROTATION_JITTER_DEGREES if is_second_lens else 0.0),
                horizontal_foreshortening=max(MINIMUM_FORESHORTENING, lens.width_pixels / nominal_lens_width),
                vertical_squash=vertical_squash,
                barrel_distortion_k1=base_k1 * random_generator.uniform(0.8, 1.2),
                perspective_coefficients=random_generator.uniform(-PERSPECTIVE_COEFFICIENT_LIMIT, PERSPECTIVE_COEFFICIENT_LIMIT, size=2),
                intensity_multiplier=random_generator.uniform(*SECOND_LENS_INTENSITY_RATIO_RANGE) if is_second_lens else 1.0,
            )
        )
    return placements


def build_ghost_placements(random_generator: np.random.Generator, placements: list[ReflectionPlacement],
                           ghost_scale_range: tuple[float, float] = GHOST_SCALE_RANGE) -> list[ReflectionPlacement]:
    """Inner-surface reflection: same orientation, rescaled, offset, much fainter."""
    ghost_scale = random_generator.uniform(*ghost_scale_range)
    ghost_offset = random_generator.uniform(-GHOST_OFFSET_RANGE, GHOST_OFFSET_RANGE, size=2)
    ghost_intensity = random_generator.uniform(*GHOST_INTENSITY_RANGE)
    return [
        ReflectionPlacement(
            lens_index=placement.lens_index,
            center_offset_normalized=placement.center_offset_normalized * ghost_scale + ghost_offset,
            scale_normalized_per_source_unit=placement.scale_normalized_per_source_unit * ghost_scale,
            rotation_degrees=placement.rotation_degrees,
            horizontal_foreshortening=placement.horizontal_foreshortening,
            vertical_squash=placement.vertical_squash,
            barrel_distortion_k1=placement.barrel_distortion_k1,
            perspective_coefficients=placement.perspective_coefficients,
            intensity_multiplier=placement.intensity_multiplier * ghost_intensity,
        )
        for placement in placements
    ]


def sample_light_source(random_generator: np.random.Generator, lens_geometries: list[LensGeometry], focal_length_m: float,
                        camera_distance_m: float, lens_width_m: float, config: GlareSamplingConfig) -> SampledLightSource:
    """Sample one light source end to end: type, severity, size, canvas, placements, blur."""
    light_type = sample_from_probability_dict(random_generator, config.light_type_probabilities)
    severity = sample_from_probability_dict(random_generator, config.severity_probabilities)
    spec = LIGHT_SOURCE_SPECS[light_type]
    source_distance_m = log_uniform(random_generator, spec.source_distance_m_range)
    if light_type == LightSourceType.RING_LIGHT:
        source_distance_m = camera_distance_m * random_generator.uniform(0.95, 1.15)  # the camera sits in the ring
    size_ratio = reflection_size_relative_to_lens(log_uniform(random_generator, spec.physical_size_m_range), source_distance_m,
                                                  focal_length_m, camera_distance_m, lens_width_m)
    size_ratio = max(size_ratio, spec.minimum_size_ratio)
    nominal_lens_width = max(lens.width_pixels for lens in lens_geometries)
    canvas_size = int(np.clip(round(size_ratio * nominal_lens_width), *CANVAS_SIZE_LIMITS))
    placements = build_lens_placements(random_generator, light_type, size_ratio, lens_geometries, config)
    ghost_probability = spec.ghost_probability if spec.ghost_probability is not None else config.ghost_reflection_probability
    has_ghost = spec.allows_ghost_reflection and random_generator.random() < ghost_probability
    has_motion_blur = random_generator.random() < config.motion_blur_probability
    return SampledLightSource(
        light_type=light_type,
        severity=severity,
        canvas_linear_rgb=draw_colored_source_canvas(light_type, random_generator, canvas_size),
        peak_linear=sample_peak_linear(random_generator, light_type, severity, config),
        extends_beyond_canvas=spec.extends_beyond_canvas,
        placements=placements,
        ghost_placements=build_ghost_placements(random_generator, placements, spec.ghost_scale_range or GHOST_SCALE_RANGE) if has_ghost else [],
        # A small source's virtual image sits near the lens focal point, so its defocus cannot exceed its own size.
        defocus_radius_pixels=min(nominal_lens_width * random_generator.uniform(*config.defocus_radius_fraction_of_lens_width_range),
                                  DEFOCUS_CAP_FRACTION_OF_REFLECTION_SIZE * size_ratio * nominal_lens_width),
        motion_blur_length_pixels=nominal_lens_width * random_generator.uniform(0, MOTION_BLUR_MAX_FRACTION_OF_LENS_WIDTH) if has_motion_blur else 0.0,
        motion_blur_angle_degrees=random_generator.uniform(0, 180),
    )


def extra_emitter_config(config: GlareSamplingConfig) -> GlareSamplingConfig:
    """The config narrowed to the compact emitters that ride along with an HDRI reflection."""
    emitter_probabilities = {light_type: config.light_type_probabilities.get(light_type, 0.0) for light_type in HDRI_COMPANION_EMITTER_TYPES}
    if sum(emitter_probabilities.values()) <= 0:
        emitter_probabilities = {light_type: 1.0 for light_type in HDRI_COMPANION_EMITTER_TYPES}
    return dataclasses.replace(config, light_type_probabilities=emitter_probabilities)


def sample_glare_scene(lens_geometries: list[LensGeometry], random_generator: np.random.Generator, config: GlareSamplingConfig,
                       hdri_library: HdriEnvironmentLibrary | None = None) -> GlareScene:
    """Draw every random choice for one sample. No lenses, or the no-glare draw, gives a glare-free scene.

    - The no-glare draw happens first and consumes exactly one random number, so the glare-free
      share is exact and independent of the rest of the config.
    - With a library, `config.hdri_reflection_share` of glared samples mirror an HDRI (plus,
      sometimes, one compact procedural emitter); the rest use 1-3 procedural sources.
    """
    if random_generator.random() < config.no_glare_probability or not lens_geometries:
        return GlareScene(is_glare_free=True)
    nominal_lens_width = max(lens.width_pixels for lens in lens_geometries)
    focal_length_m = sample_mirror_focal_length_m(random_generator, config)
    camera_distance_m = log_uniform(random_generator, config.camera_distance_m_range)
    lens_width_m = random_generator.uniform(*LENS_PHYSICAL_WIDTH_M_RANGE)
    is_ar_coated = random_generator.random() < config.ar_coating_probability
    hdri_reflection = None
    if hdri_library is not None and random_generator.random() < config.hdri_reflection_share:
        hdri_reflection = sample_hdri_reflection(lens_geometries, random_generator, config, hdri_library, is_ar_coated)
        source_count = 1 if random_generator.random() < config.hdri_extra_emitter_probability else 0
        source_config = extra_emitter_config(config)
    else:
        source_count = sample_from_probability_dict(random_generator, config.source_count_probabilities)
        source_config = config
    light_sources = [sample_light_source(random_generator, lens_geometries, focal_length_m, camera_distance_m, lens_width_m, source_config)
                     for _ in range(source_count)]
    scene = GlareScene(is_glare_free=False, light_sources=light_sources, hdri_reflection=hdri_reflection)
    resolution_scale = lens_geometries[0].lens_pixels.shape[1] / REFERENCE_CROP_WIDTH
    scene.lens_mask_erosion_pixels = random_generator.uniform(*config.lens_mask_erosion_pixels_range) * resolution_scale
    scene.lens_mask_feather_sigma_pixels = random_generator.uniform(*config.lens_mask_feather_sigma_pixels_range) * resolution_scale
    if random_generator.random() < config.edge_gain_probability:
        scene.edge_gain_strength = random_generator.uniform(*config.edge_gain_strength_range)
        scene.edge_gain_angle_radians = random_generator.uniform(0, 2 * np.pi)
    if is_ar_coated:
        scene.coating_tint_linear_rgb = sample_ar_coating_tint(random_generator)
        if random_generator.random() < config.tint_gradient_probability:
            scene.gradient_tint_linear_rgb = sample_ar_coating_tint(random_generator)
            scene.tint_gradient_angle_radians = random_generator.uniform(0, 2 * np.pi)
    if random_generator.random() < config.veil_probability:
        scene.veil = LensVeil(
            veil_linear_rgb=color_temperature_to_linear_rgb(random_generator.uniform(4000.0, 12000.0)),
            peak_linear=random_generator.uniform(*VEIL_PEAK_LINEAR_RANGE),
            gradient_angle_radians=random_generator.uniform(0, 2 * np.pi),
            gradient_floor=random_generator.uniform(0.0, 0.7),
        )
    if random_generator.random() < config.coating_texture_probability:
        scene.coating_texture_amplitude = random_generator.uniform(*COATING_TEXTURE_AMPLITUDE_RANGE)
        scene.coating_texture_grid_cells = int(random_generator.integers(*COATING_TEXTURE_GRID_CELLS_RANGE))
    scene.coating_texture_seed = int(random_generator.integers(2**31))
    scene.noise_seed = int(random_generator.integers(2**31))
    scene.rim_falloff_floor = random_generator.uniform(*RIM_FALLOFF_FLOOR_RANGE)
    if random_generator.random() < config.transmission_attenuation_probability:
        scene.transmission_attenuation = random_generator.uniform(*config.transmission_attenuation_range)
    if random_generator.random() < config.soft_knee_probability:
        scene.soft_knee_start = random_generator.uniform(*SOFT_KNEE_START_RANGE)
    if random_generator.random() < config.bloom_probability:
        scene.bloom_sigma_pixels = nominal_lens_width * random_generator.uniform(*config.bloom_sigma_fraction_of_lens_width_range)
        scene.bloom_strength = random_generator.uniform(*config.bloom_strength_range)
    scene.glare_noise_scale = random_generator.uniform(*config.glare_noise_scale_range)
    if random_generator.random() < config.jpeg_probability:
        scene.jpeg_quality = int(random_generator.integers(config.jpeg_quality_range[0], config.jpeg_quality_range[1] + 1))
    return scene
