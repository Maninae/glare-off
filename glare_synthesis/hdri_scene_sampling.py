"""Draw the random choices of an image-based (HDRI) reflection: which scene, how it is aimed, how bright.

Brightness model: a face of albedo SKIN_ALBEDO lit by an environment of mean radiance 1 has linear
level ~ albedo x exposure, so a mirror of reflectance rho shows the environment at
(face linear level / albedo) x rho x radiance. The renderer supplies the face level; this module
samples rho (coated 0.5-1.5%, uncoated 4-8%) and an exposure jitter that covers backlit faces,
fill flash and local lighting differences. Bright emitters then clip on their own.
"""

import numpy as np

from glare_synthesis.glare_sampling_config import GlareSamplingConfig
from glare_synthesis.glare_scene import HdriReflection
from glare_synthesis.hdri_environment_library import HdriEnvironmentLibrary
from glare_synthesis.hdri_reflection_rendering import lens_points_in_metres, mirror_reflected_directions, solve_environment_rotation_for_target
from glare_synthesis.lens_geometry import LensGeometry

SKIN_ALBEDO = 0.35
TOOLING_INDEX_MINUS_ONE = 0.53
LENS_PHYSICAL_WIDTH_M_RANGE = (0.045, 0.058)
UNAIMED_PITCH_STANDARD_DEVIATION = np.radians(12.0)
AIM_POINT_EXTENT = 0.6
GHOST_RADIUS_RATIO_RANGE = (0.35, 0.8)
GHOST_TILT_LIMIT = np.radians(4.0)
GHOST_INTENSITY_RANGE = (0.3, 0.9)
SECOND_LENS_INTENSITY_RANGE = (0.75, 1.0)
MAXIMUM_DEFOCUS_FRACTION_OF_LENS_WIDTH = 0.15


def log_uniform(random_generator: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Sample log-uniformly between two positive bounds."""
    return float(np.exp(random_generator.uniform(np.log(value_range[0]), np.log(value_range[1]))))


def sample_front_surface_radius_m(random_generator: np.random.Generator, config: GlareSamplingConfig) -> float:
    """Front-surface radius from a base curve (R = 0.53 / BC), or the rare flat-lens tail (R = 2 f)."""
    if random_generator.random() < config.flat_lens_probability:
        return 2.0 * log_uniform(random_generator, config.flat_lens_focal_length_m_range)
    return TOOLING_INDEX_MINUS_ONE / random_generator.uniform(*config.base_curve_diopter_range)


def sample_lens_normal_tilts(random_generator: np.random.Generator, lens_geometries: list[LensGeometry], config: GlareSamplingConfig) -> list[np.ndarray]:
    """Per-lens (yaw, pitch) of the surface normal: mirrored face-form wrap plus a shared head pose and pantoscopic tilt."""
    wrap = np.radians(random_generator.uniform(*config.face_form_wrap_degrees_range))
    head_yaw = np.radians(random_generator.normal(0.0, config.head_yaw_standard_deviation_degrees))
    shared_pitch = np.radians(random_generator.uniform(*config.lens_pitch_degrees_range))
    tilts = []
    for lens in lens_geometries:
        # Wrap turns each lens's normal toward its own temple: image-left lens toward -x.
        wrap_sign = 1.0 if lens.lens_label == 1 else -1.0
        tilts.append(np.array([wrap_sign * wrap + head_yaw, shared_pitch + np.radians(random_generator.normal(0.0, 1.0))]))
    return tilts


def sample_hdri_reflection(lens_geometries: list[LensGeometry], random_generator: np.random.Generator, config: GlareSamplingConfig,
                           library: HdriEnvironmentLibrary, is_ar_coated: bool) -> HdriReflection:
    """All random choices for one image-based reflection (rendering is deterministic afterwards)."""
    hdri_id = library.sample_hdri_id(random_generator, config.hdri_category_probabilities)
    environment = library.load_environment(hdri_id)
    nominal_lens_width_pixels = max(lens.width_pixels for lens in lens_geometries)
    front_radius_m = sample_front_surface_radius_m(random_generator, config)
    lens_width_m = random_generator.uniform(*LENS_PHYSICAL_WIDTH_M_RANGE)
    camera_distance_m = log_uniform(random_generator, config.camera_distance_m_range)
    lens_normal_tilts = sample_lens_normal_tilts(random_generator, lens_geometries, config)
    if len(environment.emitter_weights) > 0 and random_generator.random() < config.hdri_emitter_aim_probability:
        # Aim a bright emitter at a random point of a random lens: those directions cause visible glare.
        aim_lens_index = int(random_generator.integers(len(lens_geometries)))
        aim_lens = lens_geometries[aim_lens_index]
        aim_x = random_generator.uniform(-AIM_POINT_EXTENT, AIM_POINT_EXTENT)
        aim_y = random_generator.uniform(-AIM_POINT_EXTENT, AIM_POINT_EXTENT) * aim_lens.height_pixels / nominal_lens_width_pixels
        foreshortening = aim_lens.width_pixels / nominal_lens_width_pixels
        point_x_m, point_y_m = lens_points_in_metres(np.float32(aim_x), np.float32(aim_y), lens_width_m, foreshortening)
        target_direction = np.array(mirror_reflected_directions(point_x_m, point_y_m, np.zeros(2), front_radius_m,
                                                                lens_normal_tilts[aim_lens_index], camera_distance_m), dtype=np.float64)
        emitter_index = random_generator.choice(len(environment.emitter_weights), p=environment.emitter_weights / environment.emitter_weights.sum())
        environment_yaw, environment_pitch = solve_environment_rotation_for_target(target_direction, environment.emitter_lon_lat[emitter_index])
    else:
        environment_yaw = random_generator.uniform(-np.pi, np.pi)
        environment_pitch = random_generator.normal(0.0, UNAIMED_PITCH_STANDARD_DEVIATION)
    reflectance = random_generator.uniform(*(config.coated_reflectance_range if is_ar_coated else config.uncoated_reflectance_range))
    exposure_jitter = log_uniform(random_generator, config.hdri_exposure_jitter_range)
    # Image of infinity sits f = R/2 behind the lens: blur circle at the face = aperture * f / (D + f).
    focal_length_m = front_radius_m / 2.0
    blur_circle_m = log_uniform(random_generator, config.camera_aperture_mm_range) / 1000.0 * focal_length_m / (camera_distance_m + focal_length_m)
    defocus_radius_pixels = min(0.5 * blur_circle_m / lens_width_m, MAXIMUM_DEFOCUS_FRACTION_OF_LENS_WIDTH) * nominal_lens_width_pixels
    has_ghost = random_generator.random() < config.hdri_ghost_probability
    second_lens_intensity = random_generator.uniform(*SECOND_LENS_INTENSITY_RANGE)
    return HdriReflection(
        hdri_id=hdri_id,
        environment_yaw=float(environment_yaw),
        environment_pitch=float(environment_pitch),
        front_surface_radius_m=front_radius_m,
        lens_width_m=lens_width_m,
        camera_distance_m=camera_distance_m,
        lens_normal_tilts=lens_normal_tilts,
        lens_intensity_multipliers=[1.0 if lens_index == 0 else second_lens_intensity for lens_index in range(len(lens_geometries))],
        relative_exposure=reflectance * exposure_jitter / SKIN_ALBEDO,
        defocus_radius_pixels=defocus_radius_pixels,
        ghost_surface_radius_m=front_radius_m * random_generator.uniform(*GHOST_RADIUS_RATIO_RANGE) if has_ghost else None,
        ghost_intensity=random_generator.uniform(*GHOST_INTENSITY_RANGE) if has_ghost else 0.0,
        ghost_normal_tilt=random_generator.uniform(-GHOST_TILT_LIMIT, GHOST_TILT_LIMIT, size=2),
    )
