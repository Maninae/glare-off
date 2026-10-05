"""Render a `GlareScene`'s reflected light (linear RGB, before the camera) inside the lenses.

Light per lens = procedural sources + image-based HDRI reflection (`hdri_reflection_rendering`) + veil,
each blurred by its own defocus and the photo's blur. Then: coating tint (saturated only where the
reflection is faint; bright light whitens), a strength ramp across the lens, rim falloff, coating
wear, and the per-sample soft lens mask so the light ends at the rim with a 1-3 px soft edge.

Procedural sources: for every lens pixel we compute where it looks in the source's canvas
(inverse mapping) and sample it with `cv2.remap`.

Inverse mapping per pixel p (normalized lens coords q = (p - lens_center) / half_lens_width):
1. undo barrel distortion: q_u = q * (1 + k1 |q|^2)   (convex mirror compresses toward the rim)
2. undo keystone (wrap/tilt):  q_p = q_u / (1 + a . q_u)
3. into the source frame: subtract the reflection center, divide by foreshortening and squash,
   rotate by -theta, divide by the source scale -> unit source plane -> canvas pixels.
"""

import cv2
import numpy as np

from glare_synthesis.color_science import REC709_LUMINANCE_WEIGHTS
from glare_synthesis.glare_scene import GlareScene, ReflectionPlacement, SampledLightSource
from glare_synthesis.hdri_environment_library import HdriEnvironment
from glare_synthesis.hdri_reflection_rendering import render_hdri_reflection_radiance
from glare_synthesis.lens_geometry import LensGeometry, build_soft_lens_mask
from glare_synthesis.procedural_noise import smooth_random_field

RIM_FALLOFF_RAMP_END = 0.6
MINIMUM_BLUR_RADIUS_PIXELS = 0.5
REGION_MARGIN_PIXELS = 2
VEIL_LUMP_GRID_CELLS = 4
VEIL_LUMP_SEED_OFFSET = 7919
# Coating colour survives on faint reflections and washes out to white as the reflection brightens.
TINT_FULL_BELOW_LUMINANCE = 0.08
TINT_GONE_ABOVE_LUMINANCE = 0.8


def defocus_disc_kernel(radius_pixels: float) -> np.ndarray:
    """Normalized, anti-aliased disc (pillbox) kernel: the out-of-focus point spread function."""
    supersample = 4
    kernel_radius = int(np.ceil(radius_pixels))
    kernel_size = 2 * kernel_radius + 1
    big_kernel = np.zeros((kernel_size * supersample, kernel_size * supersample), np.float32)
    center = kernel_size * supersample / 2.0
    cv2.circle(big_kernel, (int(center * 16), int(center * 16)), int(radius_pixels * supersample * 16), 1.0, -1, cv2.LINE_AA, shift=4)
    kernel = cv2.resize(big_kernel, (kernel_size, kernel_size), interpolation=cv2.INTER_AREA)
    return kernel / kernel.sum()


def motion_blur_kernel(length_pixels: float, angle_degrees: float) -> np.ndarray:
    """Normalized line kernel for hand-shake motion blur."""
    kernel_size = 2 * int(np.ceil(length_pixels / 2.0)) + 1
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    half_vector = 0.5 * length_pixels * np.array([np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))])
    center = np.array([kernel_size // 2, kernel_size // 2], np.float64)
    start, end = np.round((center - half_vector) * 16).astype(int), np.round((center + half_vector) * 16).astype(int)
    cv2.line(kernel, tuple(start), tuple(end), 1.0, 1, cv2.LINE_AA, shift=4)
    return kernel / max(kernel.sum(), 1e-6)


def normalized_lens_coordinates(lens: LensGeometry, region_xyxy: tuple[int, int, int, int], nominal_lens_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized lens coordinates (1 unit = half nominal lens width) for every pixel of a region."""
    x0, y0, x1, y1 = region_xyxy
    half_nominal_width = nominal_lens_width / 2.0
    column_coordinates = (np.arange(x0, x1, dtype=np.float32) - np.float32(lens.center_xy[0])) / np.float32(half_nominal_width)
    row_coordinates = (np.arange(y0, y1, dtype=np.float32) - np.float32(lens.center_xy[1])) / np.float32(half_nominal_width)
    return np.meshgrid(column_coordinates, row_coordinates)


def sample_canvas_through_mirror(canvas: np.ndarray, placement: ReflectionPlacement, lens_x: np.ndarray, lens_y: np.ndarray,
                                 extends_beyond_canvas: bool) -> np.ndarray:
    """Remap a source canvas into a lens region through the inverse mirror mapping (module docstring)."""
    radial_factor = 1.0 + np.float32(placement.barrel_distortion_k1) * (lens_x * lens_x + lens_y * lens_y)
    undistorted_x, undistorted_y = lens_x * radial_factor, lens_y * radial_factor
    keystone_divisor = 1.0 + np.float32(placement.perspective_coefficients[0]) * undistorted_x + np.float32(placement.perspective_coefficients[1]) * undistorted_y
    relative_x = (undistorted_x / keystone_divisor - np.float32(placement.center_offset_normalized[0])) / np.float32(placement.horizontal_foreshortening)
    relative_y = (undistorted_y / keystone_divisor - np.float32(placement.center_offset_normalized[1])) / np.float32(placement.vertical_squash)
    cosine, sine = np.cos(np.radians(placement.rotation_degrees)), np.sin(np.radians(placement.rotation_degrees))
    canvas_size = canvas.shape[0]
    pixels_per_normalized_unit = np.float32(0.5 * canvas_size / placement.scale_normalized_per_source_unit)
    canvas_center = np.float32(0.5 * canvas_size - 0.5)
    map_x = (np.float32(cosine) * relative_x + np.float32(sine) * relative_y) * pixels_per_normalized_unit + canvas_center
    map_y = (np.float32(-sine) * relative_x + np.float32(cosine) * relative_y) * pixels_per_normalized_unit + canvas_center
    border_mode = cv2.BORDER_REFLECT_101 if extends_beyond_canvas else cv2.BORDER_CONSTANT
    return cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=0)


def blur_reflection(reflection: np.ndarray, light_source: SampledLightSource, photo_blur_sigma_pixels: float) -> np.ndarray:
    """Defocus disc, optional motion streak, then the photo's own blur (so glare matches its sharpness)."""
    if light_source.defocus_radius_pixels >= MINIMUM_BLUR_RADIUS_PIXELS:
        reflection = cv2.filter2D(reflection, -1, defocus_disc_kernel(light_source.defocus_radius_pixels), borderType=cv2.BORDER_CONSTANT)
    if light_source.motion_blur_length_pixels >= 1.0:
        reflection = cv2.filter2D(reflection, -1, motion_blur_kernel(light_source.motion_blur_length_pixels, light_source.motion_blur_angle_degrees),
                                  borderType=cv2.BORDER_CONSTANT)
    if photo_blur_sigma_pixels >= 0.3:
        reflection = cv2.GaussianBlur(reflection, (0, 0), photo_blur_sigma_pixels, borderType=cv2.BORDER_CONSTANT)
    return reflection


def apply_strength_dependent_tint(scene: GlareScene, lens_light: np.ndarray, lens_x: np.ndarray, lens_y: np.ndarray) -> np.ndarray:
    """Multiply by the coating tint (with drift), fading it to white where the reflection is bright."""
    tint = np.broadcast_to(scene.coating_tint_linear_rgb.astype(np.float32), lens_x.shape + (3,))
    if scene.gradient_tint_linear_rgb is not None:
        drift = np.clip(0.5 + 0.5 * (lens_x * np.cos(scene.tint_gradient_angle_radians) + lens_y * np.sin(scene.tint_gradient_angle_radians)), 0.0, 1.0)[..., None]
        tint = (1.0 - drift) * tint + drift * scene.gradient_tint_linear_rgb.astype(np.float32)
    ramp = np.clip((lens_light @ REC709_LUMINANCE_WEIGHTS - TINT_FULL_BELOW_LUMINANCE) / (TINT_GONE_ABOVE_LUMINANCE - TINT_FULL_BELOW_LUMINANCE), 0.0, 1.0)
    whitening = (ramp * ramp * (3.0 - 2.0 * ramp))[..., None]
    return lens_light * (tint + (1.0 - tint) * whitening)


def lens_modulation(scene: GlareScene, lens: LensGeometry, lens_index: int, region_xyxy: tuple[int, int, int, int],
                    lens_x: np.ndarray, lens_y: np.ndarray) -> np.ndarray:
    """Per-pixel scalar multiplier (H, W, 1): strength ramp across the lens, rim falloff, coating wear, soft lens mask."""
    x0, y0, x1, y1 = region_xyxy
    rim_ramp = np.clip(lens.rim_distance_normalized[y0:y1, x0:x1] / RIM_FALLOFF_RAMP_END, 0.0, 1.0)
    rim_falloff = scene.rim_falloff_floor + (1.0 - scene.rim_falloff_floor) * rim_ramp * rim_ramp * (3.0 - 2.0 * rim_ramp)
    soft_lens_mask = build_soft_lens_mask(lens.lens_pixels[y0:y1, x0:x1], scene.lens_mask_erosion_pixels, scene.lens_mask_feather_sigma_pixels)
    scalar_modulation = rim_falloff * soft_lens_mask
    if scene.edge_gain_strength > 0:
        along_gain_axis = lens_x * np.cos(scene.edge_gain_angle_radians) + lens_y * np.sin(scene.edge_gain_angle_radians)
        scalar_modulation = scalar_modulation * np.clip(1.0 + scene.edge_gain_strength * along_gain_axis, 0.2, None)
    if scene.coating_texture_amplitude > 0:
        wear_generator = np.random.default_rng(scene.coating_texture_seed + lens_index)
        wear_field = smooth_random_field(wear_generator, y1 - y0, x1 - x0, scene.coating_texture_grid_cells)
        scalar_modulation = scalar_modulation * (1.0 + scene.coating_texture_amplitude * (2.0 * wear_field - 1.0))
    return scalar_modulation[..., None]


def lens_veil_light(scene: GlareScene, lens_index: int, lens_x: np.ndarray, lens_y: np.ndarray) -> np.ndarray:
    """Low-frequency haze across the lens (before tint and mask), linear RGB: a gradient with soft lumps."""
    veil = scene.veil
    directional_ramp = np.clip(0.5 + 0.5 * (lens_x * np.cos(veil.gradient_angle_radians) + lens_y * np.sin(veil.gradient_angle_radians)), 0.0, 1.0)
    # A flat tinted veil reads as a colour filter; real haze is a blurred room, so add soft lumps.
    lump_generator = np.random.default_rng(scene.coating_texture_seed + VEIL_LUMP_SEED_OFFSET + lens_index)
    veil_lumps = smooth_random_field(lump_generator, lens_x.shape[0], lens_x.shape[1], VEIL_LUMP_GRID_CELLS)
    veil_profile = (veil.gradient_floor + (1.0 - veil.gradient_floor) * directional_ramp) * (0.3 + 1.4 * veil_lumps)
    return (veil.peak_linear * veil_profile)[..., None] * veil.veil_linear_rgb.astype(np.float32)


def render_hdri_light_for_lens(scene: GlareScene, environment: HdriEnvironment, lens: LensGeometry, lens_index: int, lens_x: np.ndarray,
                               lens_y: np.ndarray, nominal_lens_width: float, crop_center_xy: np.ndarray, face_linear_luminance: float,
                               photo_blur_sigma_pixels: float) -> np.ndarray:
    """HDRI mirror light for one lens region in linear units, defocused like the virtual image."""
    hdri_reflection = scene.hdri_reflection
    metres_per_pixel = hdri_reflection.lens_width_m / nominal_lens_width
    lens_center_offset_m = (lens.center_xy - crop_center_xy) * metres_per_pixel
    radiance = render_hdri_reflection_radiance(hdri_reflection, environment, lens_index, lens_x, lens_y, lens_center_offset_m,
                                               lens.width_pixels / nominal_lens_width)
    light_scale = hdri_reflection.relative_exposure * face_linear_luminance * hdri_reflection.lens_intensity_multipliers[lens_index]
    hdri_light = radiance * np.float32(light_scale)
    if hdri_reflection.defocus_radius_pixels >= MINIMUM_BLUR_RADIUS_PIXELS:
        hdri_light = cv2.filter2D(hdri_light, -1, defocus_disc_kernel(hdri_reflection.defocus_radius_pixels), borderType=cv2.BORDER_REFLECT)
    if photo_blur_sigma_pixels >= 0.3:
        hdri_light = cv2.GaussianBlur(hdri_light, (0, 0), photo_blur_sigma_pixels, borderType=cv2.BORDER_REFLECT)
    return hdri_light


def render_reflection_layer(scene: GlareScene, lens_geometries: list[LensGeometry], region_of_interest_xyxy: tuple[int, int, int, int],
                            photo_blur_sigma_pixels: float, intensity_scale: float = 1.0, hdri_environment: HdriEnvironment | None = None,
                            face_linear_luminance: float = 0.2, crop_center_xy: np.ndarray | None = None) -> np.ndarray:
    """Total reflected linear light inside the region of interest, float32 (H_roi, W_roi, 3).

    - `intensity_scale` multiplies every source's brightness (used by the severity ladder).
    - `hdri_environment` must be the scene's HDRI when `scene.hdri_reflection` is set.
    - `face_linear_luminance` sets the HDRI exposure; `crop_center_xy` anchors lens parallax.
    """
    roi_x0, roi_y0, roi_x1, roi_y1 = region_of_interest_xyxy
    reflection_layer = np.zeros((roi_y1 - roi_y0, roi_x1 - roi_x0, 3), np.float32)
    nominal_lens_width = max(lens.width_pixels for lens in lens_geometries)
    for lens_index, lens in enumerate(lens_geometries):
        bx0, by0, bx1, by1 = lens.bounding_box_xyxy
        region_xyxy = (max(bx0 - REGION_MARGIN_PIXELS, roi_x0), max(by0 - REGION_MARGIN_PIXELS, roi_y0),
                       min(bx1 + REGION_MARGIN_PIXELS, roi_x1), min(by1 + REGION_MARGIN_PIXELS, roi_y1))
        lens_x, lens_y = normalized_lens_coordinates(lens, region_xyxy, nominal_lens_width)
        lens_light = np.zeros(lens_x.shape + (3,), np.float32)
        for light_source in scene.light_sources:
            source_light = np.zeros_like(lens_light)
            for placement in light_source.placements + light_source.ghost_placements:
                if placement.lens_index != lens_index:
                    continue
                source_light += placement.intensity_multiplier * sample_canvas_through_mirror(
                    light_source.canvas_linear_rgb, placement, lens_x, lens_y, light_source.extends_beyond_canvas)
            if source_light.any():
                lens_light += np.float32(light_source.peak_linear * intensity_scale) * blur_reflection(source_light, light_source, photo_blur_sigma_pixels)
        if scene.hdri_reflection is not None:
            lens_light += np.float32(intensity_scale) * render_hdri_light_for_lens(scene, hdri_environment, lens, lens_index, lens_x, lens_y, nominal_lens_width,
                                                                                    crop_center_xy, face_linear_luminance, photo_blur_sigma_pixels)
        if scene.veil is not None:
            lens_light += np.float32(intensity_scale) * lens_veil_light(scene, lens_index, lens_x, lens_y)
        lens_light = apply_strength_dependent_tint(scene, lens_light, lens_x, lens_y)
        lens_light *= lens_modulation(scene, lens, lens_index, region_xyxy, lens_x, lens_y)
        x0, y0, x1, y1 = region_xyxy
        reflection_layer[y0 - roi_y0:y1 - roi_y0, x0 - roi_x0:x1 - roi_x0] += lens_light
    return reflection_layer
