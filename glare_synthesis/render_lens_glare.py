"""Entry point: paint physically modeled lens glare onto a clean eye crop and return the training data_dict.

`render_lens_glare_sample` is the pure function the DataLoader calls. It splits into
`sample_glare_scene` (all randomness) and `render_glare_scene` (deterministic), so a fixed scene
can be re-rendered, e.g. at several brightnesses for the severity ladder.

What differs between `glare_eye_crop` (input) and `clean_eye_crop` (target), and nothing else:
- added reflected light (sources, ghosts, veil), clipped by each lens's soft mask;
- highlight roll-off/clipping of those pixels, and bloom up to ~3 bloom sigmas outside the lens;
- local darkening of the transmitted image under strong glare (transmission attenuation);
- shot noise proportional to the added light, at the photo's own noise gain;
- JPEG artifacts of the glare delta only (16x16 blocks the glare does not touch are unchanged).
Every other pixel is bit-identical, so the model is never taught to denoise or de-JPEG the face.
"""

import numpy as np

from glare_synthesis.camera_response import (
    apply_glare_forward_model,
    bloom_light,
    encode_glared_linear,
    estimate_photo_blur_sigma,
    estimate_photo_noise_sigma,
    glare_shot_noise,
    jpeg_compress_glare_delta,
    transmission_attenuation_map,
)
from glare_synthesis.color_science import REC709_LUMINANCE_WEIGHTS, srgb_to_linear
from glare_synthesis.glare_masks import compute_glare_mask, compute_lost_detail_mask, compute_remaining_detail_contrast
from glare_synthesis.glare_sampling_config import GlareSamplingConfig
from glare_synthesis.glare_scene import GlareScene
from glare_synthesis.glare_scene_sampling import sample_glare_scene
from glare_synthesis.hdri_environment_library import HdriEnvironmentLibrary, get_hdri_environment_library
from glare_synthesis.lens_geometry import LensGeometry, analyze_lens_label_mask
from glare_synthesis.reflection_layer_rendering import render_reflection_layer

JPEG_BLOCK_ALIGNMENT = 16
BLOOM_REACH_IN_SIGMAS = 3.0
ROI_SAFETY_MARGIN_PIXELS = 4
# The photo's blur and noise are measured on the lenses plus this share of crop width around them.
STATISTICS_REGION_MARGIN_FRACTION = 0.03
DEFAULT_SAMPLING_CONFIG = GlareSamplingConfig()
# Face linear luminance used for HDRI exposure is clamped to this range (very dark or blown crops).
FACE_LINEAR_LUMINANCE_LIMITS = (0.02, 0.6)


def validate_render_inputs(clean_eye_crop: np.ndarray, lens_label_mask: np.ndarray) -> np.ndarray:
    """Check shapes and value range; return the crop as float32."""
    clean_eye_crop = np.asarray(clean_eye_crop, dtype=np.float32)
    if clean_eye_crop.ndim != 3 or clean_eye_crop.shape[2] != 3:
        raise ValueError(f"clean_eye_crop must be (H, W, 3), got {clean_eye_crop.shape}")
    if lens_label_mask.shape != clean_eye_crop.shape[:2]:
        raise ValueError(f"lens_label_mask shape {lens_label_mask.shape} does not match crop {clean_eye_crop.shape[:2]}")
    if float(clean_eye_crop.min()) < 0.0 or float(clean_eye_crop.max()) > 1.0:
        raise ValueError("clean_eye_crop must hold sRGB values in [0, 1]")
    return clean_eye_crop


def build_glare_data_dict(clean_eye_crop: np.ndarray, glare_eye_crop: np.ndarray, glare_mask: np.ndarray,
                          lost_detail_mask: np.ndarray, lens_label_mask: np.ndarray, source_id: str) -> dict:
    """Assemble the synthesis data_dict (CLAUDE.md contract); masks become (H, W, 1) float32."""
    return {
        "clean_eye_crop": clean_eye_crop,
        "glare_eye_crop": glare_eye_crop,
        "glare_mask": glare_mask[..., None].astype(np.float32),
        "lost_detail_mask": lost_detail_mask[..., None].astype(np.float32),
        "lens_mask": (lens_label_mask > 0)[..., None].astype(np.float32),
        "source_id": source_id,
    }


def glare_region_of_interest(scene: GlareScene, lens_geometries: list[LensGeometry], photo_blur_sigma: float,
                             crop_height: int, crop_width: int) -> tuple[int, int, int, int]:
    """Box holding every pixel the glare can touch (lenses + blur + bloom reach), snapped to JPEG blocks."""
    blur_reach = max((source.defocus_radius_pixels + source.motion_blur_length_pixels for source in scene.light_sources), default=0.0)
    margin = int(np.ceil(BLOOM_REACH_IN_SIGMAS * (scene.bloom_sigma_pixels + photo_blur_sigma) + blur_reach)) + ROI_SAFETY_MARGIN_PIXELS
    x0 = min(lens.bounding_box_xyxy[0] for lens in lens_geometries) - margin
    y0 = min(lens.bounding_box_xyxy[1] for lens in lens_geometries) - margin
    x1 = max(lens.bounding_box_xyxy[2] for lens in lens_geometries) + margin
    y1 = max(lens.bounding_box_xyxy[3] for lens in lens_geometries) + margin
    x0, y0 = max(0, x0 // JPEG_BLOCK_ALIGNMENT * JPEG_BLOCK_ALIGNMENT), max(0, y0 // JPEG_BLOCK_ALIGNMENT * JPEG_BLOCK_ALIGNMENT)
    x1 = min(crop_width, -(-x1 // JPEG_BLOCK_ALIGNMENT) * JPEG_BLOCK_ALIGNMENT)
    y1 = min(crop_height, -(-y1 // JPEG_BLOCK_ALIGNMENT) * JPEG_BLOCK_ALIGNMENT)
    return x0, y0, x1, y1


def glasses_statistics_region(clean_eye_crop: np.ndarray, lens_geometries: list[LensGeometry]) -> np.ndarray:
    """The crop around both lenses (plus frame), where the photo's blur and noise are measured."""
    crop_height, crop_width = clean_eye_crop.shape[:2]
    margin = int(round(STATISTICS_REGION_MARGIN_FRACTION * crop_width))
    x0 = max(0, min(lens.bounding_box_xyxy[0] for lens in lens_geometries) - margin)
    y0 = max(0, min(lens.bounding_box_xyxy[1] for lens in lens_geometries) - margin)
    x1 = min(crop_width, max(lens.bounding_box_xyxy[2] for lens in lens_geometries) + margin)
    y1 = min(crop_height, max(lens.bounding_box_xyxy[3] for lens in lens_geometries) + margin)
    return clean_eye_crop[y0:y1, x0:x1]


def hdri_library_for_config(sampling_config: GlareSamplingConfig) -> HdriEnvironmentLibrary | None:
    """The per-process HDRI library when the config uses image-based reflections, else None."""
    if sampling_config.hdri_reflection_share <= 0:
        return None
    return get_hdri_environment_library(sampling_config.hdri_baked_manifest_path)


def render_glare_scene(clean_eye_crop: np.ndarray, lens_label_mask: np.ndarray, lens_geometries: list[LensGeometry],
                       scene: GlareScene, source_id: str = "", intensity_scale: float = 1.0,
                       hdri_library: HdriEnvironmentLibrary | None = None) -> dict:
    """Deterministically render a sampled scene onto the crop and return the data_dict.

    `hdri_library` is required when `scene.hdri_reflection` is set.
    """
    crop_height, crop_width = clean_eye_crop.shape[:2]
    empty_mask = np.zeros((crop_height, crop_width), np.float32)
    if scene.is_glare_free or not lens_geometries:
        return build_glare_data_dict(clean_eye_crop, clean_eye_crop.copy(), empty_mask, empty_mask, lens_label_mask, source_id)
    glasses_region = glasses_statistics_region(clean_eye_crop, lens_geometries)
    photo_blur_sigma = estimate_photo_blur_sigma(glasses_region)
    x0, y0, x1, y1 = glare_region_of_interest(scene, lens_geometries, photo_blur_sigma, crop_height, crop_width)
    clean_srgb = clean_eye_crop[y0:y1, x0:x1]
    clean_linear = srgb_to_linear(clean_srgb)
    hdri_environment = hdri_library.load_environment(scene.hdri_reflection.hdri_id) if scene.hdri_reflection is not None else None
    face_linear_luminance = float(np.clip((srgb_to_linear(glasses_region) @ REC709_LUMINANCE_WEIGHTS).mean(), *FACE_LINEAR_LUMINANCE_LIMITS))
    crop_center_xy = np.array([crop_width / 2.0, crop_height / 2.0])
    reflection_linear = render_reflection_layer(scene, lens_geometries, (x0, y0, x1, y1), photo_blur_sigma, intensity_scale,
                                                hdri_environment, face_linear_luminance, crop_center_xy)
    attenuation = transmission_attenuation_map(reflection_linear, scene.transmission_attenuation)
    bloom_linear = bloom_light(clean_linear * (1.0 - attenuation) + reflection_linear, scene.bloom_sigma_pixels, scene.bloom_strength)

    glared_noise_free_linear = apply_glare_forward_model(clean_linear, reflection_linear, attenuation, bloom_linear, scene.soft_knee_start)
    glared_noise_free_srgb = encode_glared_linear(clean_srgb, clean_linear, glared_noise_free_linear)
    glare_mask_roi = compute_glare_mask(clean_srgb, glared_noise_free_srgb)
    remaining_contrast = compute_remaining_detail_contrast(clean_srgb, clean_linear, glared_noise_free_linear, reflection_linear, attenuation, scene.soft_knee_start)
    lost_detail_mask_roi = compute_lost_detail_mask(remaining_contrast, glare_mask_roi)

    photo_noise_sigma = estimate_photo_noise_sigma(glasses_region)
    noise_linear = glare_shot_noise(reflection_linear, clean_linear, photo_noise_sigma, scene.glare_noise_scale, scene.noise_seed)
    glared_linear = apply_glare_forward_model(clean_linear, reflection_linear + noise_linear, attenuation, bloom_linear, scene.soft_knee_start)
    glared_srgb = encode_glared_linear(clean_srgb, clean_linear, glared_linear)
    if scene.jpeg_quality is not None:
        glared_srgb = jpeg_compress_glare_delta(clean_srgb, glared_srgb, scene.jpeg_quality)

    glare_eye_crop = clean_eye_crop.copy()
    glare_eye_crop[y0:y1, x0:x1] = glared_srgb
    glare_mask = empty_mask.copy()
    glare_mask[y0:y1, x0:x1] = glare_mask_roi
    lost_detail_mask = empty_mask.copy()
    lost_detail_mask[y0:y1, x0:x1] = lost_detail_mask_roi
    return build_glare_data_dict(clean_eye_crop, glare_eye_crop, glare_mask, lost_detail_mask, lens_label_mask, source_id)


def render_lens_glare_sample(
    clean_eye_crop: np.ndarray,
    lens_label_mask: np.ndarray,
    random_generator: np.random.Generator,
    sampling_config: GlareSamplingConfig = DEFAULT_SAMPLING_CONFIG,
    source_id: str = "",
) -> dict:
    """Paint random lens glare on a clean eye crop; returns the synthesis data_dict (CLAUDE.md).

    Args:
        clean_eye_crop: float32 (H, W, 3) sRGB in [0, 1]; any size (512x256 default, 1024x512 works).
        lens_label_mask: uint8 (H, W): 0 background, 1 image-left lens, 2 image-right lens; either may be absent.
        random_generator: the only source of randomness; the same seed gives the same sample.
        sampling_config: the sampling distribution (defaults = training distribution).
        source_id: copied into the data_dict.
    Returns:
        dict with clean_eye_crop, glare_eye_crop, glare_mask, lost_detail_mask, lens_mask, source_id.

    - `sampling_config.no_glare_probability` of samples (and every crop without lenses) come back
      glare-free: glare_eye_crop equals clean_eye_crop and both masks are empty.
    - No torch, no global state; safe to call inside DataLoader workers (set cv2.setNumThreads(1) there).
    """
    clean_eye_crop = validate_render_inputs(clean_eye_crop, lens_label_mask)
    lens_geometries = analyze_lens_label_mask(lens_label_mask)
    hdri_library = hdri_library_for_config(sampling_config)
    scene = sample_glare_scene(lens_geometries, random_generator, sampling_config, hdri_library)
    return render_glare_scene(clean_eye_crop, lens_label_mask, lens_geometries, scene, source_id, hdri_library=hdri_library)
