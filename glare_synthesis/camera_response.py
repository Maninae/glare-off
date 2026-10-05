"""Camera-side effects of glare: highlight roll-off and clipping, bloom, the glare's own shot noise,
JPEG on the glare, and estimators of the clean photo's blur and noise so the glare matches it.

Identity rule: every function here leaves a pixel bit-identical when no reflection light reaches it.
The forward model therefore works on DELTAS: out = clean + T(clean_attenuated + R) - T(clean), where T
is the highlight response, so non-glare pixels keep the clean photo's own tone, noise and JPEG.
"""

import cv2
import numpy as np

from glare_synthesis.color_science import REC709_LUMINANCE_WEIGHTS, SRGB_GAMMA, linear_to_srgb

# Soft knee reaches the clip level at knee + this many knee-widths above the knee.
SOFT_KNEE_SATURATION_WIDTHS = 3.0
# Reflection luminance at which local tone-mapping attenuation reaches its full strength.
ATTENUATION_FULL_EFFECT_LUMINANCE = 0.2
# Re-blur blur estimator (Zhuo & Sim 2011 style): known Gaussian re-blur and the edge share used.
REBLUR_SIGMA_PIXELS = 1.0
SHARPEST_EDGE_PERCENTILE = 98.0
MINIMUM_EDGE_GRADIENT = 0.04
PHOTO_BLUR_SIGMA_LIMITS = (0.3, 3.0)
# Immerkaer (1996) fast noise estimator, restricted to the flatter half of the image.
IMMERKAER_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
PHOTO_NOISE_SIGMA_LIMITS = (0.002, 0.05)
MINIMUM_MEAN_LINEAR_FOR_GAIN = 0.02


def estimate_photo_blur_sigma(clean_srgb: np.ndarray) -> float:
    """Gaussian sigma (pixels) of the photo's sharpest edges, from the gradient drop under a known re-blur.

    For a Gaussian-blurred step of sigma s, re-blurring by b lowers the peak gradient by
    r = sqrt(s^2 + b^2) / s, so s = b / sqrt(r^2 - 1). Median over the sharpest edges.
    """
    gray = cv2.cvtColor(clean_srgb, cv2.COLOR_RGB2GRAY)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    reblurred = cv2.GaussianBlur(gray, (0, 0), REBLUR_SIGMA_PIXELS)
    reblurred_gradient = cv2.magnitude(cv2.Sobel(reblurred, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(reblurred, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    edge_threshold = max(float(np.percentile(gradient, SHARPEST_EDGE_PERCENTILE)), MINIMUM_EDGE_GRADIENT)
    edge_pixels = gradient >= edge_threshold
    if int(edge_pixels.sum()) < 10:
        return PHOTO_BLUR_SIGMA_LIMITS[1]
    gradient_ratio = np.clip(gradient[edge_pixels] / np.maximum(reblurred_gradient[edge_pixels], 1e-6), 1.01, None)
    blur_sigma = float(np.median(REBLUR_SIGMA_PIXELS / np.sqrt(gradient_ratio**2 - 1.0)))
    return float(np.clip(blur_sigma, *PHOTO_BLUR_SIGMA_LIMITS))


def estimate_photo_noise_sigma(clean_srgb: np.ndarray) -> float:
    """Noise standard deviation (sRGB units) of the photo, measured on its flatter half."""
    gray = cv2.cvtColor(clean_srgb, cv2.COLOR_RGB2GRAY)
    laplacian_response = np.abs(cv2.filter2D(gray, cv2.CV_32F, IMMERKAER_KERNEL))[1:-1, 1:-1]
    gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    flat_pixels = gradient[1:-1, 1:-1] <= np.median(gradient)
    noise_sigma = float(np.sqrt(np.pi / 2.0) / 6.0 * laplacian_response[flat_pixels].mean())
    return float(np.clip(noise_sigma, *PHOTO_NOISE_SIGMA_LIMITS))


def highlight_response(linear_values: np.ndarray, soft_knee_start: float | None) -> np.ndarray:
    """Sensor + tone curve near white: identity below the knee, exponential roll-off, hard clip at 1."""
    if soft_knee_start is None:
        return np.minimum(linear_values, 1.0)
    knee_width = 1.0 - soft_knee_start
    normalizer = 1.0 - np.exp(-SOFT_KNEE_SATURATION_WIDTHS)
    rolled_off = soft_knee_start + knee_width * (1.0 - np.exp(-(linear_values - soft_knee_start) / knee_width)) / normalizer
    return np.minimum(np.where(linear_values > soft_knee_start, rolled_off, linear_values), 1.0)


def highlight_response_slope(linear_values: np.ndarray, soft_knee_start: float | None) -> np.ndarray:
    """Derivative of `highlight_response`: 1 below the knee, decaying through the roll-off, 0 once clipped."""
    if soft_knee_start is None:
        return (linear_values < 1.0).astype(np.float32)
    knee_width = 1.0 - soft_knee_start
    normalizer = 1.0 - np.exp(-SOFT_KNEE_SATURATION_WIDTHS)
    rolled_off_slope = np.exp(-(linear_values - soft_knee_start) / knee_width) / normalizer
    slope = np.where(linear_values > soft_knee_start, rolled_off_slope, 1.0)
    return np.where(highlight_response(linear_values, soft_knee_start) >= 1.0, 0.0, slope).astype(np.float32)


def transmission_attenuation_map(reflection_linear: np.ndarray, attenuation: float) -> np.ndarray:
    """Local tone-mapping darkening (H, W, 1): zero where no reflection, `attenuation` under strong glare."""
    if attenuation <= 0:
        return np.zeros(reflection_linear.shape[:2] + (1,), np.float32)
    reflection_luminance = reflection_linear @ REC709_LUMINANCE_WEIGHTS
    return (attenuation * np.clip(reflection_luminance / ATTENUATION_FULL_EFFECT_LUMINANCE, 0.0, 1.0))[..., None].astype(np.float32)


def bloom_light(exposed_linear: np.ndarray, bloom_sigma_pixels: float, bloom_strength: float) -> np.ndarray:
    """Light scattered around over-exposed pixels: a Gaussian of the energy above the clip level."""
    if bloom_strength <= 0 or bloom_sigma_pixels <= 0:
        return np.zeros_like(exposed_linear)
    over_exposure = np.maximum(exposed_linear - 1.0, 0.0)
    if not over_exposure.any():
        return np.zeros_like(exposed_linear)
    return bloom_strength * cv2.GaussianBlur(over_exposure, (0, 0), bloom_sigma_pixels, borderType=cv2.BORDER_CONSTANT)


def apply_glare_forward_model(clean_linear: np.ndarray, reflection_linear: np.ndarray, attenuation: np.ndarray,
                              bloom_linear: np.ndarray, soft_knee_start: float | None) -> np.ndarray:
    """Glared linear image; exactly `clean_linear` wherever reflection, attenuation and bloom are zero."""
    exposed_linear = clean_linear * (1.0 - attenuation) + reflection_linear
    glare_delta = highlight_response(exposed_linear, soft_knee_start) - highlight_response(clean_linear, soft_knee_start)
    glared_linear = clean_linear + glare_delta + bloom_linear
    has_glare = (reflection_linear != 0) | (bloom_linear != 0) | (attenuation != 0)
    return np.where(has_glare, np.clip(glared_linear, 0.0, 1.0), clean_linear)


def glare_shot_noise(reflection_linear: np.ndarray, clean_linear: np.ndarray, photo_noise_sigma_srgb: float,
                     noise_scale: float, noise_seed: int) -> np.ndarray:
    """Zero-mean Poisson-like noise for the added light, with the photo's own gain (photon transfer).

    The photo's sRGB noise sigma is converted to linear at its mean level m, giving a gain
    g = sigma_lin^2 / m; the reflection R then adds variance g * R. Zero where R is zero.
    """
    mean_linear = max(float(clean_linear.mean()), MINIMUM_MEAN_LINEAR_FOR_GAIN)
    srgb_slope_at_mean = 1.055 / SRGB_GAMMA * mean_linear ** (1.0 / SRGB_GAMMA - 1.0)
    noise_gain = (photo_noise_sigma_srgb / srgb_slope_at_mean) ** 2 / mean_linear * noise_scale**2
    noise_generator = np.random.default_rng(noise_seed)
    standard_normal = noise_generator.standard_normal(reflection_linear.shape, dtype=np.float32)
    return standard_normal * np.sqrt(noise_gain * np.maximum(reflection_linear, 0.0))


def jpeg_round_trip(srgb_image: np.ndarray, jpeg_quality: int) -> np.ndarray:
    """Encode an RGB float image as JPEG (4:2:0) at the given quality and decode it back to float."""
    bgr_uint8 = cv2.cvtColor(np.round(np.clip(srgb_image, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, encoded_bytes = cv2.imencode(".jpg", bgr_uint8, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    return cv2.cvtColor(cv2.imdecode(encoded_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def jpeg_compress_glare_delta(clean_srgb: np.ndarray, glared_srgb: np.ndarray, jpeg_quality: int) -> np.ndarray:
    """Give the glare JPEG character without re-compressing the clean photo.

    Returns clean + (jpeg(glared) - jpeg(clean)). JPEG codes 16x16 blocks independently, so every
    block the glare does not touch encodes identically and its delta is exactly zero.
    """
    jpeg_glare_delta = jpeg_round_trip(glared_srgb, jpeg_quality) - jpeg_round_trip(clean_srgb, jpeg_quality)
    return np.clip(clean_srgb + jpeg_glare_delta, 0.0, 1.0)


def encode_glared_linear(clean_srgb: np.ndarray, clean_linear: np.ndarray, glared_linear: np.ndarray) -> np.ndarray:
    """sRGB-encode the glared image; pixels the glare left untouched are copied bit-identical from the clean crop."""
    glared_srgb = clean_srgb.copy()
    changed_pixels = (glared_linear != clean_linear).any(axis=-1)
    glared_srgb[changed_pixels] = linear_to_srgb(glared_linear[changed_pixels])
    return glared_srgb
