"""Physical parameters per light-source type, and the canvas drawer each type uses.

Sizes and distances are world metres; the mirror model in `glare_scene_sampling` turns them into
reflection sizes in pixels. Ranges follow the research doc (Q3 table) except where a reference
photo disagreed; deviations are explained in `glare_synthesis/CLAUDE.md`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from glare_synthesis.color_science import REC709_LUMINANCE_WEIGHTS, color_temperature_to_linear_rgb
from glare_synthesis.emitter_canvases import (
    draw_ceiling_grid_canvas,
    draw_point_glint_canvas,
    draw_ring_light_canvas,
    draw_softbox_canvas,
    draw_strip_light_canvas,
)
from glare_synthesis.glare_sampling_config import LightSourceType
from glare_synthesis.scene_source_canvases import draw_environment_canvas, draw_screen_canvas, draw_sky_wash_canvas, draw_window_canvas


@dataclass(frozen=True)
class LightSourceSpec:
    """How one light type is drawn, sized, placed, coloured and capped in brightness."""

    draw_canvas: Callable[[np.random.Generator, int], np.ndarray]
    physical_size_m_range: tuple[float, float]
    source_distance_m_range: tuple[float, float]
    color_temperature_kelvin_range: tuple[float, float]
    # Max offset of the reflection center from the lens center, as a fraction of lens width.
    placement_spread_fraction: float
    # In-plane rotation range in degrees (0 = upright in the eye-level crop).
    rotation_degrees_range: tuple[float, float]
    # Brightness ceiling (linear, 1 = clip) regardless of the sampled severity.
    peak_linear_ceiling: float = 1e9
    # Peak boost for small or thin sources: blur conserves their energy but spreads it, so their
    # radiance (physically 50-1000x the face) must be higher for the same visible severity.
    small_source_peak_multiplier_range: tuple[float, float] = (1.0, 1.0)
    # Texture continues past the canvas edge (environment) instead of ending at black.
    extends_beyond_canvas: bool = False
    allows_ghost_reflection: bool = True
    # Smallest reflection, as a fraction of lens width; below ~5% a glint drowns in the photo's own blur.
    minimum_size_ratio: float = 0.015
    # Per-type inner-surface reflection overrides (None = the config / sampler defaults).
    ghost_probability: float | None = None
    ghost_scale_range: tuple[float, float] | None = None


LIGHT_SOURCE_SPECS: dict[LightSourceType, LightSourceSpec] = {
    LightSourceType.RING_LIGHT: LightSourceSpec(draw_ring_light_canvas, (0.25, 0.5), (0.3, 1.0), (4000.0, 6500.0), 0.25, (-180.0, 180.0),
                                                ghost_probability=0.7, ghost_scale_range=(0.2, 0.5)),
    LightSourceType.SOFTBOX: LightSourceSpec(draw_softbox_canvas, (0.4, 1.2), (0.8, 3.0), (3200.0, 6500.0), 0.4, (-12.0, 12.0)),
    LightSourceType.WINDOW: LightSourceSpec(draw_window_canvas, (0.8, 2.0), (0.8, 4.0), (5000.0, 7500.0), 0.45, (-10.0, 10.0), peak_linear_ceiling=2.0),
    LightSourceType.SCREEN: LightSourceSpec(draw_screen_canvas, (0.15, 0.7), (0.3, 0.8), (6500.0, 9500.0), 0.35, (-15.0, 15.0), peak_linear_ceiling=1.0),
    LightSourceType.POINT_GLINT: LightSourceSpec(draw_point_glint_canvas, (0.01, 0.12), (0.5, 5.0), (2400.0, 6500.0), 0.45, (-180.0, 180.0),
                                                 small_source_peak_multiplier_range=(2.0, 6.0), minimum_size_ratio=0.05),
    LightSourceType.STRIP_LIGHT: LightSourceSpec(draw_strip_light_canvas, (0.6, 1.5), (0.8, 3.0), (3500.0, 6500.0), 0.45, (-90.0, 90.0),
                                                 small_source_peak_multiplier_range=(1.0, 2.5)),
    LightSourceType.CEILING_GRID: LightSourceSpec(draw_ceiling_grid_canvas, (1.2, 3.5), (1.5, 3.5), (3500.0, 6500.0), 0.4, (-30.0, 30.0)),
    LightSourceType.SKY_WASH: LightSourceSpec(draw_sky_wash_canvas, (3.0, 10.0), (3.0, 20.0), (6000.0, 12000.0), 0.35, (-20.0, 20.0),
                                              peak_linear_ceiling=1.3, allows_ghost_reflection=False),
    LightSourceType.ENVIRONMENT: LightSourceSpec(draw_environment_canvas, (2.0, 6.0), (1.5, 6.0), (3000.0, 7000.0), 0.3, (-10.0, 10.0),
                                                 peak_linear_ceiling=0.2, extends_beyond_canvas=True, allows_ghost_reflection=False),
}


def draw_colored_source_canvas(light_type: LightSourceType, random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """Draw a type's canvas, colour it by a sampled CCT, and normalize its peak luminance to 1.

    Returns float32 (N, N, 3) linear RGB.
    """
    spec = LIGHT_SOURCE_SPECS[light_type]
    canvas = spec.draw_canvas(random_generator, canvas_size)
    light_rgb = color_temperature_to_linear_rgb(random_generator.uniform(*spec.color_temperature_kelvin_range))
    colored_canvas = canvas[..., None] * light_rgb if canvas.ndim == 2 else canvas * light_rgb
    peak_luminance = float((colored_canvas @ REC709_LUMINANCE_WEIGHTS).max())
    return (colored_canvas / max(peak_luminance, 1e-6)).astype(np.float32)
