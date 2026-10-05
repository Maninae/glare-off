"""Radiance patterns of extended, textured sources: window, screen, sky wash, room environment.

Same canvas convention as `emitter_canvases` (unit plane [-1, 1]^2, peak about 1), but these
return float32 (N, N, 3) linear RGB because their content has its own colours (sky blue, foliage,
UI blocks). Scene content is deliberately low-detail: a reflection of a scene a metre or more
away is defocused when the camera focuses on the face.
"""

import cv2
import numpy as np

from glare_synthesis.emitter_canvases import downsample_canvas, fill_unit_polygon, new_supersampled_canvas, rounded_rectangle_polygon
from glare_synthesis.procedural_noise import fractal_random_field, smooth_random_field

SKY_ZENITH_RGB = np.array([0.55, 0.72, 1.0], np.float32)
SKY_HORIZON_RGB = np.array([0.95, 0.97, 1.0], np.float32)
FOLIAGE_RGB = np.array([0.16, 0.24, 0.12], np.float32)
BUILDING_RGB = np.array([0.45, 0.43, 0.42], np.float32)
SCREEN_WHITE_RGB = np.array([0.92, 0.96, 1.0], np.float32)
# UI accent colours seen on screens (blue links, dark mode panels, photo thumbnails).
SCREEN_ACCENT_COLORS_RGB = np.array([[0.2, 0.4, 0.9], [0.1, 0.1, 0.12], [0.85, 0.3, 0.25], [0.3, 0.7, 0.4], [0.9, 0.8, 0.4]], np.float32)


def vertical_gradient(canvas_size: int, top_rgb: np.ndarray, bottom_rgb: np.ndarray, start_fraction: float = 0.0, end_fraction: float = 1.0) -> np.ndarray:
    """(N, N, 3) gradient from top_rgb to bottom_rgb between the given canvas-height fractions."""
    row_fraction = (np.arange(canvas_size, dtype=np.float32) + 0.5) / canvas_size
    blend = np.clip((row_fraction - start_fraction) / max(end_fraction - start_fraction, 1e-3), 0.0, 1.0)[:, None, None]
    gradient_column = (1.0 - blend) * top_rgb + blend * bottom_rgb
    return np.broadcast_to(gradient_column, (canvas_size, canvas_size, 3)).copy()


def draw_outdoor_scene(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """Bright sky over a darker band of trees or buildings, blurred like a far-away scene."""
    horizon_fraction = random_generator.uniform(0.25, 0.75)
    scene = vertical_gradient(canvas_size, SKY_ZENITH_RGB, SKY_HORIZON_RGB, 0.0, horizon_fraction)
    horizon_profile = horizon_fraction + 0.12 * (smooth_random_field(random_generator, 1, canvas_size, 6)[0] - 0.5)
    row_fraction = ((np.arange(canvas_size, dtype=np.float32) + 0.5) / canvas_size)[:, None]
    below_horizon = (row_fraction > horizon_profile[None, :]).astype(np.float32)
    ground_texture = fractal_random_field(random_generator, canvas_size, canvas_size, 3)
    ground_base = FOLIAGE_RGB if random_generator.random() < 0.6 else BUILDING_RGB
    ground_rgb = ground_base[None, None, :] * (0.5 + ground_texture[..., None]) * random_generator.uniform(0.4, 1.2)
    below_horizon = cv2.GaussianBlur(below_horizon, (0, 0), canvas_size * 0.01 + 0.5)[..., None]
    return scene * (1.0 - below_horizon) + ground_rgb * below_horizon


def draw_window_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """A window pane quad with 1-3 vertical and 0-2 horizontal mullions over an outdoor scene."""
    aspect_ratio = random_generator.uniform(0.55, 1.0)
    half_width, half_height = (1.0, aspect_ratio) if random_generator.random() < 0.4 else (aspect_ratio, 1.0)
    pane_canvas = new_supersampled_canvas(canvas_size)
    fill_unit_polygon(pane_canvas, rounded_rectangle_polygon(half_width, half_height, 0.0), 1.0)
    mullion_half_thickness = random_generator.uniform(0.015, 0.05)
    for vertical_index in range(random_generator.integers(1, 4)):
        mullion_x = random_generator.uniform(-0.7, 0.7) * half_width if vertical_index else 0.0
        fill_unit_polygon(pane_canvas, rounded_rectangle_polygon(mullion_half_thickness, 1.2, 0.0, center=(mullion_x, 0.0)), 0.0)
    for _ in range(random_generator.integers(0, 3)):
        mullion_y = random_generator.uniform(-0.7, 0.7) * half_height
        fill_unit_polygon(pane_canvas, rounded_rectangle_polygon(1.2, mullion_half_thickness, 0.0, center=(0.0, mullion_y)), 0.0)
    pane_mask = cv2.GaussianBlur(downsample_canvas(pane_canvas, canvas_size), (0, 0), canvas_size * random_generator.uniform(0.005, 0.03) + 0.3)
    if random_generator.random() < 0.5:
        outdoor_scene = draw_outdoor_scene(random_generator, canvas_size)
    else:
        # Overcast or frosted glass: near-uniform bright pane with a soft gradient.
        outdoor_scene = vertical_gradient(canvas_size, SKY_HORIZON_RGB, SKY_HORIZON_RGB * random_generator.uniform(0.6, 0.95))
    return outdoor_scene * pane_mask[..., None]


def draw_screen_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """A monitor, laptop or phone: cool bright rectangle, often with darker UI blocks or text lines."""
    if random_generator.random() < 0.7:
        half_width, half_height = 1.0, random_generator.uniform(0.55, 0.75)
    else:
        half_width, half_height = random_generator.uniform(0.45, 0.6), 1.0
    screen_mask_canvas = new_supersampled_canvas(canvas_size)
    fill_unit_polygon(screen_mask_canvas, rounded_rectangle_polygon(half_width, half_height, random_generator.uniform(0.0, 0.08)), 1.0)
    screen_mask = downsample_canvas(screen_mask_canvas, canvas_size)
    screen_content = np.broadcast_to(SCREEN_WHITE_RGB, (canvas_size, canvas_size, 3)).copy()
    if random_generator.random() < 0.65:
        for _ in range(random_generator.integers(2, 9)):
            block_center = random_generator.uniform(-0.8, 0.8, size=2) * (half_width, half_height)
            block_half_size = random_generator.uniform(0.05, 0.4, size=2) * (half_width, half_height)
            if random_generator.random() < 0.4:
                block_half_size[1] = random_generator.uniform(0.015, 0.04)  # a line of text
            block_canvas = np.zeros((canvas_size, canvas_size), np.float32)
            fill_unit_polygon(block_canvas, rounded_rectangle_polygon(block_half_size[0], block_half_size[1], 0.0, center=tuple(block_center)), 1.0)
            # Screen content reflects at low contrast: accents are mixed halfway back to the backlight white.
            accent_rgb = 0.5 * SCREEN_ACCENT_COLORS_RGB[random_generator.integers(len(SCREEN_ACCENT_COLORS_RGB))] + 0.5 * SCREEN_WHITE_RGB
            screen_content = screen_content * (1.0 - block_canvas[..., None]) + accent_rgb * block_canvas[..., None]
    return screen_content * screen_mask[..., None]


def draw_sky_wash_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """Broad bright daylight wash: a smooth gradient with a soft dark boundary (roofline, trees)."""
    # Mostly white daylight; only a little of the sky's blue survives the bright wash.
    far_end_rgb = SKY_HORIZON_RGB * (1.0 - random_generator.uniform(0.0, 0.35)) + SKY_ZENITH_RGB * random_generator.uniform(0.0, 0.25)
    wash = vertical_gradient(canvas_size, SKY_HORIZON_RGB, far_end_rgb, 0.0, 1.0)
    if random_generator.random() < 0.5:
        wash = wash[::-1].copy()
    radius = np.hypot(*np.meshgrid(np.linspace(-1, 1, canvas_size), np.linspace(-1, 1, canvas_size)))
    soft_edge = np.clip((1.1 - radius) / random_generator.uniform(0.4, 0.9), 0.0, 1.0)
    # Never a constant tint: large-scale lumps plus bright streaks (window frames, building edges).
    structure = 0.45 + 0.55 * fractal_random_field(random_generator, canvas_size, canvas_size, 2)
    streak_profile = smooth_random_field(random_generator, 1, canvas_size, int(random_generator.integers(4, 10)))[0]
    structure = structure * (0.5 + 1.0 * streak_profile[None, :] ** 2)
    return wash * (soft_edge * structure)[..., None]


def desaturated_random_color(random_generator: np.random.Generator) -> np.ndarray:
    """Random room colour: a grey level with a little hue (real interiors are mostly neutral)."""
    grey_level = random_generator.uniform(0.05, 1.0)
    hue_rgb = random_generator.uniform(0.0, 1.0, size=3)
    saturation = random_generator.uniform(0.0, 0.35)
    return (grey_level * ((1.0 - saturation) + saturation * hue_rgb / max(hue_rgb.mean(), 1e-3))).astype(np.float32)


def draw_environment_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """A defocused room or street: coloured blocks and gradients with modest contrast."""
    base_rgb = desaturated_random_color(random_generator)
    environment = vertical_gradient(canvas_size, base_rgb, base_rgb * random_generator.uniform(0.3, 0.8))
    for _ in range(random_generator.integers(3, 9)):
        block_canvas = np.zeros((canvas_size, canvas_size), np.float32)
        block_center = random_generator.uniform(-1.0, 1.0, size=2)
        block_half_size = random_generator.uniform(0.1, 0.6, size=2)
        fill_unit_polygon(block_canvas, rounded_rectangle_polygon(block_half_size[0], block_half_size[1], 0.0, center=tuple(block_center)), 1.0)
        block_rgb = desaturated_random_color(random_generator)
        environment = environment * (1.0 - block_canvas[..., None]) + block_rgb * block_canvas[..., None]
    if random_generator.random() < 0.5:
        environment = environment * 0.6 + draw_outdoor_scene(random_generator, canvas_size) * 0.4
    return cv2.GaussianBlur(environment, (0, 0), canvas_size * random_generator.uniform(0.01, 0.04) + 0.5)
