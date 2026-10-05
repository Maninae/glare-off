"""Radiance patterns of compact emitters: ring light, softbox, point glints, strips, ceiling grids.

Each `draw_*` function paints one source on a square canvas covering the unit source plane
[-1, 1] x [-1, 1] (x right, y down) and returns float32 (N, N) relative radiance, peak about 1.
The longest dimension of the source spans the canvas; the mirror mapping later sets its size,
orientation and distortion in each lens. Colour is applied by the caller (`light_source_registry`).
"""

import cv2
import numpy as np

from glare_synthesis.procedural_noise import smooth_random_field

CANVAS_SUPERSAMPLE = 3
FIXED_POINT_SHIFT_BITS = 4
FIXED_POINT_SCALE = 2**FIXED_POINT_SHIFT_BITS
ROUNDED_CORNER_ARC_POINTS = 8


def unit_points_to_fixed_point_pixels(points_unit: np.ndarray, canvas_size: int) -> np.ndarray:
    """Map unit-plane points to cv2 fixed-point pixel coordinates (pixel centers at i + 0.5)."""
    pixel_points = (np.asarray(points_unit, np.float64) + 1.0) * 0.5 * canvas_size - 0.5
    return np.round(pixel_points * FIXED_POINT_SCALE).astype(np.int32)


def fill_unit_polygon(canvas: np.ndarray, points_unit: np.ndarray, value: float) -> None:
    """Fill a polygon given in unit-plane coordinates, anti-aliased, in place."""
    pixel_points = unit_points_to_fixed_point_pixels(points_unit, canvas.shape[0])
    cv2.fillPoly(canvas, [pixel_points], float(value), lineType=cv2.LINE_AA, shift=FIXED_POINT_SHIFT_BITS)


def fill_unit_ellipse(canvas: np.ndarray, center_unit: tuple[float, float], radii_unit: tuple[float, float], value: float, angle_degrees: float = 0.0) -> None:
    """Fill an ellipse given in unit-plane coordinates, anti-aliased, in place."""
    center_pixels = unit_points_to_fixed_point_pixels(np.array(center_unit), canvas.shape[0])
    radii_pixels = np.round(np.array(radii_unit) * 0.5 * canvas.shape[0] * FIXED_POINT_SCALE).astype(np.int32)
    cv2.ellipse(canvas, (int(center_pixels[0]), int(center_pixels[1])), (max(1, int(radii_pixels[0])), max(1, int(radii_pixels[1]))),
                angle_degrees, 0, 360, float(value), thickness=-1, lineType=cv2.LINE_AA, shift=FIXED_POINT_SHIFT_BITS)


def rounded_rectangle_polygon(half_width: float, half_height: float, corner_radius: float, center: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Boundary points of a rounded rectangle in unit-plane coordinates."""
    corner_radius = min(corner_radius, half_width, half_height)
    corner_centers = [(half_width - corner_radius, half_height - corner_radius), (-half_width + corner_radius, half_height - corner_radius),
                      (-half_width + corner_radius, -half_height + corner_radius), (half_width - corner_radius, -half_height + corner_radius)]
    boundary_points = []
    for corner_index, (corner_x, corner_y) in enumerate(corner_centers):
        arc_angles = np.linspace(corner_index * np.pi / 2, (corner_index + 1) * np.pi / 2, ROUNDED_CORNER_ARC_POINTS)
        boundary_points.extend(zip(corner_x + corner_radius * np.cos(arc_angles), corner_y + corner_radius * np.sin(arc_angles)))
    return np.array(boundary_points) + np.array(center)


def new_supersampled_canvas(canvas_size: int) -> np.ndarray:
    """Blank float32 canvas at CANVAS_SUPERSAMPLE x the output resolution."""
    return np.zeros((canvas_size * CANVAS_SUPERSAMPLE, canvas_size * CANVAS_SUPERSAMPLE), np.float32)


def downsample_canvas(supersampled_canvas: np.ndarray, canvas_size: int) -> np.ndarray:
    """Area-average the supersampled canvas back to (canvas_size, canvas_size)."""
    return cv2.resize(supersampled_canvas, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)


def unit_plane_polar_coordinates(canvas_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Radius and angle of every canvas pixel center in the unit plane."""
    pixel_centers = (np.arange(canvas_size, dtype=np.float32) + 0.5) / canvas_size * 2.0 - 1.0
    grid_x, grid_y = np.meshgrid(pixel_centers, pixel_centers)
    return np.hypot(grid_x, grid_y), np.arctan2(grid_y, grid_x)


def draw_ring_light_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """Thick annulus with uneven width, LED dots or diffuser grain, a soft glow, sometimes a partial arc.

    The radii are evaluated per angle (smooth random wobble), so the ring is never a perfect
    vector circle; the glow is a wide faint halo like light scattered in the diffuser and lens.
    """
    radius, angle = unit_plane_polar_coordinates(canvas_size * CANVAS_SUPERSAMPLE)
    wobble_generator_field = smooth_random_field(random_generator, 1, 64, 6)[0]
    angle_index = ((angle + np.pi) / (2 * np.pi) * 63).astype(np.int32)
    wobble = (wobble_generator_field[angle_index] - 0.5) * random_generator.uniform(0.04, 0.14)
    inner_radius_ratio = random_generator.uniform(0.45, 0.75)
    outer_radius = 1.0 - abs(wobble) * 0.5
    inner_radius = inner_radius_ratio + wobble
    edge_softness = 2.0 / (canvas_size * CANVAS_SUPERSAMPLE)
    ring = np.clip((outer_radius - radius) / edge_softness, 0.0, 1.0) * np.clip((radius - inner_radius) / edge_softness, 0.0, 1.0)
    if random_generator.random() < 0.3:
        # Bare LED ring: discrete bright dots along the band.
        led_count = int(random_generator.integers(36, 90))
        led_phase = np.cos(angle * led_count + random_generator.uniform(0, 2 * np.pi))
        ring = ring * (0.35 + 0.65 * np.clip(led_phase, 0.0, 1.0) ** 2)
    else:
        grain = smooth_random_field(random_generator, ring.shape[0], ring.shape[1], max(8, canvas_size // 4))
        ring = ring * (0.88 + 0.12 * grain)
    if random_generator.random() < 0.35:
        # Partial arc: a hand, phone or the lens rim hides a sector of the ring.
        gap_center = random_generator.uniform(-np.pi, np.pi)
        gap_half_width = random_generator.uniform(np.pi / 6, np.pi / 1.6)
        angular_distance = np.abs((angle - gap_center + np.pi) % (2 * np.pi) - np.pi)
        ring = ring * np.clip((angular_distance - gap_half_width) / 0.08, 0.0, 1.0)
    pattern = downsample_canvas(ring.astype(np.float32), canvas_size)
    _, coarse_angle = unit_plane_polar_coordinates(canvas_size)
    brightness_drift = 1.0 - random_generator.uniform(0.0, 0.35) * (0.5 + 0.5 * np.cos(coarse_angle - random_generator.uniform(0, 2 * np.pi)))
    glow = cv2.GaussianBlur(pattern, (0, 0), canvas_size * random_generator.uniform(0.05, 0.12)) * random_generator.uniform(0.05, 0.2)
    return pattern * brightness_drift + glow


def draw_softbox_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """Rounded rectangle, octagon or ribbed umbrella; slightly hotter in the middle."""
    canvas = new_supersampled_canvas(canvas_size)
    shape_choice = random_generator.random()
    if shape_choice < 0.55:
        aspect_ratio = random_generator.uniform(0.55, 1.0)
        half_width, half_height = (1.0, aspect_ratio) if random_generator.random() < 0.5 else (aspect_ratio, 1.0)
        fill_unit_polygon(canvas, rounded_rectangle_polygon(half_width, half_height, random_generator.uniform(0.0, 0.35)), 1.0)
    else:
        side_count = 8 if shape_choice < 0.85 else random_generator.integers(12, 17)
        polygon_angles = np.arange(side_count) * 2 * np.pi / side_count + random_generator.uniform(0, np.pi)
        fill_unit_polygon(canvas, np.stack([np.cos(polygon_angles), np.sin(polygon_angles)], axis=1), 1.0)
        if shape_choice >= 0.85:
            # Umbrella: dark ribs from the center to each vertex.
            rib_width = canvas.shape[0] * random_generator.uniform(0.01, 0.025)
            center_pixel = canvas.shape[0] // 2
            for polygon_angle in polygon_angles:
                end_pixel = (int(center_pixel * (1 + np.cos(polygon_angle))), int(center_pixel * (1 + np.sin(polygon_angle))))
                cv2.line(canvas, (center_pixel, center_pixel), end_pixel, 0.55, max(1, int(rib_width)), lineType=cv2.LINE_AA)
    pattern = downsample_canvas(canvas, canvas_size)
    radius, _ = unit_plane_polar_coordinates(canvas_size)
    hot_spot_falloff = random_generator.uniform(0.1, 0.4)
    # Baffle: the recessed diffuser edge reads as a darker band just inside the outline.
    baffle_band_width = max(1.0, canvas_size * random_generator.uniform(0.03, 0.08))
    inner_region = cv2.GaussianBlur(cv2.erode(pattern, np.ones((3, 3), np.uint8), iterations=int(baffle_band_width)), (0, 0), baffle_band_width * 0.5)
    baffle_darkening = random_generator.uniform(0.15, 0.4) * np.clip(pattern - inner_region, 0.0, 1.0)
    return (pattern - baffle_darkening) * (1.0 - hot_spot_falloff * np.clip(radius, 0.0, 1.0) ** 2)


def draw_point_glint_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """A small bright disc with a scatter halo, or a loose cluster of tiny dots (string lights)."""
    radius, _ = unit_plane_polar_coordinates(canvas_size)
    if random_generator.random() < 0.25:
        canvas = new_supersampled_canvas(canvas_size)
        for _ in range(random_generator.integers(3, 11)):
            dot_center = random_generator.uniform(-0.85, 0.85, size=2)
            dot_radius = random_generator.uniform(0.04, 0.09)
            fill_unit_ellipse(canvas, (dot_center[0], dot_center[1]), (dot_radius, dot_radius), random_generator.uniform(0.5, 1.0))
        return downsample_canvas(canvas, canvas_size)
    canvas = new_supersampled_canvas(canvas_size)
    core_radius = random_generator.uniform(0.5, 0.8)
    fill_unit_ellipse(canvas, (0.0, 0.0), (core_radius, core_radius * random_generator.uniform(0.7, 1.0)), 1.0, random_generator.uniform(0, 180))
    halo_width = random_generator.uniform(0.5, 0.9)
    halo = random_generator.uniform(0.05, 0.2) * np.exp(-0.5 * (radius / halo_width) ** 2)
    return np.maximum(downsample_canvas(canvas, canvas_size), halo)


def draw_strip_light_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """One or two long thin capsules (tube or LED strip)."""
    canvas = new_supersampled_canvas(canvas_size)
    strip_half_thickness = random_generator.uniform(0.012, 0.045)
    strip_offsets = [0.0] if random_generator.random() < 0.7 else [-random_generator.uniform(0.08, 0.25), random_generator.uniform(0.08, 0.25)]
    for strip_offset in strip_offsets:
        fill_unit_polygon(canvas, rounded_rectangle_polygon(1.0, strip_half_thickness, strip_half_thickness, center=(0.0, strip_offset)), 1.0)
    pattern = downsample_canvas(canvas, canvas_size)
    # A tube's diffuser glows around it, and brightness sags toward the ends.
    glow = cv2.GaussianBlur(pattern, (0, 0), canvas_size * random_generator.uniform(0.02, 0.06)) * random_generator.uniform(0.1, 0.3)
    column_position = np.linspace(-1.0, 1.0, canvas_size, dtype=np.float32)[None, :]
    end_sag = 1.0 - random_generator.uniform(0.0, 0.4) * column_position**4
    return (pattern + glow) * end_sag


def draw_ceiling_grid_canvas(random_generator: np.random.Generator, canvas_size: int) -> np.ndarray:
    """A lattice of 2-4 x 2-4 panel lights, shrinking toward the far rows (perspective)."""
    canvas = new_supersampled_canvas(canvas_size)
    column_count, row_count = random_generator.integers(2, 5), random_generator.integers(2, 5)
    panel_fill_fraction = random_generator.uniform(0.35, 0.7)
    perspective_shrink = random_generator.uniform(0.0, 0.5)
    for row_index in range(row_count):
        row_position = -1.0 + (2.0 * row_index + 1.0) / row_count
        row_scale = 1.0 - perspective_shrink * (row_index / max(row_count - 1, 1))
        for column_index in range(column_count):
            column_position = (-1.0 + (2.0 * column_index + 1.0) / column_count) * row_scale
            half_width = panel_fill_fraction / column_count * row_scale
            half_height = panel_fill_fraction / row_count * row_scale
            panel_brightness = random_generator.uniform(0.75, 1.0)
            fill_unit_polygon(canvas, rounded_rectangle_polygon(half_width, half_height, 0.02, center=(column_position, row_position)), panel_brightness)
    pattern = downsample_canvas(canvas, canvas_size)
    return pattern * (0.85 + 0.15 * smooth_random_field(random_generator, canvas_size, canvas_size, 4))
