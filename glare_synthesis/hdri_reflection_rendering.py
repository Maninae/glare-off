"""Image-based lens reflection: what a spherical spectacle front surface mirrors from an HDRI.

Camera frame: x right, y down, z from the camera toward the face. Per lens pixel:
1. The lens point p (metres, relative to the lens center) lies on a sphere of radius R facing the
   camera, so its outward normal is normalize(p_x / R, p_y / R, -1), then tilted by the lens's wrap,
   pantoscopic tilt and the head pose.
2. The camera ray v runs from the camera (distance D) to the lens point; the mirror ray is
   r = v - 2 (v . n) n. Both lenses use the same sphere and environment, so the reflected scene has
   the same handedness in both; their different positions and tilts give parallax.
3. r goes to the environment frame (y up), is rotated by the scene's environment yaw/pitch, and is
   looked up in the equirect map. Minification and barrel distortion come out of the sphere.
"""

import cv2
import numpy as np

from glare_synthesis.glare_scene import HdriReflection
from glare_synthesis.hdri_environment_library import HdriEnvironment


def rotate_about_vertical_axis(x: np.ndarray, z: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (x, z) about the y axis by `angle` radians."""
    cosine, sine = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    return x * cosine + z * sine, -x * sine + z * cosine


def rotate_about_horizontal_axis(y: np.ndarray, z: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (y, z) about the x axis by `angle` radians (positive tips a -z normal toward +y)."""
    cosine, sine = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    return y * cosine - z * sine, y * sine + z * cosine


def mirror_reflected_directions(lens_point_x_m: np.ndarray, lens_point_y_m: np.ndarray, lens_center_offset_m: np.ndarray,
                                surface_radius_m: float, normal_tilt: np.ndarray, camera_distance_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit mirror directions (camera frame) for lens points on a sphere of the given radius and tilt."""
    inverse_radius = np.float32(1.0 / surface_radius_m)
    normal_x, normal_y = lens_point_x_m * inverse_radius, lens_point_y_m * inverse_radius
    normal_z = np.full_like(normal_x, -1.0)
    normal_x, normal_z = rotate_about_vertical_axis(normal_x, normal_z, float(normal_tilt[0]))
    normal_y, normal_z = rotate_about_horizontal_axis(normal_y, normal_z, float(normal_tilt[1]))
    normal_length = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal_x, normal_y, normal_z = normal_x / normal_length, normal_y / normal_length, normal_z / normal_length
    view_x = lens_point_x_m + np.float32(lens_center_offset_m[0])
    view_y = lens_point_y_m + np.float32(lens_center_offset_m[1])
    view_z = np.float32(camera_distance_m)
    view_length = np.sqrt(view_x * view_x + view_y * view_y + view_z * view_z)
    view_x, view_y, view_z = view_x / view_length, view_y / view_length, view_z / view_length
    twice_projection = 2.0 * (view_x * normal_x + view_y * normal_y + view_z * normal_z)
    return view_x - twice_projection * normal_x, view_y - twice_projection * normal_y, view_z - twice_projection * normal_z


def camera_to_environment_directions(reflected_x: np.ndarray, reflected_y: np.ndarray, reflected_z: np.ndarray,
                                     environment_yaw: float, environment_pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame directions -> environment frame (y up), then the scene's pitch and yaw."""
    environment_y, environment_z = rotate_about_horizontal_axis(-reflected_y, reflected_z, environment_pitch)
    environment_x, environment_z = rotate_about_vertical_axis(reflected_x, environment_z, environment_yaw)
    return environment_x, environment_y, environment_z


def environment_directions_to_lon_lat(environment_x: np.ndarray, environment_y: np.ndarray, environment_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longitude in [-pi, pi] and latitude in [-pi/2, pi/2] of environment-frame directions."""
    return np.arctan2(environment_x, environment_z), np.arcsin(np.clip(environment_y, -1.0, 1.0))


def solve_environment_rotation_for_target(target_direction_camera: np.ndarray, emitter_lon_lat: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) that rotate a camera-frame mirror direction onto the emitter's environment direction.

    Pitch first sets the latitude exactly (Y cos t + Z sin t... solved in closed form); yaw then matches
    longitude. Pitch is limited to +-70 degrees so the reflected horizon never flips over.
    """
    start_y, start_z = -float(target_direction_camera[1]), float(target_direction_camera[2])
    amplitude = float(np.hypot(start_y, start_z))
    target_sin_latitude = float(np.sin(emitter_lon_lat[1]))
    phase = float(np.arctan2(start_z, start_y))
    # After pitch t: Y' = Y cos t - Z sin t = amplitude * cos(t + phase).
    cosine_argument = float(np.clip(target_sin_latitude / max(amplitude, 1e-6), -1.0, 1.0))
    pitch_candidates = [np.arccos(cosine_argument) - phase, -np.arccos(cosine_argument) - phase]
    pitch_candidates = [float((candidate + np.pi) % (2 * np.pi) - np.pi) for candidate in pitch_candidates]
    pitch = float(np.clip(min(pitch_candidates, key=abs), -np.radians(70.0), np.radians(70.0)))
    pitched_y, pitched_z = rotate_about_horizontal_axis(np.float32(start_y), np.float32(start_z), pitch)
    start_longitude = float(np.arctan2(float(target_direction_camera[0]), float(pitched_z)))
    # Yaw about y adds to longitude: atan2(x cos + z sin, -x sin + z cos) = lon + yaw.
    yaw = float(emitter_lon_lat[0]) - start_longitude
    return yaw, pitch


def lens_points_in_metres(lens_x: np.ndarray, lens_y: np.ndarray, lens_width_m: float, horizontal_foreshortening: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized lens coordinates (half nominal width units) -> physical lens-surface offsets in metres."""
    metres_per_unit = np.float32(lens_width_m / 2.0)
    return lens_x * metres_per_unit / np.float32(horizontal_foreshortening), lens_y * metres_per_unit


def sample_environment(environment: HdriEnvironment, environment_x: np.ndarray, environment_y: np.ndarray, environment_z: np.ndarray) -> np.ndarray:
    """Bilinear equirect lookup with horizontal wrap; returns (H, W, 3) mean-normalized radiance."""
    longitude, latitude = environment_directions_to_lon_lat(environment_x, environment_y, environment_z)
    map_height, map_width = environment.environment_rgb.shape[:2]
    column = (longitude + np.float32(np.pi)) * np.float32(map_width / (2.0 * np.pi)) - np.float32(0.5)
    row = (np.float32(np.pi / 2.0) - latitude) * np.float32(map_height / np.pi) - np.float32(0.5)
    return cv2.remap(environment.environment_rgb, column.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def render_hdri_reflection_radiance(hdri_reflection: HdriReflection, environment: HdriEnvironment, lens_index: int, lens_x: np.ndarray,
                                    lens_y: np.ndarray, lens_center_offset_m: np.ndarray, horizontal_foreshortening: float) -> np.ndarray:
    """Mirror radiance (front surface + optional back-surface ghost) for one lens region, mean-normalized units."""
    point_x_m, point_y_m = lens_points_in_metres(lens_x, lens_y, hdri_reflection.lens_width_m, horizontal_foreshortening)
    normal_tilt = hdri_reflection.lens_normal_tilts[lens_index]
    reflected = mirror_reflected_directions(point_x_m, point_y_m, lens_center_offset_m, hdri_reflection.front_surface_radius_m,
                                            normal_tilt, hdri_reflection.camera_distance_m)
    radiance = sample_environment(environment, *camera_to_environment_directions(*reflected, hdri_reflection.environment_yaw, hdri_reflection.environment_pitch))
    if hdri_reflection.ghost_surface_radius_m is not None and hdri_reflection.ghost_intensity > 0:
        ghost_reflected = mirror_reflected_directions(point_x_m, point_y_m, lens_center_offset_m, hdri_reflection.ghost_surface_radius_m,
                                                      normal_tilt + hdri_reflection.ghost_normal_tilt, hdri_reflection.camera_distance_m)
        ghost_radiance = sample_environment(environment, *camera_to_environment_directions(*ghost_reflected, hdri_reflection.environment_yaw,
                                                                                           hdri_reflection.environment_pitch))
        radiance = radiance + np.float32(hdri_reflection.ghost_intensity) * ghost_radiance
    return radiance
