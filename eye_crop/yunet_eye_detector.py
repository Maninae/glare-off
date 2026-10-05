"""YuNet face detection, reduced to what the eye crop needs: two eye centers per face.

YuNet (OpenCV Zoo, MIT, 233 KB) is also the detector the browser app runs, so crops built here
match what the model sees at inference time. The ONNX file lives in `app/models/`.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

YUNET_ONNX_PATH = Path(__file__).resolve().parent.parent / "app" / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_SCORE_THRESHOLD = 0.6
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 50
# YuNet is trained for faces up to roughly this size; larger photos are downscaled before detection.
YUNET_MAX_DETECTION_SIDE = 1024


@dataclass
class DetectedFaceEyes:
    """One detected face: eye centers in photo pixels, ordered by image position."""

    image_left_eye_xy: np.ndarray
    image_right_eye_xy: np.ndarray
    face_box_xywh: np.ndarray
    detection_score: float

    @property
    def eye_distance_pixels(self) -> float:
        """Distance between the two eye centers in photo pixels."""
        return float(np.linalg.norm(self.image_right_eye_xy - self.image_left_eye_xy))


def detect_face_eyes_in_photo(photo_bgr: np.ndarray, yunet_onnx_path: Path = YUNET_ONNX_PATH) -> list[DetectedFaceEyes]:
    """Detect faces in a BGR photo and return their eye centers, highest score first.

    - Large photos are downscaled to YUNET_MAX_DETECTION_SIDE for detection; coordinates are
      returned in the ORIGINAL photo's pixel grid.
    """
    photo_height, photo_width = photo_bgr.shape[:2]
    detection_scale = min(1.0, YUNET_MAX_DETECTION_SIDE / max(photo_height, photo_width))
    detection_image = photo_bgr
    if detection_scale < 1.0:
        detection_image = cv2.resize(photo_bgr, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    detection_height, detection_width = detection_image.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(yunet_onnx_path), "", (detection_width, detection_height), YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K
    )
    _, detections = detector.detect(detection_image)
    if detections is None:
        return []
    detected_faces = []
    for detection in detections:
        # YuNet row: box xywh, then 5 landmarks (subject's right eye, left eye, nose, mouth corners), score.
        first_eye = detection[4:6] / detection_scale
        second_eye = detection[6:8] / detection_scale
        image_left_eye, image_right_eye = sorted((first_eye, second_eye), key=lambda eye_xy: eye_xy[0])
        detected_faces.append(
            DetectedFaceEyes(
                image_left_eye_xy=np.asarray(image_left_eye, dtype=np.float64),
                image_right_eye_xy=np.asarray(image_right_eye, dtype=np.float64),
                face_box_xywh=np.asarray(detection[0:4] / detection_scale, dtype=np.float64),
                detection_score=float(detection[14]),
            )
        )
    detected_faces.sort(key=lambda face: face.detection_score, reverse=True)
    return detected_faces
