/**
 * Eye-crop geometry: the similarity transform between a photo and the aligned glasses crop.
 *
 * Line-for-line port of `eye_crop/eye_crop_geometry.py`, which is THE definition of the crop
 * the glare model sees. Change the Python first, then this file, then re-run
 * `tests/app/test_eye_crop_geometry_parity.mjs` (JS affine must equal Python to 1e-6).
 *
 * - The crop is EYE_CROP_WIDTH x EYE_CROP_HEIGHT, eyes level, eye midpoint at the crop center.
 * - The distance between the eye centers spans EYE_DISTANCE_FRACTION of the crop width.
 * - Affines are `[[a, b, c], [d, e, f]]` (row-major 2x3, like OpenCV) mapping photo -> crop.
 * - The pixel warps themselves live in `pipeline/eye_crop_warp.js`.
 */

export const EYE_CROP_WIDTH = 512;
export const EYE_CROP_HEIGHT = 256;
// Eye-to-eye distance as a fraction of crop width; 0.36 leaves room for wide frames and temples.
export const EYE_DISTANCE_FRACTION = 0.36;

/**
 * Port of `cv2.getRotationMatrix2D(center, angleDegrees, scale)`.
 * Positive angles rotate counter-clockwise on screen (y points down), as in OpenCV.
 * OpenCV takes the center as Point2f, so it is rounded to float32 (visible at 1e-6 on big photos).
 */
export function getRotationMatrix2D(centerXDouble, centerYDouble, angleDegrees, scale) {
  const centerX = Math.fround(centerXDouble);
  const centerY = Math.fround(centerYDouble);
  const angleRadians = angleDegrees * (Math.PI / 180);
  const alpha = Math.cos(angleRadians) * scale;
  const beta = Math.sin(angleRadians) * scale;
  return [
    [alpha, beta, (1 - alpha) * centerX - beta * centerY],
    [-beta, alpha, beta * centerX + (1 - alpha) * centerY],
  ];
}

/**
 * Return the 2x3 affine mapping photo pixel coordinates to eye-crop pixel coordinates.
 *
 * Args:
 *   imageLeftEyeXY: [x, y] of the eye that appears on the LEFT side of the image.
 *   imageRightEyeXY: [x, y] of the eye that appears on the RIGHT side of the image.
 *   cropWidth, cropHeight: output crop size; pass 2x values for a high-resolution pass.
 * Returns:
 *   [[a, b, c], [d, e, f]]: rotation + uniform scale + translation.
 *
 * - Eye order is by image position, not anatomy, so the crop is never flipped upside down.
 */
export function computePhotoToEyeCropAffine(imageLeftEyeXY, imageRightEyeXY, cropWidth = EYE_CROP_WIDTH, cropHeight = EYE_CROP_HEIGHT) {
  const leftEye = [Number(imageLeftEyeXY[0]), Number(imageLeftEyeXY[1])];
  const rightEye = [Number(imageRightEyeXY[0]), Number(imageRightEyeXY[1])];
  const eyeVector = [rightEye[0] - leftEye[0], rightEye[1] - leftEye[1]];
  const eyeDistance = Math.hypot(eyeVector[0], eyeVector[1]);
  if (eyeDistance < 1e-6) {
    throw new Error("eye centers coincide; cannot build an eye crop");
  }
  const rollAngleDegrees = Math.atan2(eyeVector[1], eyeVector[0]) * (180 / Math.PI);
  const scale = (cropWidth * EYE_DISTANCE_FRACTION) / eyeDistance;
  const eyeMidpoint = [(leftEye[0] + rightEye[0]) / 2.0, (leftEye[1] + rightEye[1]) / 2.0];
  const affine = getRotationMatrix2D(eyeMidpoint[0], eyeMidpoint[1], rollAngleDegrees, scale);
  affine[0][2] += cropWidth / 2.0 - eyeMidpoint[0];
  affine[1][2] += cropHeight / 2.0 - eyeMidpoint[1];
  return affine;
}

/** Return crop pixels per photo pixel (below 1 means the crop is a downscale of the photo). */
export function eyeCropScale(photoToEyeCropAffine) {
  return Math.hypot(photoToEyeCropAffine[0][0], photoToEyeCropAffine[0][1]);
}

/** Invert a 2x3 affine (port of `cv2.invertAffineTransform`). */
export function invertAffineTransform(affine) {
  const [[a, b, c], [d, e, f]] = affine;
  const determinant = a * e - b * d;
  const inverseDeterminant = determinant !== 0 ? 1.0 / determinant : 0.0;
  const inverseA = e * inverseDeterminant;
  const inverseB = -b * inverseDeterminant;
  const inverseD = -d * inverseDeterminant;
  const inverseE = a * inverseDeterminant;
  return [
    [inverseA, inverseB, -inverseA * c - inverseB * f],
    [inverseD, inverseE, -inverseD * c - inverseE * f],
  ];
}

/** Apply a 2x3 affine to one point. */
export function applyAffineToPoint(affine, x, y) {
  return [affine[0][0] * x + affine[0][1] * y + affine[0][2], affine[1][0] * x + affine[1][1] * y + affine[1][2]];
}
