/**
 * Parity of the app's crop math with the Python reference (fixtures from generate_app_fixtures.py):
 *
 * - affine: app/js/eye_crop_geometry.js vs eye_crop_geometry.py, every entry to 1e-6.
 * - INTER_AREA downscale: app/js/pipeline/area_downscale.js vs cv2.resize, at most 1 level off.
 * - crop extraction: app/js/pipeline/eye_crop_warp.js vs extract_eye_crop (OpenCV's fixed-point
 *   bilinear), small mean difference, max a few levels.
 * - warp back: vs warp_eye_crop_layer_back_to_photo (float), tight tolerance.
 *
 * Run: node tests/app/test_eye_crop_geometry_parity.mjs
 */

import { computePhotoToEyeCropAffine, eyeCropScale, EYE_CROP_HEIGHT, EYE_CROP_WIDTH } from "../../app/js/eye_crop_geometry.js";
import { areaDownscaleRgba } from "../../app/js/pipeline/area_downscale.js";
import {
  computePhotoRegionCoveredByEyeCrop,
  computeSourceRegionForEyeCrop,
  extractEyeCropFromRegion,
  warpCropLayersBackToRegion,
} from "../../app/js/pipeline/eye_crop_warp.js";
import { bgrToRgba, check, decodeBase64Array, finish, readJsonFixture } from "./node_test_support.mjs";

const AFFINE_TOLERANCE = 1e-6;

function maxAffineDifference(firstAffine, secondAffine) {
  let worst = 0;
  for (let row = 0; row < 2; row += 1) for (let column = 0; column < 3; column += 1) worst = Math.max(worst, Math.abs(firstAffine[row][column] - secondAffine[row][column]));
  return worst;
}

// 1. Affine parity.
let worstAffineDifference = 0;
for (const affineCase of readJsonFixture("eye_crop_affine_cases.json")) {
  const affine1x = computePhotoToEyeCropAffine(affineCase.image_left_eye_xy, affineCase.image_right_eye_xy);
  const affine2x = computePhotoToEyeCropAffine(affineCase.image_left_eye_xy, affineCase.image_right_eye_xy, EYE_CROP_WIDTH * 2, EYE_CROP_HEIGHT * 2);
  worstAffineDifference = Math.max(
    worstAffineDifference,
    maxAffineDifference(affine1x, affineCase.affine_1x),
    maxAffineDifference(affine2x, affineCase.affine_2x),
    Math.abs(eyeCropScale(affine1x) - affineCase.eye_crop_scale_1x),
  );
}
check(`affine: ${readJsonFixture("eye_crop_affine_cases.json").length} eye pairs at 1x and 2x match Python to ${AFFINE_TOLERANCE}`, worstAffineDifference <= AFFINE_TOLERANCE, `worst ${worstAffineDifference.toExponential(2)}`);
let threwOnCoincidentEyes = false;
try {
  computePhotoToEyeCropAffine([5, 5], [5, 5]);
} catch {
  threwOnCoincidentEyes = true;
}
check("affine: coincident eyes raise, like the Python ValueError", threwOnCoincidentEyes);

// 2. INTER_AREA downscale parity.
for (const downscaleCase of readJsonFixture("area_downscale_cases.json")) {
  const sourceRgba = bgrToRgba(decodeBase64Array(downscaleCase.photo_bgr_base64, Uint8Array), downscaleCase.width, downscaleCase.height);
  const expectedRgba = bgrToRgba(decodeBase64Array(downscaleCase.resized_bgr_base64, Uint8Array), downscaleCase.resized_width, downscaleCase.resized_height);
  const resizedRgba = areaDownscaleRgba({
    sourceWidth: downscaleCase.width,
    sourceHeight: downscaleCase.height,
    destinationWidth: downscaleCase.resized_width,
    destinationHeight: downscaleCase.resized_height,
    inverseScale: downscaleCase.factor,
    readSourceRows: (firstRow, rowCount) => sourceRgba.subarray(firstRow * downscaleCase.width * 4, (firstRow + rowCount) * downscaleCase.width * 4),
  });
  let worst = 0;
  let mismatched = 0;
  for (let index = 0; index < expectedRgba.length; index += 1) {
    const difference = Math.abs(expectedRgba[index] - resizedRgba[index]);
    worst = Math.max(worst, difference);
    if (difference > 0) mismatched += 1;
  }
  check(`INTER_AREA ${downscaleCase.width}x${downscaleCase.height} x${downscaleCase.factor}: within 1 level of cv2`, worst <= 1, `max ${worst}, ${mismatched} of ${expectedRgba.length} values differ`);
}

// 3 + 4. Crop extraction and warp back.
const [warpFixture] = readJsonFixture("eye_crop_warp_cases.json");
const photoRgba = bgrToRgba(decodeBase64Array(warpFixture.photo_bgr_base64, Uint8Array), warpFixture.photo_width, warpFixture.photo_height);
for (const [caseIndex, warpCase] of warpFixture.cases.entries()) {
  const affine = computePhotoToEyeCropAffine(warpCase.image_left_eye_xy, warpCase.image_right_eye_xy, warpFixture.crop_width, warpFixture.crop_height);
  const sourceRegion = computeSourceRegionForEyeCrop(affine, warpFixture.crop_width, warpFixture.crop_height, warpFixture.photo_width, warpFixture.photo_height);
  const regionRgba = new Uint8ClampedArray(sourceRegion.width * sourceRegion.height * 4);
  for (let row = 0; row < sourceRegion.height; row += 1) {
    const sourceStart = ((sourceRegion.y + row) * warpFixture.photo_width + sourceRegion.x) * 4;
    regionRgba.set(photoRgba.subarray(sourceStart, sourceStart + sourceRegion.width * 4), row * sourceRegion.width * 4);
  }
  const { cropRgba } = extractEyeCropFromRegion({
    regionRgba,
    region: sourceRegion,
    photoWidth: warpFixture.photo_width,
    photoHeight: warpFixture.photo_height,
    photoToEyeCropAffine: affine,
    cropWidth: warpFixture.crop_width,
    cropHeight: warpFixture.crop_height,
  });
  const expectedCropRgba = bgrToRgba(decodeBase64Array(warpCase.crop_bgr_base64, Uint8Array), warpFixture.crop_width, warpFixture.crop_height);
  let worstCrop = 0;
  let sumCrop = 0;
  for (let index = 0; index < cropRgba.length; index += 1) {
    if (index % 4 === 3) continue;
    const difference = Math.abs(cropRgba[index] - expectedCropRgba[index]);
    worstCrop = Math.max(worstCrop, difference);
    sumCrop += difference;
  }
  const meanCrop = sumCrop / ((cropRgba.length / 4) * 3);
  check(`crop extraction case ${caseIndex}: matches cv2.warpAffine (mean <= 0.6, max <= 4 levels)`, meanCrop <= 0.6 && worstCrop <= 4, `mean ${meanCrop.toFixed(3)}, max ${worstCrop}`);

  const coveredRegion = computePhotoRegionCoveredByEyeCrop(affine, warpFixture.crop_width, warpFixture.crop_height, warpFixture.photo_width, warpFixture.photo_height);
  const cropLayer = decodeBase64Array(warpCase.crop_layer_float32_base64, Float32Array);
  const expectedWarped = decodeBase64Array(warpCase.warped_layer_float32_base64, Float32Array);
  const [warpedRegionLayer] = coveredRegion
    ? warpCropLayersBackToRegion({ cropLayers: [cropLayer], cropWidth: warpFixture.crop_width, cropHeight: warpFixture.crop_height, photoToEyeCropAffine: affine, region: coveredRegion })
    : [new Float32Array(0)];
  let worstWarp = 0;
  let nonZeroOutsideRegion = 0;
  for (let photoY = 0; photoY < warpFixture.photo_height; photoY += 1) {
    for (let photoX = 0; photoX < warpFixture.photo_width; photoX += 1) {
      const expected = expectedWarped[photoY * warpFixture.photo_width + photoX];
      const insideRegion = coveredRegion && photoX >= coveredRegion.x && photoY >= coveredRegion.y && photoX < coveredRegion.x + coveredRegion.width && photoY < coveredRegion.y + coveredRegion.height;
      const actual = insideRegion ? warpedRegionLayer[(photoY - coveredRegion.y) * coveredRegion.width + (photoX - coveredRegion.x)] : 0;
      if (!insideRegion && expected !== 0) nonZeroOutsideRegion += 1;
      worstWarp = Math.max(worstWarp, Math.abs(actual - expected));
    }
  }
  // OpenCV quantizes sample positions to 1/32 px, so a unit-variance noise layer differs by ~0.1.
  check(`warp back case ${caseIndex}: matches cv2 inverse warp (max |diff| <= 0.15 on unit noise)`, worstWarp <= 0.15, `max ${worstWarp.toFixed(4)}`);
  check(`warp back case ${caseIndex}: covered region contains every non-zero cv2 pixel`, nonZeroOutsideRegion === 0, `${nonZeroOutsideRegion} outside`);
}

finish();
