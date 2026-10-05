/**
 * One face through the glare model: crop, run, turn the output into a photo-space delta.
 *
 * Pure apart from the two injected callbacks, so Node tests drive it with a fake model:
 *   readPhotoRegion(rect) -> RGBA bytes of that photo rect (the worker reads its bitmap);
 *   runGlareModel(cropPlanarRgb, cropWidth, cropHeight) -> { cleanPlanarRgb, masksPlanar }.
 *
 * Contract (CLAUDE.md "Model I/O"): delta = glare_mask * (clean_crop - glare_crop) in crop
 * space, warped back through the crop affine and added to the ORIGINAL photo pixels.
 * - Crop-space glare mask values below GLARE_MASK_ZERO_BELOW are forced to 0, so "outside
 *   the mask" is exact and the untouched photo stays bit-identical (see face_patch_blend.js).
 * - When the crop is a big downscale of the photo (crop scale < HIGH_RES_SCALE_THRESHOLD),
 *   an optional second pass at 2x crop size can replace the first; off by default
 *   (HIGH_RES_PASS_ENABLED in app_config.js) until the model is trained at that size.
 */

import { computePhotoToEyeCropAffine, eyeCropScale, EYE_CROP_HEIGHT, EYE_CROP_WIDTH } from "../eye_crop_geometry.js";
import { computePhotoRegionCoveredByEyeCrop, computeSourceRegionForEyeCrop, extractEyeCropFromRegion, warpCropLayersBackToRegion } from "./eye_crop_warp.js";

// Half a level of 8-bit: anything below cannot move a pixel, so it is treated as no glare.
export const GLARE_MASK_ZERO_BELOW = 0.5 / 255;
const MASK_ON_THRESHOLD = 0.5;
// A face counts as having glare when this share of the 512x256 crop is masked (~60 px).
const GLARE_PRESENT_MIN_FRACTION = 0.0005;
// Show the "reconstructed, not recovered" note above this share of the crop (~260 px).
export const LOST_DETAIL_NOTE_MIN_FRACTION = 0.002;
export const HIGH_RES_SCALE_THRESHOLD = 0.5;

/** 1 for the default 512x256 pass, 2 for the high-res pass when enabled and the face is large. */
export function chooseCropSizeMultiplier(photoToEyeCropAffine1x, highResPassEnabled) {
  return highResPassEnabled && eyeCropScale(photoToEyeCropAffine1x) < HIGH_RES_SCALE_THRESHOLD ? 2 : 1;
}

/** Glare-model outputs -> crop-space delta layers, cleaned glare mask, lost-detail mask, counts. */
export function computeCropSpaceDelta({ cropPlanarRgb, cleanPlanarRgb, masksPlanar, cropWidth, cropHeight }) {
  const planeSize = cropWidth * cropHeight;
  const deltaLayers = [new Float32Array(planeSize), new Float32Array(planeSize), new Float32Array(planeSize)];
  const glareMask = new Float32Array(planeSize);
  const lostDetailMask = new Float32Array(planeSize);
  let glarePixelCount = 0;
  let lostDetailPixelCount = 0;
  for (let index = 0; index < planeSize; index += 1) {
    const rawGlare = masksPlanar[index];
    const glare = rawGlare >= GLARE_MASK_ZERO_BELOW ? Math.min(rawGlare, 1) : 0;
    const lost = Math.max(0, Math.min(masksPlanar[planeSize + index], 1)) * (glare > 0 ? 1 : 0);
    glareMask[index] = glare;
    lostDetailMask[index] = lost;
    if (glare >= MASK_ON_THRESHOLD) glarePixelCount += 1;
    if (lost >= MASK_ON_THRESHOLD) lostDetailPixelCount += 1;
    if (glare === 0) continue;
    for (let channel = 0; channel < 3; channel += 1) {
      const planeIndex = channel * planeSize + index;
      deltaLayers[channel][index] = glare * (cleanPlanarRgb[planeIndex] - cropPlanarRgb[planeIndex]);
    }
  }
  return { deltaLayers, glareMask, lostDetailMask, glarePixelCount, lostDetailPixelCount };
}

async function runOnePass({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, cropSizeMultiplier }) {
  const cropWidth = EYE_CROP_WIDTH * cropSizeMultiplier;
  const cropHeight = EYE_CROP_HEIGHT * cropSizeMultiplier;
  const photoToEyeCropAffine = computePhotoToEyeCropAffine(face.imageLeftEyeXY, face.imageRightEyeXY, cropWidth, cropHeight);
  const sourceRegion = computeSourceRegionForEyeCrop(photoToEyeCropAffine, cropWidth, cropHeight, photoWidth, photoHeight);
  const { cropRgba, cropPlanarRgb } = extractEyeCropFromRegion({
    regionRgba: readPhotoRegion(sourceRegion),
    region: sourceRegion,
    photoWidth,
    photoHeight,
    photoToEyeCropAffine,
    cropWidth,
    cropHeight,
  });
  const { cleanPlanarRgb, masksPlanar } = await runGlareModel(cropPlanarRgb, cropWidth, cropHeight);
  const cropDelta = computeCropSpaceDelta({ cropPlanarRgb, cleanPlanarRgb, masksPlanar, cropWidth, cropHeight });
  return { cropWidth, cropHeight, cropSizeMultiplier, photoToEyeCropAffine, cropRgba, ...cropDelta };
}

/**
 * Run the glare model on one detected face and return everything the page needs.
 *
 * Returns:
 *   { hasGlare, showLostDetailNote, glareFraction, lostDetailFraction, cropSizeMultiplier,
 *     photoToEyeCropAffine, region, regionOriginalRgba, regionDeltaLayers: [R, G, B] (0..1 units),
 *     regionGlareMask, regionLostDetailMask }   (region* are null when the face has no glare)
 */
export async function runGlarePassOnFace({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, highResPassEnabled = false }) {
  const affine1x = computePhotoToEyeCropAffine(face.imageLeftEyeXY, face.imageRightEyeXY);
  let pass = await runOnePass({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, cropSizeMultiplier: 1 });
  const passArea = () => pass.cropWidth * pass.cropHeight;
  const hasGlareAfterFirstPass = pass.glarePixelCount / passArea() >= GLARE_PRESENT_MIN_FRACTION;
  if (hasGlareAfterFirstPass && chooseCropSizeMultiplier(affine1x, highResPassEnabled) === 2) {
    pass = await runOnePass({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, cropSizeMultiplier: 2 });
  }
  const glareFraction = pass.glarePixelCount / passArea();
  const lostDetailFraction = pass.lostDetailPixelCount / passArea();
  const summary = {
    hasGlare: glareFraction >= GLARE_PRESENT_MIN_FRACTION,
    showLostDetailNote: lostDetailFraction >= LOST_DETAIL_NOTE_MIN_FRACTION,
    glareFraction,
    lostDetailFraction,
    cropSizeMultiplier: pass.cropSizeMultiplier,
    photoToEyeCropAffine: pass.photoToEyeCropAffine,
    cropRgba: pass.cropRgba,
    cropWidth: pass.cropWidth,
    cropHeight: pass.cropHeight,
    region: null,
    regionOriginalRgba: null,
    regionDeltaLayers: null,
    regionGlareMask: null,
    regionLostDetailMask: null,
  };
  if (!summary.hasGlare) return summary;
  const region = computePhotoRegionCoveredByEyeCrop(pass.photoToEyeCropAffine, pass.cropWidth, pass.cropHeight, photoWidth, photoHeight);
  if (!region) return { ...summary, hasGlare: false };
  const [deltaRed, deltaGreen, deltaBlue, regionGlareMask, regionLostDetailMask] = warpCropLayersBackToRegion({
    cropLayers: [...pass.deltaLayers, pass.glareMask, pass.lostDetailMask],
    cropWidth: pass.cropWidth,
    cropHeight: pass.cropHeight,
    photoToEyeCropAffine: pass.photoToEyeCropAffine,
    region,
  });
  return {
    ...summary,
    region,
    regionOriginalRgba: readPhotoRegion(region),
    regionDeltaLayers: [deltaRed, deltaGreen, deltaBlue],
    regionGlareMask,
    regionLostDetailMask,
  };
}
