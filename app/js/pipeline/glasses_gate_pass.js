/**
 * The glasses gate: one face's eye crop goes to the glasses classifier first, and only a face
 * classified as wearing glasses goes on to the glare model. A face without glasses never
 * reaches the glare model, so it can never change a pixel.
 *
 * Pure apart from the injected callbacks (Node tests drive it with fakes):
 *   readPhotoRegion(rect) -> RGBA bytes; runGlareModel(...) as in glare_crop_pass.js;
 *   runGlassesClassifier(cropPlanarRgb, cropWidth, cropHeight) -> probability in [0, 1].
 *
 * - The 1x crop is computed ONCE and fed to both models (the classifier's `eye_crop` input is
 *   the same tensor as the glare model's `glare_crop`).
 * - A skipped face keeps its crop as three crop-size planes (`cropRgbPlanes`), so a later
 *   "force on" runs the glare model without re-detecting or re-warping (runSkippedFaceGlarePass).
 * - Face result fields added here: glassesProbability, glassesDetected (the automatic decision),
 *   glareRun (whether the glare model has seen this face), cropRgbPlanes (skipped faces only).
 */

import { extractFaceEyeCrop, runGlarePassOnFace } from "./glare_crop_pass.js";

/** The summary a face gets when the glare model has not run on it: nothing to blend. */
function buildSkippedFaceSummary(eyeCrop) {
  const planeSize = eyeCrop.cropWidth * eyeCrop.cropHeight;
  return {
    hasGlare: false,
    showLostDetailNote: false,
    glareFraction: 0,
    lostDetailFraction: 0,
    cropSizeMultiplier: 1,
    photoToEyeCropAffine: eyeCrop.photoToEyeCropAffine,
    cropWidth: eyeCrop.cropWidth,
    cropHeight: eyeCrop.cropHeight,
    region: null,
    cropDeltaLayers: null,
    cropGlareMask: null,
    // Copies, so the face result holds three crop-size planes and not the classifier's input buffer.
    cropRgbPlanes: [0, 1, 2].map((channel) => eyeCrop.cropPlanarRgb.slice(channel * planeSize, (channel + 1) * planeSize)),
  };
}

/** Three crop-size planes -> one planar RGB buffer (the model input layout). */
export function joinCropRgbPlanes(cropRgbPlanes) {
  const planeSize = cropRgbPlanes[0].length;
  const cropPlanarRgb = new Float32Array(planeSize * 3);
  cropRgbPlanes.forEach((plane, channel) => cropPlanarRgb.set(plane, channel * planeSize));
  return cropPlanarRgb;
}

/**
 * Classify one face and run the glare model only when it wears glasses.
 *
 * Returns { faceResult, eyeCrop }: faceResult is the glare_crop_pass.js summary plus the gate
 * fields above; eyeCrop is the 1x crop (the caller may make a thumbnail of it, then drop it).
 */
export async function runFaceThroughGlassesGate({ face, photoWidth, photoHeight, readPhotoRegion, runGlassesClassifier, runGlareModel, glassesProbabilityThreshold, highResPassEnabled = false }) {
  const eyeCrop = extractFaceEyeCrop({ face, photoWidth, photoHeight, readPhotoRegion });
  const glassesProbability = await runGlassesClassifier(eyeCrop.cropPlanarRgb, eyeCrop.cropWidth, eyeCrop.cropHeight);
  const glassesDetected = glassesProbability >= glassesProbabilityThreshold;
  const gateFields = { glassesProbability, glassesDetected };
  if (!glassesDetected) return { faceResult: { ...gateFields, glareRun: false, ...buildSkippedFaceSummary(eyeCrop) }, eyeCrop };
  const pass = await runGlarePassOnFace({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, highResPassEnabled, eyeCrop1x: eyeCrop });
  return { faceResult: { ...gateFields, glareRun: true, cropRgbPlanes: null, ...pass }, eyeCrop };
}

/**
 * "Force on" for a face the gate skipped: run the glare model on its kept crop.
 * `face` is the skipped face result (eye centers + cropRgbPlanes). Returns the new face result,
 * with glareRun true and the kept crop dropped. readPhotoRegion is only read by the 2x pass.
 */
export async function runSkippedFaceGlarePass({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, highResPassEnabled = false }) {
  const eyeCrop1x = { cropPlanarRgb: joinCropRgbPlanes(face.cropRgbPlanes), cropWidth: face.cropWidth, cropHeight: face.cropHeight, photoToEyeCropAffine: face.photoToEyeCropAffine };
  const pass = await runGlarePassOnFace({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, highResPassEnabled, eyeCrop1x });
  return { ...face, ...pass, glareRun: true, cropRgbPlanes: null };
}
