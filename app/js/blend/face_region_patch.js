/**
 * Crop-space face result -> full-resolution photo-region layers and patches, on demand.
 *
 * A face result (glare_crop_pass.js) keeps only crop-space layers, so its memory does not grow
 * with the photo. Whoever needs photo-region pixels (the visible card's cache, the export in the
 * worker, the test hook) warps here and drops the result when done.
 *
 * `face` needs: { region, photoToEyeCropAffine, cropWidth, cropHeight, cropDeltaLayers, cropGlareMask }.
 */

import { warpCropLayersBackToRegion } from "../pipeline/eye_crop_warp.js";
import { blendFacePatch } from "./face_patch_blend.js";

function warpFaceCropLayersToRegion(face, cropLayers) {
  return warpCropLayersBackToRegion({
    cropLayers,
    cropWidth: face.cropWidth,
    cropHeight: face.cropHeight,
    photoToEyeCropAffine: face.photoToEyeCropAffine,
    region: face.region,
  });
}

/** [R, G, B] Float32 delta layers over `face.region` (exactly 0 wherever the crop delta is 0). */
export function warpFaceDeltaToRegion(face) {
  return warpFaceCropLayersToRegion(face, face.cropDeltaLayers);
}

/** Float32 glare mask over `face.region`; > 0 marks the only pixels allowed to change. */
export function warpFaceGlareMaskToRegion(face) {
  return warpFaceCropLayersToRegion(face, [face.cropGlareMask])[0];
}

/** Patched RGBA bytes of `face.region` at `strength`, from the region's original bytes. */
export function buildFullResolutionFacePatch(face, regionOriginalRgba, strength) {
  return blendFacePatch(regionOriginalRgba, warpFaceDeltaToRegion(face), strength);
}
