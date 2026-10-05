/**
 * Turn a processed photo (worker "result" message) into things to show:
 * - `buildFacePatches(faces, entriesByFace, enabledByFace, strength)`: full-resolution patched
 *   regions from the visible photo's region cache (photo/visible_photo_regions.js). The download
 *   builds the same bytes in the worker (blend/face_region_patch.js).
 * - `composeAfterPreview(...)`: the preview with those patches drawn in at preview scale.
 * - `buildRegionCanvas(...)`: a full-resolution region as a canvas, for the eye zoom view.
 */

import { blendFacePatch } from "../blend/face_patch_blend.js";

/** [{ faceIndex, region, patchedRgba }] for faces that have glare and are switched on. */
export function buildFacePatches(faces, entriesByFace, enabledByFace, strength) {
  const patches = [];
  faces.forEach((face, faceIndex) => {
    if (!face.hasGlare || !enabledByFace[faceIndex]) return;
    const { regionOriginalRgba, regionDeltaLayers } = entriesByFace[faceIndex];
    patches.push({ faceIndex, region: face.region, patchedRgba: blendFacePatch(regionOriginalRgba, regionDeltaLayers, strength) });
  });
  return patches;
}

export function buildRegionCanvas(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas;
}

/** Draw the preview, then each patch scaled into place. Returns the canvas. */
export function composeAfterPreview(targetCanvas, previewBitmap, photoWidth, patches) {
  const previewScale = previewBitmap.width / photoWidth;
  if (targetCanvas.width !== previewBitmap.width || targetCanvas.height !== previewBitmap.height) {
    targetCanvas.width = previewBitmap.width;
    targetCanvas.height = previewBitmap.height;
  }
  const context = targetCanvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(previewBitmap, 0, 0);
  for (const { region, patchedRgba } of patches) {
    const patchCanvas = buildRegionCanvas(patchedRgba, region.width, region.height);
    context.drawImage(patchCanvas, region.x * previewScale, region.y * previewScale, region.width * previewScale, region.height * previewScale);
  }
  return targetCanvas;
}
