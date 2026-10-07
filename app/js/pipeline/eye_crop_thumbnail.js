/**
 * A small picture of a face's eye crop for the face picker (ui/face_picker.js). Runs in the
 * worker, where the crop already exists; the page receives an ImageBitmap of
 * EYE_CROP_THUMBNAIL_WIDTH x half that (2x the tile's CSS width, for sharp phones).
 */

import { EYE_CROP_THUMBNAIL_WIDTH } from "../app_config.js";

/** Planar RGB float crop in [0, 1] -> ImageBitmap thumbnail with the crop's aspect. */
export async function makeEyeCropThumbnail(cropPlanarRgb, cropWidth, cropHeight) {
  const planeSize = cropWidth * cropHeight;
  const cropRgba = new Uint8ClampedArray(planeSize * 4);
  for (let index = 0; index < planeSize; index += 1) {
    cropRgba[index * 4] = Math.round(cropPlanarRgb[index] * 255);
    cropRgba[index * 4 + 1] = Math.round(cropPlanarRgb[planeSize + index] * 255);
    cropRgba[index * 4 + 2] = Math.round(cropPlanarRgb[2 * planeSize + index] * 255);
    cropRgba[index * 4 + 3] = 255;
  }
  const cropCanvas = new OffscreenCanvas(cropWidth, cropHeight);
  cropCanvas.getContext("2d").putImageData(new ImageData(cropRgba, cropWidth, cropHeight), 0, 0);
  const thumbnailHeight = Math.round((EYE_CROP_THUMBNAIL_WIDTH * cropHeight) / cropWidth);
  return createImageBitmap(cropCanvas, { resizeWidth: EYE_CROP_THUMBNAIL_WIDTH, resizeHeight: thumbnailHeight, resizeQuality: "high" });
}
