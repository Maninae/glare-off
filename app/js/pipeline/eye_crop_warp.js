/**
 * Pixel warps between the photo and the eye crop, ports of the two cv2.warpAffine calls in
 * `eye_crop/eye_crop_geometry.py`:
 *
 * - `extractEyeCropFromRegion`: photo -> crop, bilinear, BORDER_REFLECT_101, rounded to uint8
 *   (cv2.warpAffine treats the Python's INTER_AREA flag as INTER_LINEAR, so bilinear IS the
 *   training-time behavior; OpenCV's fixed-point rounding makes us differ by about 1 level).
 * - `warpCropLayersBackToRegion`: crop-space float layers -> photo pixels, bilinear,
 *   WARP_INVERSE_MAP, BORDER_CONSTANT 0. Pixels the crop does not reach come back exactly 0,
 *   which is what keeps the rest of the photo bit-identical after blending.
 *
 * Everything works on a sub-rectangle ("region") of the photo, so a 50 MP photo is only ever
 * read around the eyes. Rects are `{ x, y, width, height }` in integer photo pixels.
 */

import { applyAffineToPoint, invertAffineTransform } from "../eye_crop_geometry.js";

/** BORDER_REFLECT_101 index: ...2 1 | 0 1 2 ... n-2 n-1 | n-2 n-3... */
export function reflect101Index(index, size) {
  if (size === 1) return 0;
  let reflected = index;
  while (reflected < 0 || reflected >= size) {
    reflected = reflected < 0 ? -reflected : 2 * (size - 1) - reflected;
  }
  return reflected;
}

function boundingBoxOfMappedRectangle(affine, left, top, right, bottom) {
  const corners = [
    applyAffineToPoint(affine, left, top),
    applyAffineToPoint(affine, right, top),
    applyAffineToPoint(affine, left, bottom),
    applyAffineToPoint(affine, right, bottom),
  ];
  return {
    minX: Math.min(...corners.map((corner) => corner[0])),
    maxX: Math.max(...corners.map((corner) => corner[0])),
    minY: Math.min(...corners.map((corner) => corner[1])),
    maxY: Math.max(...corners.map((corner) => corner[1])),
  };
}

function reflectedIndexBounds(firstIndex, lastIndex, size) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let index = firstIndex; index <= lastIndex; index += 1) {
    const reflected = reflect101Index(index, size);
    minimum = Math.min(minimum, reflected);
    maximum = Math.max(maximum, reflected);
  }
  return [minimum, maximum];
}

/** Photo rect holding every pixel the photo -> crop warp samples (reflection included). */
export function computeSourceRegionForEyeCrop(photoToEyeCropAffine, cropWidth, cropHeight, photoWidth, photoHeight) {
  const cropToPhotoAffine = invertAffineTransform(photoToEyeCropAffine);
  const box = boundingBoxOfMappedRectangle(cropToPhotoAffine, 0, 0, cropWidth - 1, cropHeight - 1);
  const [minX, maxX] = reflectedIndexBounds(Math.floor(box.minX), Math.floor(box.maxX) + 1, photoWidth);
  const [minY, maxY] = reflectedIndexBounds(Math.floor(box.minY), Math.floor(box.maxY) + 1, photoHeight);
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

/** Photo rect of every pixel whose bilinear crop sample can be non-zero; null if none. */
export function computePhotoRegionCoveredByEyeCrop(photoToEyeCropAffine, cropWidth, cropHeight, photoWidth, photoHeight) {
  const cropToPhotoAffine = invertAffineTransform(photoToEyeCropAffine);
  const box = boundingBoxOfMappedRectangle(cropToPhotoAffine, -1, -1, cropWidth, cropHeight);
  const left = Math.max(0, Math.floor(box.minX));
  const top = Math.max(0, Math.floor(box.minY));
  const right = Math.min(photoWidth - 1, Math.ceil(box.maxX));
  const bottom = Math.min(photoHeight - 1, Math.ceil(box.maxY));
  if (right < left || bottom < top) return null;
  return { x: left, y: top, width: right - left + 1, height: bottom - top + 1 };
}

/**
 * Photo -> eye crop.
 *
 * Args:
 *   regionRgba: RGBA bytes of `region` (from computeSourceRegionForEyeCrop).
 * Returns:
 *   { cropRgba: Uint8ClampedArray, cropPlanarRgb: Float32Array [3][H][W] in [0, 1] }.
 */
export function extractEyeCropFromRegion({ regionRgba, region, photoWidth, photoHeight, photoToEyeCropAffine, cropWidth, cropHeight }) {
  const cropToPhotoAffine = invertAffineTransform(photoToEyeCropAffine);
  const [[a, b, c], [d, e, f]] = cropToPhotoAffine;
  const planeSize = cropWidth * cropHeight;
  const cropRgba = new Uint8ClampedArray(planeSize * 4);
  const cropPlanarRgb = new Float32Array(planeSize * 3);
  const sampleOffset = (photoX, photoY) => {
    const regionX = reflect101Index(photoX, photoWidth) - region.x;
    const regionY = reflect101Index(photoY, photoHeight) - region.y;
    return (regionY * region.width + regionX) * 4;
  };
  for (let cropY = 0; cropY < cropHeight; cropY += 1) {
    for (let cropX = 0; cropX < cropWidth; cropX += 1) {
      const sourceX = a * cropX + b * cropY + c;
      const sourceY = d * cropX + e * cropY + f;
      const x0 = Math.floor(sourceX);
      const y0 = Math.floor(sourceY);
      const fractionX = sourceX - x0;
      const fractionY = sourceY - y0;
      const topLeft = sampleOffset(x0, y0);
      const topRight = sampleOffset(x0 + 1, y0);
      const bottomLeft = sampleOffset(x0, y0 + 1);
      const bottomRight = sampleOffset(x0 + 1, y0 + 1);
      const pixelIndex = cropY * cropWidth + cropX;
      for (let channel = 0; channel < 3; channel += 1) {
        const top = regionRgba[topLeft + channel] + (regionRgba[topRight + channel] - regionRgba[topLeft + channel]) * fractionX;
        const bottom = regionRgba[bottomLeft + channel] + (regionRgba[bottomRight + channel] - regionRgba[bottomLeft + channel]) * fractionX;
        const value = Math.round(top + (bottom - top) * fractionY);
        cropRgba[pixelIndex * 4 + channel] = value;
        cropPlanarRgb[channel * planeSize + pixelIndex] = value / 255;
      }
      cropRgba[pixelIndex * 4 + 3] = 255;
    }
  }
  return { cropRgba, cropPlanarRgb };
}

/**
 * Crop-space float layers -> photo region (cv2.warpAffine with WARP_INVERSE_MAP, zero border).
 *
 * Args:
 *   cropLayers: Float32Arrays of cropWidth*cropHeight each (e.g. delta R, G, B, glare mask).
 *   region: photo rect to fill (from computePhotoRegionCoveredByEyeCrop).
 * Returns:
 *   Float32Arrays of region.width*region.height, one per input layer.
 */
export function warpCropLayersBackToRegion({ cropLayers, cropWidth, cropHeight, photoToEyeCropAffine, region }) {
  const [[a, b, c], [d, e, f]] = photoToEyeCropAffine;
  const regionLayers = cropLayers.map(() => new Float32Array(region.width * region.height));
  for (let regionY = 0; regionY < region.height; regionY += 1) {
    const photoY = region.y + regionY;
    for (let regionX = 0; regionX < region.width; regionX += 1) {
      const photoX = region.x + regionX;
      const cropX = a * photoX + b * photoY + c;
      const cropY = d * photoX + e * photoY + f;
      const x0 = Math.floor(cropX);
      const y0 = Math.floor(cropY);
      if (x0 < -1 || y0 < -1 || x0 >= cropWidth || y0 >= cropHeight) continue;
      const fractionX = cropX - x0;
      const fractionY = cropY - y0;
      const weights = [(1 - fractionX) * (1 - fractionY), fractionX * (1 - fractionY), (1 - fractionX) * fractionY, fractionX * fractionY];
      const tapX = [x0, x0 + 1, x0, x0 + 1];
      const tapY = [y0, y0, y0 + 1, y0 + 1];
      const outputIndex = regionY * region.width + regionX;
      for (let tap = 0; tap < 4; tap += 1) {
        if (tapX[tap] < 0 || tapY[tap] < 0 || tapX[tap] >= cropWidth || tapY[tap] >= cropHeight || weights[tap] === 0) continue;
        const cropIndex = tapY[tap] * cropWidth + tapX[tap];
        for (let layer = 0; layer < cropLayers.length; layer += 1) {
          regionLayers[layer][outputIndex] += weights[tap] * cropLayers[layer][cropIndex];
        }
      }
    }
  }
  return regionLayers;
}
