/**
 * Blend a face's glare delta into the ORIGINAL photo pixels of its region.
 *
 *   patched = round(original + strength * 255 * delta)      per RGB channel, clamped to 0..255
 *
 * `delta` is already `glare_mask * (clean - input)` warped to photo space (glare_crop_pass.js),
 * so scaling it by `strength` is the same as scaling the mask, which is what the strength
 * slider means. Invariant the tests pin: wherever the delta is exactly 0 (outside the mask),
 * the output byte IS the input byte, alpha included; nothing is re-quantized.
 */

/** Returns a new RGBA array the size of `regionOriginalRgba`. `strength` in [0, 1]. */
export function blendFacePatch(regionOriginalRgba, regionDeltaLayers, strength) {
  const patchedRgba = new Uint8ClampedArray(regionOriginalRgba);
  if (strength <= 0) return patchedRgba;
  const deltaScale = strength * 255;
  const pixelCount = regionOriginalRgba.length / 4;
  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const delta = regionDeltaLayers[channel][pixelIndex];
      if (delta === 0) continue;
      const byteIndex = pixelIndex * 4 + channel;
      patchedRgba[byteIndex] = Math.round(regionOriginalRgba[byteIndex] + deltaScale * delta); // clamped by the array type
    }
  }
  return patchedRgba;
}

/** Count of pixels whose warped glare mask is non-zero (the only pixels allowed to change). */
export function countMaskedPixels(regionGlareMask) {
  let maskedPixelCount = 0;
  for (const maskValue of regionGlareMask) if (maskValue > 0) maskedPixelCount += 1;
  return maskedPixelCount;
}
