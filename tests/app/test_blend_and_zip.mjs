/**
 * Pure-function tests that need no browser:
 * - face_patch_blend.js: bytes with a zero delta are untouched (alpha too); strength 0 is identity.
 * - glare_crop_pass.js end to end with a fake glare model on a synthetic photo: every photo
 *   pixel whose warped glare mask is 0 is byte-identical after blending; the mask floor
 *   zeroes sub-1/510 values; no-glare faces come back empty; the 2x pass switches on.
 * - zip_store.js: CRC-32 of a known string; the archive lists cleanly with `unzip -t`.
 *
 * Run: node tests/app/test_blend_and_zip.mjs
 */

import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { blendFacePatch } from "../../app/js/blend/face_patch_blend.js";
import { chooseCropSizeMultiplier, computeCropSpaceDelta, GLARE_MASK_ZERO_BELOW, runGlarePassOnFace } from "../../app/js/pipeline/glare_crop_pass.js";
import { computePhotoToEyeCropAffine } from "../../app/js/eye_crop_geometry.js";
import { buildStoredZip, crc32 } from "../../app/js/photo/zip_store.js";
import { check, finish } from "./node_test_support.mjs";

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

// 1. Blend unit.
const random = seededRandom(1);
const pixelCount = 5000;
const original = new Uint8ClampedArray(pixelCount * 4).map(() => Math.floor(random() * 256));
const deltas = [0, 1, 2].map(() => new Float32Array(pixelCount).map(() => (random() < 0.5 ? 0 : random() - 0.5)));
const blended = blendFacePatch(original, deltas, 1);
let zeroDeltaChanged = 0;
let alphaChanged = 0;
for (let pixel = 0; pixel < pixelCount; pixel += 1) {
  for (let channel = 0; channel < 3; channel += 1) if (deltas[channel][pixel] === 0 && blended[pixel * 4 + channel] !== original[pixel * 4 + channel]) zeroDeltaChanged += 1;
  if (blended[pixel * 4 + 3] !== original[pixel * 4 + 3]) alphaChanged += 1;
}
check("blend: channels with a zero delta keep their exact byte", zeroDeltaChanged === 0, `${zeroDeltaChanged} changed`);
check("blend: alpha never changes", alphaChanged === 0);
check("blend: strength 0 returns the original bytes", blendFacePatch(original, deltas, 0).every((value, index) => value === original[index]));
const saturating = blendFacePatch(new Uint8ClampedArray([250, 5, 128, 255]), [new Float32Array([0.5]), new Float32Array([-0.5]), new Float32Array([0.1])], 1);
check("blend: clamps to 0..255 and rounds", saturating[0] === 255 && saturating[1] === 0 && saturating[2] === Math.round(128 + 25.5), Array.from(saturating).join(","));

// 2. Mask floor.
const floorDelta = computeCropSpaceDelta({
  cropPlanarRgb: new Float32Array(6).fill(0.5),
  cleanPlanarRgb: new Float32Array(6).fill(0.2),
  masksPlanar: new Float32Array([GLARE_MASK_ZERO_BELOW * 0.99, GLARE_MASK_ZERO_BELOW, 0, 0]),
  cropWidth: 2,
  cropHeight: 1,
});
check("mask floor: values below 0.5/255 become exactly 0 (and so does their delta)", floorDelta.glareMask[0] === 0 && floorDelta.deltaLayers[0][0] === 0 && floorDelta.glareMask[1] > 0 && floorDelta.deltaLayers[0][1] !== 0);

// 3. Whole face pass on a synthetic photo with a fake model.
const photoWidth = 900;
const photoHeight = 700;
const photoRgba = new Uint8ClampedArray(photoWidth * photoHeight * 4).map((_, index) => (index % 4 === 3 ? 255 : Math.floor(random() * 256)));
const readPhotoRegion = (rect) => {
  const regionRgba = new Uint8ClampedArray(rect.width * rect.height * 4);
  for (let row = 0; row < rect.height; row += 1) {
    const start = ((rect.y + row) * photoWidth + rect.x) * 4;
    regionRgba.set(photoRgba.subarray(start, start + rect.width * 4), row * rect.width * 4);
  }
  return regionRgba;
};
// Fake model: glare in a disc on the crop's left lens, soft edge, sigmoid-like tails everywhere.
const fakeGlareModel = async (cropPlanarRgb, cropWidth, cropHeight) => {
  const planeSize = cropWidth * cropHeight;
  const masksPlanar = new Float32Array(planeSize * 2);
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const distance = Math.hypot(x - cropWidth * 0.32, y - cropHeight * 0.5) / (cropWidth * 0.08);
      masksPlanar[y * cropWidth + x] = 1 / (1 + Math.exp((distance - 1) * 8)); // tails ~1e-30 far away
      masksPlanar[planeSize + y * cropWidth + x] = distance < 0.4 ? 1 : 0;
    }
  }
  return { cleanPlanarRgb: cropPlanarRgb.map((value) => value * 0.6), masksPlanar };
};
const face = { imageLeftEyeXY: [380, 330], imageRightEyeXY: [520, 345] };
const pass = await runGlarePassOnFace({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel: fakeGlareModel });
check("face pass: glare found by the fake model", pass.hasGlare && pass.region !== null);
check("face pass: lost-detail note raised for the saturated core", pass.showLostDetailNote);
const patched = blendFacePatch(pass.regionOriginalRgba, pass.regionDeltaLayers, 1);
let outsideMaskChanged = 0;
let insideMaskChanged = 0;
for (let pixel = 0; pixel < pass.region.width * pass.region.height; pixel += 1) {
  let differs = false;
  for (let channel = 0; channel < 4; channel += 1) if (patched[pixel * 4 + channel] !== pass.regionOriginalRgba[pixel * 4 + channel]) differs = true;
  if (differs && pass.regionGlareMask[pixel] === 0) outsideMaskChanged += 1;
  if (differs && pass.regionGlareMask[pixel] > 0) insideMaskChanged += 1;
}
check("face pass: zero pixels change where the warped glare mask is 0", outsideMaskChanged === 0, `${insideMaskChanged} changed inside the mask`);
check("face pass: the region's original bytes are the photo's bytes", pass.regionOriginalRgba.every((value, index) => value === readPhotoRegion(pass.region)[index]));

const noGlarePass = await runGlarePassOnFace({
  face,
  photoWidth,
  photoHeight,
  readPhotoRegion,
  runGlareModel: async (cropPlanarRgb, cropWidth, cropHeight) => ({ cleanPlanarRgb: cropPlanarRgb, masksPlanar: new Float32Array(cropWidth * cropHeight * 2).fill(0.001) }),
});
check("face pass: a face with no glare comes back empty and untouched", !noGlarePass.hasGlare && noGlarePass.region === null);

// 4. High-res pass switch.
const largeFaceAffine = computePhotoToEyeCropAffine([1000, 1000], [1600, 1000]); // crop scale ~0.31
const smallFaceAffine = computePhotoToEyeCropAffine([100, 100], [250, 100]); // crop scale ~1.23
check("2x pass: only when enabled and the face is large", chooseCropSizeMultiplier(largeFaceAffine, true) === 2 && chooseCropSizeMultiplier(largeFaceAffine, false) === 1 && chooseCropSizeMultiplier(smallFaceAffine, true) === 1);
const largeWidth = 2600;
const largeHeight = 2000;
let modelSizes = [];
const sizeRecordingModel = async (cropPlanarRgb, cropWidth, cropHeight) => {
  modelSizes.push(`${cropWidth}x${cropHeight}`);
  return fakeGlareModel(cropPlanarRgb, cropWidth, cropHeight);
};
const largeRegionReader = (rect) => new Uint8ClampedArray(rect.width * rect.height * 4).fill(200);
const highResPass = await runGlarePassOnFace({ face: { imageLeftEyeXY: [1000, 1000], imageRightEyeXY: [1600, 1000] }, photoWidth: largeWidth, photoHeight: largeHeight, readPhotoRegion: largeRegionReader, runGlareModel: sizeRecordingModel, highResPassEnabled: true });
check("2x pass: runs 512x256 then 1024x512 and keeps the 2x result", modelSizes.join(",") === "512x256,1024x512" && highResPass.cropSizeMultiplier === 2, modelSizes.join(","));

// 5. ZIP.
check("zip: CRC-32 of 'The quick brown fox jumps over the lazy dog' is 414fa339", crc32(new TextEncoder().encode("The quick brown fox jumps over the lazy dog")).toString(16) === "414fa339");
const zipBlob = buildStoredZip([
  { name: "a.jpg", bytes: new Uint8Array([1, 2, 3]) },
  { name: "a.jpg", bytes: new Uint8Array([4, 5]) },
  { name: "é photo.png", bytes: new Uint8Array(1000).map((_, index) => index % 251) },
]);
const zipPath = join(mkdtempSync(join(tmpdir(), "glare-off-zip-")), "test.zip");
writeFileSync(zipPath, Buffer.from(await zipBlob.arrayBuffer()));
let unzipReport = "";
try {
  unzipReport = execFileSync("unzip", ["-t", zipPath], { encoding: "utf8" });
} catch (error) {
  unzipReport = String(error.stdout ?? error.message);
}
check("zip: `unzip -t` reports no errors", /No errors detected/.test(unzipReport), unzipReport.split("\n").filter(Boolean).slice(-1)[0]);
check("zip: duplicate names are made unique", unzipReport.includes("a (2).jpg"));

finish();
