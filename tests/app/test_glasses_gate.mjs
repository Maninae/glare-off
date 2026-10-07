/**
 * The glasses gate without a browser (pipeline/glasses_gate_pass.js, ui/face_switch_state.js):
 * - the four switch states, the toggle round trip, and each tile label;
 * - a synthetic two-face photo where a fake classifier says glasses for face A and not for B:
 *   the glare model runs on A only, B keeps its crop, and a download built from the switched-on
 *   faces leaves every pixel of B's crop area (and everything outside A's mask) byte-identical;
 * - forcing B on runs the glare model on exactly the kept crop, with no second classifier call;
 * - a probability equal to the threshold counts as glasses.
 *
 * Run: node tests/app/test_glasses_gate.mjs
 */

import { buildFullResolutionFacePatch, warpFaceGlareMaskToRegion } from "../../app/js/blend/face_region_patch.js";
import { computePhotoToEyeCropAffine } from "../../app/js/eye_crop_geometry.js";
import { computePhotoRegionCoveredByEyeCrop } from "../../app/js/pipeline/eye_crop_warp.js";
import { extractFaceEyeCrop } from "../../app/js/pipeline/glare_crop_pass.js";
import { joinCropRgbPlanes, runFaceThroughGlassesGate, runSkippedFaceGlarePass } from "../../app/js/pipeline/glasses_gate_pass.js";
import { describeFaceTile, faceSwitchState, isFaceSwitchedOn } from "../../app/js/ui/face_switch_state.js";
import { check, finish } from "./node_test_support.mjs";

const GLASSES_THRESHOLD = 0.5;

// 1. State machine.
const stateTable = [
  [true, false, "auto-on", true],
  [false, false, "auto-off", false],
  [false, true, "user-on", true],
  [true, true, "user-off", false],
];
check(
  "switch state: glassesDetected x overridden -> auto-on / auto-off / user-on / user-off",
  stateTable.every(([glassesDetected, overridden, state, switchedOn]) => faceSwitchState(glassesDetected, overridden) === state && isFaceSwitchedOn(glassesDetected, overridden) === switchedOn),
);
const labelCases = [
  [{ glassesDetected: true, glareRun: true, hasGlare: true }, false, false, "Glare removed"],
  [{ glassesDetected: true, glareRun: true, hasGlare: false }, false, false, "No glare found"],
  [{ glassesDetected: false, glareRun: false, hasGlare: false }, false, false, "No glasses, skipped"],
  [{ glassesDetected: false, glareRun: true, hasGlare: true }, true, false, "Forced on"],
  [{ glassesDetected: false, glareRun: true, hasGlare: false }, true, false, "Forced on|No glare found"],
  [{ glassesDetected: true, glareRun: true, hasGlare: true }, true, false, "Forced off"],
  [{ glassesDetected: false, glareRun: false, hasGlare: false }, true, true, "Working"],
];
const wrongLabels = labelCases.filter(([face, overridden, isBusy, expected]) => describeFaceTile(face, overridden, isBusy).join("|") !== expected);
check("tile labels for every state, including no glare found and the working state", wrongLabels.length === 0, wrongLabels.map((labelCase) => labelCase[3]).join(", "));

// 2. Two faces on a synthetic photo.
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}
const random = seededRandom(7);
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
const faceWithGlasses = { imageLeftEyeXY: [250, 300], imageRightEyeXY: [350, 305] };
const faceWithoutGlasses = { imageLeftEyeXY: [600, 320], imageRightEyeXY: [700, 318] };

// Fake classifier: glasses only for crops whose first value matches face A's crop (identified up front).
const faceWithGlassesCrop = extractFaceEyeCrop({ face: faceWithGlasses, photoWidth, photoHeight, readPhotoRegion });
let classifierCallCount = 0;
const fakeClassifier = async (cropPlanarRgb) => {
  classifierCallCount += 1;
  return cropPlanarRgb.every((value, index) => value === faceWithGlassesCrop.cropPlanarRgb[index]) ? 0.9 : 0.1;
};
// Fake glare model: a soft disc of glare on the crop's left lens, the clean crop 40% darker there.
const glareModelInputs = [];
const fakeGlareModel = async (cropPlanarRgb, cropWidth, cropHeight) => {
  glareModelInputs.push(cropPlanarRgb.slice());
  const planeSize = cropWidth * cropHeight;
  const masksPlanar = new Float32Array(planeSize * 2);
  const cleanPlanarRgb = cropPlanarRgb.slice();
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const discMask = 1 / (1 + Math.exp((Math.hypot(x - cropWidth * 0.32, y - cropHeight * 0.5) / (cropWidth * 0.08) - 1) * 8));
      masksPlanar[y * cropWidth + x] = discMask;
      for (let channel = 0; channel < 3; channel += 1) cleanPlanarRgb[channel * planeSize + y * cropWidth + x] *= 0.6;
    }
  }
  return { cleanPlanarRgb, masksPlanar };
};

const gate = (face) => runFaceThroughGlassesGate({ face, photoWidth, photoHeight, readPhotoRegion, runGlassesClassifier: fakeClassifier, runGlareModel: fakeGlareModel, glassesProbabilityThreshold: GLASSES_THRESHOLD });
const { faceResult: resultA } = await gate(faceWithGlasses);
const { faceResult: resultB, eyeCrop: eyeCropB } = await gate(faceWithoutGlasses);
const faces = [
  { ...faceWithGlasses, ...resultA },
  { ...faceWithoutGlasses, ...resultB },
];
check("gate: face A classified as glasses, glare model ran, glare found", resultA.glassesDetected && resultA.glareRun && resultA.hasGlare && resultA.cropRgbPlanes === null);
check("gate: face B classified as no glasses, glare model never ran, nothing to blend", !resultB.glassesDetected && !resultB.glareRun && !resultB.hasGlare && resultB.region === null && resultB.cropDeltaLayers === null);
check("gate: the glare model was called once (face A only), the classifier twice", glareModelInputs.length === 1 && classifierCallCount === 2, `${glareModelInputs.length} glare runs, ${classifierCallCount} classifier runs`);
check("gate: the glare model received the same crop the classifier saw", glareModelInputs[0].every((value, index) => value === faceWithGlassesCrop.cropPlanarRgb[index]));
const planeSize = resultB.cropWidth * resultB.cropHeight;
check("gate: the skipped face keeps its crop as three crop-size planes", resultB.cropRgbPlanes.length === 3 && resultB.cropRgbPlanes.every((plane) => plane.length === planeSize));
check("gate: kept planes join back into the exact crop", joinCropRgbPlanes(resultB.cropRgbPlanes).every((value, index) => value === eyeCropB.cropPlanarRgb[index]));

// The download path: a fresh copy of the photo with each switched-on glare face pasted in.
function exportPhoto(exportFaces) {
  const exported = photoRgba.slice();
  for (const face of exportFaces) {
    const patched = buildFullResolutionFacePatch(face, readPhotoRegion(face.region), 1);
    for (let row = 0; row < face.region.height; row += 1) exported.set(patched.subarray(row * face.region.width * 4, (row + 1) * face.region.width * 4), ((face.region.y + row) * photoWidth + face.region.x) * 4);
  }
  return exported;
}
const switchedOnFaces = (overriddenByFace) => faces.filter((face, faceIndex) => face.hasGlare && isFaceSwitchedOn(face.glassesDetected, overriddenByFace[faceIndex]));
function countChangedPixels(exported, insideRect) {
  let changed = 0;
  for (let pixel = 0; pixel < photoWidth * photoHeight; pixel += 1) {
    const x = pixel % photoWidth;
    const y = Math.floor(pixel / photoWidth);
    if (insideRect && (x < insideRect.x || y < insideRect.y || x >= insideRect.x + insideRect.width || y >= insideRect.y + insideRect.height)) continue;
    for (let channel = 0; channel < 4; channel += 1) {
      if (exported[pixel * 4 + channel] !== photoRgba[pixel * 4 + channel]) {
        changed += 1;
        break;
      }
    }
  }
  return changed;
}
const coverageB = computePhotoRegionCoveredByEyeCrop(computePhotoToEyeCropAffine(faceWithoutGlasses.imageLeftEyeXY, faceWithoutGlasses.imageRightEyeXY), 512, 256, photoWidth, photoHeight);
const automaticExport = exportPhoto(switchedOnFaces([false, false]));
const maskA = warpFaceGlareMaskToRegion(faces[0]);
let changedOutsideMaskA = 0;
for (let pixel = 0; pixel < photoWidth * photoHeight; pixel += 1) {
  const x = pixel % photoWidth;
  const y = Math.floor(pixel / photoWidth);
  const regionA = faces[0].region;
  const insideRegionA = x >= regionA.x && y >= regionA.y && x < regionA.x + regionA.width && y < regionA.y + regionA.height;
  const masked = insideRegionA && maskA[(y - regionA.y) * regionA.width + (x - regionA.x)] > 0;
  if (masked) continue;
  for (let channel = 0; channel < 4; channel += 1) {
    if (automaticExport[pixel * 4 + channel] !== photoRgba[pixel * 4 + channel]) {
      changedOutsideMaskA += 1;
      break;
    }
  }
}
check("bit-exact: automatic download changes zero pixels outside face A's warped glare mask", changedOutsideMaskA === 0, `${countChangedPixels(automaticExport)} pixels changed in total`);
check("bit-exact: zero pixels change in face B's crop area (no glasses, skipped)", countChangedPixels(automaticExport, coverageB) === 0, `B covers ${coverageB.width}x${coverageB.height}`);
check("bit-exact: face A switched off (user-off): the download equals the photo", countChangedPixels(exportPhoto(switchedOnFaces([true, false]))) === 0);

// 3. Force face B on: the glare model runs on the kept crop, the classifier does not run again.
const classifierCallsBefore = classifierCallCount;
const forcedB = await runSkippedFaceGlarePass({ face: faces[1], photoWidth, photoHeight, readPhotoRegion, runGlareModel: fakeGlareModel });
faces[1] = forcedB;
check("force on: glare model received exactly face B's kept crop", glareModelInputs.length === 2 && glareModelInputs[1].every((value, index) => value === eyeCropB.cropPlanarRgb[index]));
check("force on: no second classifier run, kept crop dropped, glare found", classifierCallCount === classifierCallsBefore && forcedB.glareRun && forcedB.cropRgbPlanes === null && forcedB.hasGlare);
check("force on: face B's crop area now changes in the download (user-on)", countChangedPixels(exportPhoto(switchedOnFaces([false, true])), coverageB) > 0);

// 4. Threshold boundary.
const { faceResult: atThreshold } = await runFaceThroughGlassesGate({ face: faceWithGlasses, photoWidth, photoHeight, readPhotoRegion, runGlassesClassifier: async () => GLASSES_THRESHOLD, runGlareModel: fakeGlareModel, glassesProbabilityThreshold: GLASSES_THRESHOLD });
check("threshold: probability equal to the threshold counts as glasses", atThreshold.glassesDetected && atThreshold.glareRun);

finish();
