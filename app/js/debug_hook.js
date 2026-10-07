/**
 * The end-to-end test hook `window.__glareOffDebug`, installed only when the page URL carries
 * `?debug` (tests/app/test_app_end_to_end.mjs passes it). It reads state and never sends
 * anything anywhere; without the flag the page exposes nothing.
 */

import { warpFaceGlareMaskToRegion } from "./blend/face_region_patch.js";
import { computePhotoToEyeCropAffine, EYE_CROP_HEIGHT, EYE_CROP_WIDTH } from "./eye_crop_geometry.js";
import { computePhotoRegionCoveredByEyeCrop } from "./pipeline/eye_crop_warp.js";
import { faceSwitchState } from "./ui/face_switch_state.js";

/** Typed arrays reachable from a face result object (one level of arrays/objects deep). */
function listTypedArrays(face) {
  const found = [];
  for (const [key, value] of Object.entries(face)) {
    const candidates = Array.isArray(value) ? value : [value];
    for (const candidate of candidates) {
      if (ArrayBuffer.isView(candidate)) found.push({ key, type: candidate.constructor.name, length: candidate.length });
    }
  }
  return found;
}

/**
 * Args:
 *   jobs: main.js's live job list; visibleRegions: the one-photo region cache;
 *   exportJob(job): the download path; getEngineInfo/getRuntime: current engine state.
 */
export function installDebugHookWhenRequested({ jobs, visibleRegions, exportJob, getEngineInfo, getRuntime, pageTimings }) {
  if (!new URLSearchParams(location.search).has("debug")) return;
  window.__glareOffDebug = {
    describe: () => ({
      engineInfo: getEngineInfo(),
      runtime: getRuntime(),
      crossOriginIsolated: self.crossOriginIsolated,
      serviceWorkerControlled: Boolean(navigator.serviceWorker?.controller),
      jobs: jobs.map((job) => ({
        fileName: job.file.name,
        inputFormat: job.inputFormat,
        state: job.state,
        timings: job.timings,
        lastExport: job.lastExport,
        photoWidth: job.result?.photoWidth,
        photoHeight: job.result?.photoHeight,
        faces: job.result?.faces.map((face, faceIndex) => ({
          imageLeftEyeXY: face.imageLeftEyeXY,
          imageRightEyeXY: face.imageRightEyeXY,
          detectionScore: face.detectionScore,
          glassesProbability: face.glassesProbability,
          glassesDetected: face.glassesDetected,
          glareRun: face.glareRun,
          switchState: faceSwitchState(face.glassesDetected, job.card.overriddenByFace[faceIndex]),
          switchedOn: job.card.isFaceOn(faceIndex),
          hasGlare: face.hasGlare,
          showLostDetailNote: face.showLostDetailNote,
          glareFraction: face.glareFraction,
          region: face.region,
          // The photo rect this face's 1x eye crop covers, whether or not the glare model ran.
          cropCoverageRegion: computePhotoRegionCoveredByEyeCrop(computePhotoToEyeCropAffine(face.imageLeftEyeXY, face.imageRightEyeXY), EYE_CROP_WIDTH, EYE_CROP_HEIGHT, job.result.photoWidth, job.result.photoHeight),
        })),
      })),
      pageTimings,
    }),
    /** Warped glare mask of one face over its region, computed now (nothing keeps it). */
    warpGlareMaskToRegion: (jobIndex, faceIndex) => warpFaceGlareMaskToRegion(jobs[jobIndex].result.faces[faceIndex]),
    /** What each face result and the region cache hold, for the memory check. */
    describeMemory: () => ({
      faces: jobs.flatMap((job, jobIndex) =>
        (job.result?.faces ?? []).map((face, faceIndex) => ({ jobIndex, faceIndex, regionPixels: face.region ? face.region.width * face.region.height : 0, typedArrays: listTypedArrays(face) })),
      ),
      jobsHoldingRegionPixels: jobs.map((job, jobIndex) => (job.result && Object.hasOwn(job.result, "regionOriginalRgbaByFace") ? jobIndex : -1)).filter((jobIndex) => jobIndex >= 0),
      visibleRegionCacheJobIndex: jobs.indexOf(visibleRegions.owner),
      visibleRegionCacheBytes: visibleRegions.heldBytes(),
    }),
    exportJob: (jobIndex) => exportJob(jobs[jobIndex]),
  };
}
