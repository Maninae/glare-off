/**
 * The processing Web Worker: owns onnxruntime-web, both model sessions, and the photo's
 * decoded pixels. The page only ever receives small regions around the eyes and a preview.
 *
 * MODULE worker, started from a blob: bootstrap by worker_client.js so the page's
 * Content-Security-Policy governs it (a worker loaded from its own URL takes its policy from
 * HTTP headers, and static hosts send none). The ORT ESM build resolves its own .mjs glue
 * via import.meta.url, so no wasmPaths juggling; we pass the .wasm bytes ourselves
 * (env.wasm.wasmBinary) so the download shows real progress.
 *
 * Protocol (page -> worker): init, process-photo, read-regions, export-photo. Worker -> page:
 * progress, ready | init-failed, stage, result, regions, exported, error. Every reply carries
 * the requestId. read-regions and export-photo need no models (a model-free worker serves them).
 *
 * - YuNet always runs on the WASM CPU path (its input size changes per photo, which would
 *   recompile GPU shaders every time); the glare model uses WebGPU when the page chose it.
 * - Photos are decoded with createImageBitmap(file, { imageOrientation: "from-image" }) and
 *   read through a small OffscreenCanvas strip by strip; no full-size pixel buffer exists.
 */

import { computeDetectionSize, buildYunetInputTensor, decodeYunetOutputs, faceRowsToDetectedFaceEyes } from "./yunet_face_decode.js";
import { areaDownscaleRgba } from "./area_downscale.js";
import { runGlarePassOnFace } from "./glare_crop_pass.js";
import { buildFullResolutionFacePatch } from "../blend/face_region_patch.js";

let onnxRuntime = null;
let faceDetectorSession = null;
let glareSession = null;
let glareInputName = "glare_crop";
let activeBackend = "wasm";

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

/** fetch() a same-origin file as bytes, reporting each chunk through onChunk(byteCount). */
async function fetchBytesWithProgress(url, onChunk) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (HTTP ${response.status})`);
  const reader = response.body.getReader();
  const chunks = [];
  let totalBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    totalBytes += value.byteLength;
    onChunk(value.byteLength);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function initialize({ ortModuleUrl, wasmUrl, faceDetectorUrl, glareModelUrl, backend, numThreads, expectedTotalBytes }) {
  const initStarted = performance.now();
  let loadedBytes = 0;
  const reportChunk = (byteCount) => {
    loadedBytes += byteCount;
    post({ type: "progress", loadedBytes, expectedTotalBytes });
  };
  onnxRuntime = await import(ortModuleUrl);
  const [wasmBinary, faceDetectorBytes, glareModelBytes] = await Promise.all([
    fetchBytesWithProgress(wasmUrl, reportChunk),
    fetchBytesWithProgress(faceDetectorUrl, reportChunk),
    fetchBytesWithProgress(glareModelUrl, reportChunk),
  ]);
  const downloadedMs = performance.now() - initStarted;
  onnxRuntime.env.wasm.wasmBinary = wasmBinary.buffer;
  onnxRuntime.env.wasm.numThreads = numThreads;
  onnxRuntime.env.logLevel = "error";
  // Memory arenas keep peak allocations around for reuse; phones would rather give them back.
  const cpuSessionOptions = { executionProviders: ["wasm"], graphOptimizationLevel: "all", enableCpuMemArena: false, enableMemPattern: false };
  faceDetectorSession = await onnxRuntime.InferenceSession.create(faceDetectorBytes, cpuSessionOptions);
  activeBackend = backend;
  if (backend === "webgpu") {
    try {
      glareSession = await onnxRuntime.InferenceSession.create(glareModelBytes, { executionProviders: ["webgpu"], graphOptimizationLevel: "all" });
    } catch (webgpuError) {
      // Same build carries the CPU path, so a GPU failure degrades instead of failing.
      activeBackend = "wasm";
      post({ type: "notice", message: `WebGPU session failed, using CPU: ${webgpuError.message}` });
    }
  }
  if (!glareSession) glareSession = await onnxRuntime.InferenceSession.create(glareModelBytes, cpuSessionOptions);
  glareInputName = glareSession.inputNames[0];
  await warmUpGlareModel();
  post({
    type: "ready",
    backend: activeBackend,
    numThreads: onnxRuntime.env.wasm.numThreads,
    crossOriginIsolated: self.crossOriginIsolated === true,
    timings: { downloadMs: downloadedMs, initMs: performance.now() - initStarted },
  });
}

/** One throwaway run so the first real photo does not pay for shader compiles / buffer setup. */
async function warmUpGlareModel() {
  await runGlareModel(new Float32Array(3 * 256 * 512), 512, 256);
  glareInferenceMs = [];
}

let glareInferenceMs = [];

async function runGlareModel(cropPlanarRgb, cropWidth, cropHeight) {
  const inferenceStarted = performance.now();
  const inputTensor = new onnxRuntime.Tensor("float32", cropPlanarRgb, [1, 3, cropHeight, cropWidth]);
  const outputs = await glareSession.run({ [glareInputName]: inputTensor });
  const cleanPlanarRgb = await outputs.clean_crop.getData();
  const masksPlanar = await outputs.masks.getData();
  inputTensor.dispose?.();
  outputs.clean_crop.dispose?.();
  outputs.masks.dispose?.();
  glareInferenceMs.push(performance.now() - inferenceStarted);
  return { cleanPlanarRgb, masksPlanar };
}

/** Reads photo rects out of a decoded ImageBitmap through one reusable small canvas. */
function createPhotoRegionReader(photoBitmap) {
  const canvas = new OffscreenCanvas(1, 1);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  return (rect) => {
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    } else {
      context.clearRect(0, 0, rect.width, rect.height);
    }
    context.imageSmoothingEnabled = false;
    context.drawImage(photoBitmap, rect.x, rect.y, rect.width, rect.height, 0, 0, rect.width, rect.height);
    return context.getImageData(0, 0, rect.width, rect.height).data;
  };
}

async function detectFaces(photoBitmap, readPhotoRegion) {
  const photoWidth = photoBitmap.width;
  const photoHeight = photoBitmap.height;
  const { detectionScale, detectionWidth, detectionHeight } = computeDetectionSize(photoWidth, photoHeight);
  const detectionRgba =
    detectionScale >= 1
      ? readPhotoRegion({ x: 0, y: 0, width: photoWidth, height: photoHeight })
      : areaDownscaleRgba({
          sourceWidth: photoWidth,
          sourceHeight: photoHeight,
          destinationWidth: detectionWidth,
          destinationHeight: detectionHeight,
          inverseScale: detectionScale,
          readSourceRows: (firstRow, rowCount) => readPhotoRegion({ x: 0, y: firstRow, width: photoWidth, height: rowCount }),
        });
  const { tensorData, paddedWidth, paddedHeight } = buildYunetInputTensor(detectionRgba, detectionWidth, detectionHeight);
  const outputs = await faceDetectorSession.run({ input: new onnxRuntime.Tensor("float32", tensorData, [1, 3, paddedHeight, paddedWidth]) });
  const outputsByName = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, tensor.data]));
  const faceRows = decodeYunetOutputs(outputsByName, paddedWidth, paddedHeight);
  return faceRowsToDetectedFaceEyes(faceRows, detectionScale);
}

async function decodePhoto(file) {
  return createImageBitmap(file, { imageOrientation: "from-image", premultiplyAlpha: "none" });
}

async function makePreview(photoBitmap, previewLongSide) {
  const previewScale = Math.min(1, previewLongSide / Math.max(photoBitmap.width, photoBitmap.height));
  const resizeWidth = Math.max(1, Math.round(photoBitmap.width * previewScale));
  const resizeHeight = Math.max(1, Math.round(photoBitmap.height * previewScale));
  return createImageBitmap(photoBitmap, { resizeWidth, resizeHeight, resizeQuality: "high" });
}

async function processPhoto({ requestId, file, previewLongSide, highResPassEnabled }) {
  const timings = {};
  let started = performance.now();
  let photoBitmap;
  try {
    photoBitmap = await decodePhoto(file);
  } catch (decodeError) {
    post({ type: "error", requestId, code: "decode-failed", message: decodeError.message });
    return;
  }
  timings.decodeMs = performance.now() - started;
  try {
    const photoWidth = photoBitmap.width;
    const photoHeight = photoBitmap.height;
    const readPhotoRegion = createPhotoRegionReader(photoBitmap);
    post({ type: "stage", requestId, stage: "detecting", photoWidth, photoHeight });
    started = performance.now();
    const detectedFaces = await detectFaces(photoBitmap, readPhotoRegion);
    timings.detectMs = performance.now() - started;
    const faces = [];
    // The visible photo's region cache adopts these right away; faces themselves stay crop-space.
    const regionOriginalRgbaByFace = [];
    const transfer = [];
    timings.facePassMs = [];
    glareInferenceMs = [];
    for (const [faceIndex, face] of detectedFaces.entries()) {
      post({ type: "stage", requestId, stage: "cleaning", faceIndex, faceCount: detectedFaces.length });
      started = performance.now();
      const pass = await runGlarePassOnFace({ face, photoWidth, photoHeight, readPhotoRegion, runGlareModel, highResPassEnabled });
      timings.facePassMs.push(performance.now() - started);
      faces.push({
        imageLeftEyeXY: face.imageLeftEyeXY,
        imageRightEyeXY: face.imageRightEyeXY,
        faceBoxXYWH: face.faceBoxXYWH,
        detectionScore: face.detectionScore,
        ...pass,
      });
      regionOriginalRgbaByFace.push(pass.hasGlare ? readPhotoRegion(pass.region) : null);
      if (pass.hasGlare) {
        transfer.push(regionOriginalRgbaByFace[faceIndex].buffer, ...pass.cropDeltaLayers.map((layer) => layer.buffer), pass.cropGlareMask.buffer);
      }
    }
    started = performance.now();
    const previewBitmap = await makePreview(photoBitmap, previewLongSide);
    timings.previewMs = performance.now() - started;
    timings.glareInferenceOnlyMs = glareInferenceMs;
    transfer.push(previewBitmap);
    post({ type: "result", requestId, photoWidth, photoHeight, faces, regionOriginalRgbaByFace, previewBitmap, timings, backend: activeBackend }, transfer);
  } finally {
    photoBitmap.close();
  }
}

/** Original RGBA bytes of photo rects, from a fresh decode (the page keeps no full-size pixels). */
async function readRegions({ requestId, file, regions }) {
  const photoBitmap = await decodePhoto(file);
  try {
    const readPhotoRegion = createPhotoRegionReader(photoBitmap);
    const regionsRgba = regions.map((region) => readPhotoRegion(region)); // each read is a fresh buffer, safe to transfer
    post({ type: "regions", requestId, regionsRgba }, regionsRgba.map((rgba) => rgba.buffer));
  } finally {
    photoBitmap.close();
  }
}

/**
 * Full-resolution export: decode the original again, re-read each face region, warp and blend
 * its crop-space delta at `strength`, paste, encode. `faces`: crop-space face results that are
 * switched on. Encoding drops every metadata block (EXIF, GPS, ICC).
 */
async function exportPhoto({ requestId, file, faces, strength, mimeType, quality }) {
  const photoBitmap = await decodePhoto(file);
  try {
    const readPhotoRegion = createPhotoRegionReader(photoBitmap);
    const canvas = new OffscreenCanvas(photoBitmap.width, photoBitmap.height);
    const context = canvas.getContext("2d");
    context.drawImage(photoBitmap, 0, 0);
    for (const face of faces) {
      const patchedRgba = buildFullResolutionFacePatch(face, readPhotoRegion(face.region), strength);
      context.putImageData(new ImageData(patchedRgba, face.region.width, face.region.height), face.region.x, face.region.y);
    }
    const blob = await canvas.convertToBlob({ type: mimeType, quality });
    canvas.width = 0;
    canvas.height = 0;
    post({ type: "exported", requestId, blob, width: photoBitmap.width, height: photoBitmap.height });
  } finally {
    photoBitmap.close();
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "init") return await initialize(message);
    if (message.type === "process-photo") return await processPhoto(message);
    if (message.type === "read-regions") return await readRegions(message);
    if (message.type === "export-photo") return await exportPhoto(message);
  } catch (error) {
    if (message.type === "init") return post({ type: "init-failed", message: error.message });
    return post({ type: "error", requestId: message.requestId, code: "failed", message: error.message });
  }
  return undefined;
};
