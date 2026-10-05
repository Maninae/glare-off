/**
 * Wiring: starts the service worker and the processing worker, queues photos through it one
 * at a time, and hands results to result cards. All pixel work happens in the worker.
 *
 * - The models load as soon as the page opens, so processing a photo makes no network request.
 * - On phones and small tablets the worker is terminated once the queue drains, which is the
 *   only way to give WASM memory back; the next photo restarts it from the offline cache.
 * - `window.__glareOffDebug` is the test hook (tests/app/test_app_end_to_end.mjs). It exposes
 *   state and timings; it never sends anything anywhere.
 */

import { APP_NAME, HIGH_RES_PASS_ENABLED, MODEL_PATHS, ONNX_RUNTIME_BUILDS, PREVIEW_LONG_SIDE } from "./app_config.js";
import { GlareWorkerClient } from "./pipeline/worker_client.js";
import { chooseRuntime, isMemoryConstrainedDevice } from "./pipeline/runtime_choice.js";
import { buildDownloadFileName, chooseExportFormat, sniffImageFormat } from "./photo/photo_file_types.js";
import { buildStoredZip } from "./photo/zip_store.js";
import { EngineStatusLine } from "./ui/engine_status.js";
import { wireInputDoors } from "./ui/input_doors.js";
import { ResultCard } from "./ui/result_card.js";

const appBaseUrl = new URL("../", import.meta.url);
const resolveAppUrl = (relativePath) => new URL(relativePath, appBaseUrl).href;

const elements = {
  intake: document.getElementById("intake"),
  dropZone: document.getElementById("drop-zone"),
  dropZoneButton: document.getElementById("drop-zone-button"),
  addMoreButton: document.getElementById("add-more-button"),
  fileInput: document.getElementById("file-input"),
  dragOverlay: document.getElementById("drag-overlay"),
  results: document.getElementById("results"),
  resultList: document.getElementById("result-list"),
  resultTemplate: document.getElementById("result-template"),
  downloadAllButton: document.getElementById("download-all-button"),
};
const engineStatus = new EngineStatusLine({
  rootElement: document.getElementById("engine-status"),
  textElement: document.getElementById("engine-status-text"),
  fillElement: document.getElementById("engine-status-fill"),
  retryButton: document.getElementById("engine-retry-button"),
});

const jobs = []; // { file, inputFormat, card, state, result, timings }
let assetManifest = null;
let runtime = null;
let workerClient = null;
let engineReady = null; // Promise of the worker's "ready" message
let engineInfo = null;
let queueRunning = false;
const pageTimings = { pageStart: performance.timeOrigin };
// The shell precache is ~0.2 MB, so control normally arrives well inside this.
const SERVICE_WORKER_CONTROL_WAIT_MS = 5000;

async function loadAssetManifest() {
  const response = await fetch(resolveAppUrl("asset_manifest.json"));
  return response.json();
}

function assetBytes(relativePath) {
  return assetManifest.assets.find((asset) => asset.path === relativePath)?.bytes ?? 0;
}

function startEngine() {
  const build = ONNX_RUNTIME_BUILDS[runtime.backend];
  const glareModelPath = MODEL_PATHS.glareModel;
  const expectedTotalBytes = assetBytes(build.wasmUrl) + assetBytes(MODEL_PATHS.faceDetector) + assetBytes(glareModelPath);
  let isFromCache = false;
  window.caches?.match(resolveAppUrl(glareModelPath)).then((cachedResponse) => (isFromCache = Boolean(cachedResponse)), () => {});
  workerClient = new GlareWorkerClient();
  engineStatus.showLoading(0, expectedTotalBytes, isFromCache);
  const startedAt = performance.now();
  engineReady = workerClient
    .start(
      {
        ortModuleUrl: resolveAppUrl(build.moduleUrl),
        wasmUrl: resolveAppUrl(build.wasmUrl),
        faceDetectorUrl: resolveAppUrl(MODEL_PATHS.faceDetector),
        glareModelUrl: resolveAppUrl(glareModelPath),
        backend: runtime.backend,
        numThreads: runtime.numThreads,
        expectedTotalBytes,
      },
      {
        onProgress: (loadedBytes) => (loadedBytes >= expectedTotalBytes ? engineStatus.showStarting() : engineStatus.showLoading(loadedBytes, expectedTotalBytes, isFromCache)),
        onNotice: (notice) => console.info(notice),
      },
    )
    .then((readyMessage) => {
      engineInfo = { ...readyMessage, glareModelPath, engineStartMs: performance.now() - startedAt };
      engineStatus.showReady(engineInfo);
      return readyMessage;
    })
    .catch((error) => {
      engineStatus.showError(`Could not start: ${error.message}. Reloading the page usually fixes this.`, () => {
        workerClient.terminate();
        startEngine();
        runQueue();
      });
      throw error;
    });
  engineReady.catch(() => {}); // surfaced through the status line and each job
  return engineReady;
}

function ensureEngine() {
  if (!workerClient || !workerClient.worker) startEngine();
  return engineReady;
}

function addPhotos(files) {
  elements.intake.dataset.state = "has-photos";
  elements.addMoreButton.hidden = false;
  elements.results.hidden = false;
  for (const file of files) {
    const card = new ResultCard(elements.resultTemplate, { fileName: file.name || "Pasted photo", cardIndex: jobs.length, onDownload: downloadOne });
    const job = { file, inputFormat: null, card, state: "queued", result: null };
    card.jobIndex = jobs.length;
    jobs.push(job);
    elements.resultList.append(card.element);
    card.showWaitingThumbnail(file);
  }
  if (files.length > 0) jobs[jobs.length - files.length].card.element.scrollIntoView({ behavior: "smooth", block: "start" });
  updateDownloadAll();
  runQueue();
}

const UNSUPPORTED_FORMAT_MESSAGES = {
  heic: "This browser cannot open HEIC photos. Safari on iPhone, iPad and Mac can; elsewhere, save the photo as JPEG first (on iPhone, Settings > Camera > Formats > Most Compatible).",
  avif: "This browser could not open this AVIF photo. Try saving it as JPEG or PNG.",
  unknown: "This file does not look like a photo this browser can open. JPEG, PNG and WebP always work.",
};

async function processJob(job) {
  job.inputFormat = await sniffImageFormat(job.file);
  job.card.showStage({ stage: "starting" });
  try {
    await ensureEngine();
  } catch (error) {
    job.state = "error";
    job.card.showError("The processing tools did not start. Use Try again above.");
    return;
  }
  const startedAt = performance.now();
  try {
    const result = await workerClient.processPhoto(job.file, {
      previewLongSide: Math.min(PREVIEW_LONG_SIDE, Math.ceil(Math.max(screen.width, screen.height) * (devicePixelRatio || 1))),
      highResPassEnabled: HIGH_RES_PASS_ENABLED || new URLSearchParams(location.search).has("hires"),
      onStage: (stageMessage) => job.card.showStage(stageMessage),
    });
    job.result = result;
    job.timings = { ...result.timings, totalMs: performance.now() - startedAt };
    job.state = result.faces.some((face) => face.hasGlare) ? "done" : result.faces.length === 0 ? "no-face" : "no-glare";
    job.card.showResult(result);
  } catch (error) {
    job.state = "error";
    const formatMessage = UNSUPPORTED_FORMAT_MESSAGES[job.inputFormat] ?? UNSUPPORTED_FORMAT_MESSAGES.unknown;
    job.card.showError(error.code === "decode-failed" ? formatMessage : `Something went wrong while processing (${error.message}). Try the photo again, or a smaller copy of it.`);
  }
}

async function runQueue() {
  if (queueRunning) return;
  queueRunning = true;
  try {
    for (;;) {
      const nextJob = jobs.find((job) => job.state === "queued");
      if (!nextJob) break;
      nextJob.state = "working";
      await processJob(nextJob);
      updateDownloadAll();
    }
  } finally {
    queueRunning = false;
  }
  if (isMemoryConstrainedDevice() && workerClient?.worker && !new URLSearchParams(location.search).has("keepworker")) {
    workerClient.terminate();
    engineStatus.showSleeping();
  }
}

/** A worker for encoding downloads; it needs no models, so a sleeping engine stays asleep. */
async function withExportWorker(task) {
  if (workerClient?.worker) return task(workerClient);
  const exportWorker = new GlareWorkerClient();
  exportWorker.startWithoutModels();
  try {
    return await task(exportWorker);
  } finally {
    exportWorker.terminate();
  }
}

async function exportJob(job) {
  const { mimeType, quality, extension } = await chooseExportFormat(job.inputFormat);
  const patches = job.card.currentPatches().map(({ region, patchedRgba }) => ({ region, patchedRgba }));
  const exported = await withExportWorker((client) => client.exportPhoto(job.file, patches, mimeType, quality));
  return { blob: exported.blob, fileName: buildDownloadFileName(job.file.name || "photo", extension), width: exported.width, height: exported.height };
}

function saveBlob(blob, fileName) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
}

async function downloadOne(card) {
  const job = jobs[card.jobIndex];
  card.setDownloadBusy(true);
  try {
    const { blob, fileName, width, height } = await exportJob(job);
    job.lastExport = { fileName, width, height, bytes: blob.size, type: blob.type };
    saveBlob(blob, fileName);
  } catch (error) {
    card.showMessage(`Could not save the full-resolution photo (${error.message}). This browser may not handle an image this large.`);
  } finally {
    card.setDownloadBusy(false);
  }
}

function updateDownloadAll() {
  const doneCount = jobs.filter((job) => job.state === "done").length;
  elements.downloadAllButton.hidden = doneCount < 2;
  elements.downloadAllButton.textContent = `Download all ${doneCount}`;
}

async function downloadAll() {
  const doneJobs = jobs.filter((job) => job.state === "done");
  elements.downloadAllButton.disabled = true;
  try {
    const entries = [];
    for (const [jobNumber, job] of doneJobs.entries()) {
      elements.downloadAllButton.textContent = `Saving ${jobNumber + 1} of ${doneJobs.length}`;
      const { blob, fileName } = await exportJob(job);
      entries.push({ name: fileName, bytes: new Uint8Array(await blob.arrayBuffer()) });
    }
    saveBlob(buildStoredZip(entries), `${APP_NAME.toLowerCase().replace(/\s+/g, "-")}-photos.zip`);
  } finally {
    elements.downloadAllButton.disabled = false;
    updateDownloadAll();
  }
}

/**
 * Register sw.js and wait (briefly) until it controls this page, so the worker's model and
 * runtime downloads pass through it and get cached on the way: one download, then offline.
 */
async function registerServiceWorkerAndWaitForControl() {
  if (!("serviceWorker" in navigator) || new URLSearchParams(location.search).has("nosw")) return;
  try {
    await navigator.serviceWorker.register(resolveAppUrl("sw.js"));
  } catch (error) {
    console.warn("offline support unavailable:", error.message);
    return;
  }
  if (navigator.serviceWorker.controller) return;
  await new Promise((resolve) => {
    navigator.serviceWorker.addEventListener("controllerchange", resolve, { once: true });
    setTimeout(resolve, SERVICE_WORKER_CONTROL_WAIT_MS);
  });
}

function exposeDebugHook() {
  window.__glareOffDebug = {
    describe: () => ({
      engineInfo,
      runtime,
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
        faces: job.result?.faces.map((face) => ({
          imageLeftEyeXY: face.imageLeftEyeXY,
          imageRightEyeXY: face.imageRightEyeXY,
          detectionScore: face.detectionScore,
          hasGlare: face.hasGlare,
          showLostDetailNote: face.showLostDetailNote,
          glareFraction: face.glareFraction,
          region: face.region,
        })),
      })),
      pageTimings,
    }),
    /** Full face data (masks included) for the bit-exactness check. Test use only. */
    getJobFaces: (jobIndex) => jobs[jobIndex]?.result?.faces,
    exportJob: (jobIndex) => exportJob(jobs[jobIndex]),
  };
}

async function boot() {
  document.getElementById("app-name").textContent = APP_NAME;
  exposeDebugHook();
  wireInputDoors(
    {
      fileInput: elements.fileInput,
      chooseButtons: [elements.dropZoneButton],
      clickTargets: [elements.dropZone, elements.addMoreButton],
      dropTargets: [elements.dropZone, elements.addMoreButton],
      dragOverlay: elements.dragOverlay,
    },
    { onFiles: addPhotos },
  );
  elements.downloadAllButton.addEventListener("click", downloadAll);
  try {
    await registerServiceWorkerAndWaitForControl();
    assetManifest = await loadAssetManifest();
    runtime = await chooseRuntime();
    await startEngine();
  } catch (error) {
    if (!engineInfo) console.error(error);
  }
  pageTimings.engineReadyAt = performance.now();
}

boot();
