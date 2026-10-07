/**
 * Wiring: starts the service worker and the processing engine (pipeline/glare_engine.js),
 * queues photos through it one at a time, and hands results to result cards. All pixel work
 * happens in the worker.
 *
 * - The models load as soon as the page opens, so processing a photo makes no network request.
 * - On phones and small tablets the engine sleeps once the queue drains (glare_engine.js).
 * - Only the visible photo keeps full-resolution face regions (visibleRegions); downloads
 *   rebuild their patches in a worker from the crop-space face results.
 * - A face the glasses gate skipped can be forced on from its card; runSkippedFace queues that
 *   glare run on the engine behind any photo still being processed.
 * - With `?debug` in the URL, debug_hook.js installs the test hook `window.__glareOffDebug`.
 */

import { APP_NAME, GLASSES_PROBABILITY_THRESHOLD, HIGH_RES_PASS_ENABLED, PREVIEW_LONG_SIDE } from "./app_config.js";
import { GlareEngine } from "./pipeline/glare_engine.js";
import { chooseRuntime } from "./pipeline/runtime_choice.js";
import { buildDownloadFileName, chooseExportFormat, sniffImageFormat } from "./photo/photo_file_types.js";
import { VisiblePhotoRegionCache } from "./photo/visible_photo_regions.js";
import { buildStoredZip } from "./photo/zip_store.js";
import { installDebugHookWhenRequested } from "./debug_hook.js";
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
const queryFlags = new URLSearchParams(location.search);
const highResPassEnabled = HIGH_RES_PASS_ENABLED || queryFlags.has("hires");
// `?glassesthreshold=N` overrides the gate (tests: 2 skips every face, 0 checks every face).
const glassesProbabilityThreshold = queryFlags.has("glassesthreshold") ? Number(queryFlags.get("glassesthreshold")) : GLASSES_PROBABILITY_THRESHOLD;
let queueRunning = false;
const engine = new GlareEngine({ engineStatus, resolveAppUrl, onRetried: () => runQueue(), keepWorker: queryFlags.has("keepworker") });
const pageTimings = { pageStart: performance.timeOrigin };
const visibleRegions = new VisiblePhotoRegionCache({ readRegions: (file, regions) => engine.withPixelWorker((client) => client.readRegions(file, regions)) });
// The shell precache is ~0.2 MB, so control normally arrives well inside this.
const SERVICE_WORKER_CONTROL_WAIT_MS = 5000;

async function loadAssetManifest() {
  const response = await fetch(resolveAppUrl("asset_manifest.json"));
  return response.json();
}

/** done: some face has glare to blend; no-glare: faces were checked, none had glare; no-glasses: every face was skipped. */
function jobStateFromFaces(faces) {
  if (faces.length === 0) return "no-face";
  if (faces.some((face) => face.hasGlare)) return "done";
  return faces.some((face) => face.glareRun) ? "no-glare" : "no-glasses";
}

function addPhotos(files) {
  elements.intake.dataset.state = "has-photos";
  elements.addMoreButton.hidden = false;
  elements.results.hidden = false;
  for (const file of files) {
    const card = new ResultCard(elements.resultTemplate, {
      fileName: file.name || "Pasted photo",
      cardIndex: jobs.length,
      onDownload: downloadOne,
      onRunFace: runSkippedFace,
      onFacesChanged: (resultCard) => {
        const job = jobs[resultCard.jobIndex];
        job.state = jobStateFromFaces(job.result.faces);
        updateDownloadAll();
      },
      loadRegions: (resultCard) => visibleRegions.entriesFor(jobs[resultCard.jobIndex]),
    });
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
    await engine.ensure();
  } catch (error) {
    job.state = "error";
    job.card.showError("The processing tools did not start. Use Try again above.");
    return;
  }
  const startedAt = performance.now();
  try {
    const result = await engine.run((client) =>
      client.processPhoto(job.file, {
        previewLongSide: Math.min(PREVIEW_LONG_SIDE, Math.ceil(Math.max(screen.width, screen.height) * (devicePixelRatio || 1))),
        highResPassEnabled,
        glassesProbabilityThreshold,
        onStage: (stageMessage) => job.card.showStage(stageMessage),
      }),
    );
    const { regionOriginalRgbaByFace, thumbnailBitmapByFace, ...resultWithoutRegions } = result;
    job.result = resultWithoutRegions;
    visibleRegions.adopt(job, result.faces, regionOriginalRgbaByFace); // the newest result is the one on screen
    job.timings = { ...result.timings, totalMs: performance.now() - startedAt };
    job.state = jobStateFromFaces(result.faces);
    job.card.showResult(job.result, thumbnailBitmapByFace);
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
  engine.sleepWhenIdle(queueRunning);
}

/**
 * The visitor forced on a face the gate skipped: run the glare model on its kept crop, replace
 * the face result in place (the card reads the same object), and hand its region to the cache.
 */
async function runSkippedFace(card, faceIndex) {
  const job = jobs[card.jobIndex];
  const skippedFace = job.result.faces[faceIndex];
  try {
    const reply = await engine.run((client) => client.runFace(job.file, skippedFace, { photoWidth: job.result.photoWidth, photoHeight: job.result.photoHeight, highResPassEnabled }));
    job.result.faces[faceIndex] = reply.face;
    visibleRegions.adoptFace(job, faceIndex, reply.face, reply.regionOriginalRgba);
    job.timings.onDemandFacePassMs = [...(job.timings.onDemandFacePassMs ?? []), reply.facePassMs];
  } finally {
    engine.sleepWhenIdle(queueRunning);
  }
}

async function exportJob(job) {
  const { mimeType, quality, extension } = await chooseExportFormat(job.inputFormat);
  const { faces, strength } = job.card.currentExportSettings();
  const exported = await engine.withPixelWorker((client) => client.exportPhoto(job.file, faces, strength, mimeType, quality));
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
  if (!("serviceWorker" in navigator) || queryFlags.has("nosw")) return;
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

async function boot() {
  document.getElementById("app-name").textContent = APP_NAME;
  installDebugHookWhenRequested({ jobs, visibleRegions, exportJob, getEngineInfo: () => engine.info, getRuntime: () => engine.runtime, pageTimings });
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
    const assetManifest = await loadAssetManifest();
    await engine.start(await chooseRuntime(), assetManifest);
  } catch (error) {
    if (!engine.info) console.error(error);
  }
  pageTimings.engineReadyAt = performance.now();
}

boot();
