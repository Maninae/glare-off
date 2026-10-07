/**
 * End-to-end test of the real app in headless Chromium, served from a plain static server.
 *
 * Checks (each prints PASS/FAIL):
 *   1. First visit: engine loads; every photo (3 WebP, their 3 PNG copies, an EXIF-rotated JPEG,
 *      a 27 MP JPEG) finishes; face count and eye centers match the Python detector.
 *   2. Network isolation: every request goes to the page's own origin, and ZERO requests
 *      happen between "engine ready" and the end of processing and downloading.
 *   3. Downloads are produced at the original resolution; EXIF orientation is baked in.
 *   4. Bit-exactness (PNG in, PNG out): every pixel outside the warped glare masks of the
 *      switched-on faces is byte-identical to the input, and so is every pixel in the crop area
 *      of a face the glasses gate skipped; with every face switched off, the whole download is
 *      byte-identical.
 *   4a. Glasses gate: the face without glasses in the heavy-reflection and ring-light photos comes
 *      back auto-off (the faces with glasses auto-on); clicking a tile flips a face and the
 *      download follows; with `?glassesthreshold=2` (every face skipped) a click runs the glare
 *      model on demand, offline from the network, and only that face's pixels change.
 *   4b. Memory: after all photos are processed, no face result holds a region-size array, only
 *      one photo holds full-resolution regions, and that moves when another card is touched.
 *      JS heap (after a forced GC) goes to summary.json.
 *   5. Second visit (service worker in control): cross-origin isolated, multi-threaded.
 *   6. Offline (context offline, reload): still loads and processes a photo.
 *   7. WebGPU, when headless Chromium exposes an adapter.
 *   7b. Without `?debug` the page exposes no test hook.
 *   8. Screenshots at desktop and phone sizes, light and dark, to SCREENSHOT_DIRECTORY; no
 *      horizontal overflow on the phone.
 * Timings for every photo and backend go to SCREENSHOT_DIRECTORY/summary.json.
 *
 * Needs the private photos from `python -m tests.app.make_browser_test_photos` (vega) and
 * Playwright in the node tools directory (see app/CLAUDE.md, "Tests").
 * Every page load passes `debug` so the app installs window.__glareOffDebug.
 * Run: node tests/app/test_app_end_to_end.mjs
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

import { loadPlaywright, serveAppDirectory, waitForAppState } from "./browser_test_support.mjs";
import { checkAllFacesSwitchedOff, checkClickFlipsOneFace, checkForcedFaceOnDemand, checkGlassesGateDecisions } from "./e2e_glasses_gate_checks.mjs";
import { allDone, checkBitExactness, downloadCard as downloadCardInto, downloadedImageSize, isNetworkUrl, recordRequests } from "./e2e_page_checks.mjs";
import { REPO_ROOT, SCRATCH_DIRECTORY, check, finish } from "./node_test_support.mjs";

const E2E_PHOTO_DIRECTORY = join(SCRATCH_DIRECTORY, "app-e2e");
const REFERENCE_PHOTO_DIRECTORY = "/Volumes/vega/datasets/glare-off/reference-site-examples";
const SCREENSHOT_DIRECTORY = process.env.GLARE_OFF_SCREENSHOTS ?? "/tmp/glare-off-e2e";
const DOWNLOAD_DIRECTORY = join(SCREENSHOT_DIRECTORY, "downloads");
// Same pixels (PNG) must agree to float noise; lossy decodes (WebP/JPEG) differ slightly between
// libwebp/libjpeg builds, which moves eye centers by a fraction of a pixel.
const EYE_TOLERANCE_LOSSLESS_PIXELS = 0.05;
const EYE_TOLERANCE_LOSSY_PIXELS = 1.0;
// The 27 MP photo is detected at 1024 px and scaled up ~5.9x, so 1 px at detection scale.
const EYE_TOLERANCE_LARGE_PIXELS = 6.0;

const pythonReference = JSON.parse(readFileSync(join(E2E_PHOTO_DIRECTORY, "python_reference.json"), "utf8"));
const testPhotoPaths = [
  ...["heavy-reflection-glasses-before.webp", "ring-light-glare-glasses-before.webp", "sunlight-glare-glasses-before.webp"].map((name) => join(REFERENCE_PHOTO_DIRECTORY, name)),
  ...["heavy-reflection-glasses-before.png", "ring-light-glare-glasses-before.png", "sunlight-glare-glasses-before.png", "rotated-exif6.jpg", "large-27mp.jpg"].map((name) => join(E2E_PHOTO_DIRECTORY, name)),
];
const summary = { backends: {}, photos: {}, network: {}, screenshots: [] };

mkdirSync(DOWNLOAD_DIRECTORY, { recursive: true });
for (const photoPath of testPhotoPaths) {
  if (!existsSync(photoPath)) {
    console.log(`SKIP  missing private test photo ${photoPath}; run tests.app.make_browser_test_photos`);
    process.exit(0);
  }
}

const PYTHON = process.env.GLARE_OFF_PYTHON ?? "/Volumes/vega/datasets/glare-off/venv/bin/python";
let manifestCurrent = true;
try {
  execFileSync(PYTHON, ["-m", "tests.app.sync_asset_manifest", "--check"], { cwd: REPO_ROOT, stdio: "pipe" });
} catch {
  manifestCurrent = false;
}
check("app/asset_manifest.json and the sw.js version are current", manifestCurrent, "run python -m tests.app.sync_asset_manifest");

const { chromium, devices } = loadPlaywright();
const server = await serveAppDirectory();
const appOrigin = new URL(server.baseUrl).origin;
const browser = await chromium.launch({ args: ["--enable-unsafe-webgpu"] });

async function screenshot(target, name) {
  const path = join(SCREENSHOT_DIRECTORY, `${name}.png`);
  await target.screenshot({ path });
  summary.screenshots.push(path);
}

function compareFacesToPython(fileName, jobState) {
  const reference = pythonReference[fileName];
  const isLarge = fileName.startsWith("large");
  const tolerance = isLarge ? EYE_TOLERANCE_LARGE_PIXELS : fileName.endsWith(".png") ? EYE_TOLERANCE_LOSSLESS_PIXELS : EYE_TOLERANCE_LOSSY_PIXELS;
  check(`${fileName}: ${reference.faces.length} face(s) like Python`, jobState.faces.length === reference.faces.length, `found ${jobState.faces.length}`);
  let worstEyeDifference = 0;
  reference.faces.forEach((referenceFace, faceIndex) => {
    const face = jobState.faces[faceIndex];
    if (!face) return;
    worstEyeDifference = Math.max(
      worstEyeDifference,
      Math.hypot(face.imageLeftEyeXY[0] - referenceFace.image_left_eye_xy[0], face.imageLeftEyeXY[1] - referenceFace.image_left_eye_xy[1]),
      Math.hypot(face.imageRightEyeXY[0] - referenceFace.image_right_eye_xy[0], face.imageRightEyeXY[1] - referenceFace.image_right_eye_xy[1]),
    );
  });
  check(`${fileName}: eye centers within ${tolerance} px of Python`, worstEyeDifference <= tolerance, `max ${worstEyeDifference.toFixed(4)} px`);
  return worstEyeDifference;
}

const downloadCard = (page, cardIndex) => downloadCardInto(page, cardIndex, DOWNLOAD_DIRECTORY);

// ---------- 1-4: first visit, WASM, desktop light ----------
const firstContext = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light", acceptDownloads: true });
const firstRequests = recordRequests(firstContext);
const page = await firstContext.newPage();
const pageErrors = [];
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") pageErrors.push(message.text());
});
await page.goto(`${server.baseUrl}?backend=wasm&debug`);
const readyState = await waitForAppState(page, (state) => state.engineInfo);
check("first visit: engine ready on the WASM build", readyState.engineInfo.backend === "wasm", JSON.stringify(readyState.engineInfo.timings));
summary.backends.wasmFirstVisit = { ...readyState.engineInfo, crossOriginIsolated: readyState.crossOriginIsolated };
await page.waitForTimeout(300);
await screenshot(page, "desktop-light-empty");
const requestCountAtReady = firstRequests.length;
const readyAt = Date.now();

await page.setInputFiles("#file-input", testPhotoPaths);
const doneState = await waitForAppState(page, allDone, { timeoutMs: 300_000 });
let totalFaces = 0;
for (const [jobIndex, jobState] of doneState.jobs.entries()) {
  const fileName = basename(testPhotoPaths[jobIndex]);
  check(`${fileName}: processed (state ${jobState.state})`, ["done", "no-glare", "no-glasses"].includes(jobState.state));
  const worstEye = compareFacesToPython(fileName, jobState);
  totalFaces += jobState.faces.length;
  summary.photos[fileName] = {
    size: `${jobState.photoWidth}x${jobState.photoHeight}`,
    faces: jobState.faces.length,
    glareFaces: jobState.faces.filter((face) => face.hasGlare).length,
    faceSwitchStates: jobState.faces.map((face) => face.switchState),
    glassesProbabilities: jobState.faces.map((face) => Number(face.glassesProbability.toFixed(4))),
    worstEyeDifferencePx: worstEye,
    timingsWasm1Thread: jobState.timings,
  };
  checkGlassesGateDecisions(fileName, jobState); // 4a
}
check("reference WebP photos: 5 faces in total", doneState.jobs.slice(0, 3).reduce((total, job) => total + job.faces.length, 0) === 5);
const rotatedJob = doneState.jobs[testPhotoPaths.findIndex((path) => path.endsWith("rotated-exif6.jpg"))];
check("EXIF orientation 6 applied on load (stored 540x720, shown 720x540)", rotatedJob.photoWidth === 720 && rotatedJob.photoHeight === 540);
await page.waitForTimeout(900); // let the first result's sweep settle
await screenshot(page.locator(".result").first(), "desktop-light-result-card");

// 4b. Memory: face results are crop-space only; one photo at a time holds full-res regions.
const cropPlaneSize = 512 * 256;
const memoryAfterProcessing = await page.evaluate(() => window.__glareOffDebug.describeMemory());
const oversizedArrays = memoryAfterProcessing.faces.flatMap((face) => face.typedArrays.filter((array) => array.length > cropPlaneSize || (face.regionPixels > cropPlaneSize && array.length >= face.regionPixels)).map((array) => `job ${face.jobIndex} face ${face.faceIndex} ${array.key} ${array.type}[${array.length}]`));
const largestRegionPixels = Math.max(...memoryAfterProcessing.faces.map((face) => face.regionPixels));
check("memory: no face result holds an array larger than one 512x256 crop plane", oversizedArrays.length === 0, oversizedArrays.join("; ") || `largest region ${largestRegionPixels} px, ${memoryAfterProcessing.faces.reduce((total, face) => total + face.typedArrays.length, 0)} crop-size arrays over ${memoryAfterProcessing.faces.length} faces`);
check("memory: no job result keeps region pixels", memoryAfterProcessing.jobsHoldingRegionPixels.length === 0, memoryAfterProcessing.jobsHoldingRegionPixels.join(","));
const lastDoneJobIndex = doneState.jobs.map((job) => job.state).lastIndexOf("done");
check("memory: only the newest result holds full-res regions", memoryAfterProcessing.visibleRegionCacheJobIndex === lastDoneJobIndex, `job ${memoryAfterProcessing.visibleRegionCacheJobIndex}, ${(memoryAfterProcessing.visibleRegionCacheBytes / 1e6).toFixed(1)} MB`);
const cdpSession = await firstContext.newCDPSession(page);
await cdpSession.send("HeapProfiler.collectGarbage");
summary.memory = { afterProcessing: memoryAfterProcessing, heapUsageAfterGc: await cdpSession.send("Runtime.getHeapUsage") };
console.log(`INFO  heap after processing ${testPhotoPaths.length} photos and a forced GC: ${JSON.stringify(summary.memory.heapUsageAfterGc)}`);

// Downloads (one per photo with glare), size checks, bit-exactness on PNGs.
const downloadsByJob = {};
for (const [jobIndex, jobState] of doneState.jobs.entries()) {
  if (jobState.state !== "done") continue;
  const savedPath = await downloadCard(page, jobIndex);
  downloadsByJob[jobIndex] = savedPath;
  const size = await downloadedImageSize(page, savedPath);
  const fileName = basename(testPhotoPaths[jobIndex]);
  check(`${fileName}: download is full resolution (${jobState.photoWidth}x${jobState.photoHeight})`, size.width === jobState.photoWidth && size.height === jobState.photoHeight, `${basename(savedPath)} ${size.width}x${size.height}`);
  check(`${fileName}: download carries no EXIF rotation (raw decode matches)`, size.rawWidth === size.width && size.rawHeight === size.height);
  if (fileName.endsWith(".png")) {
    check(`${fileName}: PNG stays PNG`, savedPath.endsWith(".png"));
    const exactness = await checkBitExactness(page, jobIndex, testPhotoPaths[jobIndex], savedPath);
    summary.photos[fileName].bitExactness = exactness;
    check(`${fileName}: zero pixels changed outside the glare mask`, !exactness.sizeMismatch && exactness.changedOutsideMask === 0, JSON.stringify(exactness));
    if (exactness.skippedFaceAreaPixels > 0) check(`${fileName}: zero pixels changed in the skipped (no glasses) face's crop area`, exactness.changedInSkippedFaceArea === 0, `${exactness.skippedFaceAreaPixels} px area`);
    check(`${fileName}: some masked pixels did change (the patch was applied)`, exactness.changedInsideMask > 0);
  }
}

// All faces off by clicking their tiles -> the download must equal the input exactly.
const pngJobIndex = testPhotoPaths.findIndex((path) => path.endsWith("ring-light-glare-glasses-before.png"));
await checkAllFacesSwitchedOff(page, pngJobIndex, testPhotoPaths[pngJobIndex], DOWNLOAD_DIRECTORY);
// A click flips one face and the download follows it.
const heavyPngJobIndex = testPhotoPaths.findIndex((path) => path.endsWith("heavy-reflection-glasses-before.png"));
await checkClickFlipsOneFace(page, heavyPngJobIndex, testPhotoPaths[heavyPngJobIndex], downloadsByJob[heavyPngJobIndex], DOWNLOAD_DIRECTORY);

// Download all -> a ZIP.
const [zipDownload] = await Promise.all([page.waitForEvent("download", { timeout: 180_000 }), page.locator("#download-all-button").click()]);
const zipPath = join(DOWNLOAD_DIRECTORY, zipDownload.suggestedFilename());
await zipDownload.saveAs(zipPath);
const zipBytes = readFileSync(zipPath);
check("Download all produces a ZIP", zipBytes.readUInt32LE(0) === 0x04034b50, `${zipDownload.suggestedFilename()} ${(zipBytes.length / 1e6).toFixed(1)} MB`);

// Network isolation.
const foreignRequests = firstRequests.filter((request) => isNetworkUrl(request.url) && new URL(request.url).origin !== appOrigin);
const requestsAfterReady = firstRequests.slice(requestCountAtReady).filter((request) => isNetworkUrl(request.url));
summary.network = { requestsBeforeReady: requestCountAtReady, requestsAfterReady: requestsAfterReady.map((request) => request.url), foreignRequests: foreignRequests.map((request) => request.url), msFromReadyToEnd: Date.now() - readyAt };
check("network: every request went to the page's own origin", foreignRequests.length === 0, `${firstRequests.length} requests, ${foreignRequests.length} foreign`);
check("network: zero requests after the engine was ready (processing 8 photos, downloads, ZIP)", requestsAfterReady.length === 0, requestsAfterReady.map((request) => request.url).join(", "));
// CSP: a fetch to another origin is refused, from the page and from a blob-bootstrapped worker
// (the mechanism the processing worker uses to inherit this policy).
const cspProbe = await page.evaluate(async () => {
  const pageResult = await fetch("https://example.com/").then(() => "allowed", (error) => `blocked: ${error.name}`);
  const workerSource = 'fetch("https://example.com/").then(() => postMessage("allowed"), (e) => postMessage("blocked: " + e.name));';
  const worker = new Worker(URL.createObjectURL(new Blob([workerSource], { type: "text/javascript" })), { type: "module" });
  const workerResult = await new Promise((resolve) => (worker.onmessage = (event) => resolve(event.data)));
  worker.terminate();
  return { pageResult, workerResult };
});
check("CSP blocks a request to another origin from the page", cspProbe.pageResult.startsWith("blocked"), cspProbe.pageResult);
check("CSP blocks a request to another origin from a blob worker", cspProbe.workerResult.startsWith("blocked"), cspProbe.workerResult);
const cspLeak = firstRequests.filter((request) => request.url.startsWith("https://example.com"));
check("CSP-blocked requests never reached the network layer", cspLeak.length === 0, `${cspLeak.length} seen`);
pageErrors.splice(0, pageErrors.length, ...pageErrors.filter((text) => !text.includes("Content Security Policy") && !text.includes("Failed to fetch")));
check("no page errors on first visit", pageErrors.length === 0, pageErrors.join(" | "));

// Eye zoom view on the first card: it no longer holds the cache, so its regions are re-read.
await page.locator(".result").first().locator(".view-option").nth(1).click();
await page.waitForFunction(() => window.__glareOffDebug.describeMemory().visibleRegionCacheJobIndex === 0, null, { timeout: 30_000 });
await page.waitForTimeout(200);
await screenshot(page.locator(".result").first(), "desktop-light-eyes-zoom");
const memoryAfterSwitch = await page.evaluate(() => window.__glareOffDebug.describeMemory());
check("memory: touching another card moves the full-res cache to it", memoryAfterSwitch.visibleRegionCacheJobIndex === 0, `${(memoryAfterSwitch.visibleRegionCacheBytes / 1e6).toFixed(1)} MB`);

// ---------- 4c: every face skipped (?glassesthreshold=2), one forced on from its tile ----------
summary.forcedOnDemand = await checkForcedFaceOnDemand({ browser, baseUrl: server.baseUrl, appOrigin, photoPath: testPhotoPaths[heavyPngJobIndex], downloadDirectory: DOWNLOAD_DIRECTORY, screenshot });

// ---------- 5: second visit (service worker in control, cross-origin isolated) ----------
await page.goto(`${server.baseUrl}?backend=wasm&debug`);
const secondState = await waitForAppState(page, (state) => state.engineInfo);
summary.backends.wasmSecondVisit = { ...secondState.engineInfo, crossOriginIsolated: secondState.crossOriginIsolated };
check("second visit: page is controlled by the service worker", secondState.serviceWorkerControlled);
check("second visit: cross-origin isolated (COOP/COEP from the service worker)", secondState.crossOriginIsolated === true);
check("second visit: onnxruntime runs more than one thread", secondState.engineInfo.numThreads > 1, `${secondState.engineInfo.numThreads} threads`);
const secondRequestStart = firstRequests.length;
await page.setInputFiles("#file-input", [testPhotoPaths[1], join(E2E_PHOTO_DIRECTORY, "large-27mp.jpg")]);
const secondDone = await waitForAppState(page, allDone, { timeoutMs: 300_000 });
summary.photos["ring-light-glare-glasses-before.webp"].timingsWasmThreads = secondDone.jobs[0].timings;
summary.photos["large-27mp.jpg"].timingsWasmThreads = secondDone.jobs[1].timings;
check("second visit: photos processed", secondDone.jobs.every((job) => job.state === "done"));
const secondVisitNetwork = firstRequests.slice(secondRequestStart).filter((request) => isNetworkUrl(request.url));
check("second visit: processing made no network requests", secondVisitNetwork.length === 0, secondVisitNetwork.map((request) => request.url).join(", "));

// ---------- 6: offline ----------
await firstContext.setOffline(true);
await page.goto(`${server.baseUrl}?backend=wasm&debug`);
const offlineState = await waitForAppState(page, (state) => state.engineInfo, { timeoutMs: 60_000 });
check("offline: engine loads from the service worker cache", Boolean(offlineState.engineInfo));
await page.setInputFiles("#file-input", [testPhotoPaths[0]]);
const offlineDone = await waitForAppState(page, allDone, { timeoutMs: 120_000 });
check("offline: a photo is processed", offlineDone.jobs[0].state === "done");
await firstContext.setOffline(false);
await firstContext.close();

// ---------- 7: WebGPU ----------
// The default headless shell only offers SwiftShader (a software fallback adapter): the automatic
// choice must refuse it. The new headless mode (channel "chromium") reaches the real GPU.
const shellContext = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const shellPage = await shellContext.newPage();
await shellPage.goto(`${server.baseUrl}?nosw`);
await shellPage.waitForFunction(() => document.getElementById("app-name")?.textContent.length > 0); // boot() ran
const hookWithoutFlag = await shellPage.evaluate(() => typeof window.__glareOffDebug);
check("without ?debug the page exposes no test hook", hookWithoutFlag === "undefined", hookWithoutFlag);
await shellPage.goto(`${server.baseUrl}?nosw&debug`);
const shellState = await waitForAppState(shellPage, (state) => state.engineInfo, { timeoutMs: 60_000 });
const shellAdapter = await shellPage.evaluate(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  return adapter ? { architecture: adapter.info?.architecture, isFallbackAdapter: adapter.info?.isFallbackAdapter } : null;
});
check("automatic backend refuses a software fallback GPU adapter", !shellAdapter?.isFallbackAdapter || shellState.engineInfo.backend === "wasm", `${JSON.stringify(shellAdapter)} -> ${shellState.engineInfo.backend}`);
await shellContext.close();

const gpuBrowser = await chromium.launch({ channel: "chromium", args: ["--enable-unsafe-webgpu"] });
const gpuPage = await (await gpuBrowser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
await gpuPage.goto(`${server.baseUrl}?nosw&debug`);
const gpuAutoState = await waitForAppState(gpuPage, (state) => state.engineInfo, { timeoutMs: 120_000 });
summary.backends.webgpu = { ...gpuAutoState.engineInfo, note: "new headless mode, real Apple GPU (Metal)" };
if (gpuAutoState.engineInfo.backend === "webgpu") {
  check("real GPU: automatic choice picks WebGPU and the session runs on it", true, JSON.stringify(gpuAutoState.engineInfo.timings));
  await gpuPage.setInputFiles("#file-input", [testPhotoPaths[1], join(E2E_PHOTO_DIRECTORY, "large-27mp.jpg")]);
  const gpuDone = await waitForAppState(gpuPage, allDone, { timeoutMs: 300_000 });
  summary.photos["ring-light-glare-glasses-before.webp"].timingsWebgpu = gpuDone.jobs[0].timings;
  summary.photos["large-27mp.jpg"].timingsWebgpu = gpuDone.jobs[1].timings;
  check("WebGPU: photos processed", gpuDone.jobs.every((job) => job.state === "done"));
} else {
  console.log(`SKIP  WebGPU: no real GPU adapter in this browser (backend ${gpuAutoState.engineInfo.backend})`);
}
await gpuBrowser.close();

// ---------- 8: screenshots, desktop dark and phone light/dark ----------
for (const [label, contextOptions] of [
  ["desktop-dark", { viewport: { width: 1440, height: 900 }, colorScheme: "dark" }],
  ["phone-light", { ...devices["iPhone 13"], colorScheme: "light" }],
  ["phone-dark", { ...devices["iPhone 13"], colorScheme: "dark" }],
  ["phone-se-light", { ...devices["iPhone SE"], colorScheme: "light" }],
]) {
  const context = await browser.newContext({ ...contextOptions, acceptDownloads: true });
  const viewPage = await context.newPage();
  await viewPage.goto(`${server.baseUrl}?backend=wasm&nosw&keepworker&debug`);
  await waitForAppState(viewPage, (state) => state.engineInfo);
  await screenshot(viewPage, `${label}-empty`);
  await viewPage.setInputFiles("#file-input", [testPhotoPaths[1], testPhotoPaths[2]]);
  await waitForAppState(viewPage, allDone, { timeoutMs: 180_000 });
  await viewPage.waitForTimeout(900);
  await viewPage.evaluate(() => window.scrollTo(0, 0));
  await screenshot(viewPage, `${label}-results-top`);
  await screenshot(viewPage.locator(".result").first(), `${label}-result-card`);
  const overflowPixels = await viewPage.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  check(`${label}: no horizontal overflow`, overflowPixels <= 1, `${overflowPixels} px`);
  await context.close();
}

await browser.close();
await server.close();
summary.totalFacesAllPhotos = totalFaces;
writeFileSync(join(SCREENSHOT_DIRECTORY, "summary.json"), JSON.stringify(summary, null, 1));
console.log(`\nsummary and screenshots: ${SCREENSHOT_DIRECTORY}`);
finish();
