/**
 * End-to-end checks of the glasses gate and the face picker (called by test_app_end_to_end.mjs):
 * - checkGlassesGateDecisions: the face without glasses in the heavy-reflection and ring-light
 *   photos is auto-off (glare model never ran), the faces with glasses auto-on.
 * - checkAllFacesSwitchedOff / checkClickFlipsOneFace: tile clicks drive the download.
 * - checkForcedFaceOnDemand: with `?glassesthreshold=2` every face is skipped; pressing a tile
 *   (by keyboard) runs the glare model on demand, with zero network requests, and only that
 *   face's masked pixels change in the download.
 * Tiles are always addressed by face index: aria-pressed changes under a click.
 */

import { readFileSync } from "node:fs";

import { waitForAppState } from "./browser_test_support.mjs";
import { allDone, checkBitExactness, countChangedBytesInRegion, downloadCard, isNetworkUrl, recordRequests } from "./e2e_page_checks.mjs";
import { check } from "./node_test_support.mjs";

// The face without glasses in each two-face reference photo, by its image-left eye center.
const FACES_WITHOUT_GLASSES = {
  "heavy-reflection-glasses-before": [264, 244],
  "ring-light-glare-glasses-before": [249, 471],
};
const HEAVY_REFLECTION_FACE_WITH_GLASSES = [438, 212];
const FACE_MATCH_RADIUS_PIXELS = 20;
// Above any probability, so the gate skips every face.
const SKIP_EVERY_FACE_THRESHOLD = 2;

/** Index of the face whose image-left eye is near `eyeXY`, or -1. */
function findFaceIndexByEye(faces, eyeXY) {
  return faces.findIndex((face) => Math.hypot(face.imageLeftEyeXY[0] - eyeXY[0], face.imageLeftEyeXY[1] - eyeXY[1]) < FACE_MATCH_RADIUS_PIXELS);
}

const describeJob = async (page, jobIndex) => (await page.evaluate(() => window.__glareOffDebug.describe())).jobs[jobIndex];

/** Gate decisions for one processed photo (no-op for photos without a known glasses-free face). */
export function checkGlassesGateDecisions(fileName, jobState) {
  const faceWithoutGlassesEye = FACES_WITHOUT_GLASSES[fileName.replace(/\.(webp|png)$/, "")];
  if (!faceWithoutGlassesEye) return;
  const faceIndexWithoutGlasses = findFaceIndexByEye(jobState.faces, faceWithoutGlassesEye);
  const states = jobState.faces.map((face) => `${face.switchState} p=${face.glassesProbability.toFixed(3)}`).join(", ");
  const faceWithoutGlasses = jobState.faces[faceIndexWithoutGlasses];
  check(`${fileName}: the face without glasses comes back auto-off (glare model never ran)`, faceWithoutGlasses?.switchState === "auto-off" && !faceWithoutGlasses.glareRun, states);
  check(`${fileName}: the face with glasses comes back auto-on`, jobState.faces.every((face, faceIndex) => faceIndex === faceIndexWithoutGlasses || face.switchState === "auto-on"), states);
}

/** Switch every switched-on face off by clicking: the download must equal the input; then restore. */
export async function checkAllFacesSwitchedOff(page, jobIndex, photoPath, downloadDirectory) {
  const card = page.locator(".result").nth(jobIndex);
  const switchedOnIndices = (await describeJob(page, jobIndex)).faces.map((face, faceIndex) => (face.switchedOn ? faceIndex : -1)).filter((faceIndex) => faceIndex >= 0);
  for (const faceIndex of switchedOnIndices) await card.locator(".face-tile").nth(faceIndex).click();
  const allOffPath = await downloadCard(page, jobIndex, downloadDirectory);
  const allOff = await checkBitExactness(page, jobIndex, photoPath, allOffPath);
  check("every face switched off: download is byte-identical to the input pixels", allOff.changedOutsideMask === 0 && allOff.changedInsideMask === 0, JSON.stringify(allOff));
  for (const faceIndex of switchedOnIndices) await card.locator(".face-tile").nth(faceIndex).click();
}

/** auto-on -> user-off -> auto-on on one face with glare; the download follows each state. */
export async function checkClickFlipsOneFace(page, jobIndex, photoPath, automaticDownloadPath, downloadDirectory) {
  const faces = (await describeJob(page, jobIndex)).faces;
  const flipFaceIndex = faces.findIndex((face) => face.switchState === "auto-on" && face.hasGlare);
  if (flipFaceIndex < 0) {
    check("click flip: the photo has an auto-on face with glare to flip", false, faces.map((face) => face.switchState).join(","));
    return;
  }
  // Read now: the next download of this card saves over the same file name.
  const automaticDownloadBytes = readFileSync(automaticDownloadPath);
  const flipTile = page.locator(".result").nth(jobIndex).locator(".face-tile").nth(flipFaceIndex);
  await flipTile.click();
  const afterFlip = await describeJob(page, jobIndex);
  check("click flip: the face becomes user-off and its tile un-presses", afterFlip.faces[flipFaceIndex].switchState === "user-off" && (await flipTile.getAttribute("aria-pressed")) === "false", await flipTile.getAttribute("aria-label"));
  const flippedPath = await downloadCard(page, jobIndex, downloadDirectory);
  const flipped = await checkBitExactness(page, jobIndex, photoPath, flippedPath);
  const changedInFlippedRegion = await countChangedBytesInRegion(page, photoPath, flippedPath, faces[flipFaceIndex].region);
  check("click flip: the flipped-off face's region is byte-identical in the download", changedInFlippedRegion === 0 && flipped.changedOutsideMask === 0, `${changedInFlippedRegion} bytes differ; ${JSON.stringify(flipped)}`);
  check("click flip: the download changed versus the automatic one", readFileSync(flippedPath).compare(automaticDownloadBytes) !== 0);
  await flipTile.click();
  const restored = (await describeJob(page, jobIndex)).faces[flipFaceIndex].switchState;
  check("click flip: a second click returns the face to auto-on", restored === "auto-on", restored);
}

/**
 * Every face skipped, one forced on by keyboard. `photoPath`: the heavy-reflection PNG.
 * Returns a summary for summary.json.
 */
export async function checkForcedFaceOnDemand({ browser, baseUrl, appOrigin, photoPath, downloadDirectory, screenshot }) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: "light", acceptDownloads: true });
  const requests = recordRequests(context);
  const page = await context.newPage();
  await page.goto(`${baseUrl}?backend=wasm&nosw&keepworker&glassesthreshold=${SKIP_EVERY_FACE_THRESHOLD}&debug`);
  await waitForAppState(page, (state) => state.engineInfo);
  const requestCountAtReady = requests.length;
  await page.setInputFiles("#file-input", [photoPath]);
  const skipped = await waitForAppState(page, allDone, { timeoutMs: 120_000 });
  const skippedJob = skipped.jobs[0];
  check("threshold 2: every face skipped, card state no-glasses, glare model never ran", skippedJob.state === "no-glasses" && skippedJob.faces.every((face) => face.switchState === "auto-off" && !face.glareRun), skippedJob.faces.map((face) => face.switchState).join(","));
  const card = page.locator(".result").first();
  check("threshold 2: Download is disabled while nothing would change", await card.locator(".result-download").isDisabled());
  await page.waitForTimeout(300);
  await screenshot(card, "desktop-light-all-faces-skipped");

  const forcedFaceIndex = findFaceIndexByEye(skippedJob.faces, HEAVY_REFLECTION_FACE_WITH_GLASSES);
  const forcedTile = card.locator(".face-tile").nth(forcedFaceIndex);
  const tileLabel = await forcedTile.getAttribute("aria-label");
  check("tiles are buttons with aria-pressed and a text alternative", (await forcedTile.evaluate((element) => element.tagName)) === "BUTTON" && (await forcedTile.getAttribute("aria-pressed")) === "false" && /skipped/.test(tileLabel), tileLabel);
  await forcedTile.focus();
  await page.keyboard.press("Enter");
  const forcedState = await waitForAppState(page, (state) => state.jobs[0].faces[forcedFaceIndex].glareRun, { timeoutMs: 60_000 });
  const forcedJob = forcedState.jobs[0];
  const forcedFace = forcedJob.faces[forcedFaceIndex];
  check("force on (keyboard): the face becomes user-on and the glare model ran on demand", forcedFace.switchState === "user-on" && forcedFace.switchedOn, `hasGlare ${forcedFace.hasGlare}`);
  await page.waitForTimeout(400);
  await screenshot(card, "desktop-light-face-forced-on");
  const forcedTileLabel = await forcedTile.locator(".face-tile-label").innerText();
  check("force on: the tile reads FORCED ON", /^FORCED ON/.test(forcedTileLabel), forcedTileLabel.replace(/\n/g, " / "));
  if (forcedFace.hasGlare) {
    check("force on: the card turns into a before/after result with Download enabled", forcedJob.state === "done" && !(await card.locator(".result-download").isDisabled()));
    const forcedPath = await downloadCard(page, 0, downloadDirectory);
    const exactness = await checkBitExactness(page, 0, photoPath, forcedPath);
    check("force on: the download changes the forced face's masked pixels and nothing else (the other face stays skipped)", exactness.changedInsideMask > 0 && exactness.changedOutsideMask === 0 && exactness.changedInSkippedFaceArea === 0 && exactness.skippedFaceAreaPixels > 0, JSON.stringify(exactness));
  } else {
    check("force on: no glare found, the card says so and stays unchanged", forcedJob.state === "no-glare" && /NO GLARE FOUND/.test(forcedTileLabel));
  }
  const requestsAfterReady = requests.slice(requestCountAtReady).filter((request) => isNetworkUrl(request.url));
  const foreignRequests = requests.filter((request) => isNetworkUrl(request.url) && new URL(request.url).origin !== appOrigin);
  check("force on: zero network requests after ready (processing, on-demand run, download)", requestsAfterReady.length === 0, requestsAfterReady.map((request) => request.url).join(", "));
  check("force on: zero requests to another origin", foreignRequests.length === 0, `${requests.length} requests`);
  await context.close();
  return { faceIndex: forcedFaceIndex, switchState: forcedFace.switchState, hasGlare: forcedFace.hasGlare, onDemandFacePassMs: forcedJob.timings.onDemandFacePassMs };
}
