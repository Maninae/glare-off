/**
 * Page-level helpers shared by the end-to-end test scripts: request recording, downloads, and
 * the in-page pixel comparisons (original photo vs a download) the bit-exactness checks use.
 * Every helper that reads app state goes through `window.__glareOffDebug` (`?debug` pages only).
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** Every request a browser context makes, as { url, at, type }, appended live. */
export function recordRequests(context) {
  const requests = [];
  context.on("request", (request) => requests.push({ url: request.url(), at: Date.now(), type: request.resourceType() }));
  return requests;
}

export function isNetworkUrl(url) {
  return url.startsWith("http:") || url.startsWith("https:");
}

/** waitForAppState predicate: every photo left the queue. */
export const allDone = (state) => state.jobs.length > 0 && state.jobs.every((job) => !["queued", "working"].includes(job.state));

/** Click a card's Download button and save the file into `downloadDirectory`. Returns the saved path. */
export async function downloadCard(page, cardIndex, downloadDirectory) {
  const card = page.locator(".result").nth(cardIndex);
  const [download] = await Promise.all([page.waitForEvent("download", { timeout: 120_000 }), card.locator(".result-download").click()]);
  const savedPath = join(downloadDirectory, `${cardIndex}-${download.suggestedFilename()}`);
  await download.saveAs(savedPath);
  return savedPath;
}

/**
 * In the page: decode the original and the download and count changed pixels against the
 * warped glare masks of the faces that are switched on, and inside the crop area of every face
 * the glare model never ran on (the glasses gate skipped it).
 */
export async function checkBitExactness(page, jobIndex, originalPath, downloadedPath) {
  return page.evaluate(
    async ({ jobIndex, originalBase64, downloadedBase64 }) => {
      const decode = async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes]), { imageOrientation: "from-image", premultiplyAlpha: "none" });
        const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0);
        return { width: bitmap.width, height: bitmap.height, pixels: context.getImageData(0, 0, bitmap.width, bitmap.height).data };
      };
      const original = await decode(originalBase64);
      const downloaded = await decode(downloadedBase64);
      if (original.width !== downloaded.width || original.height !== downloaded.height) return { sizeMismatch: true };
      const insideMask = new Uint8Array(original.width * original.height);
      const faces = window.__glareOffDebug.describe().jobs[jobIndex].faces;
      const skippedFaceArea = new Uint8Array(original.width * original.height);
      for (const face of faces) {
        if (face.glareRun) continue;
        const { x, y, width, height } = face.cropCoverageRegion;
        for (let row = 0; row < height; row += 1) skippedFaceArea.fill(1, (y + row) * original.width + x, (y + row) * original.width + x + width);
      }
      for (const [faceIndex, face] of faces.entries()) {
        if (!face.hasGlare || !face.switchedOn) continue;
        const regionGlareMask = window.__glareOffDebug.warpGlareMaskToRegion(jobIndex, faceIndex);
        const { x, y, width, height } = face.region;
        for (let row = 0; row < height; row += 1) {
          for (let column = 0; column < width; column += 1) {
            if (regionGlareMask[row * width + column] > 0) insideMask[(y + row) * original.width + x + column] = 1;
          }
        }
      }
      let changedOutsideMask = 0;
      let changedInsideMask = 0;
      let changedInSkippedFaceArea = 0;
      let maskedPixelCount = 0;
      for (let pixel = 0; pixel < insideMask.length; pixel += 1) {
        let differs = false;
        for (let channel = 0; channel < 4; channel += 1) if (original.pixels[pixel * 4 + channel] !== downloaded.pixels[pixel * 4 + channel]) differs = true;
        if (insideMask[pixel]) maskedPixelCount += 1;
        if (differs && insideMask[pixel]) changedInsideMask += 1;
        if (differs && !insideMask[pixel]) changedOutsideMask += 1;
        if (differs && skippedFaceArea[pixel] && !insideMask[pixel]) changedInSkippedFaceArea += 1;
      }
      const skippedFaceAreaPixels = skippedFaceArea.reduce((total, value) => total + value, 0);
      return { sizeMismatch: false, changedOutsideMask, changedInsideMask, changedInSkippedFaceArea, skippedFaceAreaPixels, maskedPixelCount, totalPixels: insideMask.length };
    },
    { jobIndex, originalBase64: readFileSync(originalPath).toString("base64"), downloadedBase64: readFileSync(downloadedPath).toString("base64") },
  );
}

/** In the page: how many RGBA bytes differ between the original and a download inside one photo rect. */
export async function countChangedBytesInRegion(page, originalPath, downloadedPath, region) {
  return page.evaluate(
    async ({ originalBase64, downloadedBase64, region }) => {
      const decode = async (base64) => {
        const bitmap = await createImageBitmap(new Blob([Uint8Array.from(atob(base64), (character) => character.charCodeAt(0))]), { premultiplyAlpha: "none" });
        const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d", { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0);
        return context.getImageData(region.x, region.y, region.width, region.height).data;
      };
      const [original, downloaded] = await Promise.all([decode(originalBase64), decode(downloadedBase64)]);
      return original.reduce((count, value, index) => count + (value !== downloaded[index] ? 1 : 0), 0);
    },
    { originalBase64: readFileSync(originalPath).toString("base64"), downloadedBase64: readFileSync(downloadedPath).toString("base64"), region },
  );
}

/** Decoded size of a download, with and without EXIF orientation (they match when none is left). */
export async function downloadedImageSize(page, downloadedPath) {
  return page.evaluate(async (base64) => {
    const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const blob = new Blob([bytes]);
    const bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
    // A JPEG that still carried an EXIF orientation would decode differently with "none".
    const rawBitmap = await createImageBitmap(blob, { imageOrientation: "none" });
    return { width: bitmap.width, height: bitmap.height, rawWidth: rawBitmap.width, rawHeight: rawBitmap.height };
  }, readFileSync(downloadedPath).toString("base64"));
}
