/**
 * Shared plumbing for the app's Node test scripts (tests/app/test_*.mjs).
 *
 * - Every test is a plain script: it prints each check and exits non-zero on the first failure.
 * - Tool locations outside the repo (onnxruntime-web for Node, Playwright) are resolved here,
 *   overridable by env var, so no node_modules ever lands in the repo.
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const TESTS_APP_DIRECTORY = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = resolve(TESTS_APP_DIRECTORY, "..", "..");
export const APP_DIRECTORY = join(REPO_ROOT, "app");
export const FIXTURE_DIRECTORY = join(TESTS_APP_DIRECTORY, "fixtures");
export const SCRATCH_DIRECTORY = process.env.GLARE_OFF_SCRATCH ?? "/Volumes/vega/datasets/glare-off/scratch";
export const PRIVATE_FIXTURE_DIRECTORY = join(SCRATCH_DIRECTORY, "app-fixtures");
// `npm install --prefix <dir> onnxruntime-web@1.30.0 playwright@1.58.0` (see app/CLAUDE.md, "Tests").
export const NODE_TOOLS_DIRECTORY = process.env.GLARE_OFF_NODE_TOOLS ?? join(SCRATCH_DIRECTORY, "node-tools");

let failureCount = 0;

/** Record one named check; prints PASS/FAIL with detail. */
export function check(description, passed, detail = "") {
  const suffix = detail ? `  (${detail})` : "";
  console.log(`${passed ? "PASS" : "FAIL"}  ${description}${suffix}`);
  if (!passed) failureCount += 1;
}

/** Exit with a non-zero status if any check failed. */
export function finish() {
  if (failureCount > 0) {
    console.log(`\n${failureCount} check(s) failed`);
    process.exit(1);
  }
  console.log("\nall checks passed");
}

export function readJsonFixture(fileName) {
  return JSON.parse(readFileSync(join(FIXTURE_DIRECTORY, fileName), "utf8"));
}

/** Decode a base64 fixture field into a typed array of the given constructor. */
export function decodeBase64Array(base64Text, TypedArrayConstructor) {
  const bytes = Buffer.from(base64Text, "base64");
  const alignedCopy = new Uint8Array(bytes.length);
  alignedCopy.set(bytes);
  return new TypedArrayConstructor(alignedCopy.buffer, 0, bytes.length / TypedArrayConstructor.BYTES_PER_ELEMENT);
}

/** HxWx3 BGR bytes -> RGBA bytes (alpha 255), the layout the app works in. */
export function bgrToRgba(bgrBytes, width, height) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let pixelIndex = 0; pixelIndex < width * height; pixelIndex += 1) {
    rgba[pixelIndex * 4] = bgrBytes[pixelIndex * 3 + 2];
    rgba[pixelIndex * 4 + 1] = bgrBytes[pixelIndex * 3 + 1];
    rgba[pixelIndex * 4 + 2] = bgrBytes[pixelIndex * 3];
    rgba[pixelIndex * 4 + 3] = 255;
  }
  return rgba;
}

/** Import onnxruntime-web's Node build from the out-of-repo tools directory. */
export async function importOnnxRuntimeForNode() {
  return import(join(NODE_TOOLS_DIRECTORY, "node_modules", "onnxruntime-web", "dist", "ort.node.min.mjs"));
}
