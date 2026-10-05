/**
 * Browser-test plumbing: a static file server for app/ (what GitHub Pages does: files as-is,
 * no special headers) and Playwright loaded from the out-of-repo tools directory.
 */

import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { extname, join, normalize } from "node:path";

import { APP_DIRECTORY, NODE_TOOLS_DIRECTORY } from "./node_test_support.mjs";

const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".md": "text/markdown; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".woff2": "font/woff2",
};

export function loadPlaywright() {
  return createRequire(join(NODE_TOOLS_DIRECTORY, "package.json"))("playwright");
}

/** Serve app/ on a free localhost port. Resolves to { baseUrl, close(), servedPaths }. */
export function serveAppDirectory(rootDirectory = APP_DIRECTORY) {
  const servedPaths = [];
  const server = createServer((request, response) => {
    const urlPath = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    let filePath = normalize(join(rootDirectory, urlPath));
    if (!filePath.startsWith(rootDirectory)) {
      response.writeHead(403).end();
      return;
    }
    try {
      if (statSync(filePath).isDirectory()) filePath = join(filePath, "index.html");
      const fileSize = statSync(filePath).size;
      servedPaths.push(urlPath);
      response.writeHead(200, { "Content-Type": CONTENT_TYPES[extname(filePath)] ?? "application/octet-stream", "Content-Length": fileSize, "Cache-Control": "no-cache" });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404, { "Content-Type": "text/plain" }).end("not found");
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({ baseUrl: `http://127.0.0.1:${port}/`, servedPaths, close: () => new Promise((done) => server.close(done)) });
    });
  });
}

/** Poll window.__glareOffDebug.describe() until `predicate(state)` is true. */
export async function waitForAppState(page, predicate, { timeoutMs = 120_000, pollMs = 100 } = {}) {
  const started = Date.now();
  for (;;) {
    const state = await page.evaluate(() => window.__glareOffDebug?.describe());
    if (state && predicate(state)) return state;
    if (Date.now() - started > timeoutMs) throw new Error(`timed out waiting for app state; last: ${JSON.stringify(state)?.slice(0, 600)}`);
    await page.waitForTimeout(pollMs);
  }
}
