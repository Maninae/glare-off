/**
 * Main-thread handle on glare_worker.js: starts it, turns its messages into Promises.
 *
 * Started from a blob: bootstrap (`import "<worker url>"`) so the page's CSP also governs the
 * worker; see glare_worker.js. `terminate()` frees every byte of model and WASM memory, which
 * main.js does on phones once the queue drains (a WASM heap never shrinks otherwise).
 */

const WORKER_MODULE_URL = new URL("./glare_worker.js", import.meta.url).href;

export class GlareWorkerClient {
  constructor() {
    this.worker = null;
    this.nextRequestId = 1;
    this.pendingRequests = new Map(); // requestId -> { resolve, reject, onStage }
    this.readyPromise = null;
  }

  /**
   * Start the worker and load both models.
   * `initOptions`: { ortModuleUrl, wasmUrl, faceDetectorUrl, glareModelUrl, backend, numThreads, expectedTotalBytes }.
   * `onProgress(loadedBytes, expectedTotalBytes)` fires per downloaded chunk.
   * Resolves to { backend, numThreads, crossOriginIsolated, timings }.
   */
  start(initOptions, { onProgress, onNotice } = {}) {
    const bootstrapSource = `import ${JSON.stringify(WORKER_MODULE_URL)};\n`;
    const bootstrapUrl = URL.createObjectURL(new Blob([bootstrapSource], { type: "text/javascript" }));
    this.worker = new Worker(bootstrapUrl, { type: "module", name: "glare-worker" });
    this.readyPromise = new Promise((resolve, reject) => {
      this.worker.onmessage = (event) => {
        const message = event.data;
        if (message.type === "progress") return onProgress?.(message.loadedBytes, message.expectedTotalBytes);
        if (message.type === "notice") return onNotice?.(message.message);
        if (message.type === "ready") {
          URL.revokeObjectURL(bootstrapUrl);
          return resolve(message);
        }
        if (message.type === "init-failed") {
          URL.revokeObjectURL(bootstrapUrl);
          return reject(new Error(message.message));
        }
        return this.handleRequestMessage(message);
      };
      this.worker.onerror = (event) => {
        event.preventDefault?.();
        const error = new Error(`worker failed: ${event.message || "script error"}`);
        reject(error);
        this.rejectAll(error);
      };
    });
    this.worker.postMessage({ type: "init", ...initOptions });
    return this.readyPromise;
  }

  /** A worker for export only: no runtime, no models (encoding needs neither). */
  startWithoutModels() {
    const bootstrapUrl = URL.createObjectURL(new Blob([`import ${JSON.stringify(WORKER_MODULE_URL)};\n`], { type: "text/javascript" }));
    this.worker = new Worker(bootstrapUrl, { type: "module", name: "glare-export-worker" });
    this.worker.onmessage = (event) => this.handleRequestMessage(event.data);
    this.worker.onerror = (event) => this.rejectAll(new Error(`worker failed: ${event.message || "script error"}`));
    setTimeout(() => URL.revokeObjectURL(bootstrapUrl), 10_000);
  }

  handleRequestMessage(message) {
    const pending = this.pendingRequests.get(message.requestId);
    if (!pending) return;
    if (message.type === "stage") {
      pending.onStage?.(message);
      return;
    }
    this.pendingRequests.delete(message.requestId);
    if (message.type === "error") {
      const error = new Error(message.message);
      error.code = message.code;
      pending.reject(error);
      return;
    }
    pending.resolve(message);
  }

  request(payload, { transfer = [], onStage } = {}) {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject, onStage });
      this.worker.postMessage({ ...payload, requestId }, transfer);
    });
  }

  /** Detect faces and clean every one. Resolves to the worker's "result" message. */
  processPhoto(file, { previewLongSide, highResPassEnabled, onStage }) {
    return this.request({ type: "process-photo", file, previewLongSide, highResPassEnabled }, { onStage });
  }

  /** Original RGBA bytes of photo rects (fresh decode). Resolves to an array, one per rect. */
  async readRegions(file, regions) {
    const { regionsRgba } = await this.request({ type: "read-regions", file, regions });
    return regionsRgba;
  }

  /**
   * Encode the full-resolution result. `faces`: the crop-space face results to apply (copied,
   * not transferred); the worker warps and blends them at `strength` against a fresh decode.
   */
  exportPhoto(file, faces, strength, mimeType, quality) {
    return this.request({ type: "export-photo", file, faces, strength, mimeType, quality });
  }

  rejectAll(error) {
    for (const pending of this.pendingRequests.values()) pending.reject(error);
    this.pendingRequests.clear();
  }

  terminate() {
    this.worker?.terminate();
    this.worker = null;
    this.readyPromise = null;
    this.rejectAll(new Error("worker stopped"));
  }
}
