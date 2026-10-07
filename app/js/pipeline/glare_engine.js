/**
 * The processing engine's lifecycle on the page: start the worker with the right runtime build
 * and all three models, restart it on demand, serialize model work, and put it to sleep on
 * phones. Holds the one live GlareWorkerClient; main.js holds this.
 *
 * - `run(task)` queues model work: a face forced on from the face picker waits for the photo
 *   being processed, so two model runs never interleave inside the worker.
 * - `sleepWhenIdle(isQueueRunning)`: on memory-constrained devices the worker is terminated
 *   once nothing is queued (the only way to free a WASM heap); `run` restarts it from cache.
 * - `withPixelWorker(task)`: region reads and exports need no models, so they use the live
 *   worker if there is one and a throwaway model-free worker otherwise.
 */

import { MODEL_PATHS, ONNX_RUNTIME_BUILDS } from "../app_config.js";
import { isMemoryConstrainedDevice } from "./runtime_choice.js";
import { GlareWorkerClient } from "./worker_client.js";

export class GlareEngine {
  /**
   * Args:
   *   engineStatus: ui/engine_status.js line; resolveAppUrl(path) -> absolute URL;
   *   onRetried(): called after "Try again" restarts a failed engine (main.js re-runs its queue);
   *   keepWorker: never sleep (the `?keepworker` flag).
   */
  constructor({ engineStatus, resolveAppUrl, onRetried, keepWorker }) {
    this.engineStatus = engineStatus;
    this.resolveAppUrl = resolveAppUrl;
    this.onRetried = onRetried;
    this.keepWorker = keepWorker;
    this.runtime = null;
    this.assetManifest = null;
    this.workerClient = null;
    this.readyPromise = null; // Promise of the worker's "ready" message
    this.info = null;
    this.workTail = Promise.resolve();
    this.pendingTaskCount = 0;
  }

  assetBytes(relativePath) {
    return this.assetManifest.assets.find((asset) => asset.path === relativePath)?.bytes ?? 0;
  }

  /** Start the worker and load the models. `runtime`: runtime_choice.js; `assetManifest`: asset_manifest.json. */
  start(runtime = this.runtime, assetManifest = this.assetManifest) {
    this.runtime = runtime;
    this.assetManifest = assetManifest;
    const build = ONNX_RUNTIME_BUILDS[runtime.backend];
    const modelPaths = [MODEL_PATHS.faceDetector, MODEL_PATHS.glareModel, MODEL_PATHS.glassesClassifier];
    const expectedTotalBytes = [build.wasmUrl, ...modelPaths].reduce((total, path) => total + this.assetBytes(path), 0);
    let isFromCache = false;
    window.caches?.match(this.resolveAppUrl(MODEL_PATHS.glareModel)).then((cachedResponse) => (isFromCache = Boolean(cachedResponse)), () => {});
    this.workerClient = new GlareWorkerClient();
    this.engineStatus.showLoading(0, expectedTotalBytes, isFromCache);
    const startedAt = performance.now();
    this.readyPromise = this.workerClient
      .start(
        {
          ortModuleUrl: this.resolveAppUrl(build.moduleUrl),
          wasmUrl: this.resolveAppUrl(build.wasmUrl),
          faceDetectorUrl: this.resolveAppUrl(MODEL_PATHS.faceDetector),
          glareModelUrl: this.resolveAppUrl(MODEL_PATHS.glareModel),
          glassesClassifierUrl: this.resolveAppUrl(MODEL_PATHS.glassesClassifier),
          backend: runtime.backend,
          numThreads: runtime.numThreads,
          expectedTotalBytes,
        },
        {
          onProgress: (loadedBytes) => (loadedBytes >= expectedTotalBytes ? this.engineStatus.showStarting() : this.engineStatus.showLoading(loadedBytes, expectedTotalBytes, isFromCache)),
          onNotice: (notice) => console.info(notice),
        },
      )
      .then((readyMessage) => {
        this.info = { ...readyMessage, glareModelPath: MODEL_PATHS.glareModel, glassesClassifierPath: MODEL_PATHS.glassesClassifier, engineStartMs: performance.now() - startedAt };
        this.engineStatus.showReady(this.info);
        return readyMessage;
      })
      .catch((error) => {
        this.engineStatus.showError(`Could not start: ${error.message}. Reloading the page usually fixes this.`, () => {
          this.workerClient.terminate();
          this.start();
          this.onRetried();
        });
        throw error;
      });
    this.readyPromise.catch(() => {}); // surfaced through the status line and each job
    return this.readyPromise;
  }

  /** The ready promise, restarting a sleeping worker first. */
  ensure() {
    if (!this.workerClient || !this.workerClient.worker) this.start();
    return this.readyPromise;
  }

  /** Run `task(workerClient)` after every earlier model task, with the engine started. */
  run(task) {
    this.pendingTaskCount += 1;
    const taskRun = this.workTail.then(async () => {
      await this.ensure();
      return task(this.workerClient);
    });
    this.workTail = taskRun.catch(() => {});
    return taskRun.finally(() => {
      this.pendingTaskCount -= 1;
    });
  }

  sleepWhenIdle(isQueueRunning) {
    if (isQueueRunning || this.pendingTaskCount > 0 || this.keepWorker) return;
    if (isMemoryConstrainedDevice() && this.workerClient?.worker) {
      this.workerClient.terminate();
      this.engineStatus.showSleeping();
    }
  }

  async withPixelWorker(task) {
    if (this.workerClient?.worker) return task(this.workerClient);
    const pixelWorker = new GlareWorkerClient();
    pixelWorker.startWithoutModels();
    try {
      return await task(pixelWorker);
    } finally {
      pixelWorker.terminate();
    }
  }
}
