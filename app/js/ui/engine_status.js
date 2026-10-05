/**
 * The one-line engine status under the drop zone: download progress in real megabytes while
 * the runtime and the two networks load, then a quiet "ready" line, or an error with Retry.
 */

function formatMegabytes(byteCount) {
  return (byteCount / 1e6).toFixed(1);
}

export class EngineStatusLine {
  constructor({ rootElement, textElement, fillElement, retryButton }) {
    this.rootElement = rootElement;
    this.textElement = textElement;
    this.fillElement = fillElement;
    this.retryButton = retryButton;
  }

  showLoading(loadedBytes, expectedTotalBytes, isFromCache) {
    this.rootElement.dataset.tone = "loading";
    this.retryButton.hidden = true;
    const fraction = expectedTotalBytes > 0 ? Math.min(1, loadedBytes / expectedTotalBytes) : 0;
    this.fillElement.style.width = `${(fraction * 100).toFixed(1)}%`;
    const verb = isFromCache ? "Loading" : "Downloading";
    this.textElement.textContent = `Getting ready. ${verb} the tools, ${formatMegabytes(loadedBytes)} of ${formatMegabytes(expectedTotalBytes)} MB (once, then kept on this device)`;
  }

  showStarting() {
    this.rootElement.dataset.tone = "loading";
    this.fillElement.style.width = "100%";
    this.textElement.textContent = "Getting ready. Starting up";
  }

  showReady(engineInfo) {
    this.rootElement.dataset.tone = "ready";
    this.retryButton.hidden = true;
    const where = engineInfo.backend === "webgpu" ? "using your graphics chip" : `using ${engineInfo.numThreads === 1 ? "one processor core" : `${engineInfo.numThreads} processor cores`}`;
    this.textElement.textContent = `Ready, ${where}. Works offline from now on.`;
  }

  showSleeping() {
    this.rootElement.dataset.tone = "ready";
    this.textElement.textContent = "Ready. Memory freed until the next photo.";
  }

  showError(message, onRetry) {
    this.rootElement.dataset.tone = "error";
    this.textElement.textContent = message;
    this.retryButton.hidden = false;
    this.retryButton.onclick = onRetry;
  }
}
