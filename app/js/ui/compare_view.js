/**
 * Before/after compare: two stacked canvases, the "after" clipped to the right of a divider
 * (red edge on the before side, green on the after side, a plain bar for the handle). A
 * transparent native range input covers the whole view, so dragging anywhere, tapping, and
 * the keyboard (arrows, Home/End) all work with no custom pointer code, and screen readers
 * get a real slider.
 *
 * No motion here: the divider rests at the middle. The page's one fade (the after side
 * appearing) is CSS keyed on `data-ready`, and prefers-reduced-motion switches it off.
 */

const RESTING_PERCENT = 50;

function drawSourceOnto(canvas, source, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.drawImage(source, 0, 0, width, height);
}

export class CompareView {
  constructor(rootElement) {
    this.rootElement = rootElement;
    this.beforeCanvas = rootElement.querySelector(".compare-before");
    this.afterCanvas = rootElement.querySelector(".compare-after");
    this.rangeInput = rootElement.querySelector(".compare-range");
    this.workingText = rootElement.querySelector(".compare-working-text");
    this.rangeInput.addEventListener("input", () => this.applySplit(Number(this.rangeInput.value)));
  }

  applySplit(percent) {
    this.rootElement.style.setProperty("--split", `${percent}%`);
    this.rangeInput.setAttribute("aria-valuetext", `${Math.round(100 - percent)}% after, ${Math.round(percent)}% before`);
  }

  setWorkingText(text) {
    this.workingText.textContent = text;
  }

  /** Size the view to the photo's aspect (CSS keeps it within the window height). */
  setAspect(width, height) {
    this.rootElement.style.setProperty("--photo-aspect", `${width} / ${height}`);
    this.rootElement.style.setProperty("--photo-aspect-number", String(width / height));
  }

  /** Show the photo alone (while it is processed, or when there is nothing to compare). */
  showPlaceholder(source, width, height) {
    this.setAspect(width, height);
    drawSourceOnto(this.beforeCanvas, source, width, height);
  }

  /** Show a before/after pair of drawable sources (canvas or ImageBitmap) of one size. */
  showPair(beforeSource, afterSource, width, height) {
    this.setAspect(width, height);
    drawSourceOnto(this.beforeCanvas, beforeSource, width, height);
    drawSourceOnto(this.afterCanvas, afterSource, width, height);
    if (this.rootElement.dataset.ready !== "true") {
      this.rangeInput.value = String(RESTING_PERCENT);
      this.applySplit(RESTING_PERCENT);
    }
    this.rootElement.dataset.ready = "true";
  }

  /** Redraw only the "after" side (strength or toggle changed). */
  updateAfter(afterSource) {
    drawSourceOnto(this.afterCanvas, afterSource, this.afterCanvas.width, this.afterCanvas.height);
  }

  /** Give canvas memory back (Safari keeps released canvases around otherwise). */
  release() {
    for (const canvas of [this.beforeCanvas, this.afterCanvas]) {
      canvas.width = 0;
      canvas.height = 0;
    }
  }
}
