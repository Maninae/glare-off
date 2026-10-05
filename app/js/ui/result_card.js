/**
 * One photo's entry in the results list: its state, the compare view, the view picker
 * (whole photo or one face's eyes at full resolution), the strength slider, per-face
 * keep/revert toggles, the lost-detail note, and the Download button.
 *
 * States (data-state on the <li>): queued -> working -> done | no-face | no-glare | error.
 * The card never touches the worker; it calls `onDownload(card)` and main.js does the rest.
 */

import { CompareView } from "./compare_view.js";
import { buildFacePatches, buildRegionCanvas, composeAfterPreview } from "../photo/photo_compose.js";

const WHOLE_PHOTO_VIEW = "whole";
const WAITING_THUMBNAIL_LONG_SIDE = 960;

export class ResultCard {
  constructor(template, { fileName, cardIndex, onDownload }) {
    this.element = template.content.firstElementChild.cloneNode(true);
    this.cardIndex = cardIndex;
    this.fileName = fileName;
    this.onDownload = onDownload;
    this.compareView = new CompareView(this.element.querySelector(".compare"));
    this.downloadButton = this.element.querySelector(".result-download");
    this.controlsElement = this.element.querySelector(".result-controls");
    this.messageElement = this.element.querySelector(".result-message");
    this.metaElement = this.element.querySelector(".result-meta");
    this.strengthInput = this.element.querySelector(".strength-range");
    this.strengthOutput = this.element.querySelector(".strength-value");
    this.result = null; // the worker's "result" message
    this.enabledByFace = [];
    this.strength = 1;
    this.currentView = WHOLE_PHOTO_VIEW;
    this.afterPreviewCanvas = new OffscreenCanvas(1, 1);
    this.redrawScheduled = false;

    this.element.querySelector(".result-name").textContent = fileName;
    this.element.querySelector(".strength-range").id = `strength-${cardIndex}`;
    this.downloadButton.setAttribute("aria-label", `Download ${fileName} without glare`);
    this.downloadButton.addEventListener("click", () => this.onDownload(this));
    this.strengthInput.addEventListener("input", () => {
      this.strength = Number(this.strengthInput.value) / 100;
      this.strengthOutput.textContent = `${this.strengthInput.value}%`;
      this.scheduleRedraw();
    });
    this.setState("queued");
    this.compareView.setWorkingText("Waiting for the photos before it");
  }

  /** A quick low-resolution look at the photo while it waits; best effort (HEIC may not decode). */
  async showWaitingThumbnail(file) {
    try {
      const probe = await createImageBitmap(file, { imageOrientation: "from-image" });
      const scale = Math.min(1, WAITING_THUMBNAIL_LONG_SIDE / Math.max(probe.width, probe.height));
      const thumbnail = await createImageBitmap(probe, { resizeWidth: Math.max(1, Math.round(probe.width * scale)), resizeHeight: Math.max(1, Math.round(probe.height * scale)), resizeQuality: "medium" });
      probe.close();
      if (!this.result) this.compareView.showPlaceholder(thumbnail, thumbnail.width, thumbnail.height);
      thumbnail.close();
    } catch {
      // The worker reports undecodable files with a proper message.
    }
  }

  setState(state) {
    this.element.dataset.state = state;
  }

  showMessage(text) {
    this.messageElement.hidden = !text;
    this.messageElement.textContent = text ?? "";
  }

  /** Worker stage messages -> the working caption on the photo. */
  showStage(stageMessage) {
    this.setState("working");
    if (stageMessage.stage === "detecting") {
      this.metaElement.textContent = `${stageMessage.photoWidth} × ${stageMessage.photoHeight} pixels`;
      this.compareView.setWorkingText("Finding faces");
    } else if (stageMessage.stage === "cleaning") {
      this.compareView.setWorkingText(stageMessage.faceCount > 1 ? `Cleaning the lenses, face ${stageMessage.faceIndex + 1} of ${stageMessage.faceCount}` : "Cleaning the lenses");
    } else if (stageMessage.stage === "starting") {
      this.compareView.setWorkingText("Getting ready");
    }
  }

  showError(text) {
    this.setState("error");
    this.compareView.setWorkingText("Could not process this photo");
    this.showMessage(text);
  }

  /** The worker finished this photo. */
  showResult(result) {
    this.result = result;
    const { faces, photoWidth, photoHeight, previewBitmap } = result;
    this.enabledByFace = faces.map((face) => face.hasGlare);
    const glareFaceCount = faces.filter((face) => face.hasGlare).length;
    this.metaElement.textContent = describeOutcome(photoWidth, photoHeight, faces.length, glareFaceCount);
    if (faces.length === 0) {
      this.setState("no-face");
      this.compareView.showPlaceholder(previewBitmap, previewBitmap.width, previewBitmap.height);
      this.compareView.setWorkingText("No face found");
      this.showMessage("No face was found, so nothing was changed. The tool needs a face turned roughly toward the camera, at least about 30 pixels across.");
      return;
    }
    if (glareFaceCount === 0) {
      this.setState("no-glare");
      this.compareView.showPlaceholder(previewBitmap, previewBitmap.width, previewBitmap.height);
      this.compareView.setWorkingText("No glare found");
      this.showMessage(faces.length === 1 ? "No glare was found on these lenses (or no glasses), so the photo is unchanged." : "No glare was found on any of the lenses, so the photo is unchanged.");
      return;
    }
    this.setState("done");
    this.downloadButton.disabled = false;
    this.controlsElement.hidden = false;
    this.buildViewPicker();
    this.buildFaceList();
    this.redraw();
  }

  buildViewPicker() {
    const optionsElement = this.element.querySelector(".view-options");
    const groupName = `view-${this.cardIndex}`;
    const addOption = (value, label) => {
      const optionLabel = document.createElement("label");
      optionLabel.className = "view-option";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = groupName;
      radio.value = value;
      radio.checked = value === this.currentView;
      radio.addEventListener("change", () => {
        this.currentView = value;
        this.redraw();
      });
      const text = document.createElement("span");
      text.textContent = label;
      optionLabel.append(radio, text);
      optionsElement.append(optionLabel);
    };
    addOption(WHOLE_PHOTO_VIEW, "Whole photo");
    const glareFaceIndices = this.result.faces.map((face, faceIndex) => (face.hasGlare ? faceIndex : -1)).filter((faceIndex) => faceIndex >= 0);
    for (const faceIndex of glareFaceIndices) {
      addOption(String(faceIndex), glareFaceIndices.length === 1 ? "Eyes up close" : `Face ${faceIndex + 1} up close`);
    }
  }

  buildFaceList() {
    const listElement = this.element.querySelector(".face-list");
    const faces = this.result.faces;
    faces.forEach((face, faceIndex) => {
      const item = document.createElement("li");
      item.className = "face-item";
      const row = document.createElement("div");
      row.className = "face-item-row";
      const title = document.createElement("span");
      title.className = "face-item-title";
      title.textContent = faces.length === 1 ? "Glasses" : `Face ${faceIndex + 1}`;
      row.append(title);
      if (face.hasGlare) {
        const toggle = document.createElement("label");
        toggle.className = "face-toggle";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.addEventListener("change", () => {
          this.enabledByFace[faceIndex] = checkbox.checked;
          this.scheduleRedraw();
        });
        toggle.append(checkbox, document.createTextNode("Remove glare"));
        row.append(toggle);
      } else {
        const status = document.createElement("span");
        status.className = "face-item-status";
        status.textContent = "No glare found";
        row.append(status);
      }
      item.append(row);
      if (face.hasGlare && face.showLostDetailNote) {
        const note = document.createElement("p");
        note.className = "face-note";
        note.textContent = "Part of a lens here was pure white, so the eye under it could not be seen. That area is filled in, not recovered; check it up close.";
        item.append(note);
      }
      listElement.append(item);
    });
  }

  scheduleRedraw() {
    if (this.redrawScheduled) return;
    this.redrawScheduled = true;
    requestAnimationFrame(() => {
      this.redrawScheduled = false;
      this.redraw();
    });
  }

  /** Current patches at the current strength and toggles (also what the download uses). */
  currentPatches() {
    return buildFacePatches(this.result.faces, this.enabledByFace, this.strength);
  }

  redraw() {
    const patches = this.currentPatches();
    const { previewBitmap, photoWidth, faces } = this.result;
    if (this.currentView === WHOLE_PHOTO_VIEW) {
      const afterCanvas = composeAfterPreview(this.afterPreviewCanvas, previewBitmap, photoWidth, patches);
      this.compareView.showPair(previewBitmap, afterCanvas, previewBitmap.width, previewBitmap.height);
      return;
    }
    const faceIndex = Number(this.currentView);
    const face = faces[faceIndex];
    const patch = patches.find((candidate) => candidate.faceIndex === faceIndex);
    const beforeCanvas = buildRegionCanvas(face.regionOriginalRgba, face.region.width, face.region.height);
    const afterCanvas = patch ? buildRegionCanvas(patch.patchedRgba, face.region.width, face.region.height) : beforeCanvas;
    this.compareView.showPair(beforeCanvas, afterCanvas, face.region.width, face.region.height);
  }

  setDownloadBusy(isBusy) {
    this.downloadButton.disabled = isBusy;
    this.downloadButton.textContent = isBusy ? "Saving" : "Download";
  }
}

function describeOutcome(photoWidth, photoHeight, faceCount, glareFaceCount) {
  const size = `${photoWidth} × ${photoHeight} pixels`;
  if (faceCount === 0) return `${size}, no face found`;
  const faceWord = faceCount === 1 ? "1 face" : `${faceCount} faces`;
  if (glareFaceCount === 0) return `${size}, ${faceWord}, no glare`;
  return `${size}, ${faceWord}, glare removed on ${glareFaceCount}`;
}
