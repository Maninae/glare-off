/**
 * One photo's entry in the results list: its state, the compare view, the face picker (one
 * tile per face, ui/face_picker.js), the view picker (whole photo or one face's eyes at full
 * resolution), the strength slider, the lost-detail notes, and the Download button.
 *
 * States (data-state on the <li>): queued -> working -> done | no-glare | no-glasses | no-face
 * | error. "done" = at least one face has glare to blend; a forced-on face can move a
 * no-glare/no-glasses card to done.
 *
 * The card owns each face's switch (overriddenByFace, ui/face_switch_state.js); the download
 * applies exactly the faces that are switched on and have glare (currentExportSettings).
 * It never touches the worker: `onDownload(card)`, `onRunFace(card, faceIndex)` (forced-on
 * face the gate skipped; main.js replaces result.faces[faceIndex]) and `onFacesChanged(card)`
 * go to main.js. Full-resolution face regions come from `loadRegions(card)` (the one-photo
 * cache in photo/visible_photo_regions.js), so a card holds no region-size pixels of its own.
 */

import { CompareView } from "./compare_view.js";
import { FacePicker } from "./face_picker.js";
import { isFaceSwitchedOn } from "./face_switch_state.js";
import { buildFacePatches, buildRegionCanvas, composeAfterPreview } from "../photo/photo_compose.js";

const WHOLE_PHOTO_VIEW = "whole";
const WAITING_THUMBNAIL_LONG_SIDE = 960;
const LOST_DETAIL_NOTE = "part of a lens here was pure white, so the eye under it could not be seen. That area is filled in, not recovered; check it up close.";

export class ResultCard {
  constructor(template, { fileName, cardIndex, onDownload, onRunFace, onFacesChanged, loadRegions }) {
    this.element = template.content.firstElementChild.cloneNode(true);
    this.cardIndex = cardIndex;
    this.fileName = fileName;
    this.onDownload = onDownload;
    this.onRunFace = onRunFace;
    this.onFacesChanged = onFacesChanged;
    this.loadRegions = loadRegions;
    this.redrawTicket = 0; // a slower region read must not paint over a newer redraw
    this.compareView = new CompareView(this.element.querySelector(".compare"));
    this.facePicker = new FacePicker(this.element.querySelector(".face-picker"), { onToggle: (faceIndex) => this.toggleFace(faceIndex) });
    this.downloadButton = this.element.querySelector(".result-download");
    this.controlsElement = this.element.querySelector(".result-controls");
    this.messageElement = this.element.querySelector(".result-message");
    this.notesElement = this.element.querySelector(".face-notes");
    this.metaElement = this.element.querySelector(".result-meta");
    this.strengthInput = this.element.querySelector(".strength-range");
    this.strengthOutput = this.element.querySelector(".strength-value");
    this.result = null; // the worker's "result" message, without region bytes or thumbnails
    this.overriddenByFace = [];
    this.busyByFace = [];
    this.viewPickerFaceKey = "";
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
      this.compareView.setWorkingText(stageMessage.faceCount > 1 ? `Checking face ${stageMessage.faceIndex + 1} of ${stageMessage.faceCount}` : "Checking the lenses");
    } else if (stageMessage.stage === "starting") {
      this.compareView.setWorkingText("Getting ready");
    }
  }

  showError(text) {
    this.setState("error");
    this.compareView.setWorkingText("Could not process this photo");
    this.showMessage(text);
  }

  /** The worker finished this photo. `thumbnailBitmaps`: one eye-crop picture per face. */
  showResult(result, thumbnailBitmaps) {
    this.result = result;
    const { faces, previewBitmap } = result;
    this.overriddenByFace = faces.map(() => false);
    this.busyByFace = faces.map(() => false);
    if (faces.length === 0) {
      this.setState("no-face");
      this.metaElement.textContent = `${result.photoWidth} × ${result.photoHeight} pixels, no face found`;
      this.compareView.showPlaceholder(previewBitmap, previewBitmap.width, previewBitmap.height);
      this.compareView.setWorkingText("No face found");
      this.showMessage("No face was found, so nothing was changed. The tool needs a face turned roughly toward the camera, at least about 30 pixels across.");
      return;
    }
    this.facePicker.build(faces.length, thumbnailBitmaps);
    thumbnailBitmaps.forEach((bitmap) => bitmap?.close());
    this.refreshFaces();
  }

  isFaceOn(faceIndex) {
    return isFaceSwitchedOn(this.result.faces[faceIndex].glassesDetected, this.overriddenByFace[faceIndex]);
  }

  /** A tile was pressed: flip the override; a skipped face switched on runs the glare model first. */
  async toggleFace(faceIndex) {
    if (this.busyByFace[faceIndex]) return;
    this.overriddenByFace[faceIndex] = !this.overriddenByFace[faceIndex];
    if (this.isFaceOn(faceIndex) && !this.result.faces[faceIndex].glareRun) {
      this.busyByFace[faceIndex] = true;
      this.refreshFaces();
      try {
        await this.onRunFace(this, faceIndex);
      } catch (error) {
        this.overriddenByFace[faceIndex] = !this.overriddenByFace[faceIndex];
        this.showMessage(`Could not check face ${faceIndex + 1} (${error.message}).`);
      } finally {
        this.busyByFace[faceIndex] = false;
      }
    }
    this.refreshFaces();
  }

  /** Bring tiles, notes, meta line, state, controls and the picture in line with the faces. */
  refreshFaces() {
    const { faces, previewBitmap, photoWidth, photoHeight } = this.result;
    this.facePicker.update(faces, this.overriddenByFace, this.busyByFace);
    const appliedFaceCount = faces.filter((face, faceIndex) => face.hasGlare && this.isFaceOn(faceIndex)).length;
    this.metaElement.textContent = describeOutcome(photoWidth, photoHeight, faces, appliedFaceCount);
    this.renderLostDetailNotes();
    this.onFacesChanged?.(this);
    if (!faces.some((face) => face.hasGlare)) {
      const anyFaceChecked = faces.some((face) => face.glareRun);
      this.setState(anyFaceChecked ? "no-glare" : "no-glasses");
      this.compareView.showPlaceholder(previewBitmap, previewBitmap.width, previewBitmap.height);
      this.compareView.setWorkingText(anyFaceChecked ? "No glare found" : "No glasses found");
      this.showMessage(describeNothingToRemove(faces, anyFaceChecked));
      return;
    }
    if (this.element.dataset.state !== "done") {
      this.setState("done");
      this.showMessage(null);
      this.downloadButton.disabled = false;
      this.controlsElement.hidden = false;
    }
    this.buildViewPicker();
    this.scheduleRedraw();
  }

  renderLostDetailNotes() {
    const faces = this.result.faces;
    const notes = faces
      .map((face, faceIndex) => (face.hasGlare && face.showLostDetailNote && this.isFaceOn(faceIndex) ? faceIndex : -1))
      .filter((faceIndex) => faceIndex >= 0)
      .map((faceIndex) => {
        const note = document.createElement("p");
        note.className = "face-note";
        note.textContent = faces.length === 1 ? capitalize(LOST_DETAIL_NOTE) : `Face ${faceIndex + 1}: ${LOST_DETAIL_NOTE}`;
        return note;
      });
    this.notesElement.replaceChildren(...notes);
  }

  /** "Whole photo" plus one "up close" option per face with glare; rebuilt only when that set changes. */
  buildViewPicker() {
    const glareFaceIndices = this.result.faces.map((face, faceIndex) => (face.hasGlare ? faceIndex : -1)).filter((faceIndex) => faceIndex >= 0);
    const faceKey = glareFaceIndices.join(",");
    if (faceKey === this.viewPickerFaceKey) return;
    this.viewPickerFaceKey = faceKey;
    const optionsElement = this.element.querySelector(".view-options");
    optionsElement.replaceChildren();
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
    for (const faceIndex of glareFaceIndices) {
      addOption(String(faceIndex), this.result.faces.length === 1 ? "Eyes up close" : `Face ${faceIndex + 1} up close`);
    }
  }

  scheduleRedraw() {
    if (this.redrawScheduled) return;
    this.redrawScheduled = true;
    requestAnimationFrame(() => {
      this.redrawScheduled = false;
      this.redraw();
    });
  }

  enabledByFace() {
    return this.result.faces.map((face, faceIndex) => this.isFaceOn(faceIndex));
  }

  /** Crop-space faces that are switched on and have glare, and the strength: what the download applies. */
  currentExportSettings() {
    return { faces: this.result.faces.filter((face, faceIndex) => face.hasGlare && this.isFaceOn(faceIndex)), strength: this.strength };
  }

  async redraw() {
    this.redrawTicket += 1;
    const ticket = this.redrawTicket;
    let entriesByFace;
    try {
      entriesByFace = await this.loadRegions(this);
    } catch (error) {
      this.showMessage(`Could not redraw this photo (${error.message}).`);
      return;
    }
    if (ticket !== this.redrawTicket) return;
    const { previewBitmap, photoWidth, faces } = this.result;
    const patches = buildFacePatches(faces, entriesByFace, this.enabledByFace(), this.strength);
    if (this.currentView === WHOLE_PHOTO_VIEW) {
      const afterCanvas = composeAfterPreview(this.afterPreviewCanvas, previewBitmap, photoWidth, patches);
      this.compareView.showPair(previewBitmap, afterCanvas, previewBitmap.width, previewBitmap.height);
      return;
    }
    const faceIndex = Number(this.currentView);
    const face = faces[faceIndex];
    const patch = patches.find((candidate) => candidate.faceIndex === faceIndex);
    const beforeCanvas = buildRegionCanvas(entriesByFace[faceIndex].regionOriginalRgba, face.region.width, face.region.height);
    const afterCanvas = patch ? buildRegionCanvas(patch.patchedRgba, face.region.width, face.region.height) : beforeCanvas;
    this.compareView.showPair(beforeCanvas, afterCanvas, face.region.width, face.region.height);
  }

  setDownloadBusy(isBusy) {
    this.downloadButton.disabled = isBusy;
    this.downloadButton.textContent = isBusy ? "Saving" : "Download";
  }
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function describeOutcome(photoWidth, photoHeight, faces, appliedFaceCount) {
  const size = `${photoWidth} × ${photoHeight} pixels`;
  const faceWord = faces.length === 1 ? "1 face" : `${faces.length} faces`;
  if (!faces.some((face) => face.glareRun)) return `${size}, ${faceWord}, no glasses`;
  if (!faces.some((face) => face.hasGlare)) return `${size}, ${faceWord}, no glare`;
  return `${size}, ${faceWord}, glare removed on ${appliedFaceCount}`;
}

function describeNothingToRemove(faces, anyFaceChecked) {
  if (!anyFaceChecked) {
    return faces.length === 1
      ? "No glasses were found, so the photo is unchanged. If there are glasses, press the face below to check it anyway."
      : "No glasses were found on any face, so the photo is unchanged. Press a face below to check it anyway.";
  }
  if (faces.length === 1) return "No glare was found on these lenses, so the photo is unchanged.";
  const anyFaceSkipped = faces.some((face) => !face.glareRun);
  return anyFaceSkipped ? "No glare was found on the faces with glasses, so the photo is unchanged. Press a skipped face below to check it too." : "No glare was found on any of the lenses, so the photo is unchanged.";
}
