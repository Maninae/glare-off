/**
 * The face picker under a photo's compare view: one tile per detected face, showing that
 * face's eye crop (a thumbnail the worker made) and its state as a small chart label.
 *
 * - Each tile is a real <button> with aria-pressed (pressed = glare removal switched on for
 *   that face) and an aria-label ("Face 2: glare removed, press to leave this face as it was").
 * - The picker only draws; ResultCard owns the switch state (ui/face_switch_state.js) and
 *   calls `update` after every change. A click calls `onToggle(faceIndex)`.
 */

import { describeFaceTile, faceSwitchState, isFaceSwitchedOn } from "./face_switch_state.js";

export class FacePicker {
  constructor(rootElement, { onToggle }) {
    this.rootElement = rootElement;
    this.listElement = rootElement.querySelector(".face-tiles");
    this.onToggle = onToggle;
    this.tiles = [];
  }

  /** One tile per face. `thumbnailBitmaps[faceIndex]`: the eye-crop ImageBitmap (or null). */
  build(faceCount, thumbnailBitmaps) {
    this.listElement.replaceChildren();
    this.tiles = [];
    for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "face-tile";
      const cropCanvas = document.createElement("canvas");
      cropCanvas.className = "face-tile-crop";
      cropCanvas.setAttribute("aria-hidden", "true");
      const thumbnail = thumbnailBitmaps[faceIndex];
      if (thumbnail) {
        cropCanvas.width = thumbnail.width;
        cropCanvas.height = thumbnail.height;
        cropCanvas.getContext("2d").drawImage(thumbnail, 0, 0);
      }
      const nameElement = document.createElement("span");
      nameElement.className = "face-tile-name";
      nameElement.textContent = `Face ${faceIndex + 1}`;
      const labelElement = document.createElement("span");
      labelElement.className = "face-tile-label";
      const frameElement = document.createElement("span");
      frameElement.className = "face-tile-frame"; // the ink frame stays solid when the crop fades
      frameElement.append(cropCanvas);
      button.append(frameElement, nameElement, labelElement);
      button.addEventListener("click", () => this.onToggle(faceIndex));
      item.append(button);
      this.listElement.append(item);
      this.tiles.push({ button, labelElement, nameElement });
    }
    this.rootElement.hidden = faceCount === 0;
  }

  /** Redraw every tile from the face results and the card's switch state. Single-face photos hide the visible name. */
  update(faces, overriddenByFace, busyByFace) {
    faces.forEach((face, faceIndex) => {
      const tile = this.tiles[faceIndex];
      if (!tile) return;
      const overridden = overriddenByFace[faceIndex];
      const isBusy = busyByFace[faceIndex];
      const switchedOn = isFaceSwitchedOn(face.glassesDetected, overridden);
      const labelLines = describeFaceTile(face, overridden, isBusy);
      tile.labelElement.replaceChildren(...labelLines.flatMap((line, lineIndex) => (lineIndex === 0 ? [line] : [document.createElement("br"), line])));
      tile.nameElement.hidden = faces.length === 1;
      tile.button.setAttribute("aria-pressed", String(switchedOn));
      tile.button.setAttribute("aria-busy", String(isBusy));
      tile.button.disabled = isBusy;
      tile.button.dataset.state = faceSwitchState(face.glassesDetected, overridden);
      tile.button.dataset.switchedOn = String(switchedOn);
      // The label contains the visible text (label-in-name) plus which face and what a press does.
      const pressHint = isBusy ? "" : switchedOn ? ", press to leave this face as it was" : ", press to remove glare on this face";
      tile.button.setAttribute("aria-label", `Face ${faceIndex + 1}: ${labelLines.join(", ").toLowerCase()}${pressHint}`);
    });
  }
}
