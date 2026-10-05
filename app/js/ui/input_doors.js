/**
 * The three ways photos arrive (file picker, drag-and-drop anywhere on the page, paste) and
 * nothing else. Every door ends in `onFiles(File[])` with image-like files only.
 *
 * - Drag-and-drop covers the whole window with an overlay; drags that carry no files
 *   (text, a link) are ignored.
 * - A file counts as image-like by MIME type or, when the type is empty (HEIC on Windows),
 *   by extension; the real format check happens later by sniffing bytes.
 */

const IMAGE_EXTENSION_PATTERN = /\.(jpe?g|png|webp|heic|heif|avif|gif|bmp)$/i;

function isImageLike(file) {
  return file.type.startsWith("image/") || (file.type === "" && IMAGE_EXTENSION_PATTERN.test(file.name));
}

function dragCarriesFiles(event) {
  return Array.from(event.dataTransfer?.types ?? []).includes("Files");
}

/**
 * Wire every door. `elements`: { fileInput, chooseButtons: [HTMLElement], dropTargets: [HTMLElement], dragOverlay }.
 * `onFiles(files)` receives a non-empty array; `onNothingUsable()` fires when a drop/paste had no images.
 */
export function wireInputDoors({ fileInput, chooseButtons, clickTargets, dropTargets, dragOverlay }, { onFiles, onNothingUsable }) {
  const deliver = (fileList) => {
    const imageFiles = Array.from(fileList ?? []).filter(isImageLike);
    if (imageFiles.length > 0) onFiles(imageFiles);
    else onNothingUsable?.();
  };

  for (const button of chooseButtons) {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      fileInput.click();
    });
  }
  for (const target of clickTargets) {
    target.addEventListener("click", () => fileInput.click());
  }
  fileInput.addEventListener("change", () => {
    deliver(fileInput.files);
    fileInput.value = "";
  });

  let dragDepth = 0;
  const setDragging = (isDragging) => {
    dragOverlay.hidden = !isDragging;
    for (const target of dropTargets) target.dataset.drag = isDragging ? "over" : "";
  };
  window.addEventListener("dragenter", (event) => {
    if (!dragCarriesFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    setDragging(true);
  });
  window.addEventListener("dragover", (event) => {
    if (!dragCarriesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  window.addEventListener("dragleave", (event) => {
    if (!dragCarriesFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setDragging(false);
  });
  window.addEventListener("drop", (event) => {
    if (!dragCarriesFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    setDragging(false);
    deliver(event.dataTransfer.files);
  });

  document.addEventListener("paste", (event) => {
    const pastedFiles = Array.from(event.clipboardData?.files ?? []);
    if (pastedFiles.length === 0) return; // plain text paste: not ours
    event.preventDefault();
    deliver(pastedFiles);
  });
}
