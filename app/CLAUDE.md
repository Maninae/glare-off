# app/: the Glare Off static site

The whole deployed artifact: plain HTML/CSS/ES-module JS, no framework, no bundler, no build step, no node_modules. Serve this directory as-is. Product contracts (eye crop, model I/O) live in the repo-root `CLAUDE.md`; this file covers the site.

## Pipeline (all in one Web Worker)

```mermaid
flowchart LR
  F[File] -->|createImageBitmap, EXIF applied| B[bitmap in worker]
  B -->|strips, INTER_AREA port, long side <= 1024| D[YuNet on WASM]
  D -->|eye centers, by image x| C[512x256 eye crop per face]
  C --> G{glasses classifier: p >= threshold?}
  G -->|no: skip, keep the crop| K[face picker: NO GLASSES, SKIPPED]
  K -->|visitor forces it on: run-face| M
  G -->|yes| M[glare model: WebGPU or WASM]
  M -->|crop-space delta + mask, region rect| P[page: face results stay crop-space]
  P -->|visible photo only| V[region cache: original bytes + warped delta; blend at strength]
  P -->|faces switched on + strength| E[worker: re-decode, re-read regions, warp, blend, paste, encode]
```

## Module map

| Path | Owns |
|---|---|
| `index.html` | Page shell, the CSP meta tag (commented in place), every static DOM node, the result-card `<template>`. |
| `js/main.js` | Wiring: service-worker control, runtime choice, engine start, the sequential photo queue, the visible-photo region cache, downloads and "Download all". |
| `js/debug_hook.js` | `window.__glareOffDebug` (e2e test hook), installed only with `?debug` in the URL: state, timings, on-demand warped masks, a memory description. |
| `js/app_config.js` | Product name, model and runtime paths, switches (`HIGH_RES_PASS_ENABLED`), export qualities. |
| `js/eye_crop_geometry.js` | **Contract.** Line-for-line port of `eye_crop/eye_crop_geometry.py` (affine, scale, inverse). |
| `js/pipeline/glare_worker.js` | The module worker: loads ORT + the three sessions, decodes photos, reads pixels in strips/regions, runs detection and gated face passes (+ an eye-crop thumbnail per face), runs a skipped face on demand (`run-face`), re-reads regions for the page (`read-regions`), exports. |
| `js/pipeline/glare_engine.js` | The engine's page-side lifecycle: start/restart the worker with all three models, serialize model work (`run`), sleep on phones, the model-free pixel worker for reads and exports. |
| `js/pipeline/worker_client.js` | Main-thread side: blob bootstrap (so the CSP governs the worker), request ids to Promises, terminate. |
| `js/pipeline/runtime_choice.js` | WebGPU (real adapters only) vs WASM; thread count (iPhone/iPad 1, other mobile <= 2); which devices free memory between batches. |
| `js/pipeline/yunet_face_decode.js` | YuNet pre/post-processing ported from OpenCV FaceDetectorYN (pad to 32, BGR 0-255, anchor decode, sqrt(cls*obj), integer-box NMS). |
| `js/pipeline/area_downscale.js` | OpenCV INTER_AREA port, streamed over row strips (the detector's downscale for big photos). |
| `js/pipeline/eye_crop_warp.js` | Photo -> crop (bilinear, reflect-101) and crop layers -> photo region (bilinear, zero border). |
| `js/pipeline/glare_crop_pass.js` | One face: crop (`extractFaceEyeCrop`, reusable), model, crop-space delta with the half-level cut, optional 2x pass. Returns crop-space layers + the region rect only. |
| `js/pipeline/glasses_gate_pass.js` | The glasses gate: crop once, classify, glare pass only when `p >= GLASSES_PROBABILITY_THRESHOLD`; a skipped face keeps its crop as three planes; `runSkippedFaceGlarePass` runs it later. |
| `js/pipeline/eye_crop_thumbnail.js` | The face picker's thumbnail of an eye crop (ImageBitmap, made in the worker). |
| `js/blend/face_patch_blend.js` | `round(original + strength * 255 * delta)`; zero delta keeps the exact byte. |
| `js/blend/face_region_patch.js` | On-demand warp of a face's crop-space delta/mask to its photo region, and the full-res patch (used by the cache, the export and the test hook). |
| `js/photo/photo_file_types.js` | Format sniffing (HEIC etc.), output format choice, download names. |
| `js/photo/photo_compose.js` | Full-res patches from the region cache, the composed "after" preview, region canvases for the eye zoom. |
| `js/photo/visible_photo_regions.js` | The ONE-photo cache of full-res region bytes + warped deltas (the photo last landed or touched); switching photos drops it and re-reads through a worker. |
| `js/photo/zip_store.js` | STORE-only ZIP writer with CRC-32 for "Download all". |
| `js/ui/input_doors.js` | File picker, whole-page drag-and-drop, paste. |
| `js/ui/engine_status.js` | The status line: real-MB download progress, ready, error with Try again. |
| `js/ui/compare_view.js` | Before/after canvases, the red/green divider with its plain handle, the transparent range input that drives it. No animation (the one fade is CSS). |
| `js/ui/result_card.js` | One photo: states, owns each face's override, face picker, view picker (whole photo / eyes up close), strength, lost-detail notes, Download. Redraws are async (they may wait for a region re-read). |
| `js/ui/face_picker.js` | The tile row under the compare view: one `<button aria-pressed>` per face with its eye crop and a chart label. Draws only. |
| `js/ui/face_switch_state.js` | The per-face state machine and tile labels (pure; tested in Node). |
| `sw.js` | Service worker: shell precache, runtime caching of models and ORT, COOP/COEP/CORP headers on everything it serves. |
| `asset_manifest.json` | Generated list of every served file with sizes; feeds the SW precache, the cache version and the progress bar. |
| `styles/tokens.css`, `page.css`, `results.css` | Tokens (the only color values, light and dark, and the `@font-face`), the eye chart and its collapsed header row, the contact sheet. |
| `fonts/` | Optician Sans (OFL 1.1) woff2 and its license text. Self-hosted; see `THIRD_PARTY.md`. |
| `models/` | `face_detection_yunet_2023mar_dynamic_input.onnx` (served), the stock YuNet (Python only, not served), `glare_removal.onnx` (owned by `glare_model/`; see its README), `glasses_classifier.onnx` (owned by `glasses_classifier/`; input `eye_crop` [1,3,256,512], output `glasses_probability` [1,1]). |
| `vendor/onnxruntime-web-1.30.0/` | Two ORT builds, verbatim from the npm tarball. See `THIRD_PARTY.md`. |
| `icons/`, `manifest.webmanifest`, `404.html` | Favicon set rendered from `icons/favicon.svg` (black tile, white chart glasses), install manifest, link preview card (a crop of the chart's top rows). |

## The glasses gate and the face picker

Every detected face is cropped once; the classifier and the glare model get the same tensor. Per-face state (`ui/face_switch_state.js`), from the gate's decision and one override bit a tile press flips:

| State | Glasses detected | Overridden | Glare model | In the blend and download |
|---|---|---|---|---|
| `auto-on` | yes | no | ran at processing | yes (label GLARE REMOVED, or NO GLARE FOUND if its mask was empty) |
| `auto-off` | no | no | never ran | no, pixels untouched (NO GLASSES, SKIPPED) |
| `user-on` | no | yes | runs on demand from the kept crop (`run-face`), label WORKING meanwhile | yes (FORCED ON) |
| `user-off` | yes | yes | ran | no (FORCED OFF) |

- A second press returns to the automatic state. A face is blended only when switched on AND its glare mask is non-empty; `currentExportSettings()` sends exactly those faces, so the download follows the picker.
- Card states: `done` (some face has glare), `no-glare` (faces checked, none had glare), `no-glasses` (every face skipped; Download disabled until a forced face finds glare), `no-face`.
- `GLASSES_PROBABILITY_THRESHOLD` lives in `js/app_config.js`; `?glassesthreshold=N` overrides it (the e2e uses 2 to skip every face).

## Invariants (do not relax while iterating)

- **No other origin, ever.** CSP `default-src 'none'`, `connect-src 'self'`. The page's privacy row ("Once loaded, it makes no network requests") is pinned by the e2e test; any change that makes it false is out of scope.
- **The worker must start from the blob bootstrap** in `worker_client.js`. A worker loaded from its own URL takes its CSP from HTTP headers, and static hosts send none.
- **Untouched pixels stay byte-identical.** A crop pixel whose mask is below 0.5/255, or whose `mask * (clean - crop)` is below half a level on all three channels, gets delta 0 and mask 0 (and does not count toward "glare found" or the lost-detail note). So an idle mask head (~0.02 everywhere) touches nothing by construction, the warped delta is exactly 0 outside the mask, and `blendFacePatch` leaves those bytes alone. Downloads are a fresh decode of the original with only the patched regions pasted.
- **Per-photo memory does not scale with the photo.** Face results keep crop-space layers only (~2 MB per glare face at 512x256; a skipped face keeps its 1.5 MB crop). Full-res region bytes and warped deltas (~16 B per region pixel) exist for one photo at a time (`visible_photo_regions.js`) and inside the worker during an export. The e2e test asserts no face result holds a region-size array.
- **Faithful ports.** Geometry, YuNet decode, INTER_AREA and the warps reproduce the Python/OpenCV reference. Change the Python first, regenerate fixtures, re-run the parity tests; never "improve" the JS alone. One non-obvious detail: OpenCV takes the rotation center as float32, so the JS rounds it with `Math.fround`.
- **Nothing depends on the glare model's behavior.** The app reads only the contract shapes; the stub and the trained model are interchangeable.
- **No pixels in storage.** Photos live in memory only (worker bitmaps, page canvases). Nothing goes to localStorage or IndexedDB.
- **Bump the manifest after any file change:** `python -m tests.app.sync_asset_manifest` (the e2e test fails while it is stale). It stamps the cache version into `sw.js`.
- **Never use the word "AI" as a selling point** in UI copy.

## Runtime decisions

- **Build choice:** WebGPU build only when `requestAdapter()` returns a real GPU. Software fallbacks (SwiftShader, `isFallbackAdapter`) are refused: in testing they ran the model ~50x slower than threaded WASM. Otherwise the CPU-only build (14 MB vs 27 MB of WASM). If the WebGPU session fails to create, the same build runs on its CPU path. `?backend=wasm|webgpu` overrides.
- **YuNet always on WASM** (its input size changes per photo; on WebGPU every size recompiles shaders).
- **Multithreading: yes, via our own service worker, without a reload.** `sw.js` adds COOP/COEP to everything it serves, so from the second visit the page is cross-origin isolated and ORT uses up to 4 threads. Measured gain: ~2.2x per face. The usual coi-serviceworker shim forces a reload on first visit; we skip that, so the first visit simply runs one thread. iPhone/iPad stay at one thread (Safari tab memory limits); every other mobile user agent is capped at 2. `?threads=N` overrides (uncapped, tests only).
- **Memory:** on phones, small tablets and devices reporting <= 4 GB, the worker is terminated when the queue drains (the only way to free a WASM heap); downloads then use a model-free worker. ORT memory arenas are off.
- **Big photos:** the worker never holds a full-size pixel buffer. Detection streams 128-row strips through the INTER_AREA port; crops read only their source region. Export needs one full-size canvas (browser canvas limits apply, see Known limits).

## Measured timings (M4 Mac, Chromium, untrained stub NAFNet 2.93M params, 512x256 crop)

| Runtime | Inference per face | 720 px photo, 2 faces | 27 MP photo, 2 faces |
|---|---|---|---|
| WASM, 1 thread (first visit) | ~350 ms | ~0.8 s | ~1.4 s |
| WASM, 4 threads (second visit) | ~150 ms | ~0.4 s | ~1.0-1.2 s |
| WebGPU, Apple GPU | ~70 ms | ~0.3 s | ~0.8-1.0 s |

Engine start: ~0.5-1 s on WASM, ~1-2.3 s on WebGPU (shader compile), once per visit. On the 27 MP photo, ~400 ms is the INTER_AREA downscale for detection.

## Download size (first visit; later visits come from the cache)

| Part | Raw | gzip |
|---|---|---|
| App shell (31 files) | 0.17 MB | 0.08 MB |
| YuNet | 0.23 MB | 0.20 MB |
| Glare model (stub, fp32) | 11.8 MB | 10.7 MB |
| Glasses classifier (fp16) | 0.37 MB | |
| ORT CPU build (or) | 14.3 MB | 3.7 MB |
| ORT WebGPU build | 26.9 MB | 6.7 MB |
| **Total, CPU path / GPU path** | **26.5 / 39.1 MB** | **14.7 / 17.7 MB** |

The glare model dominates the gzip total; the fp16 export (`glare_removal_fp16.onnx`, about half) is the next lever.

## Design: the eye chart

The page IS a Snellen chart: everyone who wears glasses has stood in front of one. Owner's rule: commit to it; do not drift back toward cards, pills, serif headlines or a soft accent.

- **Canvas:** white `#ffffff`, ink `#111`; dark mode is the chart inverted (`#000` ground, `#f4f4f4` ink) via `prefers-color-scheme`, no toggle. One centered column (`--chart-width` 820 px); the contact sheet widens to `--sheet-width` 1240 px.
- **Rows shrink as you read down**, each a 3-column grid `[gutter | centered text | acuity label]`: the glasses glyph (row 1, the "big E"), the headline, DROP A PHOTO ANYWHERE ON THIS PAGE with the CHOOSE PHOTOS block, the two promise rows, then one fact per row down to 20/10. Row sizes use container units so each row keeps its line count; one line per row on desktop. Acuity labels (20/200 ... 20/10) are the one wink: small gray system sans, `aria-hidden`, hidden below 480 px. The engine status is the bottom-most tiny gray row.
- **Type:** Optician Sans (built from eye-chart optotypes), uppercase via CSS, tracked `0.12em`, for every chart row and control label. System sans, small, gray, sentence case only for acuity labels, sublabels, status, file meta, and numbers. The glyph's stroke is a fifth of the lens height, matching the letters.
- **Color:** black and white only. The duochrome red `#e8380d` / green `#0f9d58` (black letters on them, as on a chart's duochrome panel) appear in exactly one place: the BEFORE/AFTER blocks and the divider's two edges. Errors and notes are plain ink. Focus: 2 px ink outline.
- **Forbidden:** borders on containers, radii, shadows, gradients, pills, icons other than the glasses glyph, custom-drawn toggles. The one solid control is a black rectangle with white chart letters (CHOOSE PHOTOS, DOWNLOAD); secondary actions are underlined chart text (ADD PHOTOS, DOWNLOAD ALL, TRY AGAIN).
- **Results:** the chart collapses to a header row (glyph, GLARE OFF, ADD PHOTOS, DOWNLOAD ALL); photos form a contact sheet (two-up when wide, `auto-fit`), each in a 2 px ink frame drawn with `outline` so the photo keeps its exact aspect. Controls are small uppercase chart rows with native radios and range (`accent-color: ink`). Under each compare view, the face picker: 96 px eye-crop tiles in a 2 px ink `outline`, a chart label below; a switched-off face fades its crop to 40% and its label to gray. No color, no radius.
- **Motion:** none except the divider following the pointer and a 320 ms fade of the cleaned side when a result lands (off under reduced motion).
- **Whole page is the drop target** (window listeners in `input_doors.js`); row 3 (`#drop-zone`) also opens the picker on click; paste works anywhere.
- Accessibility: the compare is a real `<input type=range>`; chart text stays sentence case in the DOM (screen readers do not spell out capitals); text contrast >= 4.5:1 (gray `#767676` / `#8c8c8c`, black on red 5.0:1, on green 6.0:1); tap targets >= 44 px.

## Empty state and the sample slot

There are no demo photos yet (licensing). `#sample-slot` in chart row 3 is hidden and empty; to add "Try a sample", put CC-licensed or consented images under `samples/`, list them with attribution in `THIRD_PARTY.md`, render a button into the slot that feeds the file through `addPhotos([file])` in `main.js`, and re-run the manifest sync.

## Run locally

```
cd app
python3 -m http.server 8000
```

Open `http://localhost:8000/`. Query flags: `?backend=wasm|webgpu`, `?threads=N`, `?nosw` (no service worker), `?hires` (2x crop pass), `?keepworker` (never free the engine), `?glassesthreshold=N` (override the gate), `?debug` (install the test hook).

## Tests (all in `tests/app/`, plain scripts that print PASS/FAIL and exit non-zero on failure)

Tools live outside the repo: `npm install --no-save --prefix /Volumes/vega/datasets/glare-off/scratch/node-tools onnxruntime-web@1.30.0 playwright@1.58.0` (override with `GLARE_OFF_NODE_TOOLS`). Python: `/Volumes/vega/datasets/glare-off/venv/bin/python`, from the repo root.

| Script | Checks |
|---|---|
| `python -m tests.app.generate_app_fixtures` | Writes committed parity fixtures (`tests/app/fixtures/`) and private YuNet references (vega). |
| `node tests/app/test_eye_crop_geometry_parity.mjs` | Affine == Python to 1e-6 (46 eye pairs, 1x and 2x); INTER_AREA within 1 level of cv2; crop warp and warp-back vs cv2. |
| `node tests/app/test_yunet_decode_parity.mjs` | JS YuNet decode vs OpenCV FaceDetectorYN on identical pixels (eye centers, scores, every box/landmark). |
| `node tests/app/test_glasses_gate.mjs` | The switch state machine and tile labels; a two-face photo with a fake classifier (glasses / none): the glare model never sees the skipped face, its crop area stays byte-identical in the download, forcing it on reuses the kept crop, threshold boundary. |
| `node tests/app/test_blend_and_zip.mjs` | Blend bit-exactness, the whole face pass with a fake model (incl. an idle 0.02 mask with a sub-half-level delta: zero changed pixels), crop-size-only face results, the 2x pass switch, the ZIP writer (`unzip -t`). |
| `python -m tests.app.make_browser_test_photos` | Private e2e photos on vega (PNG copies, EXIF-rotated JPEG, 27 MP JPEG) + Python detections. |
| `node tests/app/test_app_end_to_end.mjs` | Real headless Chromium (`?debug` on every load): all photos, Python parity, full-res downloads, PNG bit-exactness (incl. a skipped face's crop area), the glasses gate (the glasses-free face in two photos is `auto-off`; a tile click flips a face and the download follows; with every face skipped, a forced face runs on demand with zero requests), memory (no region-size arrays in face results, one-photo region cache, heap after GC), zero foreign and zero post-ready requests, CSP blocking, no test hook without `?debug`, SW isolation + threads, offline, WebGPU on the real GPU, screenshots (`/tmp/glare-off-e2e/`) and `summary.json` with timings. |
| `python -m tests.app.make_yunet_dynamic_input_model` | Regenerates the served YuNet (symbolic H/W) and checks it against the stock file. |
| `python -m tests.app.sync_asset_manifest [--check]` | Regenerates (or verifies) `asset_manifest.json` and the `sw.js` version. |

## Known limits

- **Wide gamut:** photos are decoded to sRGB canvases. Display-P3 iPhone photos lose their extra gamut on export (colors outside sRGB clip). Fix path: P3 canvases where supported, converting to sRGB only for the model.
- **Transparent PNGs:** canvases premultiply alpha, so semi-transparent pixels can shift by a level on export. Opaque photos are exact.
- **Canvas size limits:** export draws one full-size canvas. Older iOS Safari caps canvas area at 16.7 MP (iOS 18: 67 MP); a larger photo shows "could not save" rather than a downscaled file. Not tested on a real iPhone.
- **HEIC:** works only where the browser decodes it (Safari). Elsewhere the card explains how to get a JPEG. No WASM HEIC decoder is bundled.
- **Untested here:** Safari, Firefox, real iOS/Android devices, onnxruntime-web WebGPU on iOS (its iOS support issue is open; the CPU path is the floor there).
- **Crop sampling:** the crop is bilinear without a prefilter (matching the Python training crops), so very large faces alias in the crop. The 2x pass (`HIGH_RES_PASS_ENABLED`) is wired and tested but off until the model is trained at 1024x512.
- **Cache on update:** a new app version re-downloads `glare_removal.onnx` (models sit in the versioned app cache); ORT files are kept across versions.
- **Link preview:** `og:image` is a relative path; set it to the absolute deployed URL when the host is known.
- **Glasses gate threshold:** 0.3 (classifier docs default to 0.5). Missed glasses faces scored 0.37-0.38 on test (one a heavy-glare close-up); the highest bare face on val scored 0.09. A missed glasses face loses its fix, while a bare face let through only costs one glare run that returns an empty mask.
- **Stub model:** the current `glare_removal.onnx` is untrained; it flags bright pixels, so the lost-detail note fires on ring-light reflections. That is the stub, not the app.
