# glare-off

A free, open-source web tool that removes glare and reflections from eyeglass lenses in a photo, with all processing in the visitor's browser. The photo never leaves the device: the site is static files, there is no backend, and after the page and model load no network request is made.

Python 3.12. Absolute imports from the repo root (`from eye_crop.eye_crop_geometry import ...`), run modules with `python -m` from the repo root. Interpreter: `/Volumes/vega/datasets/glare-off/venv/bin/python`.

## How it works

1. Find each face's two eye centers (YuNet, 233 KB).
2. Cut an aligned 512x256 crop around the glasses (`eye_crop/`).
3. Run the glare model on the crop. It returns a cleaned crop and two masks.
4. Blend the cleaned crop back into the photo only where the glare mask is on. Every other pixel of the photo is untouched.

No public glare/no-glare eyeglass dataset is freely licensed, so the model trains on synthetic pairs: clean photos of people wearing glasses, with physically modeled lens glare painted on (`glare_synthesis/`). Research behind these choices: `docs/research/`.

## Directory map

| Directory | Responsibility |
|---|---|
| `eye_crop/` | Crop geometry shared by everything, plus the YuNet eye detector wrapper |
| `training_sources/` | Acquire clean source photos, license filter, lens masks, eye crops, the source manifest |
| `glare_synthesis/` | Procedural lens-glare renderer: clean crop + lens masks in, glared crop + masks out |
| `glare_model/` | Network architecture, training loop, ONNX export, configs |
| `evaluation/` | Real-photo eval set, side-by-side sheets, metrics on held-out synthetic pairs |
| `app/` | The static site (HTML/CSS/JS, onnxruntime-web, models in `app/models/`) |
| `tests/` | pytest suite |
| `docs/research/` | Research reports the design rests on |

Heavy assets never enter the repo. Datasets: `/Volumes/vega/datasets/glare-off/`. Checkpoints and third-party weights: `/Volumes/vega/ai-models/glare-off/`. Training runs: `/Volumes/vega/datasets/glare-off/runs/`.

## Contracts (change these only in one commit that updates every side)

### Eye crop
- Defined in `eye_crop/eye_crop_geometry.py`; `app/js/eye_crop_geometry.js` is a line-for-line port.
- 512 wide x 256 high, eyes level, eye midpoint at the crop center, eye distance = 0.36 of crop width.
- Eye centers come from YuNet in both Python and the browser. Training crops jitter the eye centers to cover detector noise.
- **Crop resampling** (`extract_eye_crop_with_area_prefilter`; training uses it, the app must mirror it exactly):
  - `s = hypot(A[0][0], A[0][1])` for the photo->crop affine `A` (crop px per photo px).
  - `s >= 0.5`: one bilinear warp of the photo with `A`, BORDER_REFLECT_101. Never cubic, also for upscales.
  - `s < 0.5`: `n = floor(1/s + 1e-9)` (an integer >= 2, so `s*n` is in [0.5, 1]). Shrink the photo with INTER_AREA at `fx = fy = 1/n` (OpenCV's `cv2.resize(photo, None, fx=1/n, fy=1/n, INTER_AREA)`: n x n box averages, blocks aligned to the photo origin, output size `round(W/n) x round(H/n)`). Then bilinear warp, BORDER_REFLECT_101, with `A' = A @ [[n, 0, (n-1)/2], [0, n, (n-1)/2], [0, 0, 1]]`.
  - In the app's region-based warp, the region read from the photo must start at multiples of `n` so the blocks match the whole-photo grid. Rounding the shrunk pixels to uint8 before the warp is fine (training's phone path does the same).
  - Warping layers back to the photo still uses the ORIGINAL `A` (unchanged).

### Source manifest (`training_sources/` -> everyone)
- One JSONL row per source face at `/Volumes/vega/datasets/glare-off/sources/source_manifest.jsonl`.
- Keys: `source_id`, `split` (`train`/`val`/`test`, split by person/photo so no face crosses splits), `photo_path`, `license`, `attribution`, `image_left_eye_xy`, `image_right_eye_xy`, `lens_mask_path` (uint8 PNG on the photo's pixel grid: 0 background, 1 lens on the image's left side, 2 lens on the image's right side), `source_glare_score` (0 = clean lens, higher = existing glare; sources above the documented threshold are excluded from training).

### Synthesis data_dict (`glare_synthesis/` -> `glare_model/`)
All arrays are crop-space, float32, HxWxC, sRGB-encoded values in [0, 1] unless stated.
- `clean_eye_crop` (H, W, 3): the target.
- `glare_eye_crop` (H, W, 3): the model input, clean crop with synthetic glare.
- `glare_mask` (H, W, 1): soft, 1 where glare changed the pixel noticeably.
- `lost_detail_mask` (H, W, 1): 1 where glare saturated so the underlying pixels are unrecoverable.
- `lens_mask` (H, W, 1): 1 inside either lens.
- `source_id` (str).
A share of samples carry NO glare (`glare_eye_crop == clean_eye_crop`, empty masks), and a share come from faces without glasses, so the model learns to leave clean input alone.

### Model I/O (`glare_model/` -> `app/`)
- ONNX file: `app/models/glare_removal.onnx`, opset 17, only ops in onnxruntime-web's WebGPU operator list (no GroupNorm).
- Input `glare_crop`: float32 [1, 3, H, W], RGB, sRGB values in [0, 1]. H and W are dynamic multiples of 16; the default is 256x512.
- Output `clean_crop`: float32 [1, 3, H, W] in [0, 1].
- Output `masks`: float32 [1, 2, H, W] in [0, 1]; channel 0 = glare mask, channel 1 = lost-detail mask.
- Blend (done by the caller, not the model): `result = glare_crop + glare_mask * (clean_crop - glare_crop)`, applied to the photo as a delta warped back through the crop affine.

## Rules

- Licensing: nothing trained on CelebA/CelebAMask-HQ, SHIQ, ReyeR, Face Synthetics, or the Unsplash Dataset ships. Code is MIT. Weights trained on FFHQ are CC BY-NC-SA 4.0. Every third-party file in `app/` is listed with its license in `app/THIRD_PARTY.md`.
- The app makes no request to any origin other than its own. No analytics, no CDN, no fonts from elsewhere.
- No photos of real private people in the repo. Demo images must be CC-licensed with attribution, generated, or consented.
- Public repo hygiene: describe the tool generically; no personal or situational details.
- Use `trash`, never `rm`.
- Training on this machine (16 GB M4): one training job at a time, launched with `nice -n 10`, DataLoader workers <= 4.
