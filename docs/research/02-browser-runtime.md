# 02 — Browser runtime, face locating, prior art, and hosting (researched 2026-10-05)

## Takeaway (5 lines)

1. **onnxruntime-web 1.30.0 (Sep 14 2026, MIT) is still the pragmatic runtime for a custom image-to-image CNN.** WebGPU covers Conv, ConvTranspose, Resize, DepthToSpace, LayerNorm, InstanceNorm and attention primitives; GroupNormalization is not in the WebGPU op table, so do not use it (use LayerNorm-over-channels or InstanceNorm, as NAFNet already does).
2. **Run the restoration net only on an aligned eye-region crop (256x256 to 512x256), never on the full photo.** Then blend a predicted residual back into the full-resolution original; untouched pixels stay byte-identical except for the final encode.
3. **Face locating: YuNet ONNX (232,589 bytes, MIT) on the same onnxruntime-web session gives eye centers for free; MediaPipe Face Landmarker (3.76 MB .task + 11.8 MB wasm, Apache-2.0) is the upgrade path if we need lens outlines from the 478-point mesh.** No browser-ready, permissively licensed eyeglass-lens segmenter exists off the shelf.
4. **iPhone is the hard target.** WebGPU is on by default in iOS 26 Safari, but onnxruntime-web's own iOS support issue is still open, WASM threads need cross-origin isolation (coi-serviceworker works only in `require-corp` mode on Safari), and Safari kills tabs at a few hundred MB. Design for single-threaded WASM on iPhone as the floor.
5. **The tool appears to be novel.** GitHub and Hacker News searches found no free, open-source, in-browser eyeglass-glare remover; every hit is either a server-upload SaaS with credits or a local paid desktop plugin (Retouch4me).

## Recommended stack

| Decision | Recommendation | Why |
|---|---|---|
| Runtime | onnxruntime-web 1.30.x, `onnxruntime-web/webgpu` build, fallback `['webgpu','wasm']` | Proven in our prior app; custom-model friendly; WebGPU EP is the supported path (JSEP and WebGL deprecated) |
| Face locator | YuNet 2023mar ONNX (0.23 MB) + 5-point similarity alignment | Same runtime, tiny, MIT; eye centers + inter-ocular distance are enough for a tight rotated crop |
| Optional refinement | MediaPipe Face Landmarker, self-hosted (`FilesetResolver.forVisionTasks('/vendor/mediapipe/')`) | Only if we need per-lens polygons; costs ~15.5 MB extra download |
| Restoration model budget | **3 to 8 MB on disk (fp16 weights), hard cap 15 MB**; ~1-4 M params; Conv/ConvTranspose/DepthToSpace/LayerNorm/SimpleGate only; static input shape | Keeps first load under ~10 s on mobile, fits iPhone memory, avoids WebGPU fallbacks |
| Input resolution | Aligned crop at **512x256** (both lenses, wide aspect) at full source resolution when the eye region is that small; downscale crop to 512x256 if larger, predict residual + soft mask, upsample residual to crop size and apply | Bounded compute regardless of photo size |
| Latency target (estimate, not measured) | Laptop WebGPU: ~20-80 ms per crop; laptop WASM 4 threads: ~0.3-1 s; recent iPhone WASM 1 thread: ~1-4 s; plus one-time model compile/shader warm-up 0.2-1 s | Extrapolated from published ISNet / Real-ESRGAN numbers below, scaled by FLOPs; **must be benchmarked on our model** |
| Hosting | Same-origin GitHub Pages (or Cloudflare Pages), models committed as plain files (not LFS) under ~25 MB each, cached by our service worker | Keeps the "zero third-party requests" promise; per-file cap is 100 MiB on GitHub, 25 MiB on Cloudflare Pages |
| Isolation | coi-serviceworker in `require-corp` mode (not `credentialless`) | Credentialless COEP is not supported by WebKit; require-corp works on Safari and costs nothing when all assets are same-origin |

## 1. onnxruntime-web today

| Topic | Finding | Source |
|---|---|---|
| Latest version | onnxruntime-web 1.30.0, published 2026-09-14, MIT; dev 1.31.0-dev.20260918 | npm registry (`registry.npmjs.org/onnxruntime-web`, queried 2026-10-05); [ORT v1.30.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0) |
| Direction of travel | "onnxruntime-web has announced the deprecation of WebGL and JSEP. The native WebGPU EP is the recommended path going forward." (v1.29.0 notes) | [ORT v1.29.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0) |
| 1.30 WebGPU changes | Prepacked Conv weights for im2col-matmul path, extended conv optimizations | [newreleases summary of v1.30.0](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.30.0) |
| WASM binary sizes (1.30.0) | `ort-wasm-simd-threaded.wasm` 14.2 MB; `.jsep.wasm` 28.3 MB; `.asyncify.wasm` 26.8 MB; `.jspi.wasm` 16.8 MB (uncompressed; GitHub Pages serves gzip) | [unpkg onnxruntime-web/dist](https://app.unpkg.com/onnxruntime-web/files/dist) |
| WebGPU op coverage | Conv and ConvTranspose supported ("need perf optimization; 3d not supported; need implementing activation"); Resize supported except `align_corners` with downsampling; LayerNormalization, InstanceNormalization, DepthToSpace (PixelShuffle), MatMul, Softmax, Sigmoid, Gelu, Erf, Pad, Concat, Add, Mul, ReduceMean supported; MultiHeadAttention supported without mask/past-present; **GroupNormalization not listed** | [webgpu-operators.md](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md) |
| Fallback behavior | Unsupported ops fall back to CPU silently and kill performance; check with `env.logLevel='verbose'`, `env.debug=true`. Example: Clip falling back on mobilenet-v2 | [ORT issue #25274](https://github.com/microsoft/onnxruntime/issues/25274); [ORT issue #24475](https://github.com/microsoft/onnxruntime/issues/24475) |
| fp16 | Op table does not document dtype support (could not verify a per-op fp16 matrix). Practitioners ship fp16 weights: img.ly's ISNet fp16 (84 MB vs 168 MB fp32) ran ~100 ms on WebGPU; QUINT8 "produced visible artifacts" on image output | [img.ly blog](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/) |
| GPU IO binding / graph capture | Supported: keep tensors on GPU (`preferredOutputLocation`), graph capture for static-shape models where all kernels run on WebGPU | [ORT WebGPU tutorial](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) |
| WASM threads | Default `numThreads` = min(hardwareConcurrency/2, 4); "Only when the browser supports WebAssembly multi-threading and `crossOriginIsolated` mode is enabled, multi-threading will be enabled." Proxy worker cannot be used with WebGPU and fails under strict CSP | [ORT env flags docs](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html) |
| Cost of no isolation | Same model: 2,133 ms with isolation and 4 threads vs 6,318 ms without isolation (silent fallback to 1 thread) | [dev.to: numThreads silently falls back](https://dev.to/iterandum/onnxruntime-web-numthreads-silently-falls-back-to-1-without-coopcoep-6318-ms-vs-2133-ms-8i2) |
| GitHub Pages + COOP/COEP | GitHub Pages cannot set custom headers ([community #13309](https://github.com/orgs/community/discussions/13309)). coi-serviceworker (MIT) injects them: one reload on first visit, must be a standalone same-origin file (no CDN, no bundling), HTTPS/localhost only, defaults to `credentialless` with `require-corp` fallback | [coi-serviceworker README](https://github.com/gzuidhof/coi-serviceworker); [tomayac blog](https://blog.tomayac.com/2025/03/08/setting-coop-coep-headers-on-static-hosting-like-github-pages/) |
| Safari + COEP | WebKit does not support `credentialless`; `require-corp` works on Safari/iOS | [StackBlitz cross-browser COOP/COEP](https://blog.stackblitz.com/posts/cross-browser-with-coop-coep/); [sandburg PR #10](https://github.com/swissspidy/sandburg/pull/10) |
| Verified GitHub Pages headers | `access-control-allow-origin: *`, `cache-control: max-age=600`, no COOP/COEP (curl of maninae.github.io, 2026-10-05). The 10-minute cache means our service worker, not HTTP cache, must own model caching | measured |

### Speeds (published numbers; none are for our model)

| Model | Input | Hardware | WebGPU | WASM | Source |
|---|---|---|---|---|---|
| ISNet (conv-heavy segmentation), fp16 | 1024x1024 | M4 Mac, Chromium 149, ORT 1.27 | 209 ms (first run 258 ms) | 1,960 ms (4 threads) | [dev.to WebGPU vs WASM](https://dev.to/yue_shu_c621a4a637f22396f/webgpu-vs-wasm-in-onnxruntime-web-14x-to-94x-faster-on-the-same-mac-depending-on-the-model-li5) |
| ISNet, int8 | 1024x1024 | same | 359 ms (first 537 ms) | 2,133 ms | same |
| Real-ESRGAN x4v3 (4.9 MB, compact) | 184x184 in, 4x out | same | 331 ms | 485 ms | same |
| Real-ESRGAN x4v3 | 120x120 | same | 150 ms | 211 ms | same |
| ISNet fp16 (img.ly) | 1024x1024 | desktop, unspecified | ~100 ms | ~2 s (16 threads + SIMD); ~53 s single-thread | [img.ly blog](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/) |

- Lesson from the Real-ESRGAN rows: small, shallow models gain only 1.4-1.5x from WebGPU; dispatch overhead dominates. Our crop-sized model will sit in that regime, so WASM is a realistic primary path, not just a fallback.
- Phone numbers for a U-Net at 256/512 on ORT: **could not verify** any published benchmark. The only iPhone data point found is a 7.5 MB MobileNetV3+U-Net segmenter on an iPhone SE2 taking 4.2 s first, 0.95 s warm, single-threaded WASM ([zenn.dev iOS memory article](https://zenn.dev/kaz_sakai/articles/ios-safari-onnx-memory?locale=en)).

### Safari / iOS limits

| Limit | Finding | Source |
|---|---|---|
| WebGPU on iOS | Shipped enabled by default in Safari on iOS 26 (Sep 15 2025); all iOS browsers use WebKit, so they get it too | [caniuse WebGPU](https://caniuse.com/webgpu); [cinevva guide](https://app.cinevva.com/guides/webgl-webgpu-not-supported-fix) |
| ORT WebGPU on iOS | ORT issue "Support iOS devices" still open; maintainers stated onnxruntime-web WebGPU "is not supported in iOS devices regardless of browser type" (statement predates iOS 26; current state **could not verify**) | [ORT issue #22776](https://github.com/microsoft/onnxruntime/issues/22776) |
| JSEP on Safari 26 | Severe CPU (400%+) and memory (1 GB+, up to 14 GB) after inference with JSEP builds (ORT 1.20-1.23); plain WASM unaffected | [ORT issue #26827](https://github.com/microsoft/onnxruntime/issues/26827) |
| Tab memory | "on iOS devices Safari tab memory is limited to <500 MB" (uncited claim in paper); transformers.js + Safari WebGPU leak grew to 10 GB before kill | [Llamas on the Web, arXiv 2605.20706](https://arxiv.org/html/2605.20706v1) |
| WASM heap | Emscripten heap never shrinks; fix is run inference in a Worker and `terminate()` it; also disable `enableCpuMemArena`/`enableMemPattern`, `numThreads=1`, zero out canvases | [zenn.dev iOS memory article](https://zenn.dev/kaz_sakai/articles/ios-safari-onnx-memory?locale=en) |
| WASM ceiling | 4 GB addressable (wasm32); ORT cannot run >4 GB models | [ORT large models doc](https://onnxruntime.ai/docs/tutorials/web/large-models.html) |
| WASM load OOM on iOS 17 | Reported model-load failures on iOS 17 WASM | [ORT issue #22086](https://github.com/microsoft/onnxruntime/issues/22086) |

## 2. Alternatives to onnxruntime-web

| Runtime | 2026 status | Fit for a custom small image-to-image CNN | Source |
|---|---|---|---|
| **onnxruntime-web 1.30** | Active, MIT, WebGPU + WASM | **Best fit.** Direct PyTorch to ONNX export, tensor-level control, already proven in our stack | npm registry; [pkgpulse comparison](https://www.pkgpulse.com/guides/transformersjs-vs-onnx-runtime-web-2026) |
| LiteRT.js (`@litertjs/core` 2.5.3, Apache-2.0) | Launched Jul 9 2026; XNNPACK WASM, WebGPU via ML Drift, experimental WebNN; Google claims up to ~3x over other web runtimes | Credible challenger. Limits: no dynamic shapes, only int32/float32 I/O, **no partial delegation** (whole graph must run on one backend), PyTorch conversion via LiteRT Torch. Worth a benchmark bake-off once the model exists | [MarkTechPost](https://www.marktechpost.com/2026/07/15/google-releases-litert-js-a-javascript-binding-of-litert-that-runs-tflite-models-in-browsers-via-webgpu/); [kingy.ai limits writeup](https://kingy.ai/news/litert-js-local-browser-ai-backends-benchmarks-limits/); npm registry |
| MediaPipe Tasks (`@mediapipe/tasks-vision` 1.0.1, Apache-2.0) | Active; task-level APIs (face, segmentation) | Not a general runtime for our own net; use only for face landmarks if needed | npm registry; [unpkg listing](https://app.unpkg.com/@mediapipe/tasks-vision/files/wasm) |
| transformers.js (`@huggingface/transformers` 4.3.0) | Active; wraps onnxruntime-web | Adds pipeline/tokenizer layers we do not need; use ORT directly for a custom model | npm registry; [pkgpulse comparison](https://www.pkgpulse.com/guides/transformersjs-vs-onnx-runtime-web-2026) |
| TensorFlow.js 4.22.0 | Last npm release 2024-10-21; effectively superseded by LiteRT.js | Avoid for new work (UpscalerJS and web-realesrgan still use it) | npm registry; [pasqualepillitteri LiteRT.js article](https://pasqualepillitteri.it/en/news/7699/litert-js-google-web-ai-runtime) |
| WebNN | W3C Candidate Recommendation Jan 2026; Chrome origin trial M147-M149 (not Android); no Safari/Firefox | Not production in 2026 | [utsubo frontier APIs 2026](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready); [Phoronix Chrome 146](https://www.phoronix.com/news/Chrome-146-Beta) |
| WONNX / burn / candle (Rust to wasm) | Could not verify 2026 maturity or browser image-model benchmarks in this pass | Not recommended without evidence; more build complexity | could not verify |

## 3. Face and eye locating

| Option | Download | License | Output | Notes | Source |
|---|---|---|---|---|---|
| **YuNet 2023mar (ONNX)** | 232,589 bytes (verified via HF `x-linked-size`) | MIT | Box + 5 points (both eye centers, nose, mouth corners) | Runs on our existing ORT session; faces ~10-300 px at its native input, so run on a downscaled copy of the photo | [HF opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet/blob/main/face_detection_yunet_2023mar.onnx); [pollen-robotics mirror](https://huggingface.co/pollen-robotics/face_detection_yunet_2023mar) |
| MediaPipe BlazeFace short range | 229,746 bytes (.tflite, float16) | Apache-2.0 (upstream) | Box + 6 keypoints | Needs the MediaPipe or LiteRT runtime, not ORT | measured via curl on `storage.googleapis.com/mediapipe-models/...`; [Luxonis card](https://models.luxonis.com/luxonis/mediapipe-face-landmarker/4632304b-91cb-4fcb-b4cc-c8c414e13f56?backTo=%2F) |
| **MediaPipe Face Landmarker** | 3,758,596 bytes .task (BlazeFace 192x192 + Face Mesh V2 256x256 + blendshapes, float16) plus `vision_wasm_internal.wasm` 11.8 MB | npm package Apache-2.0; docs CC-BY-4.0 | 478 3D points incl. iris and eye contours | Self-host: `FilesetResolver.forVisionTasks('/local/path/')`, files must keep original names; `modelAssetPath` to a local .task | [MediaPipe Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker); [FilesetResolver API](https://ai.google.dev/edge/api/mediapipe/js/tasks-text.filesetresolver); [478-point map](https://www.sanderdesnaijer.com/blog/mediapipe-face-mesh-landmarks) |
| Eyeglass segmentation: `mantasu/glasses-detector` | Could not verify per-model sizes | Code MIT; trained on CelebAMask-HQ + Roboflow/Kaggle datasets with their own licenses | Masks for full, frames, legs, **lenses**, shadows | PyTorch, not browser-ready; dataset licenses (CelebAMask-HQ is non-commercial) need checking before reuse | [glasses-detector](https://github.com/mantasu/glasses-detector) |
| Eyeglass segmentation: u2netp in ORT web | ~4.6 MB | (repo-specific, could not verify) | Lens alpha mask for product photos of frames, not faces | Proves u2netp runs in onnxruntime-web/wasm; not trained on worn glasses | [EyesGlasses-System PR #3](https://github.com/MostafaMSC/EyesGlasses-System/pull/3) |

- Conclusion: no small, permissively licensed, browser-ready model segments lenses on a worn-glasses portrait. Recommended: have the restoration net predict its own soft glare mask inside the crop (a second output head), which removes the need for a separate segmenter.

## 4. Fully client-side image tools to learn from

| Project | Model + size | Runtime | License | Reported problems | Source |
|---|---|---|---|---|---|
| lxfater/inpaint-web | MI-GAN `migan_pipeline_v2.onnx` (inpaint), `realesrgan-x4.onnx` (SR); sizes not in code | onnxruntime-web WebGPU + WASM; models fetched from Hugging Face with a Cloudflare Worker backup, cached in IndexedDB via localforage | GPL-3.0 | "Works in Chrome, errors in Safari" (#61), stuck at 99.13% (#59), session creation failure (#64), "Failed to get GPU adapter" on Intel Mac (#1) | [repo](https://github.com/lxfater/inpaint-web); [cache.ts](https://raw.githubusercontent.com/lxfater/inpaint-web/main/src/adapters/cache.ts); [issues](https://github.com/lxfater/inpaint-web/issues) |
| imgly/background-removal-js | ISNet: ~40 MB quint8, ~80 MB fp16 (default), full fp32 | onnxruntime-web, `device: 'gpu'` uses WebGPU | **AGPL-3.0** | Recommends self-hosting via `publicPath` and COOP/COEP for speed; quint8 shows artifacts | [web README](https://raw.githubusercontent.com/imgly/background-removal-js/main/packages/web/README.md); [img.ly blog](https://img.ly/blog/browser-background-removal-using-onnx-runtime-webgpu/) |
| bg0 (opencoredev) | ISNet-class at 512 px on iOS | onnxruntime-web | could not verify | iPhone tab reloads / infinite spinner from memory peaks; fixed by streaming model into a preallocated buffer, decoding once to a bounded 1280 px bitmap, releasing the worker, watchdog timeouts | [bg0 PR #31](https://github.com/opencoredev/bg0/pull/31) |
| UpscalerJS | ESRGAN-family models, multiple sizes | TensorFlow.js; built-in patch tiling | MIT | TF.js is no longer released, a maintenance risk | [repo](https://github.com/thekevinscott/UpscalerJS) |
| xororz/web-realesrgan | Real-ESRGAN / Real-CUGAN, fp16 | TF.js WebGL/WebGPU, tiled | could not verify | Notes fp16 computes no faster than fp32 in its runtime | [repo](https://github.com/xororz/web-realesrgan) |
| RestoreFormer++ web | ~147 MB fp16 weights | onnxruntime-web WebGPU/WASM | could not verify | Size alone makes it unusable on phones; a cautionary upper bound | [HF Saimon8420/restoreformer-pp-web](https://huggingface.co/Saimon8420/restoreformer-pp-web) |

- Pattern across all of them: the first-load download and iOS memory are the recurring complaints, not inference speed. A 3-8 MB model sidesteps both.

## 5. Full-resolution handling and canvas pitfalls

| Topic | Best practice / pitfall | Source |
|---|---|---|
| Where to run the net | Only on the aligned eye crop; inverse-warp the residual and feather-blend with the soft mask into the original-resolution canvas | design recommendation (no single source) |
| Tiling | For crops larger than the model input, tile with overlap and blend seams; web-realesrgan and UpscalerJS both tile, and larger tiles are faster on WebGPU | [web-realesrgan](https://github.com/xororz/web-realesrgan); [UpscalerJS](https://github.com/thekevinscott/UpscalerJS) |
| Residual at low res | Predict residual + mask at model resolution, bilinear-upsample to crop resolution, add to the original crop pixels; preserves original high-frequency detail outside glare | design recommendation |
| EXIF orientation | `createImageBitmap` defaults to `imageOrientation: 'from-image'` (applies EXIF); `'none'` ignores it. Use one decode path everywhere and strip EXIF orientation on export to avoid double rotation | [MDN createImageBitmap](https://developer.mozilla.org/en-US/docs/Web/API/Window/createImageBitmap); [WebKit commit](https://github.com/WebKit/WebKit/commit/8758b1b9f85526f462e6edb74d5c85228e15d90d) |
| Display-P3 | iPhone photos are often Display-P3. A default 2D canvas is sRGB, so drawing then exporting silently clips gamut. Use `getContext('2d', {colorSpace: 'display-p3'})` and `getImageData(..., {colorSpace})` where supported; feed the net sRGB-converted data but composite in the source space | [WebKit wide-gamut canvas](https://webkit.org/blog/12058/wide-gamut-2d-graphics-using-html-canvas/); [MDN ImageData.colorSpace](https://developer.mozilla.org/docs/Web/API/ImageData/colorSpace) |
| iOS canvas size | Max area was 16,777,216 px (4096x4096) for ~10 years; iOS 18 raised it to 8192x8192 (67 M px). Safari also has a total canvas-memory budget and holds released canvases; set width/height to 0 when done | [lionpuro: canvas finally usable](https://lionpuro.com/posts/canvas-is-finally-usable-on-safari/); [pqina](https://pqina.nl/blog/canvas-area-exceeds-the-maximum-limit/); [zenn.dev](https://zenn.dev/kaz_sakai/articles/ios-safari-onnx-memory?locale=en) |
| JPEG re-encode | Every `toBlob('image/jpeg', q)` re-encodes the whole photo; default quality varies by browser. Export at q≈0.95 or offer PNG; ideally splice only changed blocks (no standard browser API, could not verify a maintained library) | could not verify a definitive source |
| HEIC input | Safari 17+ decodes HEIC natively (img + createImageBitmap); Chrome/Firefox/Edge do not, so need a libheif WASM decoder (e.g. heic2any / libheif-js) | [caniuse HEIF](https://caniuse.com/heif); [dev.to HEIC](https://dev.to/forze-dev/why-heic-breaks-websites-and-how-i-built-a-browser-only-converter-to-fix-it-gp0) |
| iPhone photo picker | iOS Safari usually converts HEIC to JPEG when picked via `<input type=file accept="image/*">` (could not verify current behavior from a primary source) | could not verify |

## 6. Competitive landscape

| Tool | Price / free tier | Uploads to server? | Notes | Source |
|---|---|---|---|---|
| removeglassesglare.com | Free 800x600 preview; credits from $2, 1 credit = 1 full-res image, no subscription; batch up to 100 | Yes ("deleted within 24 hours") | 10-30 s processing; does not remove frames | [site](https://removeglassesglare.com/) |
| Evoto | 3 free HD web exports for new users; paid plans after signup | Web upload; also desktop app | Claims "private processing" | [Evoto glare remover](https://www.evoto.ai/features/photo-glare-remover) |
| Fotor | Free credits, earn more daily; Pro ~$7.19-8.99/mo, Pro+ $13.99/mo | Yes | Glare and full glasses removal | [Fotor glare page](https://www.fotor.com/features/remove-glare-from-photo/) |
| EzEnhancer | "Free to try with daily credits"; paid tiers not verified | Yes | "EModel-Glare V1" | [ezenhancer](https://ezenhancer.ai/remove-glare/) |
| mew.design | Could not verify pricing | Yes (chat-edit agent) | Generative edit, prompt-driven | [mew.design](https://mew.design/create/ai-remove-glasses-glare) |
| EditThisPic | 1 free/week; 10 edits $4.99 | Yes | | [EditThisPic](https://editthispic.com/edit/ai-glasses-glare-remover) |
| YouCam Perfect / YouCam Online Editor | Subscription ~$4.99-5.99/mo, $69.99/yr | Yes (app / web) | Generic object removal | [weedit roundup](https://weedit.photos/glare-removal-app/); [YouCam online](https://yce.perfectcorp.com/features/remove-glare-from-photo) |
| Picsart | Free with 7-day trial upsell, ads | Yes | Remove Object / AI Replace / Clone, no dedicated glare model | [removeglassesglare Picsart guide](https://removeglassesglare.com/articles/how-to-remove-glare-from-glasses-in-picsart) |
| Facetune | Subscription; mobile only | App (could not verify on-device vs cloud) | Has an "Anti Glare" retouch tool | [Facetune blog](https://www.facetuneapp.com/blog/how-to-remove-glare-selfie-photo) |
| PhotoRoom | Could not verify a dedicated glasses-glare feature | | | could not verify |
| Retouch4me (Apex) | Paid; licenses from ~$124 per plugin, Apex pricing from $169 | **No, runs locally** (desktop plugin) | Closest privacy analogue, but paid and closed | [Retouch4me Apex](https://retouch4.me/blog/retouch4me-apex-v-1-207); [pricing](https://retouch4.me/pricing) |
| moireremoval.com, media.io, Somake, Musely, Pokecut, freebeat | Free tiers with daily limits | Yes | SEO-farm style pages | [moireremoval](https://moireremoval.com/remove-glasses-reflection); [media.io](https://www.media.io/ai/image-to-image/remove-glasses-glare-from-photo) |

**Is there an existing free, open-source, fully local in-browser lens-glare remover? No evidence of one.**
- `gh search repos` for "glasses glare", "glare glasses", "eyeglass reflection", "glasses reflection", "deglare", "spectacle reflection" (2026-10-05) found only research/student repos: `paulchi-intel/glasses_glare_remover` (video, CV pipeline, 1 star), `Aym-S/eyeglass-reflection-removal` (ERRNet, MIT, 0 stars), `longyan97/EyeglassFilter` (blurs lenses in video for privacy). None runs in a browser.
- HN Algolia search for "glasses glare" returned no Show HN or open-source tool ([HN Algolia query](https://hn.algolia.com/api/v1/search?query=glasses%20glare&tags=story)).
- Research prior art exists: De-Glared (U-Net for eyeglass glare and reflection removal) ([ResearchGate](https://www.researchgate.net/publication/378539880_De-Glared_Eyeglasses_Glare_and_Reflection_Removal_Using_Deep_Neural_Networks)); ByeGlassesGAN removes whole glasses ([arXiv 2008.11042](https://arxiv.org/pdf/2008.11042)).

## 7. Hosting

| Option | Limits | CORS / third-party | Source |
|---|---|---|---|
| **GitHub Pages, same repo** | Site ≤1 GB; soft 100 GB/month bandwidth; soft 10 builds/hour; Git warns >50 MiB, blocks >100 MiB per file; browser upload ≤25 MiB | Same-origin, zero third parties. **Git LFS files are served as pointer text**, so commit models as plain blobs (or materialize LFS in an Actions deploy) | [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits); [large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github); [git-lfs #1342](https://github.com/git-lfs/git-lfs/issues/1342) |
| GitHub Releases | Large assets allowed | **Not browser-fetchable**: download URL 302s to `release-assets.githubusercontent.com` with no `Access-Control-Allow-Origin` | [pocket-dial #138](https://github.com/GlomarGadaffi/pocket-dial/issues/138); [corsfix](https://corsfix.com/blog/fetch-github-release) |
| Hugging Face Hub | Free | CORS works: verified 302 then 200 with `access-control-allow-origin: *` from `us.aws.cdn.hf.co` (2026-10-05). But it is a third-party request that breaks the privacy promise and would need CORP for `require-corp` COEP | measured via curl |
| Cloudflare Pages | 25 MiB per asset, 20,000 files free; `_headers` file can set real COOP/COEP (no service-worker trick) | Same-origin | [CF Pages limits](https://developers.cloudflare.com/pages/platform/limits/) |
| Cloudflare R2 | Free: 10 GB-month storage, 1 M Class A, 10 M Class B ops, free egress | Separate origin unless fronted by the same domain | [R2 pricing](https://developers.cloudflare.com/r2/pricing/) |

- Recommendation: serve everything (HTML, ORT wasm, YuNet, our model, coi-serviceworker) from one origin. GitHub Pages works if each model file is under ~50 MB and committed as a plain file; Cloudflare Pages is the better host if we want true COOP/COEP headers (Safari threads without the reload trick), and it also caps files at 25 MiB, which our budget already respects.
- Ship only the ORT wasm variants we load (`ort-wasm-simd-threaded.wasm` + the WebGPU one), and precache them plus the model in the service worker with a versioned cache name.

## Open questions to benchmark ourselves

- Does onnxruntime-web 1.30's WebGPU EP actually run on iOS 26 Safari for a Conv-only net? (Issue #22776 is stale; test on a real iPhone.)
- WASM single-thread vs WebGPU crossover point for our model size on iPhone and mid-range Android.
- LiteRT.js vs onnxruntime-web on the same architecture once a trained checkpoint exists.
