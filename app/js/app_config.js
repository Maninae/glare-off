/**
 * App-wide constants: the product name, file locations, and the few tuning switches.
 * The name lives here and in index.html/manifest text only; change all three together.
 */

export const APP_NAME = "Glare Off";
export const APP_SLUG = "glare-off";

// Paths are relative to app/ (index.html's directory); resolved with import.meta.url by users.
export const MODEL_PATHS = {
  faceDetector: "models/face_detection_yunet_2023mar_dynamic_input.onnx",
  glareModel: "models/glare_removal.onnx",
};

// onnxruntime-web 1.30.0, vendored from the npm tarball (app/THIRD_PARTY.md). Two builds:
// the WebGPU build (its WASM carries both the GPU and CPU paths) and the smaller CPU-only one.
// A visitor downloads exactly one pair.
export const ONNX_RUNTIME_BUILDS = {
  webgpu: { moduleUrl: "vendor/onnxruntime-web-1.30.0/ort.webgpu.min.mjs", wasmUrl: "vendor/onnxruntime-web-1.30.0/ort-wasm-simd-threaded.asyncify.wasm" },
  wasm: { moduleUrl: "vendor/onnxruntime-web-1.30.0/ort.wasm.min.mjs", wasmUrl: "vendor/onnxruntime-web-1.30.0/ort-wasm-simd-threaded.wasm" },
};

// The 2x (1024x512) crop pass for large faces. Off until the model is trained at that size.
export const HIGH_RES_PASS_ENABLED = false;
export const MAX_WASM_THREADS = 4;
export const JPEG_EXPORT_QUALITY = 0.95;
export const WEBP_EXPORT_QUALITY = 0.95;
// Long side of the on-screen before/after preview; the download is always full resolution.
export const PREVIEW_LONG_SIDE = 2048;
