/**
 * Which onnxruntime-web build and how many threads this browser gets.
 *
 * - WebGPU build when a GPU adapter is actually available, else the smaller CPU-only build.
 *   `?backend=wasm|webgpu` overrides (tests and benchmarks use it).
 * - Threads need cross-origin isolation, which sw.js provides from the second visit on (it adds
 *   COOP/COEP headers to what it serves; no forced reload). First visit runs one thread.
 * - iPhone and iPad stay at one thread: Safari kills tabs at a few hundred MB and shared WASM
 *   memory makes that likelier (docs/research/02-browser-runtime.md, "Safari / iOS limits").
 *   Every other mobile browser (Android etc.) gets at most MOBILE_MAX_WASM_THREADS.
 * - `?threads=N` is an explicit override for tests and benchmarks; caps do not apply to it.
 */

import { MAX_WASM_THREADS } from "../app_config.js";

// Each WASM thread adds stack and scratch memory; phones get few cores' worth.
const MOBILE_MAX_WASM_THREADS = 2;

export function isAppleTouchDevice() {
  return /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

/** Any phone or tablet browser (UA-CH `mobile` where offered, else the user-agent string). */
export function isMobileUserAgent() {
  return navigator.userAgentData?.mobile === true || /Android|Mobi|iPhone|iPad|iPod/i.test(navigator.userAgent) || isAppleTouchDevice();
}

/** Phones and small tablets: free model memory between batches. */
export function isMemoryConstrainedDevice() {
  const deviceMemoryGigabytes = navigator.deviceMemory ?? 8;
  return isAppleTouchDevice() || deviceMemoryGigabytes <= 4 || matchMedia("(pointer: coarse)").matches;
}

/**
 * A real GPU adapter only. Software fallbacks (SwiftShader, handed out when the GPU or its
 * driver is blocklisted) ran our model ~50x slower than threaded WASM in testing.
 */
async function hasUsableWebGpuAdapter() {
  if (!("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    const isFallback = adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter ?? false;
    const isSoftware = /swiftshader|cpu/i.test(adapter.info?.architecture ?? "");
    return !isFallback && !isSoftware;
  } catch {
    return false;
  }
}

/** Resolves to { backend: "webgpu" | "wasm", numThreads }. */
export async function chooseRuntime(searchParams = new URLSearchParams(location.search)) {
  const requestedBackend = searchParams.get("backend");
  let backend = "wasm";
  if (requestedBackend === "webgpu" || requestedBackend === "wasm") backend = requestedBackend;
  else if (await hasUsableWebGpuAdapter()) backend = "webgpu";
  const requestedThreads = Number(searchParams.get("threads"));
  let numThreads = 1;
  if (self.crossOriginIsolated && !isAppleTouchDevice()) {
    const threadCap = isMobileUserAgent() ? MOBILE_MAX_WASM_THREADS : MAX_WASM_THREADS;
    numThreads = Math.max(1, Math.min(threadCap, Math.floor((navigator.hardwareConcurrency || 2) / 2)));
  }
  if (requestedThreads >= 1 && self.crossOriginIsolated) numThreads = Math.min(requestedThreads, MAX_WASM_THREADS);
  return { backend, numThreads };
}
