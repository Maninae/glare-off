/**
 * What kind of image a dropped file is, and what format its download should be.
 *
 * - Sniffs the container from the first bytes; the browser-reported MIME type is often empty
 *   (HEIC on Windows, files from some pickers).
 * - Output keeps the input's format where that is a sensible photo format the browser can
 *   encode: PNG stays PNG, JPEG stays JPEG, WebP stays WebP when this browser encodes WebP.
 *   Everything else (HEIC, AVIF, GIF, BMP) comes out as JPEG.
 */

import { JPEG_EXPORT_QUALITY, WEBP_EXPORT_QUALITY } from "../app_config.js";

const HEIF_BRANDS = ["heic", "heix", "hevc", "hevx", "heim", "heis", "mif1", "msf1"];

/** Resolves to "jpeg" | "png" | "webp" | "heic" | "avif" | "gif" | "bmp" | "unknown". */
export async function sniffImageFormat(file) {
  const header = new Uint8Array(await file.slice(0, 32).arrayBuffer());
  const ascii = (start, end) => String.fromCharCode(...header.slice(start, end));
  if (header[0] === 0xff && header[1] === 0xd8 && header[2] === 0xff) return "jpeg";
  if (header[0] === 0x89 && ascii(1, 4) === "PNG") return "png";
  if (ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") return "webp";
  if (ascii(0, 3) === "GIF") return "gif";
  if (ascii(0, 2) === "BM") return "bmp";
  if (ascii(4, 8) === "ftyp") {
    const brand = ascii(8, 12);
    if (brand === "avif" || brand === "avis") return "avif";
    if (HEIF_BRANDS.includes(brand)) return "heic";
  }
  return "unknown";
}

let webpEncodeSupport = null;

/** Whether this browser's canvas can encode WebP (Safari cannot; it silently returns PNG). */
export async function canEncodeWebp() {
  if (webpEncodeSupport !== null) return webpEncodeSupport;
  try {
    const probeBlob = await new OffscreenCanvas(1, 1).convertToBlob({ type: "image/webp" });
    webpEncodeSupport = probeBlob.type === "image/webp";
  } catch {
    webpEncodeSupport = false;
  }
  return webpEncodeSupport;
}

/** Resolves to { mimeType, quality, extension } for the download of a photo of `inputFormat`. */
export async function chooseExportFormat(inputFormat) {
  if (inputFormat === "png") return { mimeType: "image/png", quality: undefined, extension: "png" };
  if (inputFormat === "webp" && (await canEncodeWebp())) return { mimeType: "image/webp", quality: WEBP_EXPORT_QUALITY, extension: "webp" };
  return { mimeType: "image/jpeg", quality: JPEG_EXPORT_QUALITY, extension: "jpg" };
}

/** "IMG_0423.HEIC" -> "IMG_0423-glare-off.jpg". */
export function buildDownloadFileName(originalFileName, extension) {
  const stem = originalFileName.replace(/\.[^.]+$/, "") || "photo";
  return `${stem}-glare-off.${extension}`;
}
