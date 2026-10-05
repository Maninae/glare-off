/**
 * Full-resolution face regions for ONE photo at a time: the one the visitor last saw land or
 * touched (strength, face toggles, view picker). Every other photo keeps only crop-space data.
 *
 * Per glare face the cache holds the region's original bytes and its warped delta layers
 * (~16 bytes per region pixel), so dragging the strength slider is a blend, not a warp.
 * Switching photos drops the old entry and re-reads the new photo's regions through the
 * `readRegions(file, regions)` callback (a worker re-decodes the file; nothing is stored).
 */

import { warpFaceDeltaToRegion } from "../blend/face_region_patch.js";

function buildEntries(faces, regionOriginalRgbaByFace) {
  return faces.map((face, faceIndex) => {
    if (!face.hasGlare) return null;
    return { regionOriginalRgba: regionOriginalRgbaByFace[faceIndex], regionDeltaLayers: warpFaceDeltaToRegion(face) };
  });
}

export class VisiblePhotoRegionCache {
  /** `readRegions(file, regions)` resolves to one RGBA Uint8ClampedArray per rect. */
  constructor({ readRegions }) {
    this.readRegions = readRegions;
    this.owner = null;
    this.entriesByFace = null; // [{ regionOriginalRgba, regionDeltaLayers } | null] per face
    this.pendingOwner = null;
    this.pendingEntries = null;
  }

  /** Take regions the worker already read while processing (no second decode for a new result). */
  adopt(owner, faces, regionOriginalRgbaByFace) {
    this.drop();
    this.owner = owner;
    this.entriesByFace = buildEntries(faces, regionOriginalRgbaByFace);
  }

  /**
   * Entries for `owner` ({ file, result }), reading them if another photo holds the cache.
   * Concurrent calls for the same owner share one read.
   */
  async entriesFor(owner) {
    if (this.owner === owner) return this.entriesByFace;
    if (this.pendingOwner !== owner) {
      this.pendingOwner = owner;
      const { faces } = owner.result;
      const glareFaceIndices = faces.map((face, faceIndex) => (face.hasGlare ? faceIndex : -1)).filter((faceIndex) => faceIndex >= 0);
      this.pendingEntries = this.readRegions(owner.file, glareFaceIndices.map((faceIndex) => faces[faceIndex].region)).then((regionsRgba) => {
        const regionOriginalRgbaByFace = [];
        glareFaceIndices.forEach((faceIndex, readIndex) => (regionOriginalRgbaByFace[faceIndex] = regionsRgba[readIndex]));
        return buildEntries(faces, regionOriginalRgbaByFace);
      });
    }
    const pendingEntries = this.pendingEntries;
    let entries;
    try {
      entries = await pendingEntries;
    } catch (error) {
      if (this.pendingEntries === pendingEntries) this.pendingOwner = this.pendingEntries = null; // let the next call retry
      throw error;
    }
    if (this.pendingEntries === pendingEntries) {
      this.drop();
      this.owner = owner;
      this.entriesByFace = entries;
    }
    return entries;
  }

  drop() {
    this.owner = null;
    this.entriesByFace = null;
    this.pendingOwner = null;
    this.pendingEntries = null;
  }

  /** Bytes held, for the test hook's memory check. */
  heldBytes() {
    let bytes = 0;
    for (const entry of this.entriesByFace ?? []) {
      if (!entry) continue;
      bytes += entry.regionOriginalRgba.byteLength + entry.regionDeltaLayers.reduce((total, layer) => total + layer.byteLength, 0);
    }
    return bytes;
  }
}
