/**
 * A minimal ZIP writer for "Download all": STORE method (no compression; the photos are
 * already JPEG/PNG/WebP), CRC-32, UTF-8 names. Enough for a handful of photos; no ZIP64,
 * so the archive must stay under 4 GB.
 */

const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let byteValue = 0; byteValue < 256; byteValue += 1) {
    let crc = byteValue;
    for (let bit = 0; bit < 8; bit += 1) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
    table[byteValue] = crc >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) crc = CRC32_TABLE[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** Make every name unique ("a.jpg", "a (2).jpg") so no entry overwrites another on extract. */
function uniqueNames(names) {
  const seen = new Map();
  return names.map((name) => {
    const count = (seen.get(name) ?? 0) + 1;
    seen.set(name, count);
    return count === 1 ? name : name.replace(/(\.[^.]*)?$/, (extension) => ` (${count})${extension}`);
  });
}

/**
 * Build a ZIP. `entries`: [{ name, bytes: Uint8Array }]. Returns a Blob (application/zip).
 * DOS timestamps are fixed at 1980-01-01 so the archive carries no local time.
 */
export function buildStoredZip(entries) {
  const encoder = new TextEncoder();
  const names = uniqueNames(entries.map((entry) => entry.name));
  const fileParts = [];
  const centralParts = [];
  let offset = 0;
  const DOS_DATE_1980_01_01 = 0x21;
  const UTF8_NAME_FLAG = 0x0800;
  entries.forEach((entry, entryIndex) => {
    const nameBytes = encoder.encode(names[entryIndex]);
    const checksum = crc32(entry.bytes);
    const localHeader = new DataView(new ArrayBuffer(30));
    localHeader.setUint32(0, 0x04034b50, true);
    localHeader.setUint16(4, 20, true);
    localHeader.setUint16(6, UTF8_NAME_FLAG, true);
    localHeader.setUint16(8, 0, true); // STORE
    localHeader.setUint16(10, 0, true);
    localHeader.setUint16(12, DOS_DATE_1980_01_01, true);
    localHeader.setUint32(14, checksum, true);
    localHeader.setUint32(18, entry.bytes.length, true);
    localHeader.setUint32(22, entry.bytes.length, true);
    localHeader.setUint16(26, nameBytes.length, true);
    localHeader.setUint16(28, 0, true);
    fileParts.push(new Uint8Array(localHeader.buffer), nameBytes, entry.bytes);

    const centralHeader = new DataView(new ArrayBuffer(46));
    centralHeader.setUint32(0, 0x02014b50, true);
    centralHeader.setUint16(4, 20, true);
    centralHeader.setUint16(6, 20, true);
    centralHeader.setUint16(8, UTF8_NAME_FLAG, true);
    centralHeader.setUint16(10, 0, true);
    centralHeader.setUint16(12, 0, true);
    centralHeader.setUint16(14, DOS_DATE_1980_01_01, true);
    centralHeader.setUint32(16, checksum, true);
    centralHeader.setUint32(20, entry.bytes.length, true);
    centralHeader.setUint32(24, entry.bytes.length, true);
    centralHeader.setUint16(28, nameBytes.length, true);
    centralHeader.setUint32(42, offset, true);
    centralParts.push(new Uint8Array(centralHeader.buffer), nameBytes);
    offset += 30 + nameBytes.length + entry.bytes.length;
  });
  const centralSize = centralParts.reduce((total, part) => total + part.length, 0);
  const endRecord = new DataView(new ArrayBuffer(22));
  endRecord.setUint32(0, 0x06054b50, true);
  endRecord.setUint16(8, entries.length, true);
  endRecord.setUint16(10, entries.length, true);
  endRecord.setUint32(12, centralSize, true);
  endRecord.setUint32(16, offset, true);
  return new Blob([...fileParts, ...centralParts, new Uint8Array(endRecord.buffer)], { type: "application/zip" });
}
