/**
 * Port of OpenCV's INTER_AREA downscale (resize.cpp: computeResizeAreaTab + ResizeArea_Invoker),
 * the filter the Python detector uses to shrink big photos to YuNet's 1024 px working size.
 * Matching it keeps browser eye centers aligned with the ones training crops were built from.
 *
 * - Streams source rows through `readSourceRows(firstRow, rowCount)`, so a 50 MP photo never
 *   needs a full-size pixel buffer: only one strip plus the small output live at once.
 * - Sums run in float32 (OpenCV's work type for uint8); output rounds half to even, like
 *   saturate_cast<uchar>(float). Agreement with cv2 is exact or off by 1 on rare ties.
 * - Works on RGBA; alpha is set to 255 (photos are composited opaque before detection).
 */

const AREA_EDGE_EPSILON = 1e-3;
const SOURCE_STRIP_ROWS = 128;

/** OpenCV computeResizeAreaTab: per destination index, the source taps and their weights. */
export function computeResizeAreaTab(sourceSize, destinationSize, scale) {
  const taps = []; // { destinationIndex, sourceIndex, alpha }
  for (let destinationIndex = 0; destinationIndex < destinationSize; destinationIndex += 1) {
    const sourceStart = destinationIndex * scale;
    const sourceEnd = sourceStart + scale;
    const cellWidth = Math.min(scale, sourceSize - sourceStart);
    let firstWholeSource = Math.ceil(sourceStart);
    let lastWholeSourceEnd = Math.floor(sourceEnd);
    lastWholeSourceEnd = Math.min(lastWholeSourceEnd, sourceSize - 1);
    firstWholeSource = Math.min(firstWholeSource, lastWholeSourceEnd);
    if (firstWholeSource - sourceStart > AREA_EDGE_EPSILON) {
      taps.push({ destinationIndex, sourceIndex: firstWholeSource - 1, alpha: Math.fround((firstWholeSource - sourceStart) / cellWidth) });
    }
    for (let sourceIndex = firstWholeSource; sourceIndex < lastWholeSourceEnd; sourceIndex += 1) {
      taps.push({ destinationIndex, sourceIndex, alpha: Math.fround(1.0 / cellWidth) });
    }
    if (sourceEnd - lastWholeSourceEnd > AREA_EDGE_EPSILON) {
      taps.push({
        destinationIndex,
        sourceIndex: lastWholeSourceEnd,
        alpha: Math.fround(Math.min(Math.min(sourceEnd - lastWholeSourceEnd, 1.0), cellWidth) / cellWidth),
      });
    }
  }
  return taps;
}

/** saturate_cast<uchar>(float): round half to even, clamp to 0..255. */
function saturateToByte(value) {
  const floored = Math.floor(value);
  const fraction = value - floored;
  let rounded = floored;
  if (fraction > 0.5 || (fraction === 0.5 && floored % 2 !== 0)) rounded = floored + 1;
  return rounded < 0 ? 0 : rounded > 255 ? 255 : rounded;
}

/**
 * Area-downscale an RGBA source read in strips.
 *
 * Args:
 *   sourceWidth, sourceHeight: full source size.
 *   destinationWidth, destinationHeight: output size (computed by the caller like cv2 does).
 *   inverseScale: the `fx` passed to cv2.resize (OpenCV uses 1/fx, not src/dst, for the tabs).
 *   readSourceRows(firstRow, rowCount): returns RGBA bytes for those full-width rows.
 * Returns:
 *   Uint8ClampedArray RGBA of destinationWidth x destinationHeight.
 */
export function areaDownscaleRgba({ sourceWidth, sourceHeight, destinationWidth, destinationHeight, inverseScale, readSourceRows }) {
  const scale = 1.0 / inverseScale;
  const columnTaps = computeResizeAreaTab(sourceWidth, destinationWidth, scale);
  const rowTaps = computeResizeAreaTab(sourceHeight, destinationHeight, scale);
  const channelCount = 3;
  const rowLength = destinationWidth * channelCount;
  const horizontalBuffer = new Float32Array(rowLength);
  const accumulator = new Float32Array(rowLength);
  const destination = new Uint8ClampedArray(destinationWidth * destinationHeight * 4);
  let strip = null;
  let stripFirstRow = -1;
  let previousDestinationRow = -1;

  const flushRow = (destinationRow) => {
    for (let destinationX = 0; destinationX < destinationWidth; destinationX += 1) {
      const outputIndex = (destinationRow * destinationWidth + destinationX) * 4;
      for (let channel = 0; channel < channelCount; channel += 1) {
        destination[outputIndex + channel] = saturateToByte(accumulator[destinationX * channelCount + channel]);
      }
      destination[outputIndex + 3] = 255;
    }
  };

  for (const { destinationIndex: destinationRow, sourceIndex: sourceRow, alpha: beta } of rowTaps) {
    if (strip === null || sourceRow < stripFirstRow || sourceRow >= stripFirstRow + SOURCE_STRIP_ROWS) {
      stripFirstRow = sourceRow;
      strip = readSourceRows(stripFirstRow, Math.min(SOURCE_STRIP_ROWS, sourceHeight - stripFirstRow));
    }
    const rowOffset = (sourceRow - stripFirstRow) * sourceWidth * 4;
    horizontalBuffer.fill(0);
    for (const { destinationIndex: destinationX, sourceIndex: sourceX, alpha } of columnTaps) {
      const sourceIndex = rowOffset + sourceX * 4;
      const bufferIndex = destinationX * channelCount;
      horizontalBuffer[bufferIndex] += strip[sourceIndex] * alpha;
      horizontalBuffer[bufferIndex + 1] += strip[sourceIndex + 1] * alpha;
      horizontalBuffer[bufferIndex + 2] += strip[sourceIndex + 2] * alpha;
    }
    if (destinationRow !== previousDestinationRow) {
      if (previousDestinationRow >= 0) flushRow(previousDestinationRow);
      for (let index = 0; index < rowLength; index += 1) accumulator[index] = beta * horizontalBuffer[index];
      previousDestinationRow = destinationRow;
    } else {
      for (let index = 0; index < rowLength; index += 1) accumulator[index] += beta * horizontalBuffer[index];
    }
  }
  if (previousDestinationRow >= 0) flushRow(previousDestinationRow);
  return destination;
}
