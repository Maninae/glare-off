/**
 * YuNet (face_detection_yunet_2023mar) pre- and post-processing, ported from OpenCV's
 * FaceDetectorYN (modules/objdetect/src/face_detect.cpp), which is what the Python reference
 * `eye_crop/yunet_eye_detector.py` runs. Pure functions; the onnxruntime call happens in the
 * worker between `buildYunetInputTensor` and `decodeYunetOutputs`.
 *
 * - Input: BGR float32 in 0..255 (no mean, no scale), NCHW, the image padded with zeros on
 *   the right/bottom up to a multiple of 32. The ONNX we ship has symbolic H/W
 *   (tests/app/make_yunet_dynamic_input_model.py); the stock file only accepts 640x640.
 * - Output rows mirror OpenCV: [x, y, w, h, 5 landmarks (x, y), score]. Landmark order is the
 *   SUBJECT's right eye, left eye, nose tip, right and left mouth corner.
 * - score = sqrt(clamp(cls) * clamp(obj)); NMS runs on integer-truncated boxes, exactly as
 *   `cv::dnn::NMSBoxes` over `Rect2i` does, so borderline overlaps resolve the same way.
 */

export const YUNET_SCORE_THRESHOLD = 0.6;
export const YUNET_NMS_THRESHOLD = 0.3;
export const YUNET_TOP_K = 50;
// YuNet is trained for faces up to roughly this size; larger photos are downscaled before detection.
export const YUNET_MAX_DETECTION_SIDE = 1024;
const YUNET_STRIDES = [8, 16, 32];
const YUNET_PAD_DIVISOR = 32;

/** Detection-image size for a photo, mirroring the Python `detection_scale` and cv2.resize rounding. */
export function computeDetectionSize(photoWidth, photoHeight) {
  const detectionScale = Math.min(1.0, YUNET_MAX_DETECTION_SIDE / Math.max(photoWidth, photoHeight));
  if (detectionScale >= 1.0) {
    return { detectionScale: 1.0, detectionWidth: photoWidth, detectionHeight: photoHeight };
  }
  // cv2.resize(fx=s) sizes the output with saturate_cast<int>, i.e. round-half-to-even-ish rint.
  return {
    detectionScale,
    detectionWidth: roundHalfToEven(photoWidth * detectionScale),
    detectionHeight: roundHalfToEven(photoHeight * detectionScale),
  };
}

/** C's rint() (what OpenCV's saturate_cast<int>(double) uses): ties go to the even integer. */
export function roundHalfToEven(value) {
  const floored = Math.floor(value);
  const fraction = value - floored;
  if (fraction > 0.5) return floored + 1;
  if (fraction < 0.5) return floored;
  return floored % 2 === 0 ? floored : floored + 1;
}

/** Padded network size for an image (OpenCV's padWithDivisor). */
export function computeYunetPaddedSize(imageWidth, imageHeight) {
  return {
    paddedWidth: (Math.trunc((imageWidth - 1) / YUNET_PAD_DIVISOR) + 1) * YUNET_PAD_DIVISOR,
    paddedHeight: (Math.trunc((imageHeight - 1) / YUNET_PAD_DIVISOR) + 1) * YUNET_PAD_DIVISOR,
  };
}

/**
 * RGBA pixels -> the YuNet input tensor data.
 * Returns { tensorData: Float32Array(3*Hp*Wp), paddedWidth, paddedHeight } in BGR plane order.
 */
export function buildYunetInputTensor(rgbaPixels, imageWidth, imageHeight) {
  const { paddedWidth, paddedHeight } = computeYunetPaddedSize(imageWidth, imageHeight);
  const planeSize = paddedWidth * paddedHeight;
  const tensorData = new Float32Array(3 * planeSize); // zero-filled = the constant padding
  for (let y = 0; y < imageHeight; y += 1) {
    for (let x = 0; x < imageWidth; x += 1) {
      const sourceIndex = (y * imageWidth + x) * 4;
      const planeIndex = y * paddedWidth + x;
      tensorData[planeIndex] = rgbaPixels[sourceIndex + 2]; // B
      tensorData[planeSize + planeIndex] = rgbaPixels[sourceIndex + 1]; // G
      tensorData[2 * planeSize + planeIndex] = rgbaPixels[sourceIndex]; // R
    }
  }
  return { tensorData, paddedWidth, paddedHeight };
}

function clampUnit(value) {
  return Math.max(0, Math.min(1, value));
}

/**
 * Decode raw YuNet outputs into face rows (OpenCV's postProcess, before NMS).
 * `outputsByName`: { cls_8, obj_8, bbox_8, kps_8, ... } as Float32Arrays.
 */
export function decodeYunetCandidates(outputsByName, paddedWidth, paddedHeight, scoreThreshold = YUNET_SCORE_THRESHOLD) {
  const candidates = [];
  for (const stride of YUNET_STRIDES) {
    const columns = Math.trunc(paddedWidth / stride);
    const rows = Math.trunc(paddedHeight / stride);
    const classScores = outputsByName[`cls_${stride}`];
    const objectScores = outputsByName[`obj_${stride}`];
    const boxDeltas = outputsByName[`bbox_${stride}`];
    const landmarkDeltas = outputsByName[`kps_${stride}`];
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const anchorIndex = row * columns + column;
        // OpenCV computes in float32; Math.fround keeps the threshold comparison identical.
        const score = Math.fround(Math.sqrt(Math.fround(clampUnit(classScores[anchorIndex]) * clampUnit(objectScores[anchorIndex]))));
        if (score < scoreThreshold) continue;
        const centerX = (column + boxDeltas[anchorIndex * 4]) * stride;
        const centerY = (row + boxDeltas[anchorIndex * 4 + 1]) * stride;
        const boxWidth = Math.exp(boxDeltas[anchorIndex * 4 + 2]) * stride;
        const boxHeight = Math.exp(boxDeltas[anchorIndex * 4 + 3]) * stride;
        const faceRow = [centerX - boxWidth / 2, centerY - boxHeight / 2, boxWidth, boxHeight];
        for (let landmark = 0; landmark < 5; landmark += 1) {
          faceRow.push((landmarkDeltas[anchorIndex * 10 + 2 * landmark] + column) * stride);
          faceRow.push((landmarkDeltas[anchorIndex * 10 + 2 * landmark + 1] + row) * stride);
        }
        faceRow.push(score);
        candidates.push(faceRow.map((value) => Math.fround(value)));
      }
    }
  }
  return candidates;
}

/** Integer box area overlap (IoU) as cv::Rect2i does it. */
function integerBoxIntersectionOverUnion(boxA, boxB) {
  const areaA = boxA[2] * boxA[3];
  const areaB = boxB[2] * boxB[3];
  if (areaA + areaB <= 0) return 1.0;
  const left = Math.max(boxA[0], boxB[0]);
  const top = Math.max(boxA[1], boxB[1]);
  const intersectionWidth = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]) - left;
  const intersectionHeight = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]) - top;
  const intersectionArea = intersectionWidth > 0 && intersectionHeight > 0 ? intersectionWidth * intersectionHeight : 0;
  return intersectionArea / (areaA + areaB - intersectionArea);
}

/** cv::dnn::NMSBoxes(Rect2i boxes, scores, scoreThreshold, nmsThreshold, eta=1, topK). Returns kept indices. */
export function nonMaximumSuppression(faceRows, scoreThreshold, nmsThreshold, topK) {
  const integerBoxes = faceRows.map((faceRow) => faceRow.slice(0, 4).map((value) => Math.trunc(value)));
  // GetMaxScoreIndex: strictly above threshold, stable sort by score descending, keep topK.
  let order = faceRows.map((faceRow, index) => ({ score: faceRow[14], index })).filter((entry) => entry.score > scoreThreshold);
  order.sort((first, second) => second.score - first.score);
  if (topK > 0) order = order.slice(0, topK);
  const keptIndices = [];
  for (const { index } of order) {
    const overlapsKeptBox = keptIndices.some((keptIndex) => integerBoxIntersectionOverUnion(integerBoxes[index], integerBoxes[keptIndex]) > nmsThreshold);
    if (!overlapsKeptBox) keptIndices.push(index);
  }
  return keptIndices;
}

/** Full OpenCV postProcess: decode, then NMS only when there is more than one candidate. */
export function decodeYunetOutputs(outputsByName, paddedWidth, paddedHeight) {
  const candidates = decodeYunetCandidates(outputsByName, paddedWidth, paddedHeight);
  if (candidates.length <= 1) return candidates;
  return nonMaximumSuppression(candidates, YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K).map((index) => candidates[index]);
}

/**
 * Face rows at detection scale -> faces in photo pixels, mirroring `detect_face_eyes_in_photo`:
 * eyes ordered by image x, highest score first.
 */
export function faceRowsToDetectedFaceEyes(faceRows, detectionScale) {
  const detectedFaces = faceRows.map((faceRow) => {
    const firstEye = [faceRow[4] / detectionScale, faceRow[5] / detectionScale];
    const secondEye = [faceRow[6] / detectionScale, faceRow[7] / detectionScale];
    const [imageLeftEyeXY, imageRightEyeXY] = firstEye[0] <= secondEye[0] ? [firstEye, secondEye] : [secondEye, firstEye];
    return {
      imageLeftEyeXY,
      imageRightEyeXY,
      faceBoxXYWH: faceRow.slice(0, 4).map((value) => value / detectionScale),
      detectionScore: faceRow[14],
      detectionRowAtDetectionScale: faceRow,
    };
  });
  detectedFaces.sort((first, second) => second.detectionScore - first.detectionScore);
  return detectedFaces;
}
