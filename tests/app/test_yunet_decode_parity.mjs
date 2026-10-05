/**
 * YuNet decode parity: the app's JS pre/post-processing vs OpenCV's FaceDetectorYN.
 *
 * Feeds the exact pixels cv2 decoded (private fixtures from generate_app_fixtures.py) through
 * app/js/pipeline/yunet_face_decode.js + onnxruntime-web (Node build) and compares to the
 * OpenCV rows. Requirement: eye centers within 1 px at detection scale; we also report the
 * max difference over every landmark, box edge and score.
 *
 * Run: node tests/app/test_yunet_decode_parity.mjs
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { buildYunetInputTensor, decodeYunetOutputs, faceRowsToDetectedFaceEyes } from "../../app/js/pipeline/yunet_face_decode.js";
import { APP_DIRECTORY, PRIVATE_FIXTURE_DIRECTORY, check, finish, importOnnxRuntimeForNode } from "./node_test_support.mjs";

const EYE_CENTER_TOLERANCE_PIXELS = 1.0;
const referenceDirectory = join(PRIVATE_FIXTURE_DIRECTORY, "yunet_reference");

if (!existsSync(referenceDirectory)) {
  console.log(`SKIP  private YuNet fixtures missing (${referenceDirectory}); run tests.app.generate_app_fixtures`);
  process.exit(0);
}

const ort = await importOnnxRuntimeForNode();
const session = await ort.InferenceSession.create(join(APP_DIRECTORY, "models", "face_detection_yunet_2023mar_dynamic_input.onnx"));

let worstEyeDifference = 0;
let worstAnyCoordinateDifference = 0;
for (const fileName of readdirSync(referenceDirectory).filter((name) => name.endsWith(".json")).sort()) {
  const reference = JSON.parse(readFileSync(join(referenceDirectory, fileName), "utf8"));
  const rgbaBytes = new Uint8ClampedArray(readFileSync(join(referenceDirectory, fileName.replace(".json", ".rgba"))));
  const { tensorData, paddedWidth, paddedHeight } = buildYunetInputTensor(rgbaBytes, reference.width, reference.height);
  const outputs = await session.run({ input: new ort.Tensor("float32", tensorData, [1, 3, paddedHeight, paddedWidth]) });
  const outputsByName = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, tensor.data]));
  const faceRows = decodeYunetOutputs(outputsByName, paddedWidth, paddedHeight);
  const faces = faceRowsToDetectedFaceEyes(faceRows, 1.0);

  check(`${reference.photo_name}: same face count as OpenCV`, faces.length === reference.faces.length, `js ${faces.length}, opencv ${reference.faces.length}`);
  reference.faces.forEach((referenceFace, faceIndex) => {
    const face = faces[faceIndex];
    if (!face) return;
    const eyeDifference = Math.max(
      Math.hypot(face.imageLeftEyeXY[0] - referenceFace.image_left_eye_xy[0], face.imageLeftEyeXY[1] - referenceFace.image_left_eye_xy[1]),
      Math.hypot(face.imageRightEyeXY[0] - referenceFace.image_right_eye_xy[0], face.imageRightEyeXY[1] - referenceFace.image_right_eye_xy[1]),
    );
    worstEyeDifference = Math.max(worstEyeDifference, eyeDifference);
    check(`${reference.photo_name} face ${faceIndex}: eye centers within ${EYE_CENTER_TOLERANCE_PIXELS} px`, eyeDifference <= EYE_CENTER_TOLERANCE_PIXELS, `max ${eyeDifference.toExponential(2)} px`);
    check(`${reference.photo_name} face ${faceIndex}: score matches`, Math.abs(face.detectionScore - referenceFace.detection_score) < 1e-4, `${face.detectionScore.toFixed(5)} vs ${referenceFace.detection_score.toFixed(5)}`);
  });
  // Raw rows (OpenCV order is NMS order); compare every column of every row.
  const opencvRows = [...reference.opencv_face_rows].sort((first, second) => second[14] - first[14]);
  const jsRows = [...faceRows].sort((first, second) => second[14] - first[14]);
  opencvRows.forEach((opencvRow, rowIndex) => {
    if (!jsRows[rowIndex]) return;
    for (let column = 0; column < 14; column += 1) {
      worstAnyCoordinateDifference = Math.max(worstAnyCoordinateDifference, Math.abs(jsRows[rowIndex][column] - opencvRow[column]));
    }
  });
}
check("worst box/landmark coordinate difference over all faces < 0.05 px", worstAnyCoordinateDifference < 0.05, `${worstAnyCoordinateDifference.toExponential(2)} px`);
console.log(`\nworst eye-center difference: ${worstEyeDifference.toExponential(3)} px`);
finish();
