/**
 * Per-face switch state for the face picker: the gate's automatic decision, plus the
 * visitor's override.
 *
 *   glassesDetected  overridden  ->  state     switched on?
 *   true             false           auto-on   yes (glare model ran; its delta is blended)
 *   false            false           auto-off  no  (glare model never ran; pixels untouched)
 *   false            true            user-on   yes (glare model runs on demand, from the kept crop)
 *   true             true            user-off  no  (excluded from the blend and the download)
 *
 * A click flips `overridden`, so a second click always returns to the automatic decision.
 * "No glare found" is reported separately: a switched-on face whose glare model output was empty.
 */

export const FACE_SWITCH_STATES = {
  AUTO_ON: "auto-on",
  AUTO_OFF: "auto-off",
  USER_ON: "user-on",
  USER_OFF: "user-off",
};

export function isFaceSwitchedOn(glassesDetected, overridden) {
  return glassesDetected !== overridden;
}

export function faceSwitchState(glassesDetected, overridden) {
  if (glassesDetected) return overridden ? FACE_SWITCH_STATES.USER_OFF : FACE_SWITCH_STATES.AUTO_ON;
  return overridden ? FACE_SWITCH_STATES.USER_ON : FACE_SWITCH_STATES.AUTO_OFF;
}

/**
 * The tile's label lines, in sentence case.
 * `face`: the face result (glareRun, hasGlare); `isBusy`: its on-demand glare run is in flight.
 */
export function describeFaceTile(face, overridden, isBusy) {
  if (isBusy) return ["Working"];
  const noGlareFound = face.glareRun && !face.hasGlare;
  switch (faceSwitchState(face.glassesDetected, overridden)) {
    case FACE_SWITCH_STATES.AUTO_ON:
      return [noGlareFound ? "No glare found" : "Glare removed"];
    case FACE_SWITCH_STATES.AUTO_OFF:
      return ["No glasses, skipped"];
    case FACE_SWITCH_STATES.USER_ON:
      return noGlareFound ? ["Forced on", "No glare found"] : ["Forced on"];
    default:
      return ["Forced off"];
  }
}
