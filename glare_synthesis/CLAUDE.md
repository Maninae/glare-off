# glare_synthesis

Procedural, physically based lens-glare renderer. Clean aligned eye crop + lens label mask in, the synthesis data_dict (repo CLAUDE.md, "Synthesis data_dict") out. Runs on the fly in DataLoader workers: numpy + OpenCV only, no torch, no global state. Call `cv2.setNumThreads(1)` in each worker.

```python
from glare_synthesis.render_lens_glare import render_lens_glare_sample
data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), sampling_config, source_id)
```

Any crop size works (512x256 default, 1024x512 tested). The same generator seed gives the same sample.

## Module map

Dependencies flow downward; leaf modules have no repo imports.

| Module | Responsibility |
|---|---|
| `render_lens_glare.py` | Entry point. `render_lens_glare_sample` = `sample_glare_scene` (all randomness) + `render_glare_scene` (deterministic). Owns the region of interest and data_dict assembly |
| `glare_sampling_config.py` | `LightSourceType`, `GlareSeverity`, `GlareSamplingConfig`: the whole sampling distribution |
| `glare_scene.py` | Dataclasses of one sampled scene (`GlareScene`, `SampledLightSource`, `ReflectionPlacement`, `LensVeil`) |
| `glare_scene_sampling.py` | Mirror optics (reflection size), two-lens placement, ghosts, coating, camera choices |
| `light_source_registry.py` | Per-type physical size, distance, colour temperature, spread, rotation, brightness ceiling, and drawer |
| `emitter_canvases.py` | Ring light, softbox/umbrella, point glints, strips, ceiling grid canvases |
| `scene_source_canvases.py` | Window with mullions and outdoor scene, screen with UI blocks, sky wash, room environment |
| `reflection_layer_rendering.py` | Inverse mirror mapping into each lens, defocus/motion/photo blur, tint, rim falloff, wear, veil, lens mask |
| `camera_response.py` | Highlight roll-off and clip, bloom, shot noise, JPEG on the glare delta, photo blur and noise estimators |
| `glare_masks.py` | `glare_mask` and `lost_detail_mask` definitions |
| `color_science.py` | sRGB curve and its slopes, CCT to RGB, AR coating tints |
| `lens_geometry.py` | Per-lens centroid, box, soft interior mask, rim distance |
| `procedural_noise.py` | Smooth random fields |
| `review_source_crops.py` | Loads crops for review: development pairs directory, or the source manifest |
| `render_review_sheets.py` | `python -m glare_synthesis.render_review_sheets [--real-reference-dir DIR]`: writes the contact sheets |
| `benchmark_render_speed.py` | `python -m glare_synthesis.benchmark_render_speed`: single-core ms per sample |

To add a light type: add a `LightSourceType` member, a drawer returning an (N, N) or (N, N, 3) canvas on the unit plane, a `LIGHT_SOURCE_SPECS` entry, and a default probability. Then regenerate the sheets and look at them.

## Physical model

All light is added in linear RGB. Per sample:

1. **Mirror size.** The lens front surface is a convex mirror, f = R/2, with R = 0.53 / base curve (1-8 D), giving f = 3-26 cm. A 10% tail uses f = 0.3-2.7 m (Private Eye's measured range), which produces lens-filling washes. Reflection size relative to the lens is H·f/(d+f) · D/(D+d_i) / lens_width.
   - The research doc's Private Eye f (mean 110 cm) alone would make a 0.3 m ring light 2-4 lens widths wide.
   - The real ring-light reference shows it at about 0.6 lens widths, which base-curve f reproduces.
2. **Placement.** The same source appears in both lenses with the same orientation and handedness. The x offset = mirrored wrap term + common source-direction term; the second lens gets ±0.1 jitter, 0.9-1.1 scale and 0.7-1.0 intensity. 15% of samples show the source in one lens only. Per-lens barrel distortion k1 0.05-0.3, keystone ±0.1, vertical squash 0.85-1. Horizontal foreshortening = lens width / widest lens, which covers head-turned crops.
3. **Lens layers.**
   - Ghost (inner-surface reflection): 40% probability, 0.3-1.2x scale, 0.1-0.35x intensity.
   - Veil: 40% probability, 0.005-0.04 linear, gradient with soft lumps.
   - AR coating: 70% of lenses. Tint is green/blue/purple/magenta, mixed 50-90% with white so blown cores read near-white. 35% of coated lenses get a tint drift across the lens.
   - Coating wear: 25% probability.
   - Rim falloff: floor 0.6-1.0.
   - Soft lens mask: eroded 1.5 px, feathered, clamped to the lens.
4. **Blur.** Defocus disc of 0.6-6% of lens width, capped at 25% of the reflection's own size (small far sources image near f, close to the focal plane). Motion streak with 10% probability. Then the photo's own blur, estimated from the sharpest edges with the re-blur method, so glare is never sharper than the photo.
5. **Camera.**
   - out = clean + T(clean·(1-a) + R) - T(clean). T is the soft-knee roll-off (60%, knee 0.6-0.85) or a hard clip; a is local tone-map attenuation (15%, 1-5%).
   - Bloom: Gaussian of the over-exposure, sigma 1.5-6% of lens width, strength 0.08-0.3.
   - Shot noise: variance g·R at the photo's own gain g (estimated with the Immerkaer method), scaled 0.7-1.4x.
   - JPEG (50%, quality 60-95) applied as clean + jpeg(glared) - jpeg(clean).
6. **Brightness.** Peak added linear light (1 = clip). WEAK is 0.06-0.45 (53% of glared samples); STRONG is 0.8-8 (47%). Per-type ceilings: environment 0.2, screen 1.0, sky wash 1.3, window 2.0. Point glints get a 2-6x boost and strips 1-2.5x, because blur spreads their energy.
7. **Mix.** 12% glare-free; 1/2/3 sources at 60/30/10%.

## What differs between input and target

Only these, and every other pixel is bit-identical (tested):

- Reflected light inside the lenses.
- Its clipping and roll-off.
- Bloom up to ~3 bloom sigmas outside the lens.
- Local attenuation under the glare.
- Shot noise of the added light.
- JPEG artifacts of the glare delta, confined to the 16x16 blocks the glare touches.

The face's own noise and JPEG are never altered.

## Masks

Both are computed on the noise-free, JPEG-free render.

- **`glare_mask`:** smoothstep of the largest per-channel sRGB change, from 0.012 to 0.06.
- **`lost_detail_mask`:** the remaining contrast of the transmitted image, d(glared sRGB)/d(clean sRGB), computed analytically by the chain rule through the decode slope, the highlight response slope, and the encode slope.
  - It takes the maximum over channels.
  - It is 0 where the output is within half an 8-bit code of white.
  - Lost is 1 at contrast ≤ 0.1 and ramps to 0 at 0.25, then is capped by `glare_mask`, so it is always a subset.

## Speed

Measured with `benchmark_render_speed` on the M4, one thread, development crops:

| Crop size | Mean | p90 | Max |
|---|---|---|---|
| 512x256 | 22.7 ms | 35.7 ms | 46 ms |
| 1024x512 | 91.5 ms | 148 ms | |

## Review

Sheets go to `/Volumes/vega/datasets/glare-off/synthesis-review/latest/`:

- `light_type__*.png`
- `clean_glared_masks.png`
- `severity_ladder.png`
- `real_vs_synthetic.png`

Development crops and hand-drawn lens masks live in `synthesis-review/dev-crops/` (private reference images, never in the repo). Tests: `python -m pytest tests/glare_synthesis -q`.

## Known realism gaps

- Large sky washes are too uniform: a smooth tinted fill that reads like a colour filter. Real sunlight washes have streaks, scene structure, and are whiter.
- Shapes are drawn as clean vector primitives. Real ring lights and softboxes show diffuser texture, LED dots, and irregular edges.
- Window panes at high brightness become flat white rectangles with crisp mullions, which look pasted.
- No reflected scene geometry (the photographer, the phone, room objects) beyond the blurred procedural environment, which is low-detail.
- No catchlight interaction: corneal reflections are not added, and glare does not interact with the eye's own catchlight.
- AR tint is uniform per lens apart from the drift. Real coatings show angle-dependent rainbow edges near the rim.
- Calibrated only against 3 real reference crops plus by-eye judgement; there is no paired real evaluation yet.
