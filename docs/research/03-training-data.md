# 03: Training data, glare synthesis, masks, model, and evaluation for eyeglass glare removal

Research date: 2026-10-05. Every license claim links to or quotes the source. "Could not verify" means exactly that. This is not legal advice; it is the evidence needed for an informed licensing decision.

## Takeaway (5 lines)

1. There is no public, permissively licensed dataset of paired eyeglass glare and clean images. ReyeR (13,610 real pairs) is research-only and not released, and Lyu et al.'s rendered set has no license. Synthetic supervision on real photos is the only clean path, and it has strong precedent: Google's portrait shadow removal trained on 5,000 real faces plus synthesized shadows and generalized to in-the-wild photos.
2. The best real source is the FFHQ eyeglasses subset: about 11.8k to 16.9k of 70k images, filtered to images whose per-image Flickr license is CC BY, CC0, Public Domain, or US Government. Supplement it with FLUX.1-schnell (Apache-2.0) generated portraits and your own photos. CelebA, CelebAMask-HQ, Face Synthetics, MeGlass, and the Unsplash Dataset are all non-commercial or internal-only, so do not train on them.
3. Glare should be synthesized physically: additive light in linear RGB, clipped at the sensor. Each light source is shrunk by the lens acting as a convex mirror (magnification f/(d+f), with measured lens focal lengths of 10 to 268 cm), clipped to the lens shape, appears on both lenses, and on coated lenses is tinted green, blue, or purple. Real data shows five reflection shapes: point, line, area, circular, and flat.
4. Use a NAFNet-style U-Net at width 16 (2.9M parameters, measured) or width 24 (6.5M). It exports cleanly to ONNX once NAFNet's custom LayerNorm autograd function is swapped for plain ops. Train with L1 (Charbonnier), lens-weighted L1, SSIM, an FFT loss, and focal loss on a predicted glare mask. A VGG perceptual loss is optional and is the only license gray spot.
5. On this M4, a measured width-16 training step at batch 8 and 256 px takes 0.57 s, so 100k iterations is about 16 h. Plan on 3k to 5k clean source faces with on-the-fly synthesis. For evaluation, shoot your own paired set (tripod, light on and off). Stock-photo searches return very few usable glare portraits.

## Recommended data plan

- **Source images (clean targets):**
  - FFHQ images labeled "normal" glasses: FFHQ-Aging labels or a glasses classifier you run yourself.
  - Keep only per-image licenses of CC BY 2.0, CC0, Public Domain Mark, or US Government Works, read from `ffhq-dataset-v2.json`.
  - Drop images that already show lens glare (bright-pixel test inside the lens mask, then a manual contact-sheet pass).
  - Add about 500 to 1,500 FLUX.1-schnell portraits for diversity in skin tone, age, frame style, and lighting, plus any consented own photos.
  - Target: 3,000 to 5,000 clean faces, with identity-disjoint train, val, and test splits.
- **Lens masks:**
  - Primary: the `lenses` segmenter from `glasses-detector` (MIT code).
  - Fallback: SAM 2 (Apache-2.0) prompted with a box derived from MediaPipe eye landmarks (Apache-2.0).
  - Last resort: a landmark ellipse.
  - Run everything offline and QA every mask on a contact sheet. Masks are training-time only and never shipped.
- **Synthesis:** procedural and on-the-fly; the recipe is in Q3. Each source image yields unlimited distinct pairs (the shadow-removal precedent used 100 per face, 5,000 faces to 500K examples).
- **Model:**
  - NAFNet-style U-Net, width 16 (2.9M parameters), on an eye-region crop of 256x512 or 384x768.
  - Outputs: a residual RGB image plus a two-channel mask (soft reflection alpha, and saturated "lost" pixels needing inpainting).
  - Composite the result only inside the lens mask so the rest of the face is untouched by construction.
- **Losses:**
  - Charbonnier/L1: 1.0. Extra lens-region L1: 2.0. 1-SSIM: 0.2. FFT amplitude L1: 0.05. Focal BCE on the masks (alpha 0.25, gamma 2): 1.0.
  - Optional VGG perceptual loss at 0.01 to 0.1. No GAN in v1, because GANs hallucinate eyes.
- **Size and budget:** 100k to 200k iterations at batch 8 and 256 px, about 16 to 32 h on this M4, measured below. Synthesis cost is not included in that figure, so profile it.

---

## Q1. Real face datasets with eyeglass wearers

| Dataset | Size | Glasses count | Lens/glasses masks? | License (source) | OK for a public non-revenue model? |
|---|---|---|---|---|---|
| FFHQ (NVIDIA) | 70,000 at 1024x1024; per-image metadata JSON with license, author, Flickr URL, and landmarks ([repo](https://github.com/NVlabs/ffhq-dataset)) | 16,923 with eyewear, including sunglasses ([Zenodo bbox set](https://zenodo.org/records/14252074), [paper](https://doi.org/10.3390/s24237697)); 11,778 with eyeglasses via face parsing ([Lyu et al. 2022, Sec. 5](https://arxiv.org/abs/2203.10474)) | No | Images: "Creative Commons BY 2.0, Creative Commons BY-NC 2.0, Public Domain Mark 1.0, Public Domain CC0 1.0, or U.S. Government Works". Dataset as a whole: "Creative Commons BY-NC-SA 4.0 license by NVIDIA Corporation" ([repo](https://github.com/NVlabs/ffhq-dataset)) | **Gray, leaning yes.** Our use is non-commercial. Filtering to non-NC per-image licenses removes the photographer-side NC issue. NVIDIA's BY-NC-SA on the aligned compilation could attach ShareAlike to the weights, so either release the weights under CC BY-NC-SA 4.0 or re-crop from the in-the-wild originals. Per-license counts could not be verified without downloading the JSON. |
| FFHQ-Aging labels | Labels for the 70k FFHQ images | Glasses type "none, normal or dark" per image (from Face++) | 19-class semantic maps (CelebAMask-HQ scheme, from DeepLabV3) | CC BY-NC-SA 4.0 ([repo](https://github.com/royorel/FFHQ-Aging-Dataset)) | **Gray.** Fine for filtering only (labels not shipped). Its parsing maps come from a CelebAMask-HQ-trained net. |
| ffhq-features-dataset | Per-image attribute JSON | Has a glasses field ("NoGlasses" shown); other values and counts could not be verified | No | Page states CC BY-NC-SA 4.0 by NVIDIA ([repo](https://github.com/DCGM/ffhq-features-dataset)) | Gray; filtering only |
| FFHQ eyeglasses bboxes (Zenodo) | 16,923 bboxes | Same | Bboxes only, no masks | CC BY-NC-ND 4.0 ([Zenodo](https://zenodo.org/records/14252074)) | **No** for anything derived (ND) |
| CelebA | 202,599 | 13,193 with Eyeglasses attribute ([Lyu et al.](https://arxiv.org/abs/2203.10474), [Hu et al.](https://arxiv.org/pdf/1909.06989)) | No | "available for non-commercial research purposes only"; "not to further copy, publish or distribute" ([CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) | **No.** A public tool is not "research", and the images are celebrity web scrapes with no per-image license. |
| CelebAMask-HQ | 30,000 at 512 masks | Subset (count not verified) | `eye_g` class, label index 3 ([g_mask.py](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py)). Whether it covers frame plus lens or frames only is not stated in the repo; could not verify. Inspect samples before relying on it. | "RESTRICTED to non-commercial research and educational purposes" ([repo](https://github.com/switchablenorms/CelebAMask-HQ)) | **No** for training. Models trained on it (face parsers) are gray even for offline mask generation. |
| MeGlass | 47,917 at 120x120 crops (originals also available), 1,710 identities | 14,832 with black eyeglasses | No | Repo is MIT, but images are "selected and cleaned from MegaFace" ([repo](https://github.com/cleardusk/MeGlass)). MIT covers the repo, not the photographers' or MegaFace's terms. Current MegaFace terms and status could not be verified. | **No.** Tiny crops, and the license is not the repo's to give. |
| Microsoft Face Synthetics | 100,000 renders at 512x512 | Eyewear (11 items) in the asset library ([paper Sec. 3.5](https://arxiv.org/abs/2109.15102)) | `GLASSES = 16`; "Opaque eyeglass lenses are labeled as GLASSES, while transparent lenses as the class behind them" ([repo](https://github.com/microsoft/FaceSynthetics)) | Microsoft Research Use of Data Agreement, "non-commercial research purposes" ([project](https://microsoft.github.io/FaceSynthetics/), [repo](https://github.com/microsoft/FaceSynthetics)) | **No.** Its transparent-lens labeling also means no lens masks. |
| Lyu et al. "take-off-eyeglasses" synthetic | 29,200 training renders: 73 FaceScape-style scan identities, 21 artist glasses, 367 Poly Haven HDRIs ([paper](https://arxiv.org/abs/2203.10474)) | All | Glasses and shadow masks (intermediate supervision) | No license stated in the repo; GitHub API shows none ([repo](https://github.com/StoryMY/take-off-eyeglasses)) | **No.** No license means all rights reserved. Their face scans also carry their own research terms. |
| ReyeR (ER2Net, TCSVT 2024) | 13,610 real paired glare and no-glare images, 356 people, 24 glasses, 5 light types, 5 angles ([paper PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf)) | All | Derived reflection mask: threshold at 0.7 times the max of (input minus GT) | "All the privacy data have been permitted for research purpose only." No download link found ([paper](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf), [IEEE](https://ieeexplore.ieee.org/document/10539112/)) | **No.** Not released, research-only consent. Its capture protocol is the template for our eval set. |
| De-Glared (2024) | 2,541 images (synthetic plus real) ([ResearchGate](https://www.researchgate.net/publication/378539880_De-Glared_Eyeglasses_Glare_and_Reflection_Removal_Using_Deep_Neural_Networks)) | All | Could not verify (PDF returned 403) | Could not verify any release or license | **No** (unavailable) |
| Watanabe & Hasegawa (blue-light-cut lenses, GAN-generated pairs) | Could not verify | All | Could not verify | Could not verify ([RG](https://www.researchgate.net/publication/353855910_Reflection_Removal_on_Eyeglasses_Using_Deep_Learning), [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12177/1217722/Dataset-generation-with-GAN-for-reflection-image-removal-on-eyeglasses/10.1117/12.2626935.short)) | **No** (unavailable) |
| FairFace | 108,501 at 448x448 ([HF card](https://huggingface.co/datasets/HuggingFaceM4/FairFace/blob/main/README.md)) | No glasses label | No | Authors state CC BY 4.0 ([repo](https://github.com/joojs/fairface)); images from YFCC100M. Per-image licenses (some likely NC) are not tracked in the release. | **Gray.** Low resolution, no glasses label. Skip. |
| LFW | Could not verify (site DNS failed during research) | n/a | No | Could not verify any license; images are news photos | **No** (no license) |
| Open Images V7 | Millions of images | Has eyewear classes (counts not verified) | Masks for glasses could not be verified | Annotations CC BY 4.0; "The images are listed as having a CC BY 2.0 license", but Google makes no representation per image ([facts page](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)) | **Gray-yes**, but low face resolution. Not worth the effort over FFHQ. |
| Unsplash (website license) | n/a | Searchable | No | "irrevocable, nonexclusive, worldwide copyright license to download, copy, modify, distribute, perform, and use images from Unsplash for free, including for commercial purposes". Restrictions: no selling without significant modification; no "Compiling images from Unsplash to replicate a similar or competing service". No mention of ML ([license](https://unsplash.com/license)). | **Gray-yes** for a handful of hand-picked images. Do not bulk-scrape (compilation clause, and the dataset terms below cover bulk use). |
| Unsplash Dataset (Lite, 25k) | 25,000 | Some | No | Lite: "internally use ... to train machine learning models ... for your internal business purposes"; may not "disclose, deliver, disseminate, or publish any portion of the Licensed Data" ([TERMS.md](https://github.com/unsplash/datasets/blob/master/TERMS.md)) | **No** for a published model (internal-only) |
| Pexels | n/a | Searchable | No | Free to use and modify; "Identifiable people may not appear in a bad light or in a way that is offensive"; no redistribution on stock platforms. No mention of ML ([license](https://www.pexels.com/license/)). | **Gray-yes** for a handful of hand-picked eval or demo images |

How Creative Commons itself frames training: "To the extent your AI training is covered by an exception or limitation to copyright, you need not rely on CC licenses for the use", but "to the extent you are relying on CC licenses to train AI, you will need to follow the relevant requirements" ([CC, 2023](https://creativecommons.org/2023/08/18/understanding-cc-licenses-and-generative-ai/)). This is why the plan prefers the CC BY, CC0, and PD subset of FFHQ and ships an attribution file listing source images.

Datasets from eye-tracking and NIR work: [Rendering refraction and reflection of eyeglasses for synthetic eye tracker images](https://arxiv.org/abs/1511.07299) renders glasses at near-infrared (900 nm) for head-mounted trackers. That is the wrong spectrum and viewpoint for RGB portraits, and no relevant licensed glare dataset was found there.

## Q2. Fully synthetic sources

| Option | License of outputs | Glasses quality | Verdict |
|---|---|---|---|
| FLUX.1-schnell | Model is Apache-2.0; "can be used for personal, scientific, and commercial purposes"; out-of-scope uses are harm categories only ([model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)) | Good portraits; may produce frame artifacts (lensless or asymmetric frames), so QA is needed | **Yes.** Best synthetic supplement, and gives no-real-person demo images. Runs locally via mflux. |
| FLUX.1-dev | v1.1.1 Outputs clause: "You may use Output for any purpose (including for commercial purposes) ... You may not use the Output to train, fine-tune or distill a model that is competitive with the FLUX.1 [dev] Model". Non-commercial purpose excludes uses "in direct interactions with or that has impact on end users" ([LICENSE.md](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)) | Better than schnell | **Gray.** A tiny glare remover is not "competitive with" FLUX, but schnell avoids the question entirely. |
| SDXL base 1.0 | CreativeML Open RAIL++-M; licensor claims no rights in outputs, with use-based restrictions ([model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)) | Weaker faces | Yes, but no reason over schnell |
| 3D rendering (Blender + glasses assets) | Depends on assets and face models. Lyu used FaceScape-style scans plus artist glasses; Microsoft used a proprietary parametric face ([Lyu](https://arxiv.org/abs/2203.10474), [Wood](https://arxiv.org/abs/2109.15102)) | Physically exact lens reflections | **Not for v1.** No open, permissively licensed photoreal face-scan library was found; could not verify one exists. |

Evidence on synthetic-to-real transfer:

| Evidence | What it shows | Source |
|---|---|---|
| Synthetic degradations on real faces | Portrait shadow removal trained on 5,000 real in-the-wild faces with synthesized foreign shadows (500K examples) "generalize[s] to images taken in the wild". Ablations show each realism term matters (spatially varying blur, subsurface scatter, color jitter). | [Zhang et al. SIGGRAPH 2020](https://arxiv.org/abs/2005.08925) |
| Fully rendered faces | "Although the accurate 3D information and the high-end rendering technique improve the photo-realism ... the network still cannot generalize well to real images", so they added adversarial domain adaptation | [Lyu et al. CVPR 2022](https://arxiv.org/abs/2203.10474) |
| Fully rendered faces, huge investment | Synthetic-only training works for landmarks and parsing when the domain gap is minimized "at the source" (100k renders, artist assets) | [Wood et al. ICCV 2021](https://arxiv.org/abs/2109.15102) |
| Generic SIRR | Models trained on synthetic blends degrade on real photos. Zhang et al. mix 5,000 synthetic with 500 real patches. ER2Net (real pairs) still fails on sunlight reflections absent from training. | [Zhang 2018](https://arxiv.org/abs/1806.05376), [ER2Net PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf) |

Conclusion: keep the clean image real (or diffusion-photoreal) and synthesize only the glare layer. That is the configuration with the strongest transfer evidence. Fully rendered faces need domain adaptation and much more work.

## Q3. Glare synthesis

### What the SIRR literature does

| Method | Blend model | Reflection treatment | Source |
|---|---|---|---|
| CEILNet (Fan 2017) | I = T + blurred R, with "adaptive subtraction and clipping" instead of scaling to avoid overflow | Gaussian blur sigma random in [2, 5] | [repo](https://github.com/fqnchina/CEILNet), [paper](https://fqnchina.github.io/QingnanFan_files/iccv_2017.pdf) |
| Zhang 2018 | I = T + R in linear space ("We remove gamma correction ... operate in linear space") | Gaussian kernel size 3 to 17 px; varied intensity ("reflection ... could have comparable or higher intensity"); "slight vignette centered at random position in the reflection layer" | [arXiv 1806.05376](https://arxiv.org/abs/1806.05376) |
| ERRNet code | `ReflectionSythesis_1`: blur sigma 2 to 5, then B + R_blur, then if over 1 subtract `(mean(excess) - 1) * gamma` (gamma = 1.3) and clip. `ReflectionSythesis_2`: gamma 2.2 to linear, blur, add, per-channel overflow subtraction (att about 1.08), random Gaussian vignetting mask, gamma back. | | [transforms.py](https://github.com/Vandermode/ERRNet/blob/master/data/transforms.py) |
| Wen 2019 | Non-linear alpha blending mask instead of a linear sum | Learned synthesis network | [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wen_Single_Image_Reflection_Removal_Beyond_Linearity_CVPR_2019_paper.html) |
| Kim 2020 | Path-traced through glass (thickness, refractive index 1.6) | Ghosting, attenuation, defocus, multiple bounces | [arXiv 1904.11934](https://arxiv.org/abs/1904.11934), [repo (MIT)](https://github.com/sookim813/Reflection_removal_rendering) |

### What is specific to eyeglass lenses

| Property | Evidence | Source |
|---|---|---|
| The lens outer surface acts as a convex mirror and minifies. Magnification is h_i/h_o = f_g/(d_o + f_g), with f_g = R/2. "Smaller curvature leads to larger-size reflections." The outer surface dominates; the inner surface also reflects. | Measured f_g across 16 pairs: min 10 cm, max 268 cm, mean 110 cm. BLB glasses f_g = 8 cm; prescription pair f_g = 50 cm. Lens chord 5x4 cm and 6x5 cm. Higher prescriptions use flatter fronts, so they show larger reflections. | [Private Eye (Long et al.)](https://arxiv.org/abs/2205.03971), Sec. II-B, III-C, IV-E, App. A |
| Reflectance | Uncoated n = 1.5: about 4% per surface; n = 1.67: about 6.3%; good AR coating: about 1% per surface | [US patent 8425035 (spectacle AR)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8425035) |
| AR residual color | "usually green"; ZEISS coatings have "bluish residual reflection" | [ZEISS](https://www.zeiss.com/vision-care/us/eye-health-and-care/health-prevention/lens-coatings-anti-reflective-hard-layer-cleancoat-etc.html), [patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8425035) |
| Colors in real portrait data | "different glasses display different colors for reflection, including purple, blue, green" | [ER2Net PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf), Sec. V-H |
| Shape taxonomy | ReyeR categories: point, line, area, circular, and flat reflections. Sources: round light, phone panel light, strip light, desk lamp, flat light. | [ER2Net PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf), Fig. 2, Table I |
| Weak vs strong | Weak reflections are removable; strong ones need inpainting, which is why ER2Net fuses an elimination branch and an inpainting branch per pixel by reflection intensity. Its mask target: (input minus GT) above 0.7 times the max. | same, Sec. IV |
| Coating wear | Worn coatings change reflection quality (an empirical factor with correlation 0.31) | [Private Eye](https://arxiv.org/abs/2205.03971), Sec. IV-E |
| Motion and focus | Webcam reflections show defocus blur, shot noise, and tremor motion blur | [Private Eye](https://arxiv.org/abs/2205.03971), Sec. III-D |

One correction to the brief: the two lenses do not show left-right mirrored copies of each other. Each lens shows the same mirror image of the scene with the same handedness. Face-form wrap and pantoscopic tilt place the highlights roughly symmetrically about the face midline, so the positions mirror but the content does not. This is derived from mirror geometry, not taken from a cited source.

### Recipe (procedural, Python, per sample)

All compositing happens in linear RGB: `lin = srgb_to_linear(img)`, then add light, then simulate the camera, then `linear_to_srgb`.

1. **Lens geometry.**
   - Get the left and right lens masks M_L and M_R from Q4.
   - Fit each one to a box and center c and estimate the lens width w_px.
   - Erode each mask by 1 to 2 px so glare never paints onto the frame.
2. **Lens optics per sample (shared by both lenses):**
   - f_g log-uniform in [0.08, 2.7] m (Private Eye range).
   - Coating: 70% AR-coated with base reflectance 0.008 to 0.015 and a tint drawn from {green (0.35, 1.0, 0.45), blue (0.4, 0.6, 1.0), purple (0.8, 0.45, 1.0)}, jittered ±15%; 30% uncoated with reflectance 0.04 to 0.065 and neutral tint.
   - These reflectances scale source radiance relative to the face exposure. Saturation comes from the source being far brighter than the face, not from high reflectance.
3. **Choose 1 to 3 light sources.** Each has a type, a distance d in [0.3, 3] m, a physical size H, and a relative radiance L (multiple of the face's mean linear luminance):

   | Type | Shape in the lens | H | L (times face) | Notes |
   |---|---|---|---|---|
   | Ring light | Annulus (inner/outer about 0.7), sometimes a partial arc | 0.25 to 0.5 m | 20 to 200 | Often clipped by the rim. Inner hole shows the eye. |
   | Softbox / LED panel | Rounded rectangle, soft edge | 0.4 to 1.2 m | 10 to 80 | ReyeR "flat" and "area" |
   | Window | Quadrilateral with 1 to 3 dark mullion bars and a faint outdoor-scene texture (blurred random photo) | 0.8 to 2 m | 5 to 50 | Low-contrast wash outdoors, plus a sky gradient |
   | Monitor / laptop | Rectangle, cool white (6500 to 9000 K), optional blurred UI or text texture | 0.3 to 0.7 m at d 0.3 to 0.7 m | 1 to 8 | Often semi-transparent, so recoverable. Private Eye geometry. |
   | Point / bulb / phone flash | Small disc or ellipse with a bloom halo | 0.02 to 0.1 m | 50 to 1000 | ReyeR "point". Saturates. |
   | Strip / tube | Thin long rounded bar | 0.6 to 1.2 m by 0.03 m | 20 to 200 | ReyeR "line" |
   | Ceiling grid | Several small rectangles in a lattice | | 5 to 30 | Office scenes |

4. **Size and position.**
   - Virtual image size in metres: s = H·f_g/(d + f_g). In pixels: s_px = s/0.05 · w_px, taking a typical lens width of 0.05 m.
   - This produces anything from a tiny glint (f_g = 8 cm, d = 2 m gives 4% of the source size) to a reflection larger than the lens that gets clipped (f_g = 2 m).
   - Apply a curved-mirror warp: radial barrel distortion k1 in [0.05, 0.3] about the lens center, plus a small random homography (±10°) to mimic wrap and tilt.
   - Place it at c + offset (offset uniform within ±0.4·w_px). The other lens gets the same source with its offset mirrored about the face midline, plus jitter of ±0.1·w_px, a scale ratio of 0.9 to 1.1, and a 0.7 to 1.0 intensity ratio.
   - With probability 0.15, show the source in one lens only (head turned, or the source occluded by the brow or frame).
5. **Lens-specific layers.**
   - Ghost: with probability 0.4, add a second, fainter (0.1 to 0.3 times), smaller (0.6 to 0.9 times), offset copy for the inner-surface reflection ([Private Eye](https://arxiv.org/abs/2205.03971) App. A, [Kim 2020](https://arxiv.org/abs/1904.11934)).
   - Veil: with probability 0.5, add a low-frequency haze across the whole lens (a sky or room gradient at 0.02 to 0.15 times face luminance, times the tint).
   - Coating texture: with probability 0.2, multiply the reflection by low-amplitude Perlin noise to mimic coating wear and smudges (the Perlin idea is borrowed from the [shadow-synthesis model](https://arxiv.org/abs/2005.08925)).
6. **Blur.** Defocus blur, because the reflection's virtual image is not at the face's focal plane: a disc kernel of radius 0.5 to 6 px at 512-px face scale. Optionally add a Gaussian sigma of 0.5 to 3 (CEILNet uses [2, 5] for scene reflections; lens glints are often sharper, so use a lower range). Add a slight motion-blur line kernel with probability 0.1.
7. **Composite.**
   - R_lin = sum over sources (shape × L × reflectance × tint × warp), then apply the rim falloff (a soft vignette toward the lens edge: multiply by a 0.7 to 1.0 radial ramp).
   - Mask R_lin by the lens mask (with a 1 to 2 px feather).
   - I_lin = T_lin · (1 - reflectance_mean) + R_lin. The transmission loss is about 1 to 6% and can be ignored when coated.
8. **Camera.**
   - Clip at 1.0, with probability 0.6 applying a soft-knee highlight roll-off before clipping (phones tone-map).
   - Add bloom around clipped pixels (a Gaussian of the clipped excess, sigma 2 to 8 px, at 0.1 to 0.3).
   - Add sensor noise (Poisson-Gaussian), then convert back to sRGB.
   - Apply JPEG quality 70 to 95 to the input only, never to the target.
9. **Targets.**
   - Clean image: T (sRGB).
   - `alpha_mask`: normalized luminance of R_lin inside the lens, which is the soft reflection strength.
   - `lost_mask`: pixels where I_lin hit the clip or R_lin is above 0.85 of the total. Information there is gone and must be inpainted (the ER2Net thresholding idea).
10. **Curriculum and mix.**
    - Per batch: 15% of samples with no glare at all (teaches identity), 45% weak/semi-transparent glare (monitor, veil, AR tint), 40% strong/saturated glare.
    - Hold out one light type (for example ceiling grid) in validation to check generalization.

## Q4. Getting the lens mask on clean photos

| Option | What it segments | License | Offline practicality |
|---|---|---|---|
| `glasses-detector` lenses segmenter | Kinds: `full` (frames + lenses), `frames`, `legs`, `lenses`, `shadows`, `smart` (lenses only if opaque) | Code MIT ([repo](https://github.com/mantasu/glasses-detector)). Weight license not stated. Trained on Roboflow sets plus a Face Synthetics glasses set; their licenses are not documented, so could not verify. Needs Python ≥ 3.12. | **Best first pass.** Directly gives lenses. Masks are a training-time intermediate, not shipped. Gray only through the weights' training data. |
| SAM 2 prompted with box/points | Whatever you prompt; a box around each eye plus a positive point at the pupil usually returns the lens interior | Apache-2.0 ([repo](https://github.com/facebookresearch/sam2)) | **Best fallback**, and fully clean. Needs a box from landmarks. Watch for it returning the eye or iris instead of the lens; use multi-mask output and pick the mask whose area best matches the frame interior. |
| MediaPipe Face Landmarker + ellipse | Eye contour landmarks, scaled ellipse | Apache-2.0 ([repo](https://github.com/google-ai-edge/mediapipe)) | Cheap last resort. Real lens shapes (rectangular, cat-eye, aviator) fit poorly. Use it to make SAM prompts, and as the browser-side crop at inference. |
| Face parsing, CelebAMask-HQ `eye_g` (BiSeNet [zllrunning, MIT code](https://github.com/zllrunning/face-parsing.PyTorch), SegFormer [jonathandinu, NC license](https://huggingface.co/jonathandinu/face-parsing)) | One region for the glasses. Whether lenses are included could not be verified. | Weights derive from non-commercial CelebAMask-HQ | Gray-no. Usable only as a cross-check. |
| Lyu et al. mask network | Glasses and shadow masks | No license ([repo](https://github.com/StoryMY/take-off-eyeglasses)) | No |

Recommendation for a few thousand images:

1. Run MediaPipe to get eye boxes.
2. Run the `glasses-detector` `lenses` model.
3. If its mask is empty, fragmented, or covers more than about 1.6 times the eye-box area, run SAM 2 with the eye box plus a pupil point.
4. Keep a mask only if two methods agree with IoU above 0.7, or if it passes a 6x6 contact-sheet review (about 1 to 2 h by hand for 4k images).

Store masks as PNGs next to the images, with a manifest JSON.

## Q5. Architecture, export, losses, and budget

| Model | Params | Export notes | License | Fit |
|---|---|---|---|---|
| NAFNet, width 16, enc [1,1,1,1], mid 1, dec [1,1,1,1] | **1.14M** (measured here, 4.5 MB fp32) | Ops: conv, `chunk` (Split), `AdaptiveAvgPool2d(1)` (GlobalAveragePool), `PixelShuffle` (DepthToSpace), mul/add. **`LayerNorm2d` is a custom `torch.autograd.Function`** ([arch_util.py](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/arch_util.py)), so swap it for plain mean/var ops before `torch.onnx.export` (verified the swap runs forward). All these ops are listed as supported by ORT WebGPU; Resize lacks `align_corners` with downsampling ([ORT WebGPU ops](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)). | MIT (LICENSE file) ([repo](https://github.com/megvii-research/NAFNet)) | Tiny baseline |
| **NAFNet w16, enc [1,1,2,4], mid 4, dec [1,1,1,1]** | **2.93M** (11.7 MB fp32, about 6 MB fp16) | same | MIT | **Recommended** |
| NAFNet w24, same depth | **6.53M** (26 MB fp32) | same | MIT | If w16 underfits |
| NAFNet w32 GoPro config | 17.11M (measured; matches the paper's scale) | same | MIT | Too big for the browser budget |
| Restormer | About 26M (paper); transposed attention uses MatMul and Softmax over channels | Heavy for WebGPU at 512 px | MIT ([repo](https://github.com/swz30/Restormer)) | No |
| MIMO-UNet / MIMO-UNet+ | 6.8M / 16.1M ([paper](https://arxiv.org/abs/2108.05054)) | Multi-scale inputs and outputs, plus an FFT loss idea worth borrowing | **No LICENSE file** (GitHub API: none) ([repo](https://github.com/chosj95/MIMO-UNet)), so all rights reserved: do not copy the code | Idea only |
| Plain residual U-Net / MobileNet-encoder U-Net | 1 to 5M | Trivially exportable | Your own code | Fine fallback; NAFNet blocks are the better-tested default |
| ER2Net (eyeglass-specific) | Could not verify the count | Has flow and partial convolutions, which complicate export | Could not verify code release | Borrow the ideas (two branches, fusion by mask, eye-symmetry loss), not the code |

Loss evidence:

| Paper | Losses and weights |
|---|---|
| ER2Net (eyeglass) | L2 reconstruction on 3 outputs (1.0); VGG-19 perceptual (0.01); focal loss on reflection detection with alpha 0.25, gamma 2 (1.0); fusion weight loss (10.0); flow (1.0); eye-symmetry (1.0). The eye-symmetry loss beat an adversarial loss. Adam, lr 1e-4 decayed to 1e-5, 80 epochs on 13.6k pairs. ([PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf)) |
| Zhang 2018 (SIRR) | VGG-19 feature loss on conv1_2 through conv5_2 (0.1), adversarial (0.01), gradient-domain exclusion loss (1.0), plus L1 on the predicted R. Without a GAN they saw "unrealistic color degradation and undesirable subtle residuals". ([arXiv](https://arxiv.org/abs/1806.05376)) |
| NAFNet | PSNR loss only; Adam; 200K iterations for the main benchmarks ([arXiv](https://arxiv.org/abs/2204.04676)) |

Notes on the loss choice:

- **VGG license.** The torchvision docs say "The pre-trained models provided in this library may have their own licenses or terms and conditions derived from the dataset used for training. It is your responsibility to determine whether you have permission" ([torchvision models](https://docs.pytorch.org/vision/stable/models.html)). VGG is used only at training time and nothing from it ships, so this is gray. LPIPS code is BSD-2 ([repo](https://github.com/richzhang/PerceptualSimilarity)) but wraps the same ImageNet backbones. Train v1 without it; add VGG at 0.01 only if the outputs look waxy.
- **Exclusion loss.** It needs an explicit R-layer prediction. Our residual output is effectively R, so it is cheap to add as an ablation.
- **Masks.** Use a mask head with focal loss, and composite the output only through the predicted lens and glare mask so the eyes outside the glare stay bit-identical.

Data size and time:

- **Data size precedents.** Zhang 2018 used 5,000 synthetic plus 500 real patches. Portrait shadows used 5,000 real faces expanded to 500K synthetic examples. ER2Net used 13.6k real pairs. With on-the-fly synthesis, 3k to 5k diverse clean faces is the right order.
- **Measured on this Apple M4 (16 GB), PyTorch 2.12 on MPS, L1 loss plus AdamW, random tensors, warm-up excluded:**
  - w16 at batch 8, 256 px: 566 ms per iteration (100k iterations is about 15.7 h).
  - w24 at batch 8, 256 px: 873 ms per iteration (about 24.2 h).
  - w16 at batch 4, 512 px: 1,160 ms per iteration (about 32.2 h).
  - These figures exclude data-loading and synthesis cost. Run the synthesis in DataLoader workers or on the GPU in torch, and measure it.
- **ONNX export.** Not tested here, because the `onnx` package is not installed in that venv and nothing was installed. The forward pass with the swapped LayerNorm works.

## Q6. Real evaluation images

The most practical and legally clean path is to shoot your own paired set, following the ReyeR studio protocol (turn the reflection light off for ground truth, filter pairs with a large pixel offset, Reinhard color transfer to fix tone drift; [PDF](https://web.comp.polyu.edu.hk/pli/CoRR/TCSVT/TCSVT2024_2.pdf)). The tripod-video protocol also works: 100 real pairs from 20 videos of 8 subjects ([Zhang et al. 2020](https://arxiv.org/abs/2005.08925)). Shoot 20 to 50 pairs: phone on a tripod, subject still, toggling a ring light, a monitor at full brightness, a window (curtain open and closed), and a desk lamp. Use several glasses (AR-coated and uncoated). Volunteers sign a simple consent for public demo use. This also gives real PSNR and SSIM numbers.

Stock and Commons candidates are unpaired, for qualitative checks and possible demos. **Content was not visually verified**; view each one before use.

| Image | License (exact) | Notes |
|---|---|---|
| [Anti-reflective_coating_comparison.jpg](https://commons.wikimedia.org/wiki/File:Anti-reflective_coating_comparison.jpg) | CC BY-SA 3.0 / 2.5 / 2.0 / 1.0 or GFDL 1.2+ (choose one); author Justin Lebar | Lenses only, coated vs uncoated. Good explainer figure; BY-SA means a demo derivative is BY-SA. |
| [Перлик_А.А._во_время_лекции_2016_г..jpg](https://commons.wikimedia.org/wiki/File:%D0%9F%D0%B5%D1%80%D0%BB%D0%B8%D0%BA_%D0%90.%D0%90._%D0%B2%D0%BE_%D0%B2%D1%80%D0%B5%D0%BC%D1%8F_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B8_2016_%D0%B3..jpg) | CC BY-SA 4.0; photographer Danchenkova; 1200x1800 | Lecturer with glasses, in the "Reflections on eyeglasses" category |
| [Claudia_Bahamón_2019.jpg](https://commons.wikimedia.org/wiki/File:Claudia_Bahamón_2019.jpg) | CC BY 3.0; author Jon Saw; 527x729 | Low resolution; may be sunglasses |
| [Category:Reflections_on_eyeglasses](https://commons.wikimedia.org/wiki/Category:Reflections_on_eyeglasses) | Per file | 18 files plus subcategories, including [Reflected photographers in glasses](https://commons.wikimedia.org/wiki/Category:Reflected_photographers_in_glasses) (35 files, many sunglasses). Defense.gov/US military files are US public domain (for example [this one](https://commons.wikimedia.org/wiki/File:Defense.gov_photo_essay_070509-F-3961R-016.jpg)), but many show sunglasses. |
| Pexels [Woman in glasses looking at laptop screen](https://www.pexels.com/photo/woman-in-glasses-looking-at-laptop-screen-7320688/), [Focused woman wearing glasses working late](https://www.pexels.com/photo/focused-woman-wearing-glasses-working-late-36713416/) | [Pexels license](https://www.pexels.com/license/) (free use; no "bad light" depiction of identifiable people) | Likely screen reflections, from titles only. A before/after "fix" is not a bad-light depiction, but verify. |
| Unsplash free results for "glasses reflection" | [Unsplash license](https://unsplash.com/license) | The search mostly returned glasses on tables. The two best-titled hits ([this one](https://unsplash.com/photos/a-person-is-reflected-in-the-clear-eyeglasses-AYyihKddrA8) and [this one](https://unsplash.com/photos/stock-trading-woman-wearing-eyeglasses-looking-at-computer-screen-reflecting-in-glasses-analyzing-stock-trading-graph-close-up-of-eyes-refection-C62DpFQ0VvE)) are **Unsplash+ (paid)**, so do not use them. |

Recommendation: use your own paired captures for metrics and the hero demo. Use Commons CC BY / CC BY-SA / PD files for a 20 to 40 image qualitative "in the wild" set, with attribution in the site credits. Treat Pexels and Unsplash as optional extras after a visual check.
