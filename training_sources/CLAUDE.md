# training_sources/

Produces the clean source photos the glare synthesizer paints on, plus everything it needs to know about them: eye centers, lens masks, an existing-glare score, train/val/test splits, licenses, and credits. Output contract: "Source manifest" in the repo-root CLAUDE.md. Consumers use `training_sources/load_source_manifest.py`.

All data lives on vega under `/Volumes/vega/datasets/glare-off/sources/` (paths in `training_sources_paths.py`); nothing heavy enters the repo. Python: `/Volumes/vega/datasets/glare-off/venv/bin/python`, run from the repo root.

## Pipeline

| Stage | Command (`python -m ...`) | Writes | Rerun behaviour |
|---|---|---|---|
| 0 | `training_sources.acquisition.ffhq_metadata_slim_cache` | `ffhq_metadata/ffhq_metadata__slim-fields__n=70000.jsonl` | Built once; streams the 255 MB FFHQ JSON one entry at a time (peak ~0.7 GB). Never `json.load` the full file (2 GB RAM). |
| 1 | `training_sources.acquisition.select_ffhq_candidates` | `stages/ffhq_candidates__*.jsonl` + counts | Deterministic, rewrites. |
| 2 | `training_sources.acquisition.download_ffhq_images` | `ffhq_photos_1024/NNNNN.png` | Resumable; skips files whose md5 matches the official metadata. 4 threads max. |
| 3 | `training_sources.labeling.annotate_sources` | `stages/ffhq_annotations__*.jsonl`, `lens_masks/*.png` | Resumable (skips annotated ids); `--redo-status S` re-runs rows whose latest status is S. 2 CPU workers, ~0.5 GB total. |
| 4 | `training_sources.labeling.score_source_glare` | `stages/ffhq_glare_scores__per-face.jsonl` | Rewrites; ~3 min. Re-run after changing constants in `source_glare_score.py`. |
| 5 | `training_sources.build_source_manifest` | `source_manifest.jsonl` + `stages/source_manifest__stage-counts.json` | Rewrites from stage files. |
| 6 | `training_sources.review.write_review_sheets` | `review/*.jpg` | Rewrites (old sheets moved to `review/previous_sheets/`). |
| 7 | `training_sources.write_attribution_file` | `ATTRIBUTION.md` | Rewrites from the manifest. |

Annotation rows are append-only; every reader keeps the newest row per `source_id` (`stage_jsonl_io.py`).

## Module map

- `training_sources_paths.py`: every path and URL. `stage_jsonl_io.py`: JSONL helpers. `source_split_assignment.py`: SHA-256 split by Flickr account (~90/5/5).
- `acquisition/`: `ffhq_license_filter.py` (keep CC BY 2.0, CC0, PDM, US Gov; matched on license URL; unknown fails closed), `ffhq_glasses_labels.py` (agreement of two labelers), `ffhq_metadata_slim_cache.py`, `select_ffhq_candidates.py`, `download_ffhq_images.py`.
- `labeling/`: `lens_segmentation_model.py` (rebuilt segmenter), `lens_mask_assignment.py` (left/right split + plausibility), `annotate_sources.py` (stage 3 driver), `source_glare_score.py` (pure score), `score_source_glare.py` (stage 4 driver).
- `review/`: `review_contact_sheets.py` (tile and grid rendering), `write_review_sheets.py` (stage 6 driver).
- `build_source_manifest.py`, `load_source_manifest.py`, `write_attribution_file.py`.

## Key decisions

- **Glasses labels:** a face is "clear glasses" only when DCGM/ffhq-features (Azure Face, `ReadingGlasses`) AND FFHQ-Aging (Face++, `Normal`) agree; negatives need `NoGlasses` AND `None`. Labels are filter-only (both sets CC BY-NC-SA).
- **Download source:** Hugging Face mirror `marcosv/ffhq-dataset` (individual 1024 PNGs, byte-identical); every file is md5-checked against the official `ffhq-dataset-v2.json`. Official metadata came straight from Google Drive (md5 verified).
- **Eyes:** YuNet (`eye_crop.yunet_eye_detector`), highest-scoring face within 0.2 of the image size from the centre. Rejected when YuNet's eyes are more than 0.3 eye distances from FFHQ's own dlib eye landmarks (catches close-ups where YuNet puts both eyes in one lens).
- **Lens masks:** `lenses` LR-ASPP MobileNetV3 weights from mantasu/glasses-detector v1.0.0, architecture rebuilt from torchvision, loaded with `weights_only=True`; the package is never installed or run. Weights: `/Volumes/vega/ai-models/glare-off/lens-segmentation/` (sha256 `31659f30...`). Input is an eye-level square 3.0 eye distances wide (not the whole face, which merged lenses) with flip averaging. Plausibility: area, width, eye inside or near the lens, solidity >= 0.93 (catches leaks), frontal left/right area ratio >= 0.4. One lens is accepted only when DCGM yaw >= 30 deg (`lens_count: 1`). Offline labeling aid only; its training data includes CelebAMask-HQ, so it never ships.
- **Negatives:** the segmenter outlines bare eyes as "lenses" on about 70% of glasses-free faces, so it is NOT used to judge negatives. Their masks are all zero.
- **Glare score:** worst lens of `blob_fraction + max(0, veil_lift - 0.02)`. Blobs are pixels brighter than lit skin by 0.10 luma AND unlike the face's skin chroma (Lab), plus clipped pixels over the eye. Veil compares the lens median luma to the 70th percentile of a skin ring outside the frame. Two thresholds: <= 0.02 trainable; > 0.08 real-glare eval; in between dropped as ambiguous.
- **Real-glare eval set:** every face above 0.08 was reviewed by eye on `review/c_*` sheets. The 131 without clearly visible glare or with a bad mask are listed in `review/real_glare_eval__rejected-by-human-review.txt` (edit, then rerun stage 5).

## Measured counts (2026-10-05)

- Labelled clear glasses 9,875; license-kept 5,784 (CC BY 4,914, PDM 575, CC0 231, US Gov 64); downloaded 5,784, 0 md5 failures.
- Eyes: no face 53, YuNet/dlib disagreement 178. Lens masks: failed 58. Accepted 5,495 (14 single-lens).
- Glare: trainable 4,395 (train 3,991, val 209, test 195); ambiguous 828; eval candidates 272, of which 141 kept after review.
- Negatives: 1,000 selected, 978 accepted (train 879, val 57, test 42).
- Manifest: 5,514 rows. Licenses: CC BY 4,698, PDM 541, CC0 206, US Gov 69.

## Known weaknesses

- The glare score is weak at low values: on review sheets about half the faces scoring 0.02-0.035 still show some glare (mostly faint AR tints), and about 2 in 40 accepted clean faces show faint green or purple AR reflections. Training targets therefore carry some faint real glare.
- Above 0.08 the score is only about 52% precise; eval purity comes from the manual review, which is one reviewer's judgment on downscaled tiles.
- About 10% of accepted masks have visible defects on the random sheet (rimless frames approximated, slight spill past the frame). Rimless and thin metal frames fail plausibility far more often, so they are under-represented.
- The real-glare eval set skews toward moderate and strong glare (mild cases fall in the dropped 0.02-0.08 band). 89 of its 141 faces share a Flickr account with training faces; filter on `flickr_account` if identity overlap matters.
- Split grouping is by Flickr account, not true identity; the same person uploaded by two accounts can cross splits.
- FFHQ's aligned compilation is CC BY-NC-SA 4.0 (NVIDIA), so weights trained on it inherit that, whatever the per-photo license.
