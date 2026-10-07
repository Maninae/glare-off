# glasses_classifier/

The app's gate: a 175k-parameter CNN that says whether the face in an aligned eye crop wears glasses, so the browser skips the glare model on bare faces (and shows the user which faces it skipped). It reads the SAME tensor the glare model receives, so the app computes the crop once.

## Model I/O (`glasses_classifier/` -> `app/`)

- ONNX file `app/models/glasses_classifier.onnx`, opset 17, fp16 weights (Cast to fp32 at load), static shapes.
- Input `eye_crop`: float32 [1, 3, 256, 512], RGB, sRGB in [0, 1]: the glare model's `glare_crop` tensor.
- Output `glasses_probability`: float32 [1, 1] in [0, 1] (sigmoid inside the graph).
- **Threshold: 0.5.** `p >= 0.5` -> wears glasses, run the glare model; below -> skip the face.
- Ops: Conv, Relu, AveragePool, Add, ReduceMean, Gemm, Sigmoid, Cast (BatchNorm is folded into Conv by the exporter). All in the WebGPU table; `glare_model/export/webgpu_operator_inventory.py` re-checks on every export.

## Module map

```
glasses_classifier/
├── glasses_classifier_net.py   GlassesClassifierNet: 4x4 avg pool (256x512 -> 64x128), conv stem, 7 depthwise-separable blocks, global mean, linear, sigmoid
├── build_eye_crop_cache.py     CLI: every manifest face -> app-exact 256x512 uint8 crops in one .npy memmap per split
├── glasses_crop_dataset.py     GlassesCropDataset over the memmaps: random variant + photometric/grayscale/flip (train), variant 0 untouched (eval); class-balanced weights
├── config_loading.py           OmegaConf YAML + dotlist, struct-locked; workers <= 2
├── train.py                    CLI: train, pick best, evaluate, export to <run>/exported/
├── classification_metrics.py   confusion matrix, per-class precision/recall, ROC AUC, threshold rule
├── evaluate.py                 CLI: val/test/real_glare_eval report + misclassified contact sheet
├── contact_sheet.py            captioned crop grid PNG
├── checkpointing.py            save/load {weights, config}
├── glasses_onnx_export.py      export + parity + fp16 choice + latency, reusing glare_model/export/
├── export_onnx.py              CLI: checkpoint -> app/models/glasses_classifier.onnx
└── make_stub_onnx.py           CLI: untrained always-glasses stub (p = 0.993) with the real I/O
```

Config: `configs/glasses_classifier/base.yaml`. Tests: `tests/glasses_classifier/` (CPU, ~4 s).

## Design decisions

- **Reuse, not copies.** Crops come from `eye_crop.extract_eye_crop_with_area_prefilter` (the app's resampler). Eye-center jitter, phone-capture simulation, photometric jitter, the LR schedule, the MPS cap, opset 18 -> 17 downconversion, the metadata strip, the WebGPU check, and fp16-with-Cast all import from `glare_model/`.
- **Crop cache on vega** (`/Volumes/vega/datasets/glare-off/glasses-classifier/crop_cache/`, 5.6 GB, builds in ~80 s with 3 processes). Train faces get 3 jittered variants; the last one goes through the phone-capture path (scale 0.15-0.6, noise, JPEG, area pre-shrink). Eval splits get one crop at the exact manifest eye centers.
- **Class balance by sampling.** A WeightedRandomSampler draws glasses and bare faces equally (train is 3,991 : 879). The loss is plain BCE, so `p = 0.5` is the 50/50-prior decision point.
- **Best checkpoint:** highest val ROC AUC, ties broken by lower val BCE. Val AUC saturates at 1.0 from epoch 4, so the tiebreak decides.
- **Threshold rule:** keep the default 0.5 unless val glasses recall there is below 0.98; only then lower it. On val, 0.5 sits inside a clean gap (highest bare face 0.09, lowest glasses face 0.85), so tuning inside it would be noise.
- **Asymmetric cost.** A missed glasses face loses its glare fix (visible failure). A bare face let through only costs one glare-model run that should change nothing. If the app sees missed glasses faces in practice, lower the threshold to ~0.35 (both glasses misses on test/real_glare_eval scored 0.37-0.38; that observation comes from test, so it was not used to choose the shipped value).

## Measured numbers (run02, Oct 6 2026, Apple M4 shared, torch 2.14.1, ORT 1.30.0)

- 174,545 params. ONNX fp16 weights 0.37 MB (fp32 0.71 MB). Parity vs PyTorch: fp32 7e-7, fp16 1.3e-3 (max abs probability). 1-thread CPU ORT: 0.46 ms.
- Training: 12 epochs x 152 steps at batch 32, 2.5 min on MPS (peak 1.06 GiB), best = epoch 6.

| Split (threshold 0.5) | Faces (bare / glasses) | ROC AUC | Accuracy | Glasses P / R | Bare P / R | Confusion [[TN, FP], [FN, TP]] |
|---|---|---|---|---|---|---|
| val | 57 / 209 | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | [[57, 0], [0, 209]] |
| test | 42 / 195 | 0.9998 | 0.992 | 0.995 / 0.995 | 0.976 / 0.976 | [[41, 1], [1, 194]] |
| real_glare_eval (all glasses, eval only) | 0 / 141 | n/a | 0.993 | 1.000 / 0.993 | n/a | [[0, 0], [1, 140]] |

Misclassified at 0.5 (`runs/run02/evaluation/misclassified__test_and_real_glare_eval.png`):
- Thin wire-rim metal frames on a hazy, low-contrast photo (glasses, p 0.38).
- Thick frames with lenses full of reflected scenery, extreme close-up 3/4 view, one lens off the crop (glasses, p 0.37).
- Bare face with a hat brim and a hand at the brow (bare, p 0.67).
Test has only 42 bare faces, so the bare-class rates move ~2.4 points per face.

## How to run

The machine is a shared 16 GB M4: `nice -n 10`, MPS capped at 0.2 (set by the trainer; both env vars below avoid the torch 2.14 low-watermark crash), workers <= 2.

```bash
PY=/Volumes/vega/datasets/glare-off/venv/bin/python
# 1. Crop cache (once; skips splits already cached; delete with `trash` to rebuild)
nice -n 10 $PY -m glasses_classifier.build_eye_crop_cache
# 2. Train + evaluate + export into the run dir (pick a new run name)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.15 \
  nice -n 10 $PY -m glasses_classifier.train OUTPUT.RUN_NAME=run03
# 3. Re-evaluate any checkpoint (report JSON + misclassified sheet into <run>/evaluation/)
$PY -m glasses_classifier.evaluate --checkpoint /Volumes/vega/datasets/glare-off/glasses-classifier/runs/run03/checkpoints/best.pt
# 4. Ship to the app
$PY -m glasses_classifier.export_onnx --checkpoint /Volumes/vega/datasets/glare-off/glasses-classifier/runs/run03/checkpoints/best.pt
# Tests
$PY -m pytest tests/glasses_classifier -q
```

Runs live in `/Volumes/vega/datasets/glare-off/glasses-classifier/runs/<RUN_NAME>/{checkpoints,tb,evaluation,exported,config.yaml}`.
