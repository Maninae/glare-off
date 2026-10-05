# glare_model/

The network that turns an aligned 512x256 eye crop with lens glare into a clean crop plus a glare mask and a lost-detail mask; its training loop on synthetic pairs; and its export to the browser ONNX contract. Input contract: "Synthesis data_dict" in the repo CLAUDE.md. Output contract: "Model I/O" there.

## Module map

```
glare_model/
├── registry.py                       name -> builder registries (MODEL, SOURCE_FACE_PROVIDER, GLARE_SYNTHESIZER)
├── train.py                          CLI: python -m glare_model.train
├── export_onnx.py                    CLI: checkpoint -> app/models/glare_removal.onnx (+ _fp16), prints a JSON report
├── make_stub_onnx.py                 CLI: untrained-but-sane stub with the real graph and I/O
├── architecture/
│   ├── nafnet_blocks.py              ChannelLayerNorm2d (plain ops), SimpleGate, SCA, NAFBlock
│   ├── glare_removal_nafnet.py       GlareRemovalNAFNet: 4-level U-Net + restoration head + 2-channel mask head
│   └── stub_weights.py               hand-set head weights for the stub
├── data/
│   ├── source_face.py                the two injected interfaces: SourceFaceProvider, GlareSynthesizer; contract validator
│   ├── crop_augmentation.py          eye-center jitter, photometric jitter, flip (swaps lens labels 1<->2)
│   ├── phone_capture_simulation.py   re-sample the crop region to a phone-like crop scale (0.15-0.6), noise + JPEG
│   ├── glare_pair_dataset.py         GlarePairDataset: source face -> jittered crop -> synth -> CHW tensors
│   ├── fake_data_sources.py          FAKE provider + FAKE blob synthesizer (tests, smoke runs)
│   └── real_module_adapters.py       the ONLY file that reaches training_sources/ and glare_synthesis/
├── losses/
│   ├── structural_similarity.py      SSIM map (shared by loss and metrics)
│   └── glare_removal_loss.py         every loss term, on the app-style blended output
├── training/
│   ├── config_loading.py             _BASE_ + CONFIG_GROUPS + dotlist; validation (workers <= 4, sizes % 16)
│   ├── run_directory.py              run_output_<hash>/{tb,checkpoints,exported,config.yaml}
│   ├── optimization.py               AdamW, warmup-cosine, EMA, device, seeding, MPS memory cap
│   ├── checkpointing.py              atomic save, load, RNG state
│   ├── component_builders.py         config -> model, datasets, loaders, loss
│   ├── validation_metrics.py         PSNR/SSIM (all, in-lens), input-PSNR baseline, mask IoU, change guards, best-checkpoint score
│   ├── validation_pass.py            val loop + TensorBoard grid
│   └── glare_model_trainer.py        GlareModelTrainer: the coordinator owning all state
└── export/
    ├── onnx_export.py                dynamo export, metadata strip, fp16-weights variant
    ├── opset_downconversion.py       opset 18 -> 17 (ReduceMean axes, Split num_outputs)
    ├── webgpu_operator_inventory.py  the ORT WebGPU op table + loud check
    ├── onnx_verification.py          ORT parity and single-thread latency
    └── export_and_verify.py          shared export + verify + report
```

Configs: `configs/glare_model/base.yaml` (everything), `model/{small,base,large}.yaml`, `data/{fake,real}.yaml`, `smoke_fake.yaml`. Tests: `tests/glare_model/` and `tests/eye_crop/` (CPU only, ~7 s).

## Design decisions

- **Residual output, gate left to the caller.** The restoration head predicts `delta`; the graph outputs `clean_crop = clip(glare_crop + delta)` and the caller blends with the glare mask, exactly per the contract.
  - The delta head is zero-initialized, so an untrained net is the exact identity. The easy majority of pixels (no glare, no glasses) costs nothing to learn.
  - Gating inside the graph would duplicate the caller's blend.
  - Training scores the blend the user sees: `glare + sigmoid(mask_logit_0) * delta`. The mask head is therefore trained by the image loss as well as by its own focal/Dice loss.
  - `UNBLENDED_LENS_L1` (weight 0.5) trains `delta` inside the lens even where the predicted mask is still near 0. Otherwise the blend gates its gradient to 0 early in training.
- **Heads read the raw input** (decoder features concatenated with `glare_crop`, then a 3x3 conv). Brightness is directly visible to the mask head, and the stub can be built by setting head weights only.
- **Outside the lens nothing may change, except real glare.** `OUTSIDE_LENS_CHANGE` penalizes |delta| against the INPUT (the target can differ from the input there: the synthesizer may JPEG/noise the glare delta), weighted `(1 - lens_mask) * (1 - glare_mask)`. The synthesizer renders bloom and rim glints just past the lens edge (~1.7% of glare-mask mass); GLARE_L1 asks for their removal, so the penalty skips ground-truth glare pixels. Glare-free samples stay fully guarded: outside by this term, inside the lens by LENS_L1 and UNBLENDED_LENS_L1 (their target equals the input). The unblended L1 is restricted to the lens for the same JPEG/noise reason.
- **Dice is batch-level per channel.** Most samples have empty masks. Per-sample Dice punished their tiny background probabilities as hard as missed glare.
- **Export path: dynamo at opset 18, then a hand-written downconvert to 17.**
  - The legacy TorchScript exporter does produce opset 17, but it emits 226 Constant, 86 Identity, and Shape/Gather/Slice nodes, which are not WebGPU kernels.
  - `onnx.version_converter` crashes on the dynamo graph (onnx 1.23).
  - Only ReduceMean and Split changed between 17 and 18 for this net. The pass fails loudly on any other changed op.
- **Exporter debug metadata is stripped.** The dynamo exporter writes Python stack traces with the exporting machine's absolute paths into every node: a privacy leak in a public repo, plus about 1 MB.
- **Graph ops (all variants), 12 types:** Conv, Mul, Add, ReduceMean, Sub, Sqrt, Div, Split, DepthToSpace, Concat, Clip, Sigmoid. All are in the WebGPU table (ORT `webgpu-operators.md` @ 3eda902, 2026-07-29). There is no GroupNorm, Max/Min, Constant, Identity, Shape, Reshape, or Resize. `export/webgpu_operator_inventory.py` re-checks on every export.
- **fp16 weights, fp32 compute.** Initializers are stored as fp16 with a Cast back to fp32, which ORT constant-folds at session creation. No EP ever needs fp16 kernels, and ORT's CPU/WASM EP has almost none. The variant is written only if parity stays within 1/255.
- **Plain fp32, no AMP.** Autocast on MPS measured about 7% faster (1.46 to 1.36 s/step at batch 8) with slower loss decrease, so it is not offered.
- **Gradient accumulation instead of big batches.** The machine cap (below) fits micro-batch 2. The net has no BatchNorm, so accumulating 4 micro-batches is mathematically batch 8.
- **No VGG/LPIPS, no GAN.** `LOSS.WEIGHTS.PERCEPTUAL_VGG` is a hook that raises if set above 0 (licensing gray area, docs/research/03 Q5).

## Data wiring

- The dataset depends only on two injected interfaces (`data/source_face.py`), built by name from the `DATA` config group.
- `CONFIG_GROUPS.DATA=fake` (default): procedural faces + blob glare.
- `CONFIG_GROUPS.DATA=real`:
  - `ManifestSourceFaceProvider` -> `training_sources.load_source_manifest:load_source_manifest`
  - `build_glare_synthesis_module_synthesizer` -> `glare_synthesis.render_lens_glare:render_lens_glare_sample`
  - Both are `module:function` strings in `configs/glare_model/data/real.yaml`. Rename on their side means a one-line edit there.
- Both real paths are exercised by `test_real_modules_produce_contract_samples_through_the_adapters`, using a temporary manifest of fake faces. The real manifest JSONL did not exist yet when this was written.
- No-glasses rows (`has_glasses: false`) with no mask file load as all-background.
- Validation uses `DATA.VAL_SEED` per item (identical every epoch) and no photometric augmentation. Training seeds mix `torch.initial_seed()`, the index, and a draw counter.
- **Crops use the app's resampler** (`eye_crop.extract_eye_crop_with_area_prefilter`, rule in the repo CLAUDE.md "Eye crop"): bilinear, with an integer INTER_AREA pre-shrink below scale 0.5. Lens labels go through the same resampler as one-hot planes and are argmax'd back.
- **Simulated phone capture (`DATA.PHONE_CAPTURE`, default 50% of train AND val samples; val's mix is fixed by its seeds).** Manifest faces crop at scale ~0.6-1.2; phone photos land at ~0.15-0.4, which goes through the pre-shrink path.
  - A target scale is drawn from [0.15, 0.6]. Lower scale means MORE photo pixels per eye, so the photo region under the crop (never the whole photo) is ENLARGED (cubic, uint8) by `native / target`. A downsample would raise the scale instead. No detail is created; what is gained is the app's exact low-scale resampling chain plus photo-resolution noise and JPEG that the pre-shrink averages down.
  - Then Gaussian noise (sigma 0.003-0.02), then JPEG q70-95, each at 50%. Both happen before glare synthesis, so target and input share them.
  - The path runs in uint8 to bound memory (a float32 region at scale 0.15 is ~100 MB per copy). Measured on fake 512x512 faces, worst case scale 0.15: process peak 0.7 GB, 146 ms median per item with the real renderer. Default mix: 52 ms median (native-only: 33 ms).
- Real renderer cost: 26 ms median, 38 ms p90 per full dataset item at 512x256 with one OpenCV thread. One DataLoader worker keeps up with training (budget about 185 ms/item).

## How to run

The machine is a shared 16 GB M4: one training job at a time, `nice -n 10`, MPS capped at `TRAIN.MPS_HIGH_WATERMARK_RATIO` 0.2 (2.37 GiB). The trainer sets the cap env vars itself before touching MPS.

```bash
# Train on real data (rerunning the same command resumes from checkpoints/last.pt)
nice -n 10 /Volumes/vega/datasets/glare-off/venv/bin/python -m glare_model.train \
  --config configs/glare_model/base.yaml \
  CONFIG_GROUPS.DATA=real

# Smaller / larger model, other overrides
... CONFIG_GROUPS.MODEL=small SOLVER.MAX_ITER=50000

# Stop: Ctrl-C or kill -TERM <pid>; the trainer finishes the step and writes last.pt.

# Export a checkpoint to the app (EMA weights by default)
/Volumes/vega/datasets/glare-off/venv/bin/python -m glare_model.export_onnx \
  --checkpoint /Volumes/vega/datasets/glare-off/runs/run_output_<hash>/checkpoints/best_ema.pt \
  --out app/models/glare_removal.onnx

# Tests
/Volumes/vega/datasets/glare-off/venv/bin/python -m pytest tests/glare_model -q
```

- **Run directory:** `run_output_<hash>` hashes the config minus workers, resume, device, memory cap, and log cadence. `OUTPUT.RUN_DIRECTORY` overrides it.
- **TensorBoard:** `train/*` holds every loss term, grad norm, LR, s/step, and peak MPS GiB. `val/*` holds the metrics plus a grid (input | prediction | target | predicted glare mask | target glare mask | predicted lost-detail mask).
  - Change guards, all mean |blended - input| in [0, 1] units: `val/clean_sample_change` (+ `_max`) over glare-free samples, and `val/outside_lens_change` over all samples, outside the lens and outside ground-truth glare.
  - `val/clean_change_gate_passed` is 1 when this validation's checkpoint is eligible for best.
- **Best checkpoint (`best_ema.pt`):** highest `psnr_lens` among validations whose `clean_sample_change` is at most `TRAIN.BEST_MAX_CLEAN_SAMPLE_CHANGE` (0.5/255). Failing checkpoints are never kept as best. A val set with no glare-free sample cannot be gated, and `psnr_lens` alone decides.
- **On normal completion** `train` exports `checkpoints/best_ema.pt` to `<run>/exported/glare_removal.onnx`. It falls back to the final EMA weights, with a warning, if no checkpoint qualified.
- **Torch 2.14 gotcha:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.2` alone crashes MPS init ("invalid low watermark ratio 1.4"). Set `PYTORCH_MPS_LOW_WATERMARK_RATIO` at or below it (0.15).

## Measured numbers (2026-10-05, Apple M4 16 GB, shared with other jobs, torch 2.14.1, ORT 1.30.0)

| Variant | Params | ONNX fp32 | ONNX fp16 weights | Parity fp32 / fp16 (max abs, 256x512 and 512x1024) | 1-thread CPU ORT, 256x512 |
|---|---|---|---|---|---|
| small (`model/small.yaml`) | 1.14M | 4.6 MB | 2.4 MB | 1.5e-6 / 9.4e-4 | 88 ms |
| base (default) | 2.93M | 11.8 MB | 6.0 MB | 1.5e-6 / 1.3e-3 | 111 ms |
| large | 8.63M | 34.6 MB | 17.4 MB | 2.5e-6 / 1.5e-3 | 200 ms |

Parity was measured on random non-trivial weights. On the smoke-trained checkpoint it is 1.4e-7 (fp32) and 5e-5 (fp16). Latencies are under concurrent load and are a rough proxy, not a browser number.

**Memory and speed, training, base model, fp32, 512x256, MPS capped at 0.2 (2.37 GiB):**

| Micro-batch | Peak MPS driver | Peak RSS | s/step | s/sample |
|---|---|---|---|---|
| 1 | 1.24 GiB | 0.61 GiB | 0.179 | 0.18 |
| 2 | 2.16 GiB | 0.63 GiB | 0.368 | 0.18 |
| 4 | OOM at the 2.37 GiB cap | | | |

- **Full trainer at micro-batch 2:** `top` shows 2.7 GB for the main process; each DataLoader worker adds about 0.3 GB RSS. Default `NUM_WORKERS: 1` keeps the job near 3.0 GB.
- **Batch 8 (2 x 4 accumulation):** 0.18 s/sample gives about 1.45 s/step. This matches 1.46 s measured directly at batch 8 before the cap. The trainer under load measured 0.72-0.83 s/step at effective batch 4 (1.45-1.65 s per 8 samples).
- **Wall clock at effective batch 8:** 50k steps is about 20-23 h; 100k steps is about 40-46 h (plus negligible validation).

**Smoke run on FAKE data:** `smoke_fake.yaml`, 400 steps, effective batch 4, about 6 min; interrupted at step 151 with SIGINT and resumed with the same command.

- **Val (EMA, fixed seeds), in-lens PSNR:** input 15.24 dB; 16.45 at step 100, 19.55 at step 200, 20.78 at step 300, 21.04 at step 400.
- **Glare-mask IoU:** 0.18, 0.53, 0.65, 0.68 at those steps. Lost-detail IoU 0.38 at step 400.
- **Split over 64 val samples:**
  - 41 glare samples went from 15.5 to 21.0 dB in-lens (median +5.7 dB). Broad blobs are mostly removed; saturated blobs over the iris are not recovered.
  - 23 no-glare samples still changed: median max |output - input| 0.06, worst 0.41. The early model fires its glare mask on bright white sclera. Watch `val/iou_glare_mask` and the grid for this false positive on real data.
