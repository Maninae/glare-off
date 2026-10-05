# app/models

## glare_removal.onnx: currently an UNTRAINED STUB

`glare_removal.onnx` is a placeholder written by `python -m glare_model.make_stub_onnx`. It has the real architecture (NAFNet U-Net, base variant, 2.93M parameters, 11.8 MB) and the real I/O signature, so the app can be wired up and timed now. It has never seen training data and removes no glare.

| | Value |
|---|---|
| Input | `glare_crop` float32 [1, 3, H, W], RGB sRGB in [0, 1], H and W multiples of 16 (default 256x512) |
| Output 1 | `clean_crop` float32 [1, 3, H, W] in [0, 1] |
| Output 2 | `masks` float32 [1, 2, H, W] in [0, 1]: channel 0 glare mask, channel 1 lost-detail mask |
| Opset | 17; ops: Conv, Mul, Add, ReduceMean, Sub, Sqrt, Div, Split, DepthToSpace, Concat, Clip, Sigmoid (all in the onnxruntime-web WebGPU table) |

What the stub outputs, so you can tell it is working:

- `clean_crop` = the input darkened by 8%.
- `masks[0]` = a soft bright-pixel detector: about 0.5 at luminance 0.78, near 1 above about 0.9.
- `masks[1]` = a stricter near-white detector: about 0.5 at luminance 0.95.

With the contract blend `result = glare_crop + masks[0] * (clean_crop - glare_crop)`, bright spots inside the crop should come out slightly dimmer and everything else unchanged. If that is what you see, the pipeline (crop, inference, blend, warp back) is correct.

When a trained checkpoint exists, it replaces this file in place with the same name and signature: `python -m glare_model.export_onnx --checkpoint <run>/checkpoints/best_ema.pt --out app/models/glare_removal.onnx`. That also writes `glare_removal_fp16.onnx` (fp16 weights with Cast back to fp32 at load, about half the size, same outputs within 1/255) if it passes parity.
