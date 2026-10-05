# Third-party files in this site

Every file under `app/` that was not written for this project, with its license and source. The project's own code is MIT.

| File(s) | What | License | Source |
|---|---|---|---|
| `vendor/onnxruntime-web-1.30.0/ort.wasm.min.mjs`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm` | onnxruntime-web 1.30.0, CPU (WebAssembly) build | MIT, Copyright (c) Microsoft Corporation (`vendor/onnxruntime-web-1.30.0/LICENSE`) | npm `onnxruntime-web@1.30.0` tarball, `dist/`, byte-for-byte |
| `vendor/onnxruntime-web-1.30.0/ort.webgpu.min.mjs`, `ort-wasm-simd-threaded.asyncify.mjs`, `ort-wasm-simd-threaded.asyncify.wasm` | onnxruntime-web 1.30.0, WebGPU build (includes the CPU path) | MIT, as above | same tarball |
| `vendor/onnxruntime-web-1.30.0/LICENSE` | onnxruntime license text (the npm tarball ships none) | MIT | `github.com/microsoft/onnxruntime/blob/v1.30.0/LICENSE` |
| `models/face_detection_yunet_2023mar.onnx` | YuNet face detector (used by the Python side; not served to visitors) | MIT, Copyright (c) 2020 Shiqi Yu | OpenCV Zoo, `models/face_detection_yunet` (`huggingface.co/opencv/face_detection_yunet`) |
| `models/face_detection_yunet_2023mar_dynamic_input.onnx` | The same YuNet with symbolic input height/width so onnxruntime accepts any photo size; weights and ops unchanged | MIT, Copyright (c) 2020 Shiqi Yu | derived from the file above by `tests/app/make_yunet_dynamic_input_model.py` |
| `fonts/OpticianSans-Regular.woff2` | Optician Sans v1.002 by ANTI Hamar and Fábio Duarte Martins, the display face (eye-chart optotype letterforms) | SIL Open Font License 1.1 (`fonts/OpticianSans-OFL.txt`) | `github.com/anewtypeofinterference/Optician-Sans`, `Web-PS/Optician-Sans.woff2`, byte-for-byte (SHA-256 `e42c77bff89b9586b9c7b1ab2b8954e519a303f72f51af366e9b568184a35253`); license confirmed on `optician-sans.com` and the repo's `LICENSE.md` |

`models/glare_removal.onnx` is this project's own network (see `models/README.md` for its training data and license).

## Integrity of the vendored runtime

The tarball `onnxruntime-web-1.30.0.tgz` was checked against the npm registry before extraction:
`sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ==` (registry `dist.integrity`, matched by a local `openssl dgst -sha512`).

SHA-256 of each vendored file:

| File | SHA-256 |
|---|---|
| `ort.webgpu.min.mjs` | `3dffff71811bc13a3a3d9591c57ca5ee5735b8f5f3eaf789510b02bb040ae3a0` |
| `ort-wasm-simd-threaded.asyncify.mjs` | `3d1c85995364bb643302fc6fd877a0c3ba5ae72401815e0f24828a53d9191e28` |
| `ort-wasm-simd-threaded.asyncify.wasm` | `39f9f0894d478800487ed9f7dbe92618498db320cf55c8e3d89adff8dce658da` |
| `ort.wasm.min.mjs` | `219e6a1fc8a9938268d18efca3c91d310bd2f4a59bbd13744df5b2b7fc6cee3b` |
| `ort-wasm-simd-threaded.mjs` | `e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b` |
| `ort-wasm-simd-threaded.wasm` | `3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2` |

To upgrade: `npm pack onnxruntime-web@<version>`, compare its sha512 with `npm view onnxruntime-web@<version> dist.integrity`, copy the six files into a new `vendor/onnxruntime-web-<version>/`, update `js/app_config.js`, this table, and run `tests.app.sync_asset_manifest`.
