# Models served to visitors

- `face_detection_yunet_2023mar_dynamic_input.onnx`: YuNet face detector (MIT), eye centers for the crop.
- `glare_removal.onnx`: the glare model, v0.1 (Oct 6 2026). NAFNet-style U-Net, 2.9M parameters, fp16 weights (6.0 MB), opset 17, input `glare_crop` [1,3,H,W], outputs `clean_crop` and `masks`. Trained 50k steps on synthetic glare over license-filtered FFHQ faces (run `run02__base__real__50k`); weights are CC BY-NC-SA 4.0. Known gap: colored screen-content reflections.
- `face_detection_yunet_2023mar.onnx`: the original fixed-size YuNet, used by the Python side only.
- `glasses_classifier.onnx`: the glasses gate, v0.1 (Oct 6 2026). 175k-parameter CNN, fp16 weights (0.37 MB), opset 17, input `eye_crop` [1,3,256,512] (the glare model's crop tensor), output `glasses_probability` [1,1]. Threshold 0.5: below it the face wears no glasses and the app skips the glare model. Test accuracy 99.2% (1 of 42 bare faces and 1 of 195 glasses faces wrong). Trained on license-filtered FFHQ faces (run `run02`); weights are CC BY-NC-SA 4.0. Details: `glasses_classifier/CLAUDE.md`.
