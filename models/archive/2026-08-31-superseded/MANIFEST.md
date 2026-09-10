# Superseded model artifacts

Archived on 2026-08-31 after exporting the checkpoint introduced by Git commit
`dc711a1` (`Update capsule dataset and best weights`).

The active deployment artifacts remain in `models/trained/`:

- `capsule_yolo11s_obb_best.pt`
  - SHA-256: `a0a70b84f85f435e616db9334d52161f8853527815e0e50a49b5b4d8b7978b2a`
- `capsule_yolo11s_obb_best.onnx`
  - SHA-256: `ee1a37653415342837b9812427dedd85ecc1cad89df264aee80b9debe5bf39c0`
- `capsule_yolo11s_obb_best.engine`
  - SHA-256: `daf9cf11482ebffebe3086407dcbbbdc4846ed5d41d050c9726752ee48ec0256`
  - TensorRT 10.11.0.33, FP16, static input shape `1x3x1280x1280`

Archived files:

- `capsule_yolo11n_obb_best.pt`
- `capsule_yolo11n_obb_best.onnx`
- `capsule_yolo11s_obb_1280_best.pt`
- `capsule_yolo11s_obb_1280_best.onnx`
- `capsule_yolo11s_obb_1280_best.engine`
- `capsule_yolo11s_obb_1280_best.engine.trt10.3.bak`
- `capsule_yolo11s_obb_best.engine.trt10.3-host.bak`
- `training-run-capsule_yolo11s_obb_1280/best.pt`
- `training-run-capsule_yolo11s_obb_1280/last.pt`

The active engine was built and validated inside the `yolo-capsule:jetson`
deployment image. TensorRT deserialization and inference on
`labeled_data/images/025e265c-image00065.jpeg` succeeded.
