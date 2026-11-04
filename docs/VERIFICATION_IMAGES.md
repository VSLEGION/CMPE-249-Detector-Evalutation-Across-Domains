# Your image upload location

From the repository root, upload images into:

**`data/verification/images/`**

On the original Windows project this is:

`cmpe249-detector-evaluation\data\verification\images\`

The folder can contain any number of images and nested folders. Keep filenames
stable to preserve the generated manifest. Do not put COCO/KITTI downloads,
model weights, Python utilities or generated overlay images in this input folder.

## Unlabeled verification

Run `python run_pipeline.py verify-images --run-id my_images` using your CUDA
environment. On this machine use `..\dtect\Scripts\python.exe` instead of `python`.

All three models use 512, batch 1, FP32 and confidence .25. YOLO uses NMS .70;
RT-DETR is NMS-free. Outputs include per-model prediction JSON, cyan-box overlays,
image hashes and a CSV of output classes/counts. Overlay filenames use manifest
image IDs, so duplicate basenames in different folders remain distinguishable.

This command saves qualitative outputs. More detections is not evidence of more
correct detections, and no accuracy metric is generated without real labels.

## Optional real labels

If you have ground truth, place a COCO-format `annotations.json` in
`data/verification/`, beside `images/`. Its `images[].file_name` values must match
paths relative to `images/`, and every input image must be represented exactly once.
Each image needs its true width, height and unique ID. Categories need unique
IDs and actual COCO class names. Annotations need unique IDs, original-pixel xywh
boxes, positive area, matching image/category IDs and `iscrowd` values.

An explicitly annotated image with no objects has no annotation rows. A missing
image record is an error; it is not assumed to be an empty scene. Unknown class
names and mismatching dimensions are rejected. Evaluation uses the supplied IDs,
not inferred numeric mappings. At .25 this is a thresholded verification score,
not the primary assignment protocol's .001 confidence-floor result.

## Git uploads

The verification image directory is **not ignored by Git**. Your uploads can be
included in the repository. Generated outputs remain under ignored `outputs/`.
The source images are read, hashed and decoded; they are never overwritten.

