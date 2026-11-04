# Put your verification images here

Add JPG, JPEG, PNG, BMP or WebP files to **`images/`** (subfolders are supported).
This folder is the default input for verification predictions and timing. You
can use any number of images; the old 19-image set is not required.

Run from the repository root:

```powershell
..\dtect\Scripts\python.exe run_pipeline.py verify-images
```

Results are written to `outputs/<run-id>/verification/`: source manifest,
per-model original-coordinate prediction JSON, overlays and a CSV summary.
Files in `images/` are eligible for Git tracking. Keep large COCO/KITTI downloads
in `data/datasets/` instead (that directory is ignored).

If you have real labels, optionally add **`annotations.json`** beside this README
using COCO bbox format. Each `images[].file_name` must match the relative filename
under `images/`. Labels must use actual class names such as `person`, `car`, or
`dog` in `categories`. Image IDs are taken from the annotation file. All images
in the folder must be represented; missing labels are not assumed to be negatives.
Without this file, outputs are qualitative and no AP/precision/recall is claimed.

