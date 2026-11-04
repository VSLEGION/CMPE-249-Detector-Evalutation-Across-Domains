# CMPE 249 - Object Detector Evaluation

Controlled evaluation of **YOLOv8n**, **YOLO11n** and **RT-DETR-L** on full
COCO val2017, a seeded 500-image KITTI subset, and user-supplied verification images.

## Add your images

**Put your verification images in [`data/verification/images/`](data/verification/images/).**
JPG, JPEG, PNG, BMP and WebP are supported, including subfolders. That directory
is intentionally empty and ready for your uploads. Images there can be committed
to Git; large datasets, weights, local paths and generated outputs are ignored.

From this repository folder, using the existing parent-project environment:

```powershell
..\dtect\Scripts\python.exe run_pipeline.py doctor
..\dtect\Scripts\python.exe run_pipeline.py verify-images --run-id my_images
```

Predictions, image manifests, overlays and summaries appear in
`outputs/my_images/verification/`. No COCO/KITTI download is needed for this
command. Optional COCO labels can go in `data/verification/annotations.json`.
See [the image guide](docs/VERIFICATION_IMAGES.md).

## Run the evaluation

The existing **dtect** environment is reused on this machine. An ignored
`configs/local.json` points to existing checkpoints/datasets; nothing was moved.
The committed [default configuration](configs/default.json) uses repository-relative
paths and works without any reference to the original project.

```powershell
..\dtect\Scripts\python.exe run_pipeline.py test --run-id evaluator_check
..\dtect\Scripts\python.exe -m unittest discover -s tests -v
..\dtect\Scripts\python.exe run_pipeline.py all --run-id experiment_001
```

For another machine, use a compatible CUDA PyTorch environment, place the three
checkpoints in `weights/`, and configure the datasets. Optional editable installation
is `python -m pip install -e . --no-deps` if dependencies are already present;
`cmpe249-eval` then provides the same CLI. No installation is required for
`run_pipeline.py`.

## Repository layout

```text
configs/                       Portable defaults and local-path template
src/cmpe249_eval/              Reusable Python package
tests/                         CPU regression and relocation checks
data/verification/images/     YOUR VERIFICATION IMAGES GO HERE
data/datasets/                 Large COCO/KITTI data (ignored)
weights/                       Checkpoints (ignored)
outputs/                       New experiment runs (ignored)
results/published/             Tables and test evidence from the completed run
reports/                       Existing PDFs, no embedded Python source
docs/                          Architecture, protocols, image guide and migration
.github/workflows/tests.yml    CPU tests on GitHub Actions
run_pipeline.py                Single public entry point
```

## Baseline and methods

Primary accuracy: 512 square input, batch 1, FP32, CUDA device 0, confidence .001,
300 exported detections, YOLO NMS IoU .70. RT-DETR is NMS-free. Official
`pycocotools.COCOeval` uses maxDets `[1,10,100]`. Shared primary classes are
person/pedestrian, car and truck; bicycle/cyclist is approximate supplementary.
Converted KITTI scores are COCO-style, not official KITTI benchmark scores.

Timing: confidence .25, decoded images outside timing, 10 warmups, five repetitions,
explicit CUDA synchronization and a fresh process per model/configuration.
Predictions and image writing are outside timing. The verification command is an
untimed qualitative pass; labeled evaluation there uses the display threshold
and is explicitly distinguished from primary .001 evaluation.

## Existing results

The [published summary](results/published/README.md) records the original completed
experiment. The code has now been reorganized; published metrics are historical
artifacts, not new runs of the refactored package. New results always go in `outputs/`.

| Model | COCO 80-class AP50-95 | KITTI shared AP50-95 | Baseline latency |
|---|---:|---:|---:|
| YOLOv8n | 0.3518 | 0.1943 | 10.59 ms |
| YOLO11n | 0.3638 | 0.1969 | 14.10 ms |
| RT-DETR-L | 0.5082 | 0.2860 | 27.27 ms |

These timings describe the recorded Windows laptop/Silent-power-plan protocol,
not universal model performance. Full reports are in [reports/](reports/README.md).

## Further documentation

- [Architecture and script map](docs/ARCHITECTURE.md)
- [Reproduction and individual stages](docs/REPRODUCE.md)
- [Verification-image upload guide](docs/VERIFICATION_IMAGES.md)
- [Original-to-new layout mapping](docs/MIGRATION.md)
- [Checks completed on the reorganized repository](docs/VALIDATION.md)

This folder is prepared for a local Git repository. Nothing has been committed,
pushed, or published automatically. Model frameworks and datasets retain their
own licenses; no copied third-party implementation source is bundled here.

