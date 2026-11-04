# CMPE 249 - Detector Evaluation Across Domains

**Start with [READ_ME_FIRST.md](READ_ME_FIRST.md).**
The primary coursework report is [reports/main_experiment_report.pdf](reports/main_experiment_report.pdf).
Code is included in this folder and is also hosted at the project repository:
https://github.com/VSLEGION/CMPE-249-Detector-Evalutation-Across-Domains

## What is included

- Three COCO-pretrained detectors: YOLOv8n, YOLO11n and RT-DETR-L.
- Full COCO val2017 and a fixed seed-42 500-image prepared KITTI subset.
- Published metrics, timing samples, labeled ablations, figures and test evidence.
- The 19 verification images, per-image study, and separate qualitative outputs.
- A modular Python package, portable configuration, tests and reproduction instructions.

The original verified experiment completed 38 configurations. Published results
are preserved as historical evidence; reorganization is not presented as another
full COCO/KITTI run. Final packaging checks and the fresh 19-image verification
pass are recorded separately under `results/submission_checks/`.

## Entry points

Use the existing CUDA environment, or activate a compatible environment first.

```powershell
python run_pipeline.py doctor
python -m unittest discover -s tests -v
python run_pipeline.py verify-images --run-id my_images
```

On the original Windows machine, the relocation-safe helper finds the existing
`dtect` interpreter in an ancestor folder:

```powershell
.\scripts\use_existing_environment.ps1 doctor
.\scripts\use_existing_environment.ps1 verify-images --run-id my_images
```

The verification input is `data/verification/images/` (19 supplied images).
New outputs go to `outputs/<run-id>/verification/`. Optional real labels can go
in `data/verification/annotations.json`. Without labels, verification is
qualitative; no AP, precision or recall is claimed.

## Repository structure

```text
configs/                 Protocol defaults and local-path template
src/cmpe249_eval/         Data, evaluation, GPU workers, analysis and exports
tests/                   CPU regression tests
data/verification/       The 19 input images; optional real COCO annotations
data/datasets/           Large validation data (not bundled)
weights/                 Checkpoint placement guide (weights not bundled)
outputs/                 New local runs (not included in submission ZIP)
results/published/       Original metrics, samples, figures and reproducible evidence
results/submission_checks/ Final folder-level checks and fresh verification evidence
reports/                 Primary report, supplements and editable report sources
docs/                    Architecture, methods, migration and reproduction
scripts/                 Environment helper, report builder and submission packager
submission/              Generated checked ZIPs and manifests
archive/                 Preserved superseded documentation and unused helper
```

## Experimental protocol

Accuracy: 512 square input, batch one, FP32, device 0, confidence .001,
export cap 300, YOLO NMS IoU .70; RT-DETR is NMS-free. Official
pycocotools.COCOeval uses maxDets [1,10,100].
The main shared comparison uses person/pedestrian, car and truck. Bicycle/cyclist
is approximate supplementary. KITTI values are converted COCO-style scores,
not official KITTI benchmark scores.

Timing: confidence .25, predecoded inputs, 10 warmups, 5 repetitions per image,
CUDA synchronization and fresh model processes. Recorded timings describe the
original Windows laptop/Silent-power-plan conditions.

## Original results

| Model | COCO 80-class AP50-95 | KITTI shared AP50-95 | Mean full-prediction ms |
|---|---:|---:|---:|
| YOLOv8n | 0.3518 | 0.1943 | 10.59 |
| YOLO11n | 0.3638 | 0.1969 | 14.10 |
| RT-DETR-L | 0.5082 | 0.2860 | 27.27 |

Exact values and additional metrics are in `results/published/tables/`.
The main report has eight pages; the repository overview and the 19-image report
are supplementary. No Python listings are embedded in those PDFs.

## Reproduction and submission

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for inputs and individual stages,
[RUBRIC_CHECKLIST.md](RUBRIC_CHECKLIST.md) for evidence, and
[SUBMISSION_MANIFEST.md](SUBMISSION_MANIFEST.md) for archive contents.
Local configuration is intentionally ignored by Git. The included source/configuration
satisfies code availability; the project repository is also at
https://github.com/VSLEGION/CMPE-249-Detector-Evalutation-Across-Domains . Nothing is pushed,
published or submitted automatically.

