# Start here - CMPE 249 Track A

## Primary submission

**Read `reports/main_experiment_report.pdf` first.** It is the eight-page
experimental report, within the assignment's 6-10-page range.

The two other PDFs are supporting documents:
`reports/technical_methods_report.pdf` explains the repository and project workflow;
`reports/benchmark_19_report.pdf` covers the 19 unlabeled images in detail.

## Evidence and code

- `run_pipeline.py` is the current code entry point; source is in `src/cmpe249_eval/`.
- `results/published/tables/` contains exact original metrics and efficiency results.
- `results/published/figures/` contains charts, labeled KITTI panels and crops.
- `results/published/timing_samples/` contains original individual timing measurements.
- `results/reproduction/` supports a small metric-only reproduction without GPU inference.
- `results/submission_checks/` records tests of this exact copied folder.
- `reports/main_experiment_sources/` contains editable primary report source/figures.
- `RUBRIC_CHECKLIST.md` maps the requirements to artifacts.

## Main findings

COCO all-class AP50-95 was 0.3518 / 0.3638 / 0.5082 for YOLOv8n /
YOLO11n / RT-DETR-L. Their original mean full-prediction latency was 10.59 /
14.10 / 27.27 ms. Shared KITTI AP was 0.1943 / 0.1969 / 0.2860.
Detailed tables separate shared classes and approximate bicycle/cyclist results.

## Limits to explain when presenting

The KITTI evaluation is COCO-style evaluation on converted labels, not official
KITTI scoring. Its prepared labels omit DontCare/difficulty information.
The local 19 images have no ground truth. The laptop used a Silent power plan,
so its speed measurements are setup-specific. Model capacity/preprocessing differ.
Original experimental results and new submission-folder checks are labeled separately.

## Upload package

Use the verified main ZIP in `submission/`; see `SUBMISSION_MANIFEST.md`.
The optional full prediction archive is separate because of its size.
The source code is included, so no missing external GitHub URL is required to
understand or reproduce the submitted work.

No coursework was uploaded or submitted automatically. Names/IDs are not invented;
use the course portal's own student identification fields.
