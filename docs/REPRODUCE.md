# Reproduce the project

## Runtime and inputs

The original experiment used Python 3.13.6, CUDA-enabled PyTorch 2.14.1+cu130,
torchvision 0.29.1+cu130 and Ultralytics 8.4.198 in the existing `dtect`
environment. Reuse a working CUDA environment; do not replace it merely to
inspect the reports or run CPU checks. Package versions are in
`results/published/requirements.lock.txt`; the portable runtime list is
`requirements.txt`. PDF helpers use the optional `requirements-reports.txt`.

Large data and checkpoints are not included in the submission archive.
Use `weights/yolov8n.pt`, `weights/yolo11n.pt`, `weights/rtdetr-l.pt`, and:

- COCO: `data/datasets/coco/annotations/instances_val2017.json` and `images/val2017/`.
- Prepared KITTI: `data/datasets/kitti/kitti.yaml`, `images/val/`, `labels/val/`.

Alternatively, copy `configs/local.example.json` to ignored `configs/local.json`
and specify existing paths. Relative paths resolve from this repository.
On the original machine a private local override reuses the parent-project
weights and existing datasets. `scripts/use_existing_environment.ps1` searches
ancestor directories for the working interpreter, so folder depth is not assumed.

## Checks and verification

From this folder in an activated environment:

```powershell
python run_pipeline.py doctor
python -m unittest discover -s tests -v
python run_pipeline.py test --run-id evaluator_check
python run_pipeline.py verify-images --run-id verification_19
```

The 19 supplied source images are in `data/verification/images/`.
Verification is a fresh untimed .25-confidence pass. Generated overlays and counts
are not ground truth. With real `data/verification/annotations.json` labels,
the same command also evaluates those labels at the display threshold.

## Full scientific workflow

```powershell
python run_pipeline.py all --run-id experiment_001
```

Or run `prepare`, `smoke`, `coco`, `kitti`, `benchmark`, `ablation`,
`analyze`, `package` separately with the same run ID.
The CPU-only `test` command is distinct from the prepared KITTI GPU smoke stage.
Compatible inference jobs are reused by input/configuration/code/checkpoint hashes.
Do not treat saved results as compatible after a material environment change.

## Small saved-prediction reproduction (no GPU)

A converted KITTI ground truth, one original prediction export and its expected
shared-class metrics are bundled in `results/reproduction/`.

```powershell
python run_pipeline.py evaluate --annotations results/reproduction/kitti_ground_truth.json --predictions results/reproduction/yolov8n_predictions.json --categories person car truck --run-id metric_reproduction
```

Compare `outputs/metric_reproduction/evaluation/metrics.json` to
`results/reproduction/expected_shared_metrics.json`.
The submission audit performs this comparison automatically.

## Report sources and packaging

The primary PDF is `reports/main_experiment_report.pdf` (eight pages).
Its portable LaTeX source and figures are in `reports/main_experiment_sources/`.
The main PDF was generated with ReportLab; its LaTeX source was not compiled on
this machine because no TeX distribution was available.

`scripts/prepare_submission.py` checks the copied project, runs the CPU suite,
recomputes saved metrics, and creates verified ZIPs. It does not rerun full COCO
or KITTI. Invoke it with an experiment interpreter and, for PDF inspection, a
reporting interpreter that has pypdf available. See the submission guide.

The original full prediction evidence is supplied as a separate large archive,
excluded from the main submission ZIP. All exact commands, logs and provenance
records identify which checks were rerun and which outputs are historical.

