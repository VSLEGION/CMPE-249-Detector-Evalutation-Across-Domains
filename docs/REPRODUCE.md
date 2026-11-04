# Reproduction

Use the existing `dtect` environment. It is outside this repository and remains
unchanged. `requirements.txt` records the lightweight experiment packages;
CUDA-enabled PyTorch/torchvision must match the machine. The original versions
were PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130 on Python 3.13.6.
Do not replace working CUDA wheels with a generic CPU installation.

1. Populate `weights/` with the three checkpoint files, or override `weights_dir`.
2. Place COCO/KITTI validation data under `data/datasets/`, or override those paths.
3. Put verification/benchmark inputs in `data/verification/images/`.
4. Run `doctor`, then `test`, then the desired stage.

Machine-local overrides belong in `configs/local.json` (ignored). Start from
`configs/local.example.json`. Configuration paths are relative to the repository;
absolute external dataset paths also work. `--no-local` checks committed defaults.

From this repository folder on the original machine:

```powershell
..\dtect\Scripts\python.exe run_pipeline.py doctor
..\dtect\Scripts\python.exe -m unittest discover -s tests -v
..\dtect\Scripts\python.exe run_pipeline.py test --run-id smoke_cpu
..\dtect\Scripts\python.exe run_pipeline.py prepare --run-id experiment_001
..\dtect\Scripts\python.exe run_pipeline.py smoke --run-id experiment_001
..\dtect\Scripts\python.exe run_pipeline.py all --run-id experiment_001
```

The `test` command is CPU-only. The `smoke` stage additionally checks all three
models on prepared KITTI images. Reusing a run ID reuses compatible inference
jobs; changing data/configuration/code/checkpoint hashes invalidates them.
The case matrix is 38 configurations with the defaults. The benchmark stage
uses however many supported images you place in the verification directory.

Available stages are `prepare`, `smoke`, `coco`, `kitti`, `benchmark`, `ablation`,
`analyze`, `package`, and `all`. Add `--run <existing-output-folder>` for an
explicit existing run. `--run-id` accepts a name, not a path.

## Evaluate saved predictions without inference

```powershell
python run_pipeline.py evaluate --annotations ground_truth.json --predictions predictions.json --categories person car truck --run-id reevaluation
```

This uses official COCOeval and creates evaluator arrays and metric tables in
the new output folder. Undefined categories/size bins remain unavailable.

## Reports and archives

Install the optional report dependencies in an appropriate reporting interpreter
if needed (`requirements-reports.txt`). `report_python` can point to that
interpreter without altering the CUDA environment. Without the extra, report
content JSON is retained and the PDF build failure is recorded explicitly.

The `package` stage exports report/code evidence separately from bulky prediction
JSON and GT. It checks each ZIP for corruption. No GitHub upload or coursework
submission happens as part of a local stage.

