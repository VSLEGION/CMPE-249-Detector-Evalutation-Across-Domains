# Architecture

`run_pipeline.py` only bootstraps the package and delegates to the CLI.
Everything is resolved from the repository root, not from the shell's current
directory. There are no imports from the parent project or its old scripts.

```text
cli -> config -> repository-relative paths + optional local overrides
    -> pipeline -> datasets / runtime -> isolated worker
    -> verification -> image inventory / runtime / original-pixel overlays
    -> evaluation -> official COCOeval / saved arrays
    -> analysis -> tables / plots / domain and parameter comparisons
    -> artifacts -> report renderer / source snapshots / verified archives
```

| Module | Responsibility |
|---|---|
| `config.py` | Resolve paths; merge ignored machine-local overrides; validate baseline |
| `constants.py` | Shared class vocabulary and image extensions |
| `io.py` | Atomic JSON, CSV, hashes and sorted image discovery |
| `geometry.py` | Validate, convert and clip bounding boxes |
| `datasets.py` | COCO/KITTI preparation, seeded subsets, annotation audit, GT overlays |
| `evaluation.py` | Official bbox metrics, undefined-value handling and evaluator arrays |
| `validation.py` | Fifteen asserted metric/data regression cases |
| `runtime.py` | Model isolation, content-based cache signatures and one clean OOM retry |
| `worker.py` | Verified CUDA inference or warm/repeated synchronized timing |
| `pipeline.py` | Environment manifests, case matrix and labeled accuracy stages |
| `verification.py` | Arbitrary user image folders, optional labels and untimed overlays |
| `analysis.py` | Aggregate saved outputs and generate quantitative/qualitative evidence |
| `integrity.py` | Reconcile image counts, hashes, tensor shapes and metric/timing summaries |
| `report_content.py` | Report prose/tables derived from saved outputs |
| `render_report.py` | PDF rendering (optional reporting dependencies) |
| `artifacts.py` | Export run packages and check ZIP member integrity |

Dependencies flow toward small data/metric utilities. The worker is the only
module that constructs a detector. CPU evaluator tests do not need checkpoints,
validation datasets, CUDA or a populated image directory.

Large inputs and new outputs are separate from the code. `results/published/`
contains compact artifacts from the original verified run, and never acts as
an inference cache. The old prototypes were not copied into this repository.

