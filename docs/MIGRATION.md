# Migration from the original project

The original project files, environment, datasets, weights, reports and results
remain where they were. This is a new self-contained source repository.

| Original location / behavior | Repository replacement |
|---|---|
| `run_track_a.py` | `run_pipeline.py` -> `src/cmpe249_eval/cli.py` and `pipeline.py` |
| `verified_experiment/core.py` | `io.py`, `geometry.py`, `datasets.py`, `evaluation.py`, `validation.py` |
| `verified_experiment/worker.py` | `src/cmpe249_eval/worker.py` |
| `verified_experiment/reporting.py` | `analysis.py` and `artifacts.py` |
| `verified_experiment/verify_run.py` | `integrity.py` |
| `benchmark_19_report.py` | General-purpose `verify-images` command; original PDF retained as historical evidence |
| Hard-coded parent-project paths | `configs/default.json` plus ignored `configs/local.json` |
| `data/benchmark_images` | `data/verification/images/` (ready for your uploads) |
| `results_verified/<run-id>` | `outputs/<run-id>` for all new runs |
| Original verified summary outputs | `results/published/` (historical, never cached inference) |
| Old prototypes | Preserved only in the parent project, not shipped here |

The file-based batch-one image loader and official COCOeval implementation are
retained. Model/dataset logic was reorganized, not replaced by a new AP algorithm.
The new package supports any verification image count, optional real COCO labels,
standalone CPU tests, and invocation from an unrelated working directory.

Older reports describe the original filenames and original completed run. Use
the table above for current code paths. The full COCO/KITTI experiment was not
rerun simply to reorganize the source; repository validation is recorded separately.

