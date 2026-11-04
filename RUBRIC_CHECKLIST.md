# Track A rubric evidence

| Requirement | Actual evidence |
|---|---|
| At least three configurations from two families | YOLOv8n, YOLO11n, RT-DETR-L; report methods and GPU/model metadata |
| At least two datasets/domains | Full COCO val2017; seed-42 500-image prepared KITTI subset |
| Document taxonomy differences | Main report; repository overview; annotation audit; shared-class table |
| Accuracy metrics | `results/published/tables/primary.csv`, `per_class.csv`, `domain_comparison.csv` |
| Latency/FPS/memory and settings | `timing_memory_and_sweeps.csv`, raw samples, curated run manifest |
| At least two technical analyses | Main report: speed-accuracy and per-class domain transfer; labeled ablations |
| Parameter investigation | Resolution/precision/confidence/YOLO NMS timing cases; fixed 200-image labeled ablations |
| Failure analysis | KITTI labeled panels/crops, `qualitative_diagnostics.csv`, explicit mapping limits |
| Eight credible references, four papers, two official sources | Nine sources in main report and `results/published/references.json` |
| 6-10-page technical PDF | `reports/main_experiment_report.pdf`: eight pages |
| Code/configuration/commands available | Included package, configuration, requirements, original and current command logs |
| Evaluator tests and reproducibility | Original 15 checks; new CPU suite; exact saved-KITTI metric reproduction |
| Actual local GPU execution | Original per-model metadata; fresh 19-image verification in this submission folder |
| Report/source dependencies | Editable main-report LaTeX and figures; repository-report source |
| Explicit AI-use statement | Main report and repository overview |

Training and fine-tuning are optional and were not performed.
There are no invented scores, student identifiers or GitHub URLs.
The raw prediction archive is optional supporting evidence, kept separate from
the main upload. Prepared KITTI scores must not be called official KITTI scores.
The external GitHub Actions workflow is configured; no hosted CI execution is claimed.
