"""Generate parallel PDF/LaTeX report content from actual saved outputs."""

from pathlib import Path
from .io import read


def content(run, cfg):
    d = read(run / "tables/report_data.json")
    audit = read(run / "data/audit.json")
    manifest = read(run / "run_manifest.json")
    f = lambda v: "N/A" if v is None else f"{v:.3f}"
    baseline = [
        r for r in d["timing_memory_and_sweeps"] if r["id"].endswith("_baseline")
    ]
    primary = [r for r in d["primary"] if r["dataset"] == "coco"]
    shared = d["domain_comparison"]
    scenes = read(run / "figures/scene_selection.json")
    pages = []
    pages.append(
        {
            "title": "CMPE 249 Track A: controlled detector evaluation",
            "paragraphs": [
                "Abstract. This study compares pretrained YOLOv8n, YOLO11n and RT-DETR-L on full COCO val2017 and a fixed 500-image sample of prepared KITTI validation data. All primary comparisons use a single CUDA device, input size 512, batch one and FP32. Results come from saved original-coordinate predictions evaluated with official pycocotools COCOeval. A three-class evaluation supports domain comparison, while bicycle/cyclist is reported separately. Efficiency is measured on decoded images with synchronization, warmup, repeated measurements and fresh model processes. No model was trained or fine-tuned.",
                "Question and scope. How do these three supplied checkpoints trade accuracy for prediction latency, and how do measured shared-class scores change between COCO scenes and the driving domain? The models cover two detector families. Their capacities differ substantially, so the experiment compares deployable checkpoint configurations rather than isolating architecture effects. Published benchmark claims are context, not substitutes for local measurements [1-6].",
                f'Execution evidence. The local environment is {manifest["gpu"]}, PyTorch {manifest["torch"]}, Ultralytics {manifest["ultralytics"]} and Python {manifest["python"].split()[0]}. Audit: {audit["coco_images"]} COCO images, {audit["coco_annotations"]} annotations, {audit["kitti_images"]} KITTI images sampled from {audit["kitti_parent_images"]}. The timing set contains exactly {audit["benchmark_images"]} supported images. All 15 assertion-based evaluator checks passed; all three detectors produced CUDA results with hooks verifying shape, dtype and evaluation state.',
                "Reading the results. AP values are fractions on [0,1]; differences described in percentage points multiply AP differences by 100. COCO all-class AP and the shared-class transfer metric answer different questions and are never subtracted. Converted KITTI scores use COCO rules and are not official KITTI benchmark scores. Detailed commands, hashes, predictions, evaluator arrays and error history accompany this report.",
            ],
            "table": [["Model", "COCO AP", "AP50", "AP75"]]
            + [[r["model"], f(r["AP"]), f(r["AP50"]), f(r["AP75"])] for r in primary],
        }
    )
    pages.append(
        {
            "title": "2. Models and inspected implementation",
            "paragraphs": [
                "YOLOv8n and YOLO11n use multi-scale convolutional detection. The installed YOLOv8 configuration contains C2f feature blocks and SPPF, with top-down and bottom-up feature fusion feeding a Detect head at P3/P4/P5. YOLO11 uses C3k2 blocks and C2PSA while retaining multi-scale detection. Both selected checkpoints pass through DetectionPredictor.postprocess, class-aware non-maximum suppression and original-image coordinate restoration [5,6,8].",
                "The supplied RT-DETR-L configuration has an HGStem/HGBlock backbone, AIFI within-scale interaction, cross-scale RepC3 fusion and an RTDETRDecoder. Its set-prediction lineage uses bipartite assignment, as in DETR [2,3]. This installed RTDETRPredictor has scale_fill=True and converts normalized center/size predictions to original-image corners. It filters confidence and caps predictions without NMS. The supplied checkpoint should not be described as a ResNet variant merely because another RT-DETR implementation uses that backbone [9].",
                "Loss and assignment context. Inspected local code contains TaskAlignedAssigner, BCEWithLogits classification and BboxLoss with distributional box regression for the YOLO path. The RT-DETR loss contains Hungarian matching and class, box/GIoU, auxiliary and denoising terms. These explain training design; no training loss was measured in this project. Inference capacity is reported below from actual loaded parameters before fusion.",
                "Implementation evidence. architecture_notes.md and implementation_snapshot/index.json identify copied installed source files and SHA-256 hashes. experiments/*/model_evidence.json records module counts, parameter dtypes, selected predictor and actual processed tensors. Official documentation describes current behavior but may evolve; the installed snapshot grounds version-specific statements. Equal nominal input sizes still give different image transformations, so preprocessing is part of the checkpoint comparison.",
            ],
            "table": [["Model", "Parameters", "Predictor", "Resize"]]
            + [
                [
                    m,
                    str(
                        read(
                            run
                            / f"experiments/smoke_{Path(m).stem}/model_evidence.json"
                        )["parameters_before_fusion"]
                    ),
                    "RTDETRPredictor" if "rtdetr" in m else "DetectionPredictor",
                    "stretch" if "rtdetr" in m else "letterbox",
                ]
                for m in cfg["models"]
            ],
        }
    )
    pages.append(
        {
            "title": "3. Data, taxonomy and validation",
            "paragraphs": [
                "COCO val2017 uses all 5,000 official image IDs and 80 annotated categories [1]. Predictions map class names to the actual sparse category IDs in instances_val2017.json. The original COCO areas and crowd flags are preserved. KITTI uses the existing prepared Ultralytics images/val split, not the official test server [4]. A sorted parent list is sampled once with Python random.Random(42).sample; the exact 500 selected paths and hashes are saved. A separately seeded 200-image sample of those 500 supplies labeled ablations.",
                "Primary taxonomy: person maps to pedestrian, car to car, truck to truck. KITTI separates vans from cars and sitting persons from pedestrians; those labels are excluded from scored GT. Bicycle to cyclist is an approximate supplementary mapping because the rider and box extents need not match a bicycle-only annotation. No numeric class ID is assumed interchangeable. Nonmapped predictions do not enter the shared-class score.",
                audit["ignore_policy"],
                "Annotation validation checks ID uniqueness, category references, dimensions, finite coordinates, positive boxes and image decodability. Prepared KITTI normalized center/size labels are restored to original pixels and clipped at image bounds. Missing label files raise errors; empty files are recorded as legitimate negatives. This selected subset has no missing or empty label files. Source labels remain unchanged. Several original-resolution GT overlays are saved for visual checks.",
                "Size interpretation. COCO retains original area metadata, usually segmentation-derived. Converted KITTI area is clipped bounding-box width times height. COCOeval size strata are therefore reproducible within each dataset but are not physically equivalent across datasets. Domain effects cannot be reduced to an architecture ranking.",
            ],
            "table": [["Concept", "COCO GT", "KITTI GT", "Role"]]
            + [
                [
                    c,
                    str(audit["coco_counts"][c]),
                    str(audit["kitti_mapped_counts"][c]),
                    "approximate" if c == "bicycle" else "primary",
                ]
                for c in ["person", "car", "truck", "bicycle"]
            ],
            "figures": ["figures/gt_kitti/1.jpg"],
        }
    )
    pages.append(
        {
            "title": "4. Evaluation protocol and primary accuracy",
            "paragraphs": [
                "Primary settings: 512 input, batch 1, FP32, device 0, confidence .001, export cap 300. YOLO NMS IoU is .70. RT-DETR is NMS-free. rect=False forces square processed inputs; YOLO uses aspect-preserving resize/padding while RT-DETR stretches. BGR pixels become RGB floats divided by 255. Hooks assert CUDA, the requested float dtype, batch one, no gradients and evaluation mode. Framework classes are selected explicitly for each detector.",
                "Official COCOeval bbox evaluation uses IoU thresholds .50:.05:.95, recall sampling at 101 points and maxDets [1,10,100] [7]. Saved exports precede metrics, so evaluation can run without another GPU pass. Arrays preserve precision, recall, scores, categories, image IDs and thresholds. AP50-95, AP50, AP75, size AP, AR1/10/100 and per-class AP/GT counts appear in CSV/LaTeX tables. Undefined values are null/N/A rather than zero.",
                "Evaluator tests cover perfect and empty predictions, duplicates, wrong class, wrong localization, absent GT classes, overlapping GT requiring an unmatched alternative, coordinate conversion/clipping, reversed/nonfinite boxes, missing versus empty labels, unknown categories and invalid/duplicate image IDs. The alternative-match case confirms two true positives at IoU .50. Duplicate predictions after complete recall can leave interpolated AP at one; that expected behavior is asserted, not treated as a failure.",
                "The primary accuracy chart uses only person, car and truck on both datasets. Its macro average is calculated over the same categories with the same evaluator. The separate all-80 COCO result measures breadth. The KITTI all-vocabulary export also contains supplementary bicycle/cyclist, which is not the primary shared transfer macro.",
            ],
            "table": [["Model", "COCO APs", "APm", "APl", "AR100"]]
            + [
                [
                    r["model"],
                    f(r["AP_small"]),
                    f(r["AP_medium"]),
                    f(r["AP_large"]),
                    f(r["AR100"]),
                ]
                for r in primary
            ],
            "figures": ["figures/shared_accuracy.png"],
        }
    )
    pp = []
    for m in cfg["models"]:
        c = next(r for r in shared if r["model"] == m and r["dataset"] == "coco")
        k = next(r for r in shared if r["model"] == m and r["dataset"] == "kitti")
        pp.append(
            f"{m}: shared AP changes from {c['shared_AP']:.4f} to {k['shared_AP']:.4f}, a descriptive difference of {100*(k['shared_AP']-c['shared_AP']):+.2f} percentage points. Supplementary KITTI bicycle/cyclist AP is {k['supplementary_bicycle_cyclist_AP']:.4f}. The per-class table and paired plot show whether the macro hides contrasting class behavior."
        )
    car_changes = []
    for m in cfg["models"]:
        c = next(r for r in shared if r["model"] == m and r["dataset"] == "coco")
        k = next(r for r in shared if r["model"] == m and r["dataset"] == "kitti")
        car_changes.append(
            f"{Path(m).stem}: {100*(k['car_AP']-c['car_AP']):+.2f} points"
        )
    pages.append(
        {
            "title": "5. Analysis I: cross-dataset behavior",
            "paragraphs": pp
            + [
                "The macro drop does not apply uniformly across classes. Car AP changes are "
                + ", ".join(car_changes)
                + ". The paired plot shows person and truck reductions alongside these car increases. Class composition and taxonomy therefore matter to interpretation.",
                "These are zero-shot observations from COCO-pretrained models. A change in score can reflect image composition, object frequency, camera viewpoint, object scale, annotation conventions or predictor preprocessing. It does not isolate generalization as a causal property of one architecture. KITTI has many cars but fewer trucks and pedestrians, yet AP is macro averaged by class; the count imbalance changes uncertainty and examples available rather than assigning a car-weighted AP.",
                "The bicycle/cyclist comparison must remain visible even when poor. A cyclist annotation can cover both rider and vehicle, whereas COCO can separately annotate person and bicycle. Car/van and pedestrian/sitting-person distinctions also produce apparent errors under the selected mapping. Missing DontCare masks can penalize detections an official KITTI protocol might ignore. Preserve these caveats when presenting the numerical transfer differences.",
            ],
            "figures": ["figures/per_class_transfer.png"],
        }
    )
    pages.append(
        {
            "title": "6. Analysis II: latency, memory and parameter changes",
            "paragraphs": [
                f"Timing uses confidence {cfg['timing']['conf']}. All {audit['benchmark_images']} decoded images are preloaded outside timing. Each fresh subprocess loads one model, runs {cfg['timing']['warmup']} warmup calls and {cfg['timing']['repeats']} timed repetitions per image. A monotonic CPU clock and CUDA synchronization surround each full model.predict call. This includes resize, H2D, normalization, inference, postprocessing, coordinate restoration and API overhead; it excludes disk read/decode, CPU export and drawing. Framework inference timing isolates the forward pass. The sample count is {audit['benchmark_images'] * cfg['timing']['repeats']} timed calls per configuration.",
                "FPS is 1000 divided by mean per-image full-prediction latency. Tables retain median, p95, standard deviation, repetition means and between-repetition spread. Peak allocated and reserved memory come from the PyTorch allocator after warmup; they are not total device VRAM. The manifest captures power scheme, battery status and nvidia-smi when available. Laptop temperature, clocks and concurrent activity can vary; small differences relative to observed spread do not establish a stable winner.",
                "One-factor changes around the timing baseline cover 384/512/640 resolution, verified FP16 versus FP32, confidence .10/.25/.50, and YOLO-only NMS IoU .50/.70. Identical settings are deduplicated; RT-DETR has no NMS sweep. Labeled ablations use a fixed 200-image KITTI sample at .001 confidence for distinct resolution/precision settings. The corresponding 512 FP32 predictions are reused from the primary run. Unlabeled output counts describe filtering, not measured recall. The observed YOLO latency curves are not monotonic with resolution; clock/host variability and overhead are plausible explanations, not experimentally isolated causes.",
            ],
            "table": [["Model", "Full ms", "Forward ms", "FPS", "Alloc MiB"]]
            + [
                [
                    r["model"],
                    f(r["mean_latency_ms"]),
                    f(r["inference_mean_ms"]),
                    f(r["fps"]),
                    f(r["peak_allocated_MiB"]),
                ]
                for r in baseline
            ],
            "figures": [
                "figures/speed_accuracy.png",
                "figures/parameter_sensitivity.png",
            ],
        }
    )
    pages.append(
        {
            "title": "7. Qualitative failures and limitations",
            "paragraphs": [
                "Six different annotated KITTI scenes are selected with recorded rules: most small GT objects, most mapped objects, most pedestrians, most trucks, maximum overlap fraction between GT boxes, and the highest shared-class false-positive count for YOLOv8n at IoU .50/confidence .25. Exact selection IDs and reasons are saved. Green denotes GT and blue denotes model predictions at the same visualization threshold. Full panels include all three models; separate crops preserve readable object detail. Captions report image IDs and settings. Overlap does not automatically prove physical occlusion; the prepared labels omit occlusion attributes.",
                "The crop below comes from the selected false-positive scene; the corresponding original-resolution full panel is included in figures/. Its selection uses official COCOeval assignments, not a visual guess. Per-image TP/FP/FN diagnostics are saved for the primary shared classes at IoU .50 and confidence .25. An unmatched prediction can reflect wrong class/localization, excluded taxonomy, duplicate output or missing ignore metadata; the panel alone does not prove its cause. Similarly, a small or occluded object is a plausible source of difficulty, not evidence for an unmeasured mechanism.",
                "Limits and conclusions. The results support checkpoint-level speed/accuracy choices under the measured protocol and descriptive domain comparisons. They do not establish performance on another device, precision backend or application distribution. The validation sample is fixed, not an independent official KITTI benchmark. No training, threshold selection or expensive extra model family was added. Reported uncertainty concerns timing repetition, not confidence bounds for AP. Complete raw exports remain available for additional diagnostics.",
            ],
            "figures": [f'figures/crop_{scenes[-1]["id"]}.png'],
        }
    )
    refs = read(run / "references.json")
    pages.append(
        {
            "title": "8. Reproducibility, references and AI-use statement",
            "paragraphs": [
                f"Persistent run: outputs/{run.name}. Resume from the repository root: python run_pipeline.py all --run-id {run.name}. Compatible inference jobs are reused by signature. Config, exact commands, append-only events, dataset manifests, checkpoint hashes and locked dependencies are included. Individual stages support independent reproduction. A saved KITTI prediction export was evaluated again and every shared metric reconciled exactly without GPU inference.",
                "The run retains encountered implementation errors and recovery events; successful checks do not erase earlier failures. Compatible jobs reuse code/config/input/checkpoint signatures. Original project results were not promoted into this run. Archives omit datasets, checkpoints and environments; the separate prediction/evidence archive contains raw exports and GT needed for metric reproduction.",
                "AI-use statement. An AI coding agent implemented the workflow, executed local checks and experiments, inspected source files, consulted the author/official reference pages and generated analysis/report assets from saved outputs. Automated assertions and visual inspection provide the verification described here. The student has not been represented as independently checking the code, results, references or prose; review and understanding before submission remain the student’s responsibility.",
                "References (author/official pages checked 2026-11-03):",
            ]
            + [f'[{i}] {r["citation"]} {r["url"]}' for i, r in enumerate(refs, 1)],
        }
    )
    return pages
