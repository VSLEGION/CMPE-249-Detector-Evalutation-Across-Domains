"""Report assets derived exclusively from persisted experiment outputs."""

import csv
import json
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from collections import Counter
import numpy as np
from PIL import Image
from .config import ROOT, PACKAGE_DIR
from .constants import SHARED
from .io import read, write, rows, digest
from .evaluation import evaluate

REFERENCES = [
    (
        "coco",
        "Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.",
        "https://arxiv.org/abs/1405.0312",
        "original paper",
    ),
    (
        "detr",
        "Carion et al. End-to-End Object Detection with Transformers. ECCV 2020.",
        "https://arxiv.org/abs/2005.12872",
        "original paper",
    ),
    (
        "rtdetr",
        "Zhao et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR 2024.",
        "https://arxiv.org/abs/2304.08069",
        "original paper",
    ),
    (
        "kitti",
        "Geiger, Lenz and Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.",
        "https://www.cvlibs.net/publications/Geiger2012CVPR.pdf",
        "original paper",
    ),
    (
        "v8",
        "Ultralytics. YOLOv8 model documentation.",
        "https://docs.ultralytics.com/models/yolov8/",
        "official documentation",
    ),
    (
        "v11",
        "Ultralytics. YOLO11 model documentation.",
        "https://docs.ultralytics.com/models/yolo11/",
        "official documentation",
    ),
    (
        "cocoeval",
        "COCO Consortium. Official Python COCOeval implementation.",
        "https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py",
        "official repository",
    ),
    (
        "ultra",
        "Ultralytics. Official framework source repository.",
        "https://github.com/ultralytics/ultralytics",
        "official repository",
    ),
    (
        "rtimpl",
        "Ultralytics. RT-DETR model documentation.",
        "https://docs.ultralytics.com/models/rtdetr/",
        "official documentation",
    ),
]


def fmt(v, d=3):
    return (
        "N/A" if v is None else f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)
    )


def latex_table(path, data, columns):
    def esc(v):
        return fmt(v).replace("_", r"\_").replace("%", r"\%")

    lines = [
        r"% Requires graphicx. Values are AP fractions, not percent.",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(columns) + "}",
        r"\hline",
        " & ".join(esc(c) for c in columns) + r" \\",
        r"\hline",
    ]
    lines += [" & ".join(esc(r.get(c)) for c in columns) + r" \\" for r in data]
    lines += [r"\hline", r"\end{tabular}", r"}"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def qualitative(run, cfg):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    gt = read(run / "data/kitti_gt.json")
    counts = Counter(a["image_id"] for a in gt["annotations"])
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    candidates = gt["images"]
    selected = []

    # Selection rules use GT and scores; preserve exact selected IDs and reasons.
    def overlap_score(im):
        boxes = [a["bbox"] for a in anns[im["id"]]]
        score = 0
        for i, (x, y, w, h) in enumerate(boxes):
            for xx, yy, ww, hh in boxes[i + 1 :]:
                intersection = max(0, min(x + w, xx + ww) - max(x, xx)) * max(
                    0, min(y + h, yy + hh) - max(y, yy)
                )
                score = max(score, intersection / min(w * h, ww * hh))
        return score

    selectors = [
        ("small_objects", lambda im: sum(a["area"] < 32**2 for a in anns[im["id"]])),
        ("crowded_scene", lambda im: counts[im["id"]]),
        ("pedestrians", lambda im: sum(a["category_id"] == 1 for a in anns[im["id"]])),
        ("trucks", lambda im: sum(a["category_id"] == 3 for a in anns[im["id"]])),
        ("overlapping_gt", overlap_score),
    ]
    for reason, key in selectors:
        im = max(
            (im for im in candidates if im not in [x[1] for x in selected]), key=key
        )
        selected.append((reason, im))
    # False-positive scenes selected using official matching, at IoU .50 / score >= .25.
    predby = {}
    for mn in cfg["models"]:
        stem = Path(mn).stem
        predby[stem] = defaultdict(list)
        for p in read(run / f"experiments/kitti_{stem}/predictions.json"):
            predby[stem][p["image_id"]].append(p)
    fpcounts = Counter()
    state = np.load(
        run
        / f'experiments/kitti_{Path(cfg["models"][0]).stem}/shared/evaluator_arrays.npz'
    )
    # Use saved predictions with COCOeval again for per-image diagnostic assignments.
    _, ev = evaluate(
        gt,
        read(run / f'experiments/kitti_{Path(cfg["models"][0]).stem}/predictions.json'),
        run / "diagnostics/matching",
        SHARED,
    )
    diagnostic = []
    for item in ev.evalImgs:
        if item is None or item["aRng"] != [0, 1e10]:
            continue
        kept = np.array(item["dtScores"]) >= 0.25
        fp = int(((item["dtMatches"][0] == 0) & (~item["dtIgnore"][0]) & kept).sum())
        tp = int(((item["dtMatches"][0] > 0) & (~item["dtIgnore"][0]) & kept).sum())
        matched_at_threshold = set(item["dtMatches"][0][kept & (~item["dtIgnore"][0])])
        fn = sum(
            not ignore and gid not in matched_at_threshold
            for gid, ignore in zip(item["gtIds"], item["gtIgnore"])
        )
        diagnostic.append(
            {
                "image_id": item["image_id"],
                "category_id": item["category_id"],
                "TP_IoU50_conf25": tp,
                "FP_IoU50_conf25": fp,
                "FN_IoU50_conf25": fn,
            }
        )
        fpcounts[item["image_id"]] += fp
    rows(run / "tables/qualitative_diagnostics.csv", diagnostic)
    selected.append(
        (
            "highest_shared_false_positives_yolov8n",
            max(
                (im for im in candidates if im not in [x[1] for x in selected]),
                key=lambda im: fpcounts[im["id"]],
            ),
        )
    )
    write(
        run / "figures/scene_selection.json",
        [{"reason": r, **im} for r, im in selected],
    )
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    for reason, im in selected:
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), layout="constrained")
        pic = Image.open(im["path"])
        for k, ax in enumerate(axes):
            ax.imshow(pic)
            ax.set_axis_off()
            if k == 0:
                boxes = anns[im["id"]]
                color = "lime"
                title = "Ground truth (mapped vocabulary)"
            else:
                stem = Path(cfg["models"][k - 1]).stem
                boxes = [b for b in predby[stem][im["id"]] if b["score"] >= 0.25]
                color = "deepskyblue"
                title = (
                    f"{stem} | 512 FP32 | conf >= .25 | YOLO IoU .70; RT-DETR NMS-free"
                )
            for b in boxes:
                x, y, w, h = b["bbox"]
                label = (
                    "cyclist (approx bicycle)"
                    if k == 0 and b["category_id"] == 4
                    else cats[b["category_id"]]
                )
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), w, h, fill=False, edgecolor=color, linewidth=1.2
                    )
                )
                ax.text(
                    x,
                    y,
                    label + (f" {b['score']:.2f}" if "score" in b else ""),
                    color="black",
                    fontsize=7,
                    bbox=dict(facecolor=color, alpha=0.8, pad=0.5),
                )
            ax.set_title(title, fontsize=10)
        fig.suptitle(
            f'KITTI {im["file_name"]} | ID {im["id"]} | {reason} | green=GT, blue=predictions',
            fontsize=11,
        )
        fig.savefig(run / f'figures/failure_{im["id"]}.png', dpi=150)
        fig.savefig(run / f'figures/failure_{im["id"]}.pdf')
        plt.close(fig)
        # Original-scale crop around a small GT object, with identical context for every model.
        if anns[im["id"]]:
            a = min(anns[im["id"]], key=lambda x: x["area"])
            x, y, w, h = a["bbox"]
            region = (
                max(0, int(x - 45)),
                max(0, int(y - 45)),
                min(im["width"], int(x + w + 45)),
                min(im["height"], int(y + h + 45)),
            )
            fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
            for k, ax in enumerate(axes):
                ax.imshow(pic)
                ax.set_xlim(region[0], region[2])
                ax.set_ylim(region[3], region[1])
                ax.set_axis_off()
                bs = (
                    anns[im["id"]]
                    if k == 0
                    else [
                        b
                        for b in predby[Path(cfg["models"][k - 1]).stem][im["id"]]
                        if b["score"] >= 0.25
                    ]
                )
                for b in bs:
                    bx, by, bw, bh = b["bbox"]
                    ax.add_patch(
                        patches.Rectangle(
                            (bx, by),
                            bw,
                            bh,
                            fill=False,
                            edgecolor="lime" if k == 0 else "deepskyblue",
                            linewidth=1.5,
                        )
                    )
                ax.set_title("GT" if k == 0 else Path(cfg["models"][k - 1]).stem)
            fig.savefig(run / f'figures/crop_{im["id"]}.png', dpi=160)
            plt.close(fig)


def analyze(run, cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run)
    (run / "tables").mkdir(exist_ok=True)
    (run / "figures").mkdir(exist_ok=True)
    from datetime import datetime, timezone

    manifest = read(run / "run_manifest.json")
    manifest["analysis_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["dataset_manifests"] = {
        p.name: {"path": str(p), "sha256": digest(p)}
        for p in (run / "data").glob("*manifest.json")
    }
    manifest["annotation_audit"] = {
        "path": str(run / "data/audit.json"),
        "sha256": digest(run / "data/audit.json"),
    }
    manifest["code_hashes"] = {p.name: digest(p) for p in PACKAGE_DIR.glob("*.py")}
    manifest["entrypoint_sha256"] = digest(ROOT / "run_pipeline.py")
    manifest["model_evidence_files"] = [
        str(p) for p in (run / "experiments").glob("*/model_evidence.json")
    ]
    write(run / "run_manifest.json", manifest)
    primary = []
    transfer = []
    timing = []
    ablation = []
    per = []
    matrix = read(run / "experiment_matrix.json")
    accounted = []
    for case in matrix:
        dest = run / "experiments" / case["id"]
        ok = (dest / "success.json").exists() or (dest / "reuse.json").exists()
        accounted.append(
            {
                "id": case["id"],
                "status": "success" if ok else "failed_or_missing",
                "stage": case["stage"],
            }
        )
        if not ok:
            continue
        if case["stage"] == "benchmark":
            m = read(dest / "timing.json")
            timing.append(
                {
                    "id": case["id"],
                    "model": case["model"],
                    **m["settings"],
                    **{
                        k: v
                        for k, v in m.items()
                        if k not in ["settings", "repeat_means_ms"]
                    },
                    "repeat_means_ms": json.dumps(m["repeat_means_ms"]),
                }
            )
            continue
        a = read(dest / "shared/metrics.json")
        s = read(dest / "supplementary/metrics.json")
        if case["stage"] == "ablation":
            ablation.append(
                {
                    "id": case["id"],
                    "model": case["model"],
                    **case["settings"],
                    **{
                        k: v
                        for k, v in a.items()
                        if k not in ["per_class", "category_names", "maxDets"]
                    },
                }
            )
            continue
        allm = read(dest / "all/metrics.json")
        mainm = allm if case["dataset"] == "coco" else a
        primary.append(
            {
                "id": case["id"],
                "model": case["model"],
                "dataset": case["dataset"],
                "vocabulary": (
                    "COCO 80 classes"
                    if case["dataset"] == "coco"
                    else "primary shared 3 classes"
                ),
                **{
                    k: v
                    for k, v in mainm.items()
                    if k not in ["per_class", "category_names", "maxDets"]
                },
            }
        )
        for p in allm["per_class"]:
            per.append(
                {
                    "id": case["id"],
                    "model": case["model"],
                    "dataset": case["dataset"],
                    "role": (
                        "supplementary approximate"
                        if case["dataset"] == "kitti" and p["class"] == "bicycle"
                        else "primary"
                    ),
                    **p,
                }
            )
        transfer.append(
            {
                "id": case["id"],
                "model": case["model"],
                "dataset": case["dataset"],
                **{
                    "shared_" + k: v
                    for k, v in a.items()
                    if k not in ["per_class", "category_names", "maxDets"]
                },
                "supplementary_bicycle_cyclist_AP": s["AP"],
                **{p["class"] + "_AP": p["AP"] for p in a["per_class"]},
                **{p["class"] + "_GT": p["gt_count"] for p in a["per_class"]},
            }
        )
    counts = []
    audit = read(run / "data/audit.json")
    for domain, key in [
        ("COCO", "coco_counts"),
        ("KITTI native", "kitti_native_counts"),
        ("KITTI mapped", "kitti_mapped_counts"),
        ("KITTI ablation", "ablation_counts"),
    ]:
        counts.extend(
            {"dataset": domain, "class": name, "instances": n}
            for name, n in audit[key].items()
        )
    datasets = {
        "primary": primary,
        "domain_comparison": transfer,
        "timing_memory_and_sweeps": timing,
        "labeled_ablation": ablation,
        "per_class": per,
        "annotation_counts": counts,
        "experiment_status": accounted,
    }
    for name, data in datasets.items():
        rows(run / f"tables/{name}.csv", data)
        cols = {
            "primary": [
                "model",
                "dataset",
                "AP",
                "AP50",
                "AP75",
                "AP_small",
                "AP_medium",
                "AP_large",
                "AR100",
            ],
            "domain_comparison": [
                "model",
                "dataset",
                "shared_AP",
                "person_AP",
                "car_AP",
                "truck_AP",
            ],
            "timing_memory_and_sweeps": [
                "id",
                "mean_latency_ms",
                "p95_latency_ms",
                "fps",
                "inference_mean_ms",
                "peak_allocated_MiB",
            ],
            "labeled_ablation": ["model", "imgsz", "half", "AP", "AP50", "AR100"],
        }.get(name, list(data[0])[:5] if data else [])
        if data:
            latex_table(run / f"tables/{name}.tex", data, cols)
    write(run / "tables/report_data.json", datasets)

    def save(fig, name):
        fig.savefig(run / f"figures/{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(run / f"figures/{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cfg["models"]))
    width = 0.35
    for i, d in enumerate(["coco", "kitti"]):
        vals = [
            next(
                r["shared_AP"]
                for r in transfer
                if r["model"] == m and r["dataset"] == d
            )
            for m in cfg["models"]
        ]
        ax.bar(x + (i - 0.5) * width, vals, width, label=d)
    ax.set_xticks(x, [Path(m).stem for m in cfg["models"]])
    ax.set_ylabel("COCO-style shared-class AP50-95")
    ax.legend()
    ax.set_ylim(0, 1)
    save(fig, "shared_accuracy")
    baselines = [t for t in timing if t["id"].endswith("_baseline")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in baselines:
        ap = next(
            p["AP"]
            for p in primary
            if p["model"] == r["model"] and p["dataset"] == "coco"
        )
        ax.errorbar(
            r["mean_latency_ms"],
            ap,
            xerr=r["repeat_std_ms"],
            fmt="o",
            capsize=4,
            label=Path(r["model"]).stem,
        )
    ax.set_xlabel("Full-prediction latency (ms); error = repeat SD")
    ax.set_ylabel("Full COCO AP50-95")
    ax.legend()
    save(fig, "speed_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, m in zip(axes, cfg["models"]):
        v = [r for r in timing if r["model"] == m]
        ax.bar(
            range(len(v)),
            [r["mean_latency_ms"] for r in v],
            yerr=[r["repeat_std_ms"] for r in v],
            capsize=3,
        )
        ax.set_xticks(
            range(len(v)),
            [r["id"].replace("timing_" + Path(m).stem + "_", "") for r in v],
            rotation=65,
            ha="right",
        )
        ax.set_title(Path(m).stem)
        ax.set_ylabel("Latency ms +/- repeat SD")
    save(fig, "all_timing_sweeps")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, cls in zip(axes, SHARED):
        for m in cfg["models"]:
            vals = [
                next(
                    r[cls + "_AP"]
                    for r in transfer
                    if r["model"] == m and r["dataset"] == d
                )
                for d in ["coco", "kitti"]
            ]
            ax.plot(["COCO", "KITTI"], vals, "o-", label=Path(m).stem)
        ax.set_title(cls)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Per-class AP50-95")
    axes[-1].legend(fontsize=8)
    save(fig, "per_class_transfer")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m in cfg["models"]:
        v = sorted(
            [
                r
                for r in timing
                if r["model"] == m
                and (r["id"].endswith("_baseline") or "_resolution_" in r["id"])
            ],
            key=lambda r: r["imgsz"],
        )
        axes[0].errorbar(
            [r["imgsz"] for r in v],
            [r["mean_latency_ms"] for r in v],
            yerr=[r["repeat_std_ms"] for r in v],
            marker="o",
            label=Path(m).stem,
        )
        v = sorted(
            [r for r in ablation if r["model"] == m and not r["half"]],
            key=lambda r: r["imgsz"],
        )
        axes[1].plot(
            [r["imgsz"] for r in v], [r["AP"] for r in v], "o-", label=Path(m).stem
        )
    axes[0].set(xlabel="Input size", ylabel="Latency (ms)")
    axes[1].set(xlabel="Input size", ylabel="200-image KITTI shared AP")
    axes[0].legend()
    save(fig, "parameter_sensitivity")
    qualitative(run, cfg)
    write(
        run / "references.json",
        [
            {
                "key": k,
                "citation": c,
                "url": u,
                "type": t,
                "checked": "2026-11-03 via official/author page",
            }
            for k, c, u, t in REFERENCES
        ],
    )
    notes = []
    for m in cfg["models"]:
        stem = Path(m).stem
        c = next(r for r in transfer if r["model"] == m and r["dataset"] == "coco")
        k = next(r for r in transfer if r["model"] == m and r["dataset"] == "kitti")
        t = next((r for r in baselines if r["model"] == m), None)
        notes.append(
            f"- {m}: shared AP {c['shared_AP']:.4f} (COCO) to {k['shared_AP']:.4f} (KITTI), difference {100*(k['shared_AP']-c['shared_AP']):+.2f} percentage points. Sources: experiments/coco_{stem}/shared/metrics.json and experiments/kitti_{stem}/shared/metrics.json."
        )
        if t:
            notes.append(
                f"  Timing baseline: {t['mean_latency_ms']:.2f} ms, {t['fps']:.2f} FPS; repeat mean SD {t['repeat_std_ms']:.2f} ms; isolated forward {t['inference_mean_ms']:.2f} ms. Source: {t['id']}/timing.json."
            )
    baseline_model = cfg["models"][0]
    reference_ap = next(
        r["AP"]
        for r in primary
        if r["model"] == baseline_model and r["dataset"] == "coco"
    )
    reference_t = next(r for r in baselines if r["model"] == baseline_model)
    for m in cfg["models"][1:]:
        ap = next(
            r["AP"] for r in primary if r["model"] == m and r["dataset"] == "coco"
        )
        t = next(r for r in baselines if r["model"] == m)
        notes.append(
            f"- Relative to {baseline_model}, {m} adds {100*(ap-reference_ap):+.2f} COCO AP percentage points and has {t['mean_latency_ms']/reference_t['mean_latency_ms']:.2f}x measured full-prediction latency. This compares checkpoint configurations, not architecture alone; repeat spreads above bound interpretation."
        )
    diagnostic = list(
        csv.DictReader((run / "tables/qualitative_diagnostics.csv").open())
    )
    for scene in read(run / "figures/scene_selection.json"):
        subset = [r for r in diagnostic if int(r["image_id"]) == scene["id"]]
        notes.append(
            f"- Qualitative scene {scene['file_name']} (ID {scene['id']}, {scene['reason']}): YOLOv8n shared-class diagnostics at IoU .50/conf .25: {sum(int(r['TP_IoU50_conf25']) for r in subset)} TP, {sum(int(r['FP_IoU50_conf25']) for r in subset)} FP, {sum(int(r['FN_IoU50_conf25']) for r in subset)} FN. Full panel and crop are in figures/. Counts reflect the mapped protocol; missing DontCare and excluded taxonomy can explain some unmatched predictions."
        )
    limitations = "The shared-class comparison uses identical evaluator thresholds, not identical scenes or annotation taxonomies. Differences are descriptive and confounded by dataset composition, object scale, model capacity and resizing. KITTI vans and sitting persons are excluded; mapped predictions overlapping them can become false positives. DontCare/difficulty data are absent. COCO retains original area metadata whereas KITTI uses clipped box area, so size bins are not directly equivalent. Bicycle/cyclist is supplementary and approximate. Timing uncertainty is repeat variability on one laptop and one fixed image set, not a population confidence interval. No threshold was tuned on evaluation data. No retraining was performed."
    effects = []
    for m in cfg["models"]:
        b = next(
            r
            for r in ablation
            if r["model"] == m and r["imgsz"] == 512 and not r["half"]
        )
        for r in [r for r in ablation if r["model"] == m and r["id"] != b["id"]]:
            effects.append(
                f"- {r['id']}: 200-image shared AP {r['AP']:.4f}, {100*(r['AP']-b['AP']):+.2f} percentage points relative to the SAME 200-image 512 FP32 baseline ({b['AP']:.4f})."
            )
        tbase = next(r for r in baselines if r["model"] == m)
        for r in [r for r in timing if r["model"] == m and r["id"] != tbase["id"]]:
            effects.append(
                f"- {r['id']}: latency change {100*(r['mean_latency_ms']/tbase['mean_latency_ms']-1):+.1f}% vs timing baseline; repeat SD {r['repeat_std_ms']:.2f} ms, mean output count {r['mean_detections']:.2f}. Output counts alone do not measure recall."
            )
    analysis = (
        "# Verified experimental findings\n\n"
        + "\n".join(notes)
        + "\n\n## Parameter effects\n\n"
        + "\n".join(effects)
        + "\n\n"
        + limitations
        + "\n\nAll exact tables come from tables/report_data.json. Primary KITTI table values use only the three shared classes. The raw KITTI all/ evaluator directory also includes supplementary bicycle/cyclist; use shared/ and domain_comparison.csv for the primary transfer macro. Counts, mappings and input hashes are in data/audit.json. Qualitative scene selection rules and exact image IDs are in figures/scene_selection.json; matched TP/FP/FN diagnostics use official COCOeval at IoU .50 and confidence .25. A visible disagreement alone does not establish its cause.\n"
    )
    (run / "analysis.md").write_text(analysis, encoding="utf-8")
    architecture(run)
    # Re-evaluate one saved primary prediction file and reconcile metrics without GPU inference.
    one = next(r for r in primary if r["dataset"] == "kitti")
    dest = run / "experiments" / one["id"]
    re, _ = evaluate(
        run / "data/kitti_gt.json",
        dest / "predictions.json",
        run / "reproduction_check",
        SHARED,
    )
    before = read(dest / "shared/metrics.json")
    assert re == before
    write(
        run / "reproduction_check/result.json",
        {
            "status": "PASS",
            "experiment": one["id"],
            "all_shared_metrics_equal": True,
            "prediction_sha256": digest(dest / "predictions.json"),
        },
    )
    from .integrity import verify

    verify(run, cfg)


def architecture(run):
    import ultralytics

    package = Path(ultralytics.__file__).parent
    files = [
        "cfg/models/v8/yolov8.yaml",
        "cfg/models/11/yolo11.yaml",
        "cfg/models/rt-detr/rtdetr-l.yaml",
        "models/yolo/detect/predict.py",
        "models/rtdetr/predict.py",
        "engine/predictor.py",
        "nn/modules/head.py",
        "utils/loss.py",
        "models/utils/loss.py",
        "models/utils/ops.py",
        "utils/tal.py",
    ]
    evidence = []
    for rel in files:
        p = package / rel
        evidence.append({"path": str(p), "relative": rel, "sha256": digest(p)})
        target = run / "implementation_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    write(run / "implementation_snapshot/index.json", evidence)
    text = f"""# Inspected implementation: Ultralytics {ultralytics.__version__}

YOLOv8n: its model YAML uses Conv/C2f stages and SPPF, then upsampling, concatenation and downsampling to fuse P3/P4/P5 features. Detect produces dense anchor-free box/class outputs. YOLO11n substitutes C3k2 feature blocks and adds C2PSA after SPPF, retaining multi-scale Detect. Both pass through DetectionPredictor.postprocess and class-aware non_max_suppression, then ops.scale_boxes restores coordinates. This local version's v8DetectionLoss uses TaskAlignedAssigner, BCEWithLogits classification and BboxLoss with localization/distributional regression. No training was run. [v8,v11,ultra]

RT-DETR-L: inspected YAML has HGStem/HGBlock backbone, AIFI for within-scale interaction, RepC3-based cross-scale fusion and RTDETRDecoder on P3/P4/P5. This is the supplied HG-style checkpoint, not an assumed ResNet variant. RTDETRDetectionLoss and HungarianMatcher implement set matching and classification, box/GIoU plus auxiliary/denoising terms. RTDETRPredictor has scale_fill=True. Its postprocess converts normalized xywh to xyxy, thresholds scores and scales by original width/height, without NMS. The decoder's top-k format is normalized [cx,cy,w,h,score,class] in this installed version. [detr,rtdetr,rtimpl]

Primary inputs use rect=False: all forward-hook observations must show batch 1 and square 512 tensors. YOLO pads aspect-preserving resized images; RT-DETR stretches to square. Both reorder BGR to RGB and divide by 255. FP32/FP16 are verified in hooks and model parameter dtypes rather than inferred from a flag. Metadata reports parameter counts before fusion and module counts for each checkpoint; architecture and capacity are confounded.

Evidence: implementation_snapshot/index.json hashes the copied implementation files; experiments/*/model_evidence.json records predictor class, parameters, processed shapes, dtype, eval/gradient state and device. The official repository's current branch can differ from the installed package. References use documentation for context and this snapshot for version-specific claims.
"""
    (run / "architecture_notes.md").write_text(text, encoding="utf-8")
