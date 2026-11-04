"""Environment manifests, experiment planning and primary accuracy orchestration."""

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from .config import ROOT
from .constants import SHARED, SUPP
from .io import read, write, rows, digest
from .evaluation import evaluate
from .runtime import run_worker as worker


def now():
    return datetime.now(timezone.utc).isoformat()


def environment(run, cfg):
    import torch
    import ultralytics

    manifest = {
        "utc": now(),
        "root": str(ROOT),
        "run": str(run),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "config": cfg,
        "model_hashes": {
            m: digest(Path(cfg["weights_dir"]) / m) for m in cfg["models"]
        },
        "primary_protocol": "FP32 square tensors 512x512; rect=False; YOLO aspect-preserving letterbox, RTDETR scale_fill=True. RGB /255. Predictions restored to original image coordinates. COCO maxDets [1,10,100], export cap 300.",
        "timing_protocol": "separate conf=.25; decoded images outside timing, 10 warmups, 5 repetitions, explicit CUDA synchronization. Allocated/reserved memory are PyTorch allocator metrics.",
    }
    for command in [
        ["nvidia-smi"],
        ["powercfg", "/getactivescheme"],
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining | ConvertTo-Json",
        ],
    ]:
        try:
            manifest[" ".join(command)] = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            ).stdout
        except Exception as e:
            manifest[" ".join(command)] = str(e)
    manifest["dependencies"] = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    write(run / "run_manifest.json", manifest)
    (run / "requirements.lock.txt").write_text(
        "\n".join(f"{n}=={v}" for n, v in sorted(manifest["dependencies"].items()))
    )
    return manifest


def plan(run, cfg):
    matrix = []
    for model in cfg["models"]:
        stem = Path(model).stem
        for domain in ["coco", "kitti"]:
            matrix.append(
                {
                    "id": f"{domain}_{stem}",
                    "stage": domain,
                    "model": model,
                    "settings": dict(cfg["primary"]),
                    "dataset": domain,
                }
            )
        settings = []
        base = {**cfg["primary"], "conf": cfg["timing"]["conf"]}
        settings.append(("baseline", base))
        settings.extend(
            (f"resolution_{sz}", {**base, "imgsz": sz}) for sz in [384, 640]
        )
        settings.append(("fp16", {**base, "half": True}))
        settings.extend((f"conf_{c}", {**base, "conf": c}) for c in [0.1, 0.5])
        if "rtdetr" not in model:
            settings.append(("iou_0.5", {**base, "iou": 0.5}))
        for name, s in settings:
            matrix.append(
                {
                    "id": f"timing_{stem}_{name}",
                    "stage": "benchmark",
                    "model": model,
                    "settings": s,
                    "dataset": "benchmark",
                }
            )
        for sz, half in [(384, False), (512, False), (640, False), (512, True)]:
            matrix.append(
                {
                    "id": f'ablation_{stem}_{sz}_{"fp16" if half else "fp32"}',
                    "stage": "ablation",
                    "model": model,
                    "settings": {**cfg["primary"], "imgsz": sz, "half": half},
                    "dataset": "ablation",
                    "reuse_primary": sz == 512 and not half,
                }
            )
    write(run / "experiment_matrix.json", matrix)
    rows(
        run / "experiment_matrix.csv",
        [{**m, "settings": json.dumps(m["settings"])} for m in matrix],
    )
    return matrix


def run_accuracy(run, case, cfg):
    gt = read(run / f'data/{case["dataset"]}_gt.json')
    if case.get("reuse_primary"):
        primary = (
            run
            / "experiments"
            / f'kitti_{Path(case["model"]).stem}'
            / "predictions.json"
        )
        assert (
            primary.exists()
        ), "Primary predictions required for baseline ablation reuse"
        ids = {im["id"] for im in gt["images"]}
        pred = [p for p in read(primary) if p["image_id"] in ids]
        dest = run / "experiments" / case["id"]
        write(dest / "predictions.json", pred)
        write(
            dest / "reuse.json",
            {
                "source": str(primary),
                "sha256": digest(primary),
                "image_ids": sorted(ids),
            },
        )
        record(run, {"id": case["id"], "status": "reused_primary_subset"})
    else:
        dest = worker(run, case, gt["images"], gt["categories"], cfg=cfg)
    pred = read(dest / "predictions.json")
    for label, cats in [("all", None), ("shared", SHARED), ("supplementary", SUPP)]:
        evaluate(gt, pred, dest / label, cats)
