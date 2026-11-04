"""Reconcile completed experiment artifacts; never treats missing jobs as success."""

import csv
from collections import Counter
from pathlib import Path
import numpy as np
from .io import read, write, rows, digest


def verify(run, cfg):
    run = Path(run)
    results = []
    matrix = read(run / "experiment_matrix.json")
    for domain in ["coco", "kitti", "ablation"]:
        gt = read(run / f"data/{domain}_gt.json")
        manifest = read(run / f"data/{domain}_image_manifest.json")
        ids = {im["id"] for im in gt["images"]}
        assert (
            len(ids)
            == {
                "coco": 5000,
                "kitti": cfg["kitti_size"],
                "ablation": cfg["ablation_size"],
            }[domain]
        )
        assert {im["id"] for im in manifest} == ids
        for im in manifest:
            assert digest(im["path"]) == im["sha256"]
        results.append(
            {
                "check": domain + "_image_manifest_hashes",
                "status": "PASS",
                "count": len(ids),
            }
        )
    for case in matrix:
        dest = run / "experiments" / case["id"]
        if not (dest / "success.json").exists() and not (dest / "reuse.json").exists():
            continue
        if case["stage"] == "benchmark":
            summary = read(dest / "timing.json")
            samples = list(csv.DictReader((dest / "samples.csv").open()))
            expected_calls = (
                len(read(run / "data/benchmark.json")) * cfg["timing"]["repeats"]
            )
            assert summary["timed_calls"] == len(samples) == expected_calls
            assert (
                abs(
                    summary["mean_latency_ms"]
                    - np.mean([float(r["latency_ms"]) for r in samples])
                )
                < 1e-9
            )
        else:
            domain = case["dataset"]
            gt = read(run / f"data/{domain}_gt.json")
            ids = {im["id"] for im in gt["images"]}
            pred = read(dest / "predictions.json")
            counts = Counter(p["image_id"] for p in pred)
            assert set(counts) <= ids and max(counts.values(), default=0) <= 300
            a = read(dest / "shared/metrics.json")
            valid = [p["AP"] for p in a["per_class"] if p["AP"] is not None]
            assert abs(a["AP"] - np.mean(valid)) < 1e-12
        if (dest / "model_evidence.json").exists():
            e = read(dest / "model_evidence.json")
            assert all(
                t["shape"]
                == [1, 3, case["settings"]["imgsz"], case["settings"]["imgsz"]]
                for t in e["observed_tensors"]
            )
        results.append({"check": case["id"], "status": "PASS"})
    rows(run / "tests/run_integrity.csv", results)
    write(run / "tests/run_integrity.json", results)
    return results
