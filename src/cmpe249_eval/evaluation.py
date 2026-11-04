"""Official pycocotools bbox metrics and saved evaluator arrays."""

import contextlib
import copy
import io
import math
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .datasets import validate_gt
from .io import read, write, rows


def evaluate(gt, pred, out, category_names=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(gt, (str, Path)):
        gt = read(gt)
    if isinstance(pred, (str, Path)):
        pred = read(pred)
    validate_gt(gt)
    ids = {im["id"] for im in gt["images"]}
    cats = {c["id"]: c["name"] for c in gt["categories"]}
    for p in pred:
        assert (
            p["image_id"] in ids and p["category_id"] in cats
        ), "invalid prediction image/category ID"
        assert (
            len(p["bbox"]) == 4
            and all(math.isfinite(v) for v in p["bbox"])
            and min(p["bbox"][2:]) > 0
        )
        assert math.isfinite(p["score"]) and 0 <= p["score"] <= 1
    selected = [
        cid for cid, n in cats.items() if category_names is None or n in category_names
    ]
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        c = COCO()
        c.dataset = copy.deepcopy(gt)
        c.createIndex()
        if pred:
            d = c.loadRes(copy.deepcopy(pred))
        else:
            d = COCO()
            d.dataset = {
                "images": copy.deepcopy(gt["images"]),
                "categories": copy.deepcopy(gt["categories"]),
                "annotations": [],
            }
            d.createIndex()
        e = COCOeval(c, d, "bbox")
        e.params.imgIds = sorted(ids)
        e.params.catIds = sorted(selected)
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        e.summarize()
    (out / "summary.txt").write_text(capture.getvalue())
    keys = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    result = {k: float(v) if v >= 0 else None for k, v in zip(keys, e.stats)}
    per = []
    for i, cid in enumerate(e.params.catIds):
        v = e.eval["precision"][:, :, i, 0, -1]
        v = v[v >= 0]
        per.append(
            {
                "class": cats[cid],
                "category_id": cid,
                "AP": float(v.mean()) if len(v) else None,
                "gt_count": sum(a["category_id"] == cid for a in gt["annotations"]),
            }
        )
    result.update(
        per_class=per,
        image_count=len(ids),
        gt_count=sum(a["category_id"] in selected for a in gt["annotations"]),
        maxDets=e.params.maxDets,
        category_names=[cats[cid] for cid in e.params.catIds],
    )
    write(out / "metrics.json", result)
    rows(out / "per_class.csv", per)
    np.savez_compressed(
        out / "evaluator_arrays.npz",
        precision=e.eval["precision"],
        recall=e.eval["recall"],
        scores=e.eval["scores"],
        iouThrs=e.params.iouThrs,
        recThrs=e.params.recThrs,
        catIds=e.params.catIds,
        imgIds=e.params.imgIds,
    )
    return result, e
