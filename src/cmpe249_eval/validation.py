"""Asserted evaluator regression checks, runnable without a GPU."""

import copy
from pathlib import Path
import numpy as np
from .datasets import parse_label, validate_gt
from .geometry import normalized_box, box_xywh
from .evaluation import evaluate
from .io import write, rows


def smoke(run):
    run = Path(run)
    out = run / "tests"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    gt = {
        "info": {},
        "images": [{"id": 19, "file_name": "synthetic", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}, {"id": 7, "name": "absent"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 19,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
    }
    p = {"image_id": 19, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99}

    def check(name, fn):
        try:
            evidence = fn()
            results.append({"test": name, "status": "PASS", "evidence": str(evidence)})
        except Exception as ex:
            results.append({"test": name, "status": "FAIL", "evidence": repr(ex)})

    def metric(name, preds, want, g=gt):
        write(out / name / "gt.json", g)
        write(out / name / "predictions.json", preds)
        m, e = evaluate(g, preds, out / name)
        assert abs(m["AP"] - want) < 1e-8, (m, want)
        return m["AP"]

    check("perfect", lambda: metric("perfect", [p], 1))
    check("empty_predictions", lambda: metric("empty_predictions", [], 0))
    check(
        "duplicate_detection",
        lambda: metric("duplicate_detection", [p, {**p, "score": 0.9}], 1),
    )
    check("wrong_class", lambda: metric("wrong_class", [{**p, "category_id": 7}], 0))
    check(
        "wrong_localization",
        lambda: metric("wrong_localization", [{**p, "bbox": [60, 60, 20, 20]}], 0),
    )
    g2 = copy.deepcopy(gt)
    g2["annotations"].append(
        {**gt["annotations"][0], "id": 2, "bbox": [12, 10, 20, 20]}
    )

    def alternatives():
        # Both predictions prefer GT1; prediction 2 must use the valid unmatched GT2.
        m, e = evaluate(g2, [p, {**p, "score": 0.9}], out / "overlap_alternative")
        ev = next(
            x
            for x in e.evalImgs
            if x and x["category_id"] == 1 and x["aRng"] == [0, 1e10]
        )
        assert (ev["dtMatches"][0] > 0).sum() == 2, ev
        return "2 TPs at IoU .50, alternative GT matched"

    check("overlapping_gt_alternative_match", alternatives)

    def absent():
        m, e = evaluate(gt, [p], out / "absent")
        assert m["per_class"][1]["AP"] is None
        return "undefined class AP = null"

    check("absent_gt_class", absent)

    def coords():
        b = normalized_box([0.5, 0.5, 0.2, 0.4], 200, 100)
        assert np.allclose(b, [80, 30, 40, 40])
        assert np.allclose(box_xywh([10, 20, 40, 60], 100, 100), [10, 20, 30, 40])
        assert np.allclose(box_xywh([-2, -2, 20, 20], 100, 100), [0, 0, 20, 20])
        return b

    check("coordinate_conversion_and_clipping", coords)

    def rejects(fn, exc):
        try:
            fn()
        except exc:
            return "correctly rejected"
        raise AssertionError("did not reject invalid input")

    check(
        "reversed_box",
        lambda: rejects(lambda: box_xywh([20, 10, 10, 30], 100, 100), ValueError),
    )
    check(
        "nonfinite_box",
        lambda: rejects(
            lambda: normalized_box([float("nan"), 0.5, 0.2, 0.2], 100, 100), ValueError
        ),
    )
    check(
        "missing_label",
        lambda: rejects(
            lambda: parse_label(out / "nonexistent.txt", 100, 100, {0: "car"}),
            FileNotFoundError,
        ),
    )
    (out / "empty.txt").write_text("")

    def empty():
        assert parse_label(out / "empty.txt", 100, 100, {0: "car"}) == []
        return "legitimate empty label preserved"

    check("empty_label", empty)
    (out / "unknown.txt").write_text("9 0.5 0.5 0.1 0.1")
    check(
        "unknown_category",
        lambda: rejects(
            lambda: parse_label(out / "unknown.txt", 100, 100, {0: "car"}), KeyError
        ),
    )
    check(
        "unknown_image_id",
        lambda: rejects(
            lambda: evaluate(gt, [{**p, "image_id": 20}], out / "bad_id"),
            AssertionError,
        ),
    )
    dup = copy.deepcopy(gt)
    dup["images"] *= 2
    check(
        "duplicate_image_id", lambda: rejects(lambda: validate_gt(dup), AssertionError)
    )
    rows(out / "tests.csv", results)
    write(out / "results.json", results)
    if any(r["status"] != "PASS" for r in results):
        raise AssertionError(results)
    return results
