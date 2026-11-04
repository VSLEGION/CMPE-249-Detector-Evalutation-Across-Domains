"""Dataset audits, seeded manifests, KITTI conversion and GT overlays."""

import copy
import math
import random
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image, ImageDraw
from .config import ROOT
from .geometry import normalized_box
from .io import read, write, digest, image_files


def parse_label(path, width, height, names):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing label is NOT a verified negative: {path}")
    objects = []
    clipped = 0
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        z = line.split()
        if len(z) != 5:
            raise ValueError(f"{path}:{line_no}: expected normalized YOLO label")
        cid = int(z[0])
        name = names[cid]  # Reject unknown IDs.
        b = normalized_box(list(map(float, z[1:])), width, height)
        objects.append((name, b))
    return objects


def validate_gt(g, check_paths=False):
    ims = {x["id"]: x for x in g["images"]}
    cats = {x["id"]: x["name"] for x in g["categories"]}
    assert len(ims) == len(g["images"]), "duplicate image ids"
    assert len(cats) == len(g["categories"]), "duplicate category ids"
    assert len({a["id"] for a in g["annotations"]}) == len(
        g["annotations"]
    ), "duplicate annotation ids"
    counts = Counter()
    for im in ims.values():
        assert im["width"] > 0 and im["height"] > 0
        if check_paths:
            with Image.open(im["path"]) as image:
                assert image.size == (
                    im["width"],
                    im["height"],
                ), f"image dimension mismatch {im}"
                image.verify()
    for a in g["annotations"]:
        assert a["image_id"] in ims and a["category_id"] in cats, "unknown id reference"
        b = a["bbox"]
        assert (
            len(b) == 4 and all(math.isfinite(v) for v in b) and b[2] > 0 and b[3] > 0
        )
        assert a["area"] > 0 and a.get("iscrowd", 0) in (0, 1)
        counts[cats[a["category_id"]]] += 1
    return dict(counts)


def overlays(g, directory, count=6):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cats = {c["id"]: c["name"] for c in g["categories"]}
    for im in g["images"][:count]:
        with Image.open(im["path"]) as src:
            pic = src.convert("RGB")
        d = ImageDraw.Draw(pic)
        for a in g["annotations"]:
            if a["image_id"] != im["id"]:
                continue
            x, y, w, h = a["bbox"]
            d.rectangle((x, y, x + w, y + h), outline="lime", width=3)
            d.text(
                (x, max(0, y - 12)),
                cats[a["category_id"]],
                fill="lime",
                stroke_width=1,
                stroke_fill="black",
            )
        d.text(
            (5, 5),
            f"GT | image {im['id']} | green boxes | original pixels",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        pic.save(directory / f"{im['id']}.jpg")


def prepare(run, cfg):
    from datetime import datetime, timezone

    run = Path(run)
    (run / "data").mkdir(parents=True, exist_ok=True)
    cp = Path(cfg["coco_root"]) / "annotations/instances_val2017.json"
    coco = read(cp)
    assert len(coco["images"]) == 5000 and len(coco["categories"]) == 80
    for im in coco["images"]:
        im["path"] = str(Path(cfg["coco_root"]) / "images/val2017" / im["file_name"])
    cc = validate_gt(coco, True)
    write(run / "data/coco_gt.json", coco)
    kr = Path(cfg["kitti_root"])
    kc = yaml.safe_load((kr / "kitti.yaml").read_text(encoding="utf-8"))
    names = {int(k): v.lower() for k, v in kc["names"].items()}
    assert set(names.values()) == {
        "car",
        "van",
        "truck",
        "pedestrian",
        "person_sitting",
        "cyclist",
        "tram",
        "misc",
    }
    allims = image_files(kr / "images/val")
    assert len(allims) >= cfg["kitti_size"]
    chosen = sorted(random.Random(cfg["seed"]).sample(allims, cfg["kitti_size"]))
    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "truck"},
        {"id": 4, "name": "bicycle"},
    ]
    mapping = {"pedestrian": 1, "car": 2, "truck": 3, "cyclist": 4}
    gt = {"info": {}, "images": [], "categories": categories, "annotations": []}
    native = Counter()
    labels = []
    empty = []
    for i, p in enumerate(chosen, 1):
        with Image.open(p) as im:
            w, h = im.size
            im.verify()
        lp = kr / "labels/val" / f"{p.stem}.txt"
        objs = parse_label(lp, w, h, names)
        if not objs:
            empty.append(p.name)
        gt["images"].append(
            {"id": i, "file_name": p.name, "path": str(p), "width": w, "height": h}
        )
        labels.append({"path": str(lp), "sha256": digest(lp)})
        for name, b in objs:
            native[name] += 1
            if name in mapping:
                gt["annotations"].append(
                    {
                        "id": len(gt["annotations"]) + 1,
                        "image_id": i,
                        "category_id": mapping[name],
                        "bbox": b,
                        "area": b[2] * b[3],
                        "iscrowd": 0,
                    }
                )
    counts = validate_gt(gt, True)
    write(run / "data/kitti_gt.json", gt)
    subids = {
        im["id"]
        for im in random.Random(cfg["seed"]).sample(gt["images"], cfg["ablation_size"])
    }
    sub = copy.deepcopy(gt)
    sub["images"] = [im for im in gt["images"] if im["id"] in subids]
    sub["annotations"] = [a for a in gt["annotations"] if a["image_id"] in subids]
    write(run / "data/ablation_gt.json", sub)
    bench = []
    for p in image_files(ROOT / cfg["benchmark_root"]):
        with Image.open(p) as im:
            w, h = im.size
            im.verify()
        bench.append({"path": str(p), "width": w, "height": h, "sha256": digest(p)})
    assert bench
    audit = {
        "coco_images": len(coco["images"]),
        "coco_annotations": len(coco["annotations"]),
        "coco_counts": cc,
        "coco_source_sha256": digest(cp),
        "coco_area": "original COCO metadata (segmentation-derived), unchanged",
        "kitti_parent_split": "prepared Ultralytics images/val",
        "kitti_parent_images": len(allims),
        "kitti_selection": "random.Random(42).sample sorted parent file list; selected files sorted for execution",
        "kitti_images": len(chosen),
        "kitti_native_counts": dict(native),
        "kitti_mapped_counts": counts,
        "kitti_empty_labels": empty,
        "missing_labels": 0,
        "kitti_label_hashes": labels,
        "kitti_config": kc,
        "ablation_images": len(subids),
        "ablation_counts": validate_gt(sub),
        "benchmark_images": len(bench),
        "mapping": {
            "COCO person": "KITTI pedestrian",
            "COCO car": "KITTI car",
            "COCO truck": "KITTI truck",
            "SUPPLEMENTARY COCO bicycle": "KITTI cyclist (approximate; box includes rider)",
        },
        "excluded_annotations": ["van", "person_sitting", "tram", "misc"],
        "ignore_policy": "Prepared YOLO annotations lack DontCare, truncation, occlusion and difficulty. No difficulty filtering or ignore-region suppression is possible. Excluded classes are not scored as GT; mapped predictions over those objects can count as false positives. Other predicted classes are excluded only for shared-class evaluation. These are COCO-style mapped scores, not official KITTI scores.",
        "kitti_area": "clipped bounding-box width * height; not comparable to COCO segmentation area",
    }
    write(run / "data/audit.json", audit)
    write(run / "data/benchmark.json", bench)
    for kind, g in [("coco", coco), ("kitti", gt), ("ablation", sub)]:
        write(
            run / f"data/{kind}_image_manifest.json",
            [{**x, "sha256": digest(x["path"])} for x in g["images"]],
        )
        overlays(g, run / f"figures/gt_{kind}")
    return audit
