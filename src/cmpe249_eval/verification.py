"""User-supplied image verification, independent of COCO/KITTI downloads."""

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .datasets import validate_gt
from .evaluation import evaluate
from .io import digest, image_files, read, rows, write
from .runtime import run_worker

# COCO's public sparse IDs, paired with names rather than detector indices.
COCO_CATEGORIES = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (5, "airplane"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (9, "boat"),
    (10, "traffic light"),
    (11, "fire hydrant"),
    (13, "stop sign"),
    (14, "parking meter"),
    (15, "bench"),
    (16, "bird"),
    (17, "cat"),
    (18, "dog"),
    (19, "horse"),
    (20, "sheep"),
    (21, "cow"),
    (22, "elephant"),
    (23, "bear"),
    (24, "zebra"),
    (25, "giraffe"),
    (27, "backpack"),
    (28, "umbrella"),
    (31, "handbag"),
    (32, "tie"),
    (33, "suitcase"),
    (34, "frisbee"),
    (35, "skis"),
    (36, "snowboard"),
    (37, "sports ball"),
    (38, "kite"),
    (39, "baseball bat"),
    (40, "baseball glove"),
    (41, "skateboard"),
    (42, "surfboard"),
    (43, "tennis racket"),
    (44, "bottle"),
    (46, "wine glass"),
    (47, "cup"),
    (48, "fork"),
    (49, "knife"),
    (50, "spoon"),
    (51, "bowl"),
    (52, "banana"),
    (53, "apple"),
    (54, "sandwich"),
    (55, "orange"),
    (56, "broccoli"),
    (57, "carrot"),
    (58, "hot dog"),
    (59, "pizza"),
    (60, "donut"),
    (61, "cake"),
    (62, "chair"),
    (63, "couch"),
    (64, "potted plant"),
    (65, "bed"),
    (67, "dining table"),
    (70, "toilet"),
    (72, "tv"),
    (73, "laptop"),
    (74, "mouse"),
    (75, "remote"),
    (76, "keyboard"),
    (77, "cell phone"),
    (78, "microwave"),
    (79, "oven"),
    (80, "toaster"),
    (81, "sink"),
    (82, "refrigerator"),
    (84, "book"),
    (85, "clock"),
    (86, "vase"),
    (87, "scissors"),
    (88, "teddy bear"),
    (89, "hair drier"),
    (90, "toothbrush"),
]


def inventory_images(image_dir, annotations=None):
    """Decode/hash inputs and optionally bind every image to real COCO labels."""
    image_dir = Path(image_dir).resolve()
    paths = image_files(image_dir)
    if not paths:
        raise FileNotFoundError(
            f"No verification images found. Add JPG/PNG images to {image_dir}"
        )
    items = []
    for index, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        items.append(
            {
                "id": index,
                "path": str(path),
                "file_name": path.relative_to(image_dir).as_posix(),
                "width": width,
                "height": height,
                "sha256": digest(path),
            }
        )
    if not annotations:
        return items, None
    gt = read(annotations)
    validate_gt(gt)
    index = {im["file_name"]: im for im in gt["images"]}
    if len(index) != len(gt["images"]) or set(index) != {i["file_name"] for i in items}:
        raise ValueError(
            "Annotation filenames must uniquely cover every verification image"
        )
    known = {name for _, name in COCO_CATEGORIES}
    unknown = {cat["name"] for cat in gt["categories"]} - known
    if unknown:
        raise ValueError(f"Unknown COCO class names in annotations: {sorted(unknown)}")
    for item in items:
        annotated = index[item["file_name"]]
        if (item["width"], item["height"]) != (annotated["width"], annotated["height"]):
            raise ValueError(f"Annotation dimensions differ: {item['file_name']}")
        item["id"] = annotated["id"]
        annotated["path"] = item["path"]
    return items, gt


def draw_overlays(items, predictions, categories, destination, model):
    """Write labeled original-coordinate overlays, outside timed inference."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = {cat["id"]: cat["name"] for cat in categories}
    summary = []
    for item in items:
        boxes = [p for p in predictions if p["image_id"] == item["id"]]
        with Image.open(item["path"]) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(12, min(image.size) // 45))
        for box in boxes:
            x, y, width, height = box["bbox"]
            draw.rectangle(
                (x, y, x + width, y + height),
                outline="cyan",
                width=max(2, image.width // 600),
            )
            draw.text(
                (x, max(0, y - font.size)),
                f"{names[box['category_id']]} {box['score']:.2f}",
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )
        image.thumbnail((2200, 2200))
        image.save(destination / f"{item['id']}.jpg", quality=92)
        counts = Counter(names[box["category_id"]] for box in boxes)
        summary.append(
            {
                "model": model,
                "image_id": item["id"],
                "file_name": item["file_name"],
                "detections": len(boxes),
                "classes": "; ".join(f"{c}: {n}" for c, n in sorted(counts.items())),
            }
        )
    return summary


def verify_images(run, cfg, annotation_path=None):
    """Run all detectors on the folder the user fills after repository setup."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    default_annotation = Path(cfg["benchmark_root"]).parent / "annotations.json"
    annotation_path = annotation_path or (
        default_annotation if default_annotation.exists() else None
    )
    items, gt = inventory_images(cfg["benchmark_root"], annotation_path)
    categories = (
        gt["categories"]
        if gt
        else [{"id": i, "name": name} for i, name in COCO_CATEGORIES]
    )
    write(run / "verification/image_manifest.json", items)
    if gt:
        write(run / "verification/ground_truth.json", gt)
    output_rows = []
    for model in cfg["models"]:
        case = {
            "id": f"verification_{Path(model).stem}",
            "model": model,
            "settings": {**cfg["primary"], "conf": cfg["timing"]["conf"]},
        }
        destination = run_worker(run, case, items, categories, cfg=cfg)
        predictions = read(destination / "predictions.json")
        output_rows.extend(
            draw_overlays(
                items,
                predictions,
                categories,
                run / "verification/overlays" / Path(model).stem,
                model,
            )
        )
        if gt:
            evaluate(gt, predictions, destination / "evaluation")
    rows(run / "verification/summary.csv", output_rows)
    write(
        run / "verification/status.json",
        {
            "status": "complete",
            "images": len(items),
            "model_cases": len(cfg["models"]),
            "ground_truth_supplied": gt is not None,
            "confidence": cfg["timing"]["conf"],
            "metric_scope": (
                "COCO bbox at display confidence; not the .001 primary protocol"
                if gt
                else "unlabeled qualitative outputs only"
            ),
        },
    )
    return output_rows
