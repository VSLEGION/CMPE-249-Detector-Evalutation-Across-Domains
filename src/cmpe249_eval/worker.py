"""Single-model GPU worker. Parent launches each case in a fresh process."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from collections import Counter
import numpy as np
from .io import read, write, rows, digest
from .geometry import box_xywh


def main(job):
    import cv2
    import torch
    from ultralytics import YOLO, RTDETR

    dest = Path(job["destination"])
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA unavailable; CPU fallback is forbidden"
    torch.cuda.set_device(0)
    s = job["settings"]
    model_path = Path(job["model_path"])
    m = (RTDETR if "rtdetr" in job["model"] else YOLO)(str(model_path))
    params = sum(x.numel() for x in m.model.parameters())
    original_types = Counter(type(x).__name__ for x in m.model.modules())
    kw = dict(
        s,
        verbose=False,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show=False,
        augment=False,
    )
    # Setup once, outside timing, so type/precision/shape checks see final fused network.
    paths = job["images"]
    first = cv2.imread(paths[0]["path"])
    assert first is not None
    m.predict(first, **kw)
    observed = []

    def hook(mod, args):
        t = args[0]
        assert t.device.type == "cuda" and t.dtype == (
            torch.float16 if s["half"] else torch.float32
        )
        assert t.shape[0] == 1, "baseline batch must be exactly one"
        assert not torch.is_grad_enabled() and not mod.training
        if len(observed) < 12:
            observed.append(
                {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "device": str(t.device),
                    "grad_enabled": torch.is_grad_enabled(),
                    "training": mod.training,
                }
            )

    handle = m.predictor.model.model.register_forward_pre_hook(hook)
    metadata = {
        "model": job["model"],
        "checkpoint_sha256": digest(model_path),
        "parameters_before_fusion": params,
        "module_counts": dict(original_types),
        "predictor": type(m.predictor).__module__ + "." + type(m.predictor).__name__,
        "scale_fill": m.predictor.scale_fill,
        "settings": s,
        "NMS": (
            "not applicable; NMS-free"
            if "rtdetr" in job["model"]
            else "class-aware IoU " + str(s["iou"])
        ),
        "parameter_dtypes": sorted(
            {str(x.dtype) for x in m.predictor.model.model.parameters()}
        ),
        "pid": os.getpid(),
        "torch_threads": 4,
    }
    assert metadata["parameter_dtypes"] == [
        "torch.float16" if s["half"] else "torch.float32"
    ], metadata
    if job["kind"] == "timing":
        decoded = [cv2.imread(p["path"]) for p in paths]
        assert all(x is not None for x in decoded)
        for i in range(job["warmup"]):
            m.predict(decoded[i % len(decoded)], **kw)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        samples = []
        for rep in range(job["repeats"]):
            for i, im in enumerate(decoded):
                torch.cuda.synchronize(0)
                start = time.perf_counter_ns()
                result = m.predict(im, **kw)[0]
                torch.cuda.synchronize(0)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                samples.append(
                    {
                        "repeat": rep,
                        "image": i,
                        "latency_ms": elapsed,
                        "inference_ms": result.speed["inference"],
                        "preprocess_ms": result.speed["preprocess"],
                        "postprocess_ms": result.speed["postprocess"],
                        "detections": len(result.boxes),
                    }
                )
        rows(dest / "samples.csv", samples)
        v = np.array([x["latency_ms"] for x in samples])
        inf = np.array([x["inference_ms"] for x in samples])
        repmeans = [
            float(np.mean([x["latency_ms"] for x in samples if x["repeat"] == r]))
            for r in range(job["repeats"])
        ]
        summary = {
            "model": job["model"],
            "settings": s,
            "images": len(paths),
            "timed_calls": len(samples),
            "warmup": job["warmup"],
            "repeats": job["repeats"],
            "mean_latency_ms": float(v.mean()),
            "median_latency_ms": float(np.median(v)),
            "p95_latency_ms": float(np.percentile(v, 95)),
            "std_latency_ms": float(v.std(ddof=1)),
            "fps": float(1000 / v.mean()),
            "inference_mean_ms": float(inf.mean()),
            "repeat_means_ms": repmeans,
            "repeat_std_ms": float(np.std(repmeans, ddof=1)),
            "mean_detections": float(np.mean([x["detections"] for x in samples])),
            "peak_allocated_MiB": torch.cuda.max_memory_allocated(0) / 2**20,
            "peak_reserved_MiB": torch.cuda.max_memory_reserved(0) / 2**20,
            "timing_boundary": "CPU decoded BGR input -> model.predict -> synchronized GPU Results. Includes input adaptation, resize, H2D, normalization, inference, NMS/filter/coordinate restore and API overhead. Excludes disk read/decode, CPU box export, image writing. Framework inference_ms isolates forward via CUDA-synchronized Profile.",
        }
        write(dest / "timing.json", summary)
    else:
        cats = job["categories"]
        mapping = {c["name"]: c["id"] for c in cats}
        predictions = []
        count = 0
        output = dest / "predictions.jsonl"
        # A Python list of paths is treated as an in-memory image batch by this
        # Ultralytics version. A .txt source uses LoadImagesAndVideos(batch=1).
        sourcefile = dest / "source_images.txt"
        sourcefile.write_text("\n".join(p["path"] for p in paths), encoding="utf-8")
        with output.open("w", encoding="utf-8") as fp:
            with torch.inference_mode():
                for im, result in zip(
                    paths, m.predict(str(sourcefile), stream=True, **kw)
                ):
                    assert result.orig_shape == (im["height"], im["width"])
                    for x1, y1, x2, y2, score, cls in result.boxes.data.cpu().tolist():
                        name = result.names[int(cls)]
                        if name not in mapping:
                            continue
                        try:
                            b = box_xywh([x1, y1, x2, y2], im["width"], im["height"])
                        except ValueError:
                            continue
                        p = {
                            "image_id": im["id"],
                            "category_id": mapping[name],
                            "bbox": b,
                            "score": score,
                        }
                        predictions.append(p)
                        fp.write(json.dumps(p) + "\n")
                    count += 1
                    if count % 100 == 0:
                        fp.flush()
                        print(f'{job["id"]}: {count}/{len(paths)} images', flush=True)
        assert count == len(paths)
        write(dest / "predictions.json", predictions)
        write(
            dest / "inference.json", {"images": count, "detections": len(predictions)}
        )
    handle.remove()
    metadata["observed_tensors"] = observed
    assert observed
    write(dest / "model_evidence.json", metadata)
    write(dest / "success.json", {"id": job["id"], "signature": job["signature"]})


if __name__ == "__main__":
    job = read(sys.argv[1])
    try:
        main(job)
    except Exception as ex:
        write(
            Path(job["destination"]) / "failure.json",
            {
                "error": repr(ex),
                "traceback": traceback.format_exc(),
                "OOM": "out of memory" in str(ex).lower(),
                "settings": job["settings"],
            },
        )
        traceback.print_exc()
        sys.exit(1)
