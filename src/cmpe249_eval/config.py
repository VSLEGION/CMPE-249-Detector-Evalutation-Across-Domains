"""Resolve every data/output path against the repository, never the shell cwd."""

import copy
import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]
PATH_KEYS = ("weights_dir", "coco_root", "kitti_root", "benchmark_root", "output_dir")


def merge(base, override):
    """Deep-merge a local configuration without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path=None, *, use_local=True, root=ROOT):
    root = Path(root).resolve()
    config_path = Path(path).resolve() if path else root / "configs/default.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    local = root / "configs/local.json"
    if use_local and local.exists():
        cfg = merge(cfg, json.loads(local.read_text(encoding="utf-8")))
    for key in PATH_KEYS:
        value = Path(cfg[key]).expanduser()
        cfg[key] = str(
            value.resolve() if value.is_absolute() else (root / value).resolve()
        )
    if cfg.get("report_python"):
        report_python = Path(cfg["report_python"])
        cfg["report_python"] = str(
            report_python if report_python.is_absolute() else root / report_python
        )
    if cfg["primary"]["batch"] != 1 or str(cfg["primary"]["device"]) != "0":
        raise ValueError("The common protocol requires batch=1 and CUDA device=0")
    if cfg["primary"]["imgsz"] != 512 or cfg["primary"]["half"]:
        raise ValueError("The primary protocol requires imgsz=512 and FP32")
    if not 0 < cfg["ablation_size"] <= cfg["kitti_size"]:
        raise ValueError(
            "Ablation size must be positive and no larger than KITTI subset"
        )
    return cfg
