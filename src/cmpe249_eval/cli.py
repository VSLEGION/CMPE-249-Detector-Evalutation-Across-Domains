"""Small public command surface; paths are resolved by configuration."""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, load_config


COMMANDS = [
    "doctor",
    "test",
    "verify-images",
    "prepare",
    "smoke",
    "coco",
    "kitti",
    "benchmark",
    "ablation",
    "analyze",
    "package",
    "all",
    "evaluate",
]


def doctor(cfg):
    """Inventory dependencies and inputs without loading a detector or writing files."""
    for module in ["torch", "ultralytics", "pycocotools", "PIL", "numpy", "matplotlib"]:
        print(
            f"{module}: {'available' if importlib.util.find_spec(module) else 'MISSING'}"
        )
    for model in cfg["models"]:
        path = Path(cfg["weights_dir"]) / model
        print(
            f"Checkpoint {model}: {'FOUND' if path.is_file() else 'MISSING'} ({path})"
        )
    for key in ["coco_root", "kitti_root"]:
        print(
            f"{key}: {cfg[key]} ({'found' if Path(cfg[key]).exists() else 'not present'})"
        )
    print(f"ADD YOUR VERIFICATION IMAGES HERE: {cfg['benchmark_root']}")
    print(f"New outputs: {cfg['output_dir']}")
    if importlib.util.find_spec("torch"):
        import torch

        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--no-local", action="store_true", help="Ignore configs/local.json"
    )
    parser.add_argument("--run-id", help="New or resumable run name, not a path")
    parser.add_argument("--run", type=Path, help="Existing run for evaluation/analysis")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--categories", nargs="+", help="Optional class-name subset for evaluate"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, use_local=not args.no_local)
    if args.command == "doctor":
        return doctor(cfg)
    from .io import read, write
    from .runtime import record

    if args.run_id and (
        Path(args.run_id).name != args.run_id or args.run_id in [".", ".."]
    ):
        parser.error("--run-id must be a directory name, not a path")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = args.run.resolve() if args.run else Path(cfg["output_dir"]) / run_id
    run.mkdir(parents=True, exist_ok=True)
    write(
        Path(cfg["output_dir"]) / "latest_run.json",
        {"run_id": run.name, "path": str(run)},
    )
    write(run / "config.json", cfg)
    try:
        if args.command == "test":
            from .validation import smoke

            tests = smoke(run)
            print(
                f"PASS: {len(tests)} evaluator/data assertions. Evidence: {run / 'tests'}"
            )
            return 0
        if args.command == "verify-images":
            from .verification import verify_images

            verify_images(run, cfg, args.annotations)
            print(f"Verification complete: {run / 'verification'}")
            return 0
        if args.command == "evaluate":
            if not args.annotations or not args.predictions:
                parser.error("evaluate requires --annotations and --predictions")
            from .evaluation import evaluate

            metrics, _ = evaluate(
                args.annotations, args.predictions, run / "evaluation", args.categories
            )
            print(json.dumps(metrics, indent=2))
            return 0
        from .pipeline import environment, plan, run_accuracy
        from .runtime import run_worker

        matrix = plan(run, cfg)
        stages = (
            [
                "prepare",
                "smoke",
                "coco",
                "kitti",
                "benchmark",
                "ablation",
                "analyze",
                "package",
            ]
            if args.command == "all"
            else [args.command]
        )
        for stage in stages:
            record(run, {"stage": stage, "status": "started"})
            if stage == "prepare":
                from .datasets import prepare

                environment(run, cfg)
                prepare(run, cfg)
            elif stage == "smoke":
                from .validation import smoke

                smoke(run)
                gt = read(run / "data/kitti_gt.json")
                for model in cfg["models"]:
                    run_worker(
                        run,
                        {
                            "id": "smoke_" + Path(model).stem,
                            "model": model,
                            "settings": cfg["primary"],
                        },
                        gt["images"][:3],
                        gt["categories"],
                        cfg=cfg,
                    )
            elif stage in ["coco", "kitti", "benchmark", "ablation"]:
                for case in [c for c in matrix if c["stage"] == stage]:
                    try:
                        if stage == "benchmark":
                            run_worker(
                                run,
                                case,
                                read(run / "data/benchmark.json"),
                                kind="timing",
                                cfg=cfg,
                            )
                        else:
                            run_accuracy(run, case, cfg)
                    except Exception as error:
                        if stage in ["coco", "kitti"]:
                            raise
                        record(
                            run,
                            {
                                "id": case["id"],
                                "status": "optional_failed",
                                "error": str(error),
                            },
                        )
            elif stage == "analyze":
                from .analysis import analyze

                analyze(run, cfg)
            elif stage == "package":
                from .artifacts import package_run

                package_run(run, cfg)
            record(run, {"stage": stage, "status": "complete"})
            write(run / "status.json", {"stage": stage, "status": "complete"})
        print(f"Completed {args.command}: {run}")
        return 0
    except Exception as error:
        record(run, {"status": "failed", "command": args.command, "error": repr(error)})
        write(run / "status.json", {"status": "failed", "error": str(error)})
        print(f"ERROR: {error}")
        return 1
