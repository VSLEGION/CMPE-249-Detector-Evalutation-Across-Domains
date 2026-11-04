"""Portable run export; no machine-specific document-runtime path."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .config import PACKAGE_DIR, ROOT
from .io import digest, read, write


def package_run(run, cfg):
    run = Path(run)
    from .report_content import content

    pages = content(run, cfg)
    report = run / "report"
    (report / "figures").mkdir(parents=True, exist_ok=True)
    write(report / "report_content.json", pages)
    for page in pages:
        for name in page.get("figures", []):
            shutil.copy2(run / name, report / "figures" / Path(name).name)
    interpreter = cfg.get("report_python") or sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [interpreter, "-m", "cmpe249_eval.render_report", str(report)],
        env=env,
        capture_output=True,
        text=True,
    )
    write(
        report / "build_status.json",
        {
            "pdf_generated": result.returncode == 0,
            "python": interpreter,
            "stderr": result.stderr,
        },
    )
    if result.returncode:
        print(
            "Report PDF not generated. Install the reports extra or configure report_python. JSON source is saved."
        )
    code_dir = run / "code"
    shutil.copytree(
        ROOT / "src",
        code_dir / "src",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "run_pipeline.py", code_dir / "run_pipeline.py")
    shutil.copy2(ROOT / "pyproject.toml", code_dir / "pyproject.toml")
    files = [
        p
        for p in run.rglob("*")
        if p.is_file()
        and p.suffix not in [".zip", ".tmp"]
        and "__pycache__" not in p.parts
    ]
    manifests = {}
    for name, bulky in [
        ("report_ready_bundle.zip", False),
        ("prediction_evidence_bundle.zip", True),
    ]:
        selected = [
            p
            for p in files
            if (
                p.name
                in [
                    "predictions.json",
                    "predictions.jsonl",
                    "coco_gt.json",
                    "kitti_gt.json",
                    "ablation_gt.json",
                ]
            )
            == bulky
        ]
        target = run / name
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in selected:
                archive.write(path, path.relative_to(run).as_posix())
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt archive: {target}")
        manifests[name] = {"members": len(selected), "sha256": digest(target)}
    write(run / "zip_verification.json", {"status": "PASS", "archives": manifests})
