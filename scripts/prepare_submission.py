"""Audit a submission copy and create clean local ZIPs; never uploads files."""

import argparse
import configparser
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "results/submission_checks"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_original_evidence(source):
    """Copy compact historical evidence; keep its provenance explicit."""
    source = Path(source).resolve()
    published = ROOT / "results/published"
    provenance = {}
    for name in [
        "requirements.lock.txt",
        "commands.log",
        "experiment_matrix.csv",
        "experiment_matrix.json",
        "analysis.md",
        "architecture_notes.md",
    ]:
        copy(source / name, published / name)
        provenance[name] = digest(source / name)
    manifest = read(source / "run_manifest.json")
    keys = [
        "utc",
        "python",
        "platform",
        "torch",
        "cuda_build",
        "cuda_available",
        "ultralytics",
        "gpu",
        "gpu_total_memory",
        "dependencies",
        "model_hashes",
        "primary_protocol",
        "timing_protocol",
    ]
    curated = {key: manifest[key] for key in keys if key in manifest}
    curated.update(
        {
            "original_run_id": source.name,
            "source_manifest_sha256": digest(source / "run_manifest.json"),
            "scope": "Original experiment; paths and unrelated OS process listings omitted",
        }
    )
    curated["power_scheme"] = manifest.get("powercfg /getactivescheme")
    driver = re.search(r"NVIDIA-SMI\s+([\d.]+)", manifest.get("nvidia-smi", ""))
    curated["driver_version_snapshot"] = driver.group(1) if driver else None
    write(published / "run_manifest.json", curated)
    audit = read(source / "data/audit.json")
    audit["kitti_label_hashes"] = [
        {"file_name": Path(item["path"]).name, "sha256": item["sha256"]}
        for item in audit["kitti_label_hashes"]
    ]
    write(published / "annotation_audit.json", audit)
    for domain in ["coco", "kitti", "ablation"]:
        records = read(source / f"data/{domain}_image_manifest.json")
        write(
            published / f"manifests/{domain}_images.json",
            [{k: v for k, v in item.items() if k != "path"} for item in records],
        )
    for path in (source / "experiments").glob("timing_*/samples.csv"):
        copy(path, published / "timing_samples" / f"{path.parent.name}.csv")
    for path in (source / "experiments").glob("*/console_attempt_*.log"):
        copy(path, published / "logs" / path.parent.name / path.name)
    for path in (source / "figures").glob("*"):
        if path.is_file() and path.suffix in [".png", ".pdf", ".jpg", ".json"]:
            copy(path, published / "figures" / path.name)
    for path in (source / "figures/gt_kitti").glob("*.jpg"):
        copy(path, published / "figures/gt_kitti" / path.name)
    for path in (source / "benchmark_19_report/tables").glob("*.csv"):
        copy(path, published / "benchmark_19/tables" / path.name)
    for path in (source / "benchmark_19_report/figures").glob("image_*.jpg"):
        copy(path, published / "benchmark_19/figures" / path.name)
    for path in (source / "report").glob("report.*"):
        if path.suffix in [".tex", ".json"]:
            copy(path, ROOT / "reports/main_experiment_sources" / path.name)
    for path in (source / "report/figures").iterdir():
        if path.is_file():
            copy(path, ROOT / "reports/main_experiment_sources/figures" / path.name)
    gt = read(source / "data/kitti_gt.json")
    for image in gt["images"]:
        image.pop("path", None)
    write(ROOT / "results/reproduction/kitti_ground_truth.json", gt)
    copy(
        source / "experiments/kitti_yolov8n/predictions.json",
        ROOT / "results/reproduction/yolov8n_predictions.json",
    )
    copy(
        source / "experiments/kitti_yolov8n/shared/metrics.json",
        ROOT / "results/reproduction/expected_shared_metrics.json",
    )
    copy(
        source / "prediction_evidence_bundle.zip",
        ROOT / "submission/prediction_evidence_bundle.zip",
    )
    write(
        published / "evidence_provenance.json",
        {
            "original_run_id": source.name,
            "copied_original_files_sha256": provenance,
            "full_prediction_archive_sha256": digest(
                source / "prediction_evidence_bundle.zip"
            ),
            "new_full_experiment_run": False,
        },
    )


def invoke(python, arguments, log_name):
    command = [str(python), *map(str, arguments)]
    CHECKS.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (CHECKS / log_name).write_text(
        result.stdout + "\n" + result.stderr, encoding="utf-8"
    )
    with (CHECKS / "commands.log").open("a", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline(command) + "\n")
    if result.returncode:
        raise RuntimeError(
            f"Command failed; inspect results/submission_checks/{log_name}"
        )
    return result


def audit(python):
    expected = read(
        ROOT / "results/published/verification/original_benchmark_images.json"
    )
    for image in expected:
        path = ROOT / "data/verification/images" / image["file_name"]
        if not path.is_file() or digest(path) != image["sha256"]:
            raise ValueError(
                f"Benchmark image missing or changed: {image['file_name']}"
            )
    unit = invoke(
        python, ["-m", "unittest", "discover", "-s", "tests", "-v"], "unit_tests.log"
    )
    match = re.search(r"Ran\s+(\d+)\s+tests?", unit.stderr)
    invoke(python, ["run_pipeline.py", "doctor"], "doctor.log")
    invoke(
        python,
        ["run_pipeline.py", "test", "--run-id", "submission_evaluator_check"],
        "evaluator_tests.log",
    )
    invoke(
        python,
        ["run_pipeline.py", "verify-images", "--run-id", "submission_verification_19"],
        "verification_gpu.log",
    )
    verification_run = ROOT / "outputs/submission_verification_19"
    status = read(verification_run / "verification/status.json")
    if status["images"] != 19 or status["model_cases"] != 3:
        raise AssertionError(status)
    copy(
        verification_run / "verification/summary.csv",
        CHECKS / "verification/summary.csv",
    )
    manifest = read(verification_run / "verification/image_manifest.json")
    write(
        CHECKS / "verification/image_manifest.json",
        [{k: v for k, v in im.items() if k != "path"} for im in manifest],
    )
    for path in (verification_run / "verification/overlays").rglob("*.jpg"):
        copy(
            path,
            CHECKS
            / "verification/overlays"
            / path.relative_to(verification_run / "verification/overlays"),
        )
    for path in (verification_run / "experiments").glob("*/model_evidence.json"):
        copy(path, CHECKS / "verification" / f"{path.parent.name}_model_evidence.json")
    invoke(
        python,
        [
            "run_pipeline.py",
            "evaluate",
            "--annotations",
            "results/reproduction/kitti_ground_truth.json",
            "--predictions",
            "results/reproduction/yolov8n_predictions.json",
            "--categories",
            "person",
            "car",
            "truck",
            "--run-id",
            "submission_metric_reproduction",
        ],
        "metric_reproduction.log",
    )
    expected_metrics = read(ROOT / "results/reproduction/expected_shared_metrics.json")
    actual_metrics = read(
        ROOT / "outputs/submission_metric_reproduction/evaluation/metrics.json"
    )
    if actual_metrics != expected_metrics:
        raise AssertionError("Saved-prediction reproduction does not exactly match")
    copy(
        ROOT / "outputs/submission_metric_reproduction/evaluation/metrics.json",
        CHECKS / "reproduced_metrics.json",
    )
    evaluator_checks = read(
        ROOT / "outputs/submission_evaluator_check/tests/results.json"
    )
    write(CHECKS / "evaluator_results.json", evaluator_checks)
    pdfs = {
        path.name: len(PdfReader(path).pages)
        for path in (ROOT / "reports").glob("*.pdf")
    }
    if not 6 <= pdfs["main_experiment_report.pdf"] <= 10:
        raise AssertionError("Primary report is outside 6-10 pages")
    remote = configparser.ConfigParser()
    remote.read(ROOT / ".git/config", encoding="utf-8")
    url = remote.get('remote "origin"', "url", fallback="")
    parsed = urlsplit(url)
    sanitized_url = (
        urlunsplit((parsed.scheme, parsed.netloc.split("@")[-1], parsed.path, "", ""))
        if parsed.scheme
        else url
    )
    result = {
        "status": "PASS",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "repository_tests_passed": int(match.group(1)) if match else None,
        "evaluator_assertions_passed": len(evaluator_checks),
        "verification_images_matching_original": len(expected),
        "fresh_verification_gpu_models": 3,
        "fresh_verification_model_images": 57,
        "fresh_verification_scope": "unlabeled, conf=.25; not an accuracy result",
        "saved_metrics_reproduced_exactly": True,
        "pdf_page_counts": pdfs,
        "github_origin": sanitized_url or None,
        "full_coco_kitti_experiment_reexecuted": False,
        "primary_report": "reports/main_experiment_report.pdf",
    }
    write(CHECKS / "validation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def eligible(path):
    rel = path.relative_to(ROOT)
    if any(
        p
        in {".git", "__pycache__", ".venv", "dtect", "submission", "archive", "outputs"}
        for p in rel.parts
    ):
        return False
    if (
        path.suffix in {".pyc", ".zip", ".pt", ".onnx", ".engine"}
        or rel.as_posix() == "configs/local.json"
    ):
        return False
    if rel.parts[:2] == ("data", "datasets") and path.name != "README.md":
        return False
    return True


def package():
    checks = read(CHECKS / "validation.json")
    if checks["status"] != "PASS":
        raise ValueError("Submission validation has not passed")
    files = sorted(
        path for path in ROOT.rglob("*") if path.is_file() and eligible(path)
    )
    manifests = {}
    for name, compact in [
        ("cmpe249_track_a_submission.zip", False),
        ("report_ready_bundle.zip", True),
    ]:
        selected = files
        if compact:
            selected = [
                p
                for p in files
                if p.relative_to(ROOT).parts[0] != "data"
                and p.name != "benchmark_19_report.pdf"
                and "overlays" not in p.parts
                and not ("benchmark_19" in p.parts and "figures" in p.parts)
            ]
        target = ROOT / "submission" / name
        with zipfile.ZipFile(
            target, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in selected:
                archive.write(
                    path, "CMPE249_TrackA/" + path.relative_to(ROOT).as_posix()
                )
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            names = archive.namelist()
            assert "CMPE249_TrackA/reports/main_experiment_report.pdf" in names
            assert "CMPE249_TrackA/run_pipeline.py" in names
            assert not any(
                "/.git/" in n or n.endswith("configs/local.json") for n in names
            )
        manifests[name] = {
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "members": [
                {
                    "name": "CMPE249_TrackA/" + path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
                for path in selected
            ],
        }
    evidence = ROOT / "submission/prediction_evidence_bundle.zip"
    with zipfile.ZipFile(evidence) as archive:
        assert archive.testzip() is None
    manifests[evidence.name] = {
        "bytes": evidence.stat().st_size,
        "sha256": digest(evidence),
        "scope": "Original full predictions/GT; separate optional evidence",
    }
    write(
        ROOT / "submission/archive_verification.json",
        {"status": "PASS", "archives": manifests},
    )
    print("Submission archives verified.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    (ROOT / "submission").mkdir(exist_ok=True)
    if args.source_run:
        stage_original_evidence(args.source_run)
    if not args.package_only:
        if not args.python:
            parser.error("--python must point to the working experiment interpreter")
        audit(args.python)
    package()
