"""Regression checks for relocation, geometry, manifests and official metrics."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from cmpe249_eval.config import ROOT, load_config, merge
from cmpe249_eval.io import read
from cmpe249_eval.validation import smoke
from cmpe249_eval.verification import inventory_images
from cmpe249_eval.pipeline import plan
from cmpe249_eval.runtime import run_worker


class RepositoryTests(unittest.TestCase):
    def test_cli_evaluate_prints_json_safe_metric_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            gt = {
                "info": {},
                "images": [
                    {"id": 91, "file_name": "scene.jpg", "width": 100, "height": 80}
                ],
                "categories": [{"id": 5, "name": "car"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 91,
                        "category_id": 5,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
            }
            pred = [
                {
                    "image_id": 91,
                    "category_id": 5,
                    "bbox": [10, 10, 20, 20],
                    "score": 0.99,
                }
            ]
            (directory / "gt.json").write_text(json.dumps(gt))
            (directory / "pred.json").write_text(json.dumps(pred))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_pipeline.py"),
                    "evaluate",
                    "--annotations",
                    str(directory / "gt.json"),
                    "--predictions",
                    str(directory / "pred.json"),
                    "--run",
                    str(directory / "output"),
                    "--no-local",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = json.loads(result.stdout)
            self.assertGreater(metrics["AP"], 0.999)
            self.assertEqual(metrics["maxDets"], [1, 10, 100])

    def test_subprocess_contract_uses_relocated_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            weights = run / "weights"
            weights.mkdir()
            (weights / "test.pt").write_bytes(
                b"placeholder only for subprocess-contract test"
            )
            cfg = load_config(use_local=False)
            cfg["weights_dir"] = str(weights)
            case = {"id": "contract", "model": "test.pt", "settings": cfg["primary"]}
            with patch("cmpe249_eval.runtime.subprocess.run") as launch:
                launch.return_value.returncode = 0
                destination = run_worker(run, case, [], [], cfg=cfg)
                command = launch.call_args.args[0]
                self.assertEqual(command[1:3], ["-m", "cmpe249_eval.worker"])
                self.assertIn(
                    str(ROOT / "src"), launch.call_args.kwargs["env"]["PYTHONPATH"]
                )
                job = read(destination / "job.json")
                self.assertEqual(Path(job["model_path"]), weights / "test.pt")

    def test_portable_default_paths(self):
        cfg = load_config(use_local=False)
        self.assertEqual(Path(cfg["benchmark_root"]), ROOT / "data/verification/images")
        self.assertEqual(Path(cfg["output_dir"]), ROOT / "outputs")
        self.assertEqual(Path(cfg["weights_dir"]), ROOT / "weights")

    def test_paths_ignore_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                cfg = load_config(use_local=False)
                self.assertEqual(
                    Path(cfg["benchmark_root"]), ROOT / "data/verification/images"
                )
            finally:
                os.chdir(original)

    def test_explicit_local_merge_preserves_baseline(self):
        base = {"primary": {"imgsz": 512, "batch": 1}}
        override = {"primary": {"conf": 0.001}}
        merged = merge(base, override)
        self.assertEqual(merged["primary"], {"imgsz": 512, "batch": 1, "conf": 0.001})
        self.assertNotIn("conf", base["primary"])

    def test_module_help_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_pipeline.py"), "--help"],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("verify-images", result.stdout)

    def test_evaluator_regression_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = smoke(Path(tmp))
            self.assertEqual(len(checks), 15)
            self.assertTrue(all(c["status"] == "PASS" for c in checks))

    def test_empty_image_folder_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Add JPG/PNG images"):
                inventory_images(tmp)

    def test_unlabeled_images_do_not_create_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (100, 80), "white").save(Path(tmp) / "scene.jpg")
            images, gt = inventory_images(tmp)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0]["width"], 100)
            self.assertIsNone(gt)

    def test_missing_annotation_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(image_dir / "scene.jpg")
            annotation = Path(tmp) / "annotations.json"
            annotation.write_text(
                json.dumps({"images": [], "categories": [], "annotations": []})
            )
            with self.assertRaisesRegex(ValueError, "cover every verification image"):
                inventory_images(image_dir, annotation)

    def test_real_annotation_ids_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp) / "images"
            image_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(image_dir / "scene.jpg")
            annotation = Path(tmp) / "annotations.json"
            annotation.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 91,
                                "file_name": "scene.jpg",
                                "width": 100,
                                "height": 80,
                            }
                        ],
                        "categories": [{"id": 5, "name": "car"}],
                        "annotations": [],
                    }
                )
            )
            images, gt = inventory_images(image_dir, annotation)
            self.assertEqual(images[0]["id"], 91)
            self.assertEqual(gt["categories"][0]["id"], 5)

    def test_matrix_is_38_unique_configurations(self):
        with tempfile.TemporaryDirectory() as tmp:
            matrix = plan(Path(tmp), load_config(use_local=False))
            self.assertEqual(len(matrix), 38)
            self.assertEqual(len({x["id"] for x in matrix}), 38)
            nms = [x for x in matrix if "iou_0.5" in x["id"]]
            self.assertEqual(len(nms), 2)
            self.assertTrue(all("rtdetr" not in x["model"] for x in nms))


if __name__ == "__main__":
    unittest.main()
