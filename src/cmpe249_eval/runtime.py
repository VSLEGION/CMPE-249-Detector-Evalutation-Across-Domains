"""One CUDA model per process, compatible caching and explicit failure recovery."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from .config import PACKAGE_DIR
from .io import read, write, digest, fingerprint


def now():
    return datetime.now(timezone.utc).isoformat()


def record(run, obj):
    with (Path(run) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **obj}) + "\n"
        )


def run_worker(run, case, images, categories=None, kind="accuracy", cfg=None):
    model_path = Path(cfg["weights_dir"]) / case["model"]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint (no automatic download): {model_path}"
        )
    dest = run / "experiments" / case["id"]
    dest.mkdir(parents=True, exist_ok=True)
    code = {p.name: digest(p) for p in PACKAGE_DIR.glob("*.py")}
    input_hashes = {im["path"]: digest(im["path"]) for im in images}
    signature = fingerprint(
        {
            "case": case,
            "images": images,
            "input_hashes": input_hashes,
            "categories": categories,
            "code": code,
            "model": digest(Path(cfg["weights_dir"]) / case["model"]),
        }
    )
    done = dest / "success.json"
    if done.exists() and read(done)["signature"] == signature:
        record(run, {"id": case["id"], "status": "reused_compatible"})
        return dest
    if done.exists():
        old = read(done)
        done.rename(dest / ("success_previous_" + str(time.time_ns()) + ".json"))
        record(
            run,
            {
                "id": case["id"],
                "status": "invalidated_incompatible_cache",
                "old_signature": old["signature"],
                "new_signature": signature,
            },
        )
    job = {
        "model_path": str(Path(cfg["weights_dir"]) / case["model"]),
        **case,
        "kind": kind,
        "signature": signature,
        "input_hashes": input_hashes,
        "images": images,
        "categories": categories,
        "destination": str(dest),
        "warmup": cfg["timing"]["warmup"],
        "repeats": cfg["timing"]["repeats"],
    }
    jobfile = dest / "job.json"
    write(jobfile, job)
    command = [sys.executable, "-m", "cmpe249_eval.worker", str(jobfile)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    for attempt in [1, 2]:
        record(
            run,
            {
                "id": case["id"],
                "status": "running",
                "attempt": attempt,
                "command": subprocess.list2cmdline(command),
            },
        )
        with (run / "commands.log").open("a") as f:
            f.write(f"{now()} {subprocess.list2cmdline(command)}\n")
        print(f'RUN {case["id"]} attempt {attempt}', flush=True)
        with (dest / f"console_attempt_{attempt}.log").open("a", encoding="utf-8") as f:
            result = subprocess.run(
                command, stdout=f, stderr=subprocess.STDOUT, env=env
            )
        if result.returncode == 0:
            record(run, {"id": case["id"], "status": "success", "attempt": attempt})
            return dest
        failure = (
            read(dest / "failure.json")
            if (dest / "failure.json").exists()
            else {"error": f"exit {result.returncode}", "OOM": False}
        )
        record(
            run, {"id": case["id"], "status": "failed", "attempt": attempt, **failure}
        )
        if not failure.get("OOM") or attempt == 2:
            raise RuntimeError(f'Case {case["id"]} failed: {failure["error"]}')
