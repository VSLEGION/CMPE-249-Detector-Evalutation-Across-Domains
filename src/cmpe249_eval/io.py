"""Atomic JSON, CSV, hashing and deterministic image discovery."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .constants import EXT


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            allow_nan=False,
            default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(),
        ),
        encoding="utf-8",
    )
    tmp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fingerprint(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def rows(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not data:
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, sorted(set().union(*(d.keys() for d in data))))
        w.writeheader()
        w.writerows(data)


def image_files(p):
    return sorted(x for x in Path(p).rglob("*") if x.suffix.lower() in EXT)
