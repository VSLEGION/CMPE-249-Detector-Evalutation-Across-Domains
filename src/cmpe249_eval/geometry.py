"""Finite, positive original-pixel box conversion and boundary clipping."""

import math
import numpy as np


def box_xywh(xyxy, width, height):
    if len(xyxy) != 4 or not all(math.isfinite(v) for v in xyxy):
        raise ValueError("nonfinite or malformed box")
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        raise ValueError("reversed/zero-area box")
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box entirely outside image")
    return list(map(float, [x1, y1, x2 - x1, y2 - y1]))


def normalized_box(vals, width, height):
    if len(vals) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in vals):
        raise ValueError("invalid normalized box")
    x, y, w, h = vals
    return box_xywh(
        [
            (x - w / 2) * width,
            (y - h / 2) * height,
            (x + w / 2) * width,
            (y + h / 2) * height,
        ],
        width,
        height,
    )
