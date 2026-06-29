from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

import numpy as np

from .geometry import DocumentDetection, order_points, polygon_area

CornerPoints = list[list[float]]


def _point_from_value(value: object) -> list[float]:
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValueError("corner point dictionaries must contain x and y")
        point = [value["x"], value["y"]]
    else:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("each corner must be a point pair")
        if len(value) < 2:
            raise ValueError("each corner must contain x and y")
        point = [value[0], value[1]]

    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("corner coordinates must be numeric") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("corner coordinates must be finite")
    return [x, y]


def _corners_from_sequence(value: object) -> CornerPoints:
    if isinstance(value, dict):
        value = value.get("corners")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("corners must be a list of four points")
    corners = [_point_from_value(point) for point in value]
    if len(corners) != 4:
        raise ValueError("exactly four corner points are required")
    return corners


def parse_corner_points(value: str | Sequence[object] | dict[str, object]) -> CornerPoints:
    """Parse four document corners from JSON, x/y dicts, or an x,y text list."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("corners cannot be empty")
        if text[0] in "[{":
            try:
                return _corners_from_sequence(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError("corners JSON is invalid") from exc

        numbers = [float(match) for match in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)]
        if len(numbers) != 8:
            raise ValueError('corners must contain four x,y pairs, for example "10,20 300,18 310,420 8,430"')
        return [[numbers[index], numbers[index + 1]] for index in range(0, 8, 2)]

    return _corners_from_sequence(value)


def scale_corner_points(corners: Sequence[Sequence[float]], source_size: tuple[int, int], target_size: tuple[int, int]) -> CornerPoints:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("image sizes must be positive")
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    parsed = _corners_from_sequence(corners)
    return [[point[0] * scale_x, point[1] * scale_y] for point in parsed]


def manual_detection_from_corners(corners: Sequence[Sequence[float]], width: int, height: int) -> DocumentDetection:
    parsed = _corners_from_sequence(corners)
    ordered = order_points(np.asarray(parsed, dtype=np.float32))
    ordered[:, 0] = np.clip(ordered[:, 0], 0, max(0, width - 1))
    ordered[:, 1] = np.clip(ordered[:, 1], 0, max(0, height - 1))

    area_ratio = polygon_area(ordered) / max(1.0, float(width * height))
    side_lengths = [
        float(np.linalg.norm(ordered[1] - ordered[0])),
        float(np.linalg.norm(ordered[2] - ordered[1])),
        float(np.linalg.norm(ordered[2] - ordered[3])),
        float(np.linalg.norm(ordered[3] - ordered[0])),
    ]
    if area_ratio < 0.01 or min(side_lengths) < 8.0:
        raise ValueError("manual corners describe an area that is too small")

    return DocumentDetection(
        corners=np.round(ordered, 2).tolist(),
        confidence=1.0,
        area_ratio=round(float(min(1.0, area_ratio)), 3),
        method="manual_corners",
        found=True,
    )
