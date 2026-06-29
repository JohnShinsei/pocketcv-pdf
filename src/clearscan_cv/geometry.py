from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DocumentDetection:
    corners: list[list[float]]
    confidence: float
    area_ratio: float
    method: str
    found: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("image is empty")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def image_border_detection(width: int, height: int) -> DocumentDetection:
    corners = [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]]
    return DocumentDetection(corners=corners, confidence=0.1, area_ratio=1.0, method="image_border", found=False)


def looks_like_partial_bright_region(points: np.ndarray, width: int, height: int, area_ratio: float) -> bool:
    ordered = order_points(points)
    min_x = float(np.min(ordered[:, 0]))
    max_x = float(np.max(ordered[:, 0]))
    min_y = float(np.min(ordered[:, 1]))
    max_y = float(np.max(ordered[:, 1]))
    width_coverage = (max_x - min_x) / max(1.0, float(width))
    height_coverage = (max_y - min_y) / max(1.0, float(height))
    misses_top = min_y > height * 0.12
    misses_bottom = max_y < height * 0.86
    spans_width = width_coverage > 0.78
    return spans_width and height_coverage < 0.9 and area_ratio < 0.9 and (misses_top or misses_bottom)


def looks_like_unsafe_full_frame_crop(points: np.ndarray, width: int, height: int, area_ratio: float) -> bool:
    ordered = order_points(points)
    min_x = float(np.min(ordered[:, 0]))
    max_x = float(np.max(ordered[:, 0]))
    max_y = float(np.max(ordered[:, 1]))
    top_average = float(np.mean(ordered[:2, 1]))
    top_skew = float(abs(ordered[0, 1] - ordered[1, 1]))

    touches_left = min_x <= width * 0.025
    touches_right = max_x >= width * 0.975
    touches_bottom = max_y >= height * 0.975
    would_crop_top_content = top_average > height * 0.16 or top_skew > height * 0.09
    return area_ratio > 0.62 and touches_left and touches_right and touches_bottom and would_crop_top_content


def quad_is_plausible(points: np.ndarray, width: int, height: int, area_ratio: float) -> bool:
    ordered = order_points(points)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    min_side = min(top_width, bottom_width, left_height, right_height)
    return 0.12 <= area_ratio <= 0.97 and min_side >= min(width, height) * 0.15


def approximate_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in (0.015, 0.02, 0.03, 0.045, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _line_from_segment(segment: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = [float(value) for value in segment]
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    angle = (float(np.degrees(np.arctan2(dy, dx))) + 180.0) % 180.0
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = float(np.hypot(a, b))
    if norm <= 0:
        norm = 1.0
    return {
        "a": a / norm,
        "b": b / norm,
        "c": c / norm,
        "length": length,
        "mid_x": (x1 + x2) / 2.0,
        "mid_y": (y1 + y2) / 2.0,
        "angle": angle,
    }


def _line_intersection(line_a: dict[str, float], line_b: dict[str, float]) -> tuple[float, float] | None:
    a1, b1, c1 = line_a["a"], line_a["b"], line_a["c"]
    a2, b2, c2 = line_b["a"], line_b["b"], line_b["c"]
    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant
    return float(x), float(y)


def _hough_side_candidates(lines: list[dict[str, float]], side: str, width: int, height: int) -> list[dict[str, float]]:
    max_dim = float(max(width, height))
    candidates: list[tuple[float, dict[str, float]]] = []
    for line in lines:
        length_score = min(1.0, line["length"] / max_dim)
        if side == "top":
            if line["mid_y"] > height * 0.62:
                continue
            score = (1.0 - line["mid_y"] / max(1.0, height)) * 0.7 + length_score * 0.3
        elif side == "bottom":
            if line["mid_y"] < height * 0.38:
                continue
            score = (line["mid_y"] / max(1.0, height)) * 0.7 + length_score * 0.3
        elif side == "left":
            if line["mid_x"] > width * 0.62:
                continue
            score = (1.0 - line["mid_x"] / max(1.0, width)) * 0.7 + length_score * 0.3
        else:
            if line["mid_x"] < width * 0.38:
                continue
            score = (line["mid_x"] / max(1.0, width)) * 0.7 + length_score * 0.3
        candidates.append((score, line))

    return [line for _, line in sorted(candidates, key=lambda item: item[0], reverse=True)[:8]]


def detect_hough_document_region(image: np.ndarray, max_dim: int = 900) -> DocumentDetection | None:
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(width, height)))
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(0, 0.55 * median))
    upper = int(min(255, max(70, 1.35 * median)))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)

    min_line_length = max(40, int(round(min(small.shape[:2]) * 0.24)))
    max_line_gap = max(8, int(round(max(small.shape[:2]) * 0.025)))
    threshold = max(36, int(round(max(small.shape[:2]) * 0.06)))
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if raw_lines is None:
        return None

    horizontal: list[dict[str, float]] = []
    vertical: list[dict[str, float]] = []
    for raw_line in raw_lines[:, 0, :]:
        line = _line_from_segment(raw_line.astype(np.float32) / scale)
        if line["length"] < min(width, height) * 0.18:
            continue
        horizontal_angle = min(line["angle"], abs(180.0 - line["angle"]))
        vertical_angle = abs(line["angle"] - 90.0)
        if horizontal_angle <= 28.0:
            horizontal.append(line)
        elif vertical_angle <= 38.0:
            vertical.append(line)

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    top_candidates = _hough_side_candidates(horizontal, "top", width, height)
    bottom_candidates = _hough_side_candidates(horizontal, "bottom", width, height)
    left_candidates = _hough_side_candidates(vertical, "left", width, height)
    right_candidates = _hough_side_candidates(vertical, "right", width, height)
    if not top_candidates or not bottom_candidates or not left_candidates or not right_candidates:
        return None

    image_area = float(width * height)
    best: tuple[np.ndarray, float] | None = None
    margin_x = width * 0.08
    margin_y = height * 0.08
    for top in top_candidates:
        for bottom in bottom_candidates:
            if bottom["mid_y"] - top["mid_y"] < height * 0.28:
                continue
            for left in left_candidates:
                for right in right_candidates:
                    if right["mid_x"] - left["mid_x"] < width * 0.28:
                        continue
                    intersections = [
                        _line_intersection(top, left),
                        _line_intersection(top, right),
                        _line_intersection(bottom, right),
                        _line_intersection(bottom, left),
                    ]
                    if any(point is None for point in intersections):
                        continue
                    points = np.array(intersections, dtype=np.float32)
                    if (
                        np.any(points[:, 0] < -margin_x)
                        or np.any(points[:, 0] > width - 1 + margin_x)
                        or np.any(points[:, 1] < -margin_y)
                        or np.any(points[:, 1] > height - 1 + margin_y)
                    ):
                        continue
                    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                    ordered = order_points(points)
                    area_ratio = min(1.0, polygon_area(ordered) / image_area)
                    if not quad_is_plausible(ordered, width, height, area_ratio):
                        continue
                    if looks_like_partial_bright_region(ordered, width, height, area_ratio):
                        continue
                    if looks_like_unsafe_full_frame_crop(ordered, width, height, area_ratio):
                        continue

                    line_score = (top["length"] + bottom["length"] + left["length"] + right["length"]) / max(1.0, 2.0 * (width + height))
                    score = area_ratio * 0.78 + min(1.0, line_score) * 0.22
                    if best is None or score > best[1]:
                        best = (ordered, score)

    if best is None:
        return None

    ordered, score = best
    area_ratio = min(1.0, polygon_area(ordered) / image_area)
    confidence = min(0.9, 0.26 + area_ratio * 0.52 + score * 0.18)
    return DocumentDetection(
        corners=np.round(ordered, 2).tolist(),
        confidence=round(float(confidence), 3),
        area_ratio=round(float(area_ratio), 3),
        method="hough_lines",
        found=True,
    )


def detect_fallback_document_region(image: np.ndarray, max_dim: int = 900) -> DocumentDetection | None:
    hough_detection = detect_hough_document_region(image, max_dim=max_dim)
    if hough_detection is not None:
        return hough_detection
    connected_detection = detect_connected_document_region(image, max_dim=max_dim)
    if connected_detection is not None:
        return connected_detection
    return detect_bright_document_region(image, max_dim=max_dim)


def detect_connected_document_region(image: np.ndarray, max_dim: int = 900) -> DocumentDetection | None:
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(width, height)))
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(70.0, min(150.0, min(float(otsu_threshold) - 10.0, float(np.percentile(gray, 20)) + 8.0)))
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)

    kernel_size = max(9, int(round(max(gray.shape[:2]) / 55)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)
    small_area = float(small.shape[0] * small.shape[1])
    best: tuple[np.ndarray, float] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < small_area * 0.08:
            continue

        quad = approximate_quad(contour)
        if quad is None:
            continue

        quad = quad / scale
        quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
        ordered = order_points(quad)
        area_ratio = min(1.0, polygon_area(ordered) / image_area)
        if not quad_is_plausible(ordered, width, height, area_ratio):
            continue
        if looks_like_partial_bright_region(ordered, width, height, area_ratio):
            continue
        if looks_like_unsafe_full_frame_crop(ordered, width, height, area_ratio):
            continue

        if best is None or contour_area > best[1]:
            best = (ordered, contour_area)

    if best is None:
        return None

    ordered, contour_area = best
    area_ratio = min(1.0, polygon_area(ordered) / image_area)
    confidence = min(0.93, 0.34 + area_ratio * 0.5 + min(0.18, contour_area / small_area))
    return DocumentDetection(
        corners=np.round(ordered, 2).tolist(),
        confidence=round(float(confidence), 3),
        area_ratio=round(float(area_ratio), 3),
        method="connected_paper",
        found=True,
    )


def detect_bright_document_region(image: np.ndarray, max_dim: int = 900) -> DocumentDetection | None:
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(width, height)))
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    threshold = max(float(np.percentile(gray, 70)), float(gray.mean() + gray.std() * 0.1), 95.0)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)

    kernel_size = max(5, int(round(max(gray.shape[:2]) / 100)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = float(width * height)
    small_area = float(small.shape[0] * small.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < small_area * 0.08:
            continue

        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32) / scale
        box[:, 0] = np.clip(box[:, 0], 0, width - 1)
        box[:, 1] = np.clip(box[:, 1], 0, height - 1)
        ordered = order_points(box)
        area_ratio = min(1.0, polygon_area(ordered) / image_area)

        if not quad_is_plausible(ordered, width, height, area_ratio):
            continue
        if looks_like_partial_bright_region(ordered, width, height, area_ratio):
            continue
        if looks_like_unsafe_full_frame_crop(ordered, width, height, area_ratio):
            continue

        confidence = min(0.88, 0.28 + area_ratio * 0.55 + min(0.2, contour_area / small_area))
        return DocumentDetection(
            corners=np.round(ordered, 2).tolist(),
            confidence=round(float(confidence), 3),
            area_ratio=round(float(area_ratio), 3),
            method="brightness_rect",
            found=True,
        )

    return None


def detect_document_corners(image: np.ndarray, max_dim: int = 900) -> DocumentDetection:
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(width, height)))
    small = cv2.resize(bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, max(80, 1.33 * median)))

    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(width * height)
    best_rect: tuple[np.ndarray, float, str] | None = None

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        contour_area = cv2.contourArea(contour) / (scale * scale)
        if contour_area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            points = approx.reshape(4, 2).astype(np.float32) / scale
            best_rect = (points, contour_area, "contour_quad")
            break

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32) / scale
        if best_rect is None:
            best_rect = (box, contour_area, "min_area_rect")

    if best_rect is None:
        fallback_detection = detect_fallback_document_region(bgr, max_dim=max_dim)
        if fallback_detection is not None:
            return fallback_detection
        return image_border_detection(width, height)

    points, contour_area, method = best_rect
    ordered = order_points(points)
    area_ratio = min(1.0, polygon_area(ordered) / image_area)
    if not quad_is_plausible(ordered, width, height, area_ratio) or looks_like_partial_bright_region(
        ordered, width, height, area_ratio
    ) or looks_like_unsafe_full_frame_crop(ordered, width, height, area_ratio):
        fallback_detection = detect_fallback_document_region(bgr, max_dim=max_dim)
        if fallback_detection is not None:
            return fallback_detection
        return image_border_detection(width, height)

    confidence = min(0.99, 0.25 + area_ratio * 0.9 + (0.2 if method == "contour_quad" else 0.0))

    return DocumentDetection(
        corners=np.round(ordered, 2).tolist(),
        confidence=round(float(confidence), 3),
        area_ratio=round(float(area_ratio), 3),
        method=method,
        found=True,
    )


def four_point_transform(image: np.ndarray, corners: list[list[float]] | np.ndarray, padding_ratio: float = 0.015) -> np.ndarray:
    rect = order_points(np.asarray(corners, dtype=np.float32))
    if padding_ratio > 0:
        height, width = image.shape[:2]
        center = rect.mean(axis=0)
        rect = center + (rect - center) * (1.0 + padding_ratio)
        rect[:, 0] = np.clip(rect[:, 0], 0, width - 1)
        rect[:, 1] = np.clip(rect[:, 1], 0, height - 1)

    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_width = max(1, int(round(max(width_a, width_b))))
    max_height = max(1, int(round(max(height_a, height_b))))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))
