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

        top_width = np.linalg.norm(ordered[1] - ordered[0])
        bottom_width = np.linalg.norm(ordered[2] - ordered[3])
        left_height = np.linalg.norm(ordered[3] - ordered[0])
        right_height = np.linalg.norm(ordered[2] - ordered[1])
        min_side = min(top_width, bottom_width, left_height, right_height)
        if area_ratio < 0.12 or area_ratio > 0.96 or min_side < min(width, height) * 0.15:
            continue
        if looks_like_partial_bright_region(ordered, width, height, area_ratio):
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
        brightness_detection = detect_bright_document_region(bgr, max_dim=max_dim)
        if brightness_detection is not None:
            return brightness_detection
        return image_border_detection(width, height)

    points, contour_area, method = best_rect
    ordered = order_points(points)
    area_ratio = min(1.0, polygon_area(ordered) / image_area)
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
