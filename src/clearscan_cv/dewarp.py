from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import ensure_bgr


@dataclass(frozen=True)
class DewarpResult:
    image: np.ndarray
    report: dict[str, object]


def _dark_text_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(35.0, min(185.0, min(float(otsu_threshold), float(np.percentile(blur, 35))) - 2.0))
    mask = (blur <= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _normalize_projection(projection: np.ndarray) -> np.ndarray:
    signal = projection.astype(np.float32)
    signal = cv2.GaussianBlur(signal.reshape(-1, 1), (1, 9), 0).reshape(-1)
    signal -= float(np.mean(signal))
    norm = float(np.linalg.norm(signal))
    return signal / norm if norm > 1e-6 else signal


def _best_vertical_shift(reference: np.ndarray, projection: np.ndarray, max_shift: int) -> tuple[int, float]:
    best_shift = 0
    best_score = -1.0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            ref_slice = reference[:shift]
            proj_slice = projection[-shift:]
        elif shift > 0:
            ref_slice = reference[shift:]
            proj_slice = projection[:-shift]
        else:
            ref_slice = reference
            proj_slice = projection
        if ref_slice.size < 24 or proj_slice.size < 24:
            continue
        score = float(np.dot(ref_slice, proj_slice))
        if score > best_score:
            best_score = score
            best_shift = shift
    return best_shift, best_score


def estimate_textline_column_offsets(image: np.ndarray, strip_count: int = 31, max_shift_ratio: float = 0.045) -> tuple[np.ndarray, np.ndarray, float]:
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, 900.0 / float(max(height, width)))
    small = cv2.resize(gray, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    mask = _dark_text_mask(small)
    if float(np.mean(mask)) < 0.004:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0

    small_height, small_width = mask.shape[:2]
    strip_count = max(9, min(strip_count, small_width // 16 if small_width >= 160 else 9))
    centers: list[float] = []
    projections: list[np.ndarray] = []
    for index in range(strip_count):
        x0 = int(round(index * small_width / strip_count))
        x1 = int(round((index + 1) * small_width / strip_count))
        if x1 <= x0 + 2:
            continue
        projection = mask[:, x0:x1].sum(axis=1)
        if float(np.max(projection)) < max(2.0, (x1 - x0) * 0.05):
            continue
        centers.append((x0 + x1) / 2.0 / scale)
        projections.append(_normalize_projection(projection))

    if len(projections) < 7:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0

    middle = len(projections) // 2
    reference = projections[middle]
    max_shift = max(2, int(round(small_height * max_shift_ratio)))
    shifts: list[float] = []
    scores: list[float] = []
    for projection in projections:
        shift, score = _best_vertical_shift(reference, projection, max_shift=max_shift)
        shifts.append(shift / scale)
        scores.append(score)

    centers_array = np.asarray(centers, dtype=np.float32)
    shifts_array = np.asarray(shifts, dtype=np.float32)
    scores_array = np.asarray(scores, dtype=np.float32)
    if shifts_array.size < 7:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0

    kernel = min(7, shifts_array.size if shifts_array.size % 2 == 1 else shifts_array.size - 1)
    if kernel >= 3:
        shifts_array = cv2.GaussianBlur(shifts_array.reshape(1, -1), (kernel, 1), 0).reshape(-1)
    confidence = float(np.clip(np.mean(np.maximum(scores_array, 0.0)), 0.0, 1.0))
    return centers_array, shifts_array, confidence


def dewarp_by_textline_columns(image: np.ndarray) -> DewarpResult:
    bgr = ensure_bgr(image)
    height, width = bgr.shape[:2]
    centers, shifts, confidence = estimate_textline_column_offsets(bgr)
    if centers.size < 7 or shifts.size < 7:
        return DewarpResult(bgr.copy(), {"applied": False, "method": "textline_column_offsets", "confidence": round(confidence, 3), "reason": "insufficient_textlines"})

    max_offset = float(np.max(np.abs(shifts)))
    if confidence < 0.45 or max_offset < max(6.0, height * 0.006):
        return DewarpResult(
            bgr.copy(),
            {
                "applied": False,
                "method": "textline_column_offsets",
                "confidence": round(confidence, 3),
                "max_offset": round(max_offset, 2),
                "reason": "flat_or_low_confidence",
            },
        )

    x_coords = np.arange(width, dtype=np.float32)
    x_points = np.concatenate(([0.0], centers, [float(width - 1)]))
    shift_points = np.concatenate(([float(shifts[0])], shifts, [float(shifts[-1])]))
    full_shift = np.interp(x_coords, x_points, shift_points).astype(np.float32)

    grid_x, grid_y = np.meshgrid(x_coords, np.arange(height, dtype=np.float32))
    map_x = grid_x.astype(np.float32)
    map_y = (grid_y - full_shift.reshape(1, -1)).astype(np.float32)
    border_value = (255, 255, 255)
    remapped = cv2.remap(bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return DewarpResult(
        remapped,
        {
            "applied": True,
            "method": "textline_column_offsets",
            "confidence": round(confidence, 3),
            "max_offset": round(max_offset, 2),
            "strip_count": int(centers.size),
        },
    )
