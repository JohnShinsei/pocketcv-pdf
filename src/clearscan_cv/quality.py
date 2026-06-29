from __future__ import annotations

import cv2
import numpy as np

from .geometry import ensure_bgr


def _clip_score(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _target_score(value: float, target: float, tolerance: float) -> float:
    return float(np.clip(1.0 - abs(value - target) / tolerance, 0.0, 1.0))


def _odd_metric_kernel(size: int, minimum: int = 31, maximum: int = 201) -> int:
    size = max(minimum, min(maximum, size))
    return size + 1 if size % 2 == 0 else size


def _shadow_residual(gray: np.ndarray) -> float:
    height, width = gray.shape[:2]
    kernel_size = _odd_metric_kernel(int(min(height, width) / 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    low, high = np.percentile(background, [5, 95])
    return float(max(0.0, high - low))


def _boldness_risk(ink_density: float) -> float:
    return float(np.clip((ink_density - 0.085) / 0.095, 0.0, 1.0))


def assess_quality(image: np.ndarray) -> dict[str, float | int | str]:
    bgr = ensure_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, 1600.0 / float(max(height, width)))
    metric_gray = (
        cv2.resize(gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )

    sharpness = float(cv2.Laplacian(metric_gray, cv2.CV_64F).var())
    contrast = float(metric_gray.std())
    brightness = float(metric_gray.mean())
    edges = cv2.Canny(metric_gray, 80, 160)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    underexposed = float(np.mean(metric_gray < 20))
    overexposed = float(np.mean(metric_gray > 252))
    exposure_balance = max(0.0, 1.0 - underexposed * 2.0 - max(0.0, overexposed - 0.55) * 1.25)
    paper_tone = _target_score(brightness, target=222.0, tolerance=95.0)
    shadow_residual = _shadow_residual(metric_gray)
    shadow_score = float(np.clip(1.0 - (shadow_residual - 12.0) / 58.0, 0.0, 1.0))
    ink_density = float(np.mean(metric_gray < 128))
    boldness_risk = _boldness_risk(ink_density)

    score = (
        _clip_score(sharpness, 40.0, 450.0) * 24.0
        + _clip_score(contrast, 18.0, 75.0) * 21.0
        + _clip_score(edge_density, 0.015, 0.12) * 18.0
        + paper_tone * 13.0
        + exposure_balance * 8.0
        + shadow_score * 11.0
        + (1.0 - boldness_risk) * 5.0
    )
    grade = "excellent" if score >= 82 else "good" if score >= 65 else "review"

    return {
        "width": int(width),
        "height": int(height),
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "brightness": round(brightness, 2),
        "edge_density": round(edge_density, 4),
        "exposure_balance": round(exposure_balance, 4),
        "paper_tone": round(paper_tone, 4),
        "shadow_residual": round(shadow_residual, 2),
        "shadow_score": round(shadow_score, 4),
        "ink_density": round(ink_density, 4),
        "boldness_risk": round(boldness_risk, 4),
        "score": round(float(score), 2),
        "grade": grade,
    }


def compare_quality(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    before_metrics = assess_quality(before)
    after_metrics = assess_quality(after)
    return {
        "before": before_metrics,
        "after": after_metrics,
        "score_delta": round(float(after_metrics["score"]) - float(before_metrics["score"]), 2),
    }
