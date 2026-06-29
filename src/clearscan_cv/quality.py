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


def _metric_float(metrics: dict[str, object], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nested_metric_float(metrics: dict[str, object], keys: tuple[str, ...], default: float = 0.0) -> float:
    current: object = metrics
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _nested_metric_exists(metrics: dict[str, object], keys: tuple[str, ...]) -> bool:
    current: object = metrics
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is not None


def diagnose_scan_quality(
    metrics: dict[str, object],
    *,
    perspective_confidence: float | None = None,
    readability: dict[str, object] | None = None,
) -> dict[str, object]:
    score = _metric_float(metrics, "score")
    shadow_residual = _metric_float(metrics, "shadow_residual")
    shadow_score = _metric_float(metrics, "shadow_score", 1.0)
    ink_density = _metric_float(metrics, "ink_density")
    boldness_risk = _metric_float(metrics, "boldness_risk")
    issues: list[dict[str, object]] = []

    def add_issue(code: str, severity: str, message: str, action: str) -> None:
        issues.append({"code": code, "severity": severity, "message": message, "action": action})

    if perspective_confidence is not None:
        if perspective_confidence < 0.28:
            add_issue(
                "perspective_low",
                "high",
                "Document corner confidence is low.",
                "Adjust the four corners manually or retake with the whole page visible.",
            )
        elif perspective_confidence < 0.48:
            add_issue(
                "perspective_review",
                "medium",
                "Document corner confidence should be reviewed.",
                "Check the corner overlay before exporting the final scan.",
            )

    if shadow_residual > 42.0 or shadow_score < 0.48:
        add_issue(
            "shadow_residual",
            "high",
            "Uneven paper illumination remains after enhancement.",
            "Retake under flatter light or export grayscale for OCR-sensitive text.",
        )
    elif shadow_residual > 28.0:
        add_issue(
            "shadow_review",
            "medium",
            "Some uneven paper illumination remains.",
            "Review the scan report before using binary output.",
        )

    if boldness_risk > 0.55 or ink_density > 0.16:
        add_issue(
            "bold_text",
            "high",
            "Binary output may be too bold for small text.",
            "Use grayscale output or retake closer with better focus.",
        )
    elif boldness_risk > 0.28:
        add_issue(
            "bold_text_review",
            "medium",
            "Binary output may slightly thicken text strokes.",
            "Check small characters before sharing or running OCR.",
        )

    if ink_density < 0.006 and _metric_float(metrics, "edge_density") < 0.018:
        add_issue(
            "weak_text",
            "medium",
            "Very little text ink was detected.",
            "Use grayscale mode or retake with sharper focus.",
        )

    if readability is not None and _metric_float(readability, "textline_horizontal_score", 1.0) < 0.68:
        add_issue(
            "textline_tilt",
            "medium",
            "Text lines still look tilted or curved.",
            "Adjust the four corners or retake from directly above.",
        )
    if readability is not None and isinstance(readability.get("ocr_quality"), dict):
        ocr_line_count = _nested_metric_float(readability, ("ocr_quality", "line_count"))
        ocr_character_count = _nested_metric_float(readability, ("ocr_quality", "character_count"))
        mean_confidence = _nested_metric_float(readability, ("ocr_quality", "mean_confidence"), 100.0)
        low_confidence_ratio = _nested_metric_float(readability, ("ocr_quality", "low_confidence_ratio"))
        if ocr_line_count <= 0 or ocr_character_count <= 0:
            add_issue(
                "ocr_empty",
                "high",
                "OCR detected little or no text.",
                "Check the selected language, use grayscale output, or retake with sharper focus.",
            )
        elif mean_confidence < 50.0:
            add_issue(
                "ocr_low_confidence",
                "high",
                "OCR confidence is low.",
                "Use grayscale output, check language data, or retake with better focus.",
            )
        elif mean_confidence < 68.0 or low_confidence_ratio > 0.35:
            add_issue(
                "ocr_review",
                "medium",
                "OCR confidence should be reviewed.",
                "Review low-confidence lines before using the recovered document.",
            )
        if _nested_metric_exists(readability, ("ocr_quality", "character_error_rate")):
            character_error_rate = _nested_metric_float(readability, ("ocr_quality", "character_error_rate"))
            if character_error_rate > 0.25:
                add_issue(
                    "ocr_high_cer",
                    "high",
                    "OCR character error rate is high against the reference text.",
                    "Retake or tune the processing mode before trusting OCR output.",
                )
            elif character_error_rate > 0.12:
                add_issue(
                    "ocr_cer_review",
                    "medium",
                    "OCR character error rate should be reviewed.",
                    "Compare the searchable text with the original page before export.",
                )

    if score < 45.0:
        add_issue(
            "low_quality",
            "high",
            "Overall scan quality is low.",
            "Retake from directly above with flatter light.",
        )
        status = "retake"
    elif score < 65.0:
        add_issue(
            "quality_review",
            "medium",
            "Overall scan quality should be reviewed.",
            "Check the scan report before exporting.",
        )
        status = "review"
    elif issues:
        status = "review"
    else:
        status = "ready"

    return {
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
    }


def compare_quality(before: np.ndarray, after: np.ndarray) -> dict[str, object]:
    before_metrics = assess_quality(before)
    after_metrics = assess_quality(after)
    return {
        "before": before_metrics,
        "after": after_metrics,
        "score_delta": round(float(after_metrics["score"]) - float(before_metrics["score"]), 2),
    }
