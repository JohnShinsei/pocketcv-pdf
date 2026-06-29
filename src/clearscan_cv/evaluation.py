from __future__ import annotations

import re
import unicodedata

import numpy as np

from .ocr import OcrResult
from .pipeline import estimate_textline_skew


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", "", normalized)


def edit_distance(reference: str, hypothesis: str) -> int:
    source = normalize_ocr_text(reference)
    target = normalize_ocr_text(hypothesis)
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for row_index, source_char in enumerate(source, start=1):
        current = [row_index]
        for column_index, target_char in enumerate(target, start=1):
            substitution_cost = 0 if source_char == target_char else 1
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    normalized_reference = normalize_ocr_text(reference)
    normalized_hypothesis = normalize_ocr_text(hypothesis)
    if not normalized_reference:
        return 0.0 if not normalized_hypothesis else 1.0
    return round(edit_distance(normalized_reference, normalized_hypothesis) / len(normalized_reference), 4)


def _confidence_values(result: OcrResult) -> list[float]:
    values: list[float] = []
    for line in result.lines:
        if line.confidence is not None and line.confidence >= 0:
            values.append(float(line.confidence))
        for word in line.words:
            if word.confidence is not None and word.confidence >= 0:
                values.append(float(word.confidence))
    if not values and result.confidence is not None and result.confidence >= 0:
        values.append(float(result.confidence))
    return values


def _ocr_word_count(result: OcrResult) -> int:
    word_boxes = sum(len(line.words) for line in result.lines)
    if word_boxes:
        return word_boxes
    return sum(len(re.findall(r"\S+", line.text)) for line in result.lines)


def _ocr_coverage_ratio(result: OcrResult) -> float:
    page_area = max(1, result.width * result.height)
    line_area = 0
    for line in result.lines:
        _x, _y, width, height = line.bbox
        line_area += max(0, width) * max(0, height)
    return round(min(1.0, line_area / page_area), 4)


def evaluate_ocr_result(result: OcrResult, expected_text: str | None = None) -> dict[str, object]:
    text = result.text or "\n".join(line.text for line in result.lines)
    confidences = _confidence_values(result)
    low_confidence_ratio = float(np.mean([value < 60.0 for value in confidences])) if confidences else None
    metrics: dict[str, object] = {
        "engine": result.engine,
        "language": result.language,
        "line_count": len(result.lines),
        "word_count": _ocr_word_count(result),
        "character_count": len(normalize_ocr_text(text)),
        "mean_confidence": round(float(np.mean(confidences)), 2) if confidences else None,
        "min_confidence": round(float(np.min(confidences)), 2) if confidences else None,
        "low_confidence_ratio": round(low_confidence_ratio, 4) if low_confidence_ratio is not None else None,
        "ocr_coverage_ratio": _ocr_coverage_ratio(result),
    }
    if expected_text is not None:
        metrics["edit_distance"] = edit_distance(expected_text, text)
        metrics["character_error_rate"] = character_error_rate(expected_text, text)
    return metrics


def evaluate_readability(image: np.ndarray, ocr_result: OcrResult | None = None, expected_text: str | None = None) -> dict[str, object]:
    angle, confidence = estimate_textline_skew(image)
    horizontal_score = max(0.0, 1.0 - min(abs(angle), 6.0) / 6.0)
    metrics: dict[str, object] = {
        "textline_angle": round(float(angle), 3),
        "textline_confidence": round(float(confidence), 3),
        "textline_horizontal_score": round(horizontal_score, 4),
    }
    if ocr_result is not None:
        ocr_metrics = evaluate_ocr_result(ocr_result, expected_text=expected_text)
        confidence_score = (float(ocr_metrics["mean_confidence"]) / 100.0) if ocr_metrics["mean_confidence"] is not None else 0.0
        line_score = min(1.0, float(ocr_metrics["line_count"]) / 8.0)
        metrics["ocr_quality"] = ocr_metrics
        metrics["readability_score"] = round(horizontal_score * 35.0 + confidence_score * 45.0 + line_score * 20.0, 2)
    else:
        metrics["readability_score"] = round(horizontal_score * 100.0, 2)
    return metrics
