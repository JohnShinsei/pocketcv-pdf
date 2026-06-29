from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import re
from typing import Literal

import cv2
import numpy as np

from .geometry import ensure_bgr

OcrEngine = Literal["auto", "rapidocr", "tesseract", "paddleocr"]
BBox = tuple[int, int, int, int]


class OcrUnavailableError(RuntimeError):
    """Raised when the requested optional OCR engine is not installed."""


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float | None
    bbox: BBox

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "confidence": self.confidence, "bbox": list(self.bbox)}


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float | None
    bbox: BBox
    words: list[OcrWord] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True)
class OcrResult:
    engine: str
    language: str
    text: str
    confidence: float | None
    width: int
    height: int
    lines: list[OcrLine]

    def to_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "language": self.language,
            "text": self.text,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
            "lines": [line.to_dict() for line in self.lines],
        }


def _mean_confidence(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and value >= 0]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def _union_bbox(boxes: list[BBox]) -> BBox:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return (left, top, right - left, bottom - top)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _result_from_lines(engine: str, language: str, image: np.ndarray, lines: list[OcrLine]) -> OcrResult:
    height, width = image.shape[:2]
    sorted_lines = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
    text = "\n".join(line.text for line in sorted_lines if line.text)
    return OcrResult(
        engine=engine,
        language=language,
        text=text,
        confidence=_mean_confidence([line.confidence for line in sorted_lines]),
        width=width,
        height=height,
        lines=sorted_lines,
    )


def _bbox_from_points(points: object) -> BBox:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x_min = int(np.floor(float(np.min(array[:, 0]))))
    y_min = int(np.floor(float(np.min(array[:, 1]))))
    x_max = int(np.ceil(float(np.max(array[:, 0]))))
    y_max = int(np.ceil(float(np.max(array[:, 1]))))
    return (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))


def _parse_rapid_line(item: object) -> OcrLine | None:
    box: object | None = None
    text: object | None = None
    score: object | None = None

    if isinstance(item, dict):
        box = item.get("box") or item.get("bbox") or item.get("points")
        text = item.get("text") or item.get("txt")
        score = item.get("score") or item.get("confidence")
    elif isinstance(item, (list, tuple)) and len(item) >= 3:
        box, text, score = item[0], item[1], item[2]

    cleaned = _clean_text(text)
    if not cleaned or box is None:
        return None

    try:
        confidence = round(float(score) * 100 if float(score) <= 1 else float(score), 2)
    except (TypeError, ValueError):
        confidence = None
    bbox = _bbox_from_points(box)
    word = OcrWord(text=cleaned, confidence=confidence, bbox=bbox)
    return OcrLine(text=cleaned, confidence=confidence, bbox=bbox, words=[word])


def _rapid_items(raw_result: object) -> list[object]:
    if raw_result is None:
        return []
    if isinstance(raw_result, tuple) and raw_result:
        return _rapid_items(raw_result[0])
    if hasattr(raw_result, "boxes") and hasattr(raw_result, "txts"):
        boxes = list(getattr(raw_result, "boxes") or [])
        texts = list(getattr(raw_result, "txts") or [])
        scores = list(getattr(raw_result, "scores", []) or [])
        return [[box, text, scores[index] if index < len(scores) else None] for index, (box, text) in enumerate(zip(boxes, texts))]
    if hasattr(raw_result, "to_json"):
        try:
            return _rapid_items(raw_result.to_json())
        except TypeError:
            return _rapid_items(raw_result.to_json)
    if isinstance(raw_result, dict):
        for key in ("result", "results", "data"):
            if key in raw_result:
                return _rapid_items(raw_result[key])
        if {"boxes", "txts"}.issubset(raw_result):
            boxes = list(raw_result.get("boxes") or [])
            texts = list(raw_result.get("txts") or [])
            scores = list(raw_result.get("scores") or [])
            return [[box, text, scores[index] if index < len(scores) else None] for index, (box, text) in enumerate(zip(boxes, texts))]
    if isinstance(raw_result, list):
        return raw_result
    return []


def _run_rapidocr(image: np.ndarray, language: str) -> OcrResult:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        engine_name = "rapidocr_onnxruntime"
    except ImportError:
        try:
            from rapidocr import RapidOCR  # type: ignore[import-not-found,no-redef]

            engine_name = "rapidocr"
        except ImportError as exc:
            raise OcrUnavailableError(
                "RapidOCR is not installed. Install it with: pip install -e .[rapidocr]"
            ) from exc

    ocr_engine = RapidOCR()
    raw_result = ocr_engine(ensure_bgr(image))
    lines = [line for item in _rapid_items(raw_result) if (line := _parse_rapid_line(item)) is not None]
    return _result_from_lines(engine_name, language, image, lines)


def _run_tesseract(image: np.ndarray, language: str) -> OcrResult:
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pytesseract import Output  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OcrUnavailableError("pytesseract is not installed. Install it with: pip install -e .[ocr]") from exc

    rgb = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, lang=language, output_type=Output.DICT)
    grouped: OrderedDict[tuple[int, int, int], list[OcrWord]] = OrderedDict()
    total = len(data.get("text", []))
    for index in range(total):
        text = _clean_text(data["text"][index])
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and confidence < 0:
            confidence = None
        bbox = (
            int(data["left"][index]),
            int(data["top"][index]),
            int(data["width"][index]),
            int(data["height"][index]),
        )
        key = (int(data["block_num"][index]), int(data["par_num"][index]), int(data["line_num"][index]))
        grouped.setdefault(key, []).append(OcrWord(text=text, confidence=confidence, bbox=bbox))

    lines: list[OcrLine] = []
    for words in grouped.values():
        if not words:
            continue
        bbox = _union_bbox([word.bbox for word in words])
        lines.append(
            OcrLine(
                text=" ".join(word.text for word in words),
                confidence=_mean_confidence([word.confidence for word in words]),
                bbox=bbox,
                words=words,
            )
        )
    return _result_from_lines("tesseract", language, image, lines)


def _paddle_language(language: str) -> str:
    lowered = language.lower()
    if "jpn" in lowered or "japan" in lowered:
        return "japan"
    if "eng" in lowered or lowered == "en":
        return "en"
    if "ch" in lowered or "chi" in lowered or "zh" in lowered:
        return "ch"
    return language.split("+", 1)[0] or "ch"


def _run_paddleocr(image: np.ndarray, language: str) -> OcrResult:
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OcrUnavailableError("PaddleOCR is not installed. Install it with: pip install -e .[paddleocr]") from exc

    ocr_engine = PaddleOCR(use_angle_cls=True, lang=_paddle_language(language), show_log=False)
    raw_result = ocr_engine.ocr(ensure_bgr(image), cls=True)
    items = raw_result[0] if raw_result and isinstance(raw_result[0], list) else raw_result
    lines: list[OcrLine] = []
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = item[0]
        text_score = item[1]
        if isinstance(text_score, (list, tuple)) and len(text_score) >= 2:
            text, score = text_score[0], text_score[1]
        else:
            text, score = text_score, None
        line = _parse_rapid_line([box, text, score])
        if line is not None:
            lines.append(line)
    return _result_from_lines("paddleocr", language, image, lines)


def recognize_image(image: np.ndarray, language: str = "jpn+eng", engine: OcrEngine = "auto") -> OcrResult:
    if engine == "rapidocr":
        return _run_rapidocr(image, language)
    if engine == "tesseract":
        return _run_tesseract(image, language)
    if engine == "paddleocr":
        return _run_paddleocr(image, language)
    if engine != "auto":
        raise ValueError("engine must be one of: auto, rapidocr, tesseract, paddleocr")

    candidates: list[OcrEngine] = ["tesseract", "rapidocr", "paddleocr"] if "jpn" in language.lower() else ["rapidocr", "tesseract", "paddleocr"]
    errors: list[str] = []
    for candidate in candidates:
        try:
            return recognize_image(image, language=language, engine=candidate)
        except OcrUnavailableError as exc:
            errors.append(str(exc))
    raise OcrUnavailableError("No OCR engine is installed. " + " ".join(errors))


def _looks_cjk(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" for char in text)


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if _looks_cjk(left[-1:]) or _looks_cjk(right[:1]):
        return f"{left}{right}"
    return f"{left} {right}"


def _column_for_line(line: OcrLine, page_width: int, two_columns: bool) -> int:
    if not two_columns:
        return 0
    center_x = line.bbox[0] + line.bbox[2] / 2.0
    return 0 if center_x < page_width / 2.0 else 1


def _has_two_columns(lines: list[OcrLine], page_width: int) -> bool:
    if len(lines) < 6 or page_width <= 0:
        return False
    centers = np.array([line.bbox[0] + line.bbox[2] / 2.0 for line in lines], dtype=np.float32)
    left = int(np.sum(centers < page_width * 0.44))
    right = int(np.sum(centers > page_width * 0.56))
    return left >= 3 and right >= 3


def recover_layout_markdown(result: OcrResult) -> str:
    lines = [line for line in result.lines if line.text]
    if not lines:
        return ""

    heights = [max(1, line.bbox[3]) for line in lines]
    median_height = float(np.median(heights)) if heights else 12.0
    two_columns = _has_two_columns(lines, result.width)
    ordered = sorted(lines, key=lambda line: (_column_for_line(line, result.width, two_columns), line.bbox[1], line.bbox[0]))

    paragraphs: list[str] = []
    current = ""
    previous_line: OcrLine | None = None
    previous_column = 0

    for line in ordered:
        column = _column_for_line(line, result.width, two_columns)
        centered = abs((line.bbox[0] + line.bbox[2] / 2.0) - result.width / 2.0) < result.width * 0.18
        is_heading = len(line.text) <= 42 and (line.bbox[3] > median_height * 1.25 or centered)
        gap = 0.0 if previous_line is None else line.bbox[1] - (previous_line.bbox[1] + previous_line.bbox[3])
        starts_new = previous_line is None or column != previous_column or gap > median_height * 1.35 or is_heading

        if starts_new and current:
            paragraphs.append(current)
            current = ""
        if is_heading:
            paragraphs.append(f"## {line.text}")
        else:
            current = _join_text(current, line.text)
        previous_line = line
        previous_column = column

    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())
