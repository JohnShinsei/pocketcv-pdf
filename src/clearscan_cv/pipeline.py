from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .geometry import detect_document_corners, ensure_bgr, four_point_transform
from .quality import assess_quality, compare_quality

OutputMode = Literal["color", "gray", "binary"]


@dataclass
class EnhancementResult:
    image: np.ndarray
    report: dict[str, object]


def _odd_kernel(size: int, minimum: int = 15, maximum: int = 99) -> int:
    size = max(minimum, min(maximum, size))
    return size + 1 if size % 2 == 0 else size


def unsharp_mask(gray: np.ndarray, amount: float = 0.75, radius: float = 2.0) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    h, w = lightness.shape[:2]
    kernel = _odd_kernel(int(min(h, w) / 10), minimum=31, maximum=181)
    background = cv2.GaussianBlur(lightness, (kernel, kernel), 0)
    normalized = cv2.divide(lightness, np.maximum(background, 1), scale=245)
    normalized = cv2.medianBlur(normalized, 3)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    corrected = clahe.apply(normalized)
    corrected = unsharp_mask(corrected, amount=0.55, radius=1.6)

    paper_mask = corrected > 205
    corrected = corrected.copy()
    corrected[paper_mask] = np.maximum(corrected[paper_mask], 245)

    merged = cv2.merge((corrected, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def to_clean_binary(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = unsharp_mask(gray, amount=0.25, radius=1.0)

    h, w = gray.shape[:2]
    block_size = _odd_kernel(int(min(h, w) / 18), minimum=31, maximum=81)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        15,
    )

    foreground = 255 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    kept = np.zeros_like(foreground)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        touches_edge = (
            stats[label, cv2.CC_STAT_LEFT] <= 1
            or stats[label, cv2.CC_STAT_TOP] <= 1
            or stats[label, cv2.CC_STAT_LEFT] + width >= foreground.shape[1] - 2
            or stats[label, cv2.CC_STAT_TOP] + height >= foreground.shape[0] - 2
        )
        is_edge_stain = touches_edge and area > foreground.size * 0.004 and min(width, height) > 12
        if not is_edge_stain and (area >= 10 or (area >= 5 and max(width, height) >= 5)):
            kept[labels == label] = 255
    return 255 - kept


def build_side_by_side(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    left = ensure_bgr(original)
    right = ensure_bgr(processed)
    target_height = max(left.shape[0], right.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        scale = target_height / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), target_height), interpolation=cv2.INTER_AREA)

    left = resize_to_height(left)
    right = resize_to_height(right)
    gap = np.full((target_height, 18, 3), 245, dtype=np.uint8)
    return cv2.hconcat([left, gap, right])


def enhance_image(image: np.ndarray, mode: OutputMode = "color", auto_warp: bool = True) -> EnhancementResult:
    if mode not in {"color", "gray", "binary"}:
        raise ValueError("mode must be one of: color, gray, binary")

    bgr = ensure_bgr(image)
    detection = detect_document_corners(bgr)
    warped = four_point_transform(bgr, detection.corners) if auto_warp and detection.found else bgr.copy()
    enhanced = normalize_illumination(warped)

    if mode == "binary":
        output = to_clean_binary(enhanced)
    elif mode == "gray":
        output = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        output = enhanced

    quality_after = enhanced if mode == "binary" else output
    report = {
        "mode": mode,
        "auto_warp": auto_warp,
        "document_detection": detection.to_dict(),
        "quality": compare_quality(warped, quality_after),
        "output_quality": assess_quality(output),
        "pipeline": ["document_detection", "perspective_correction", "illumination_normalization", mode],
    }
    return EnhancementResult(image=output, report=report)


def process_file(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    mode: OutputMode = "color",
    auto_warp: bool = True,
    side_by_side: bool = False,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(str(input_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = enhance_image(image, mode=mode, auto_warp=auto_warp)
    output_path = output_dir / f"{input_path.stem}_clearscan.png"
    report_path = output_dir / f"{input_path.stem}_report.json"
    cv2.imwrite(str(output_path), result.image)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        **result.report,
    }

    if side_by_side:
        compare_path = output_dir / f"{input_path.stem}_comparison.png"
        cv2.imwrite(str(compare_path), build_side_by_side(image, result.image))
        report["comparison_path"] = str(compare_path)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
