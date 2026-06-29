from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .dewarp import dewarp_by_textline_columns
from .geometry import detect_document_corners, ensure_bgr, four_point_transform
from .quality import assess_quality, compare_quality

OutputMode = Literal["color", "gray", "binary"]
MAX_PROCESS_IMAGE_EDGE = 3200
MAX_PROCESS_IMAGE_PIXELS = 6_500_000


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


def limit_image_resolution(image: np.ndarray, max_edge: int = MAX_PROCESS_IMAGE_EDGE, max_pixels: int = MAX_PROCESS_IMAGE_PIXELS) -> np.ndarray:
    h, w = image.shape[:2]
    edge_scale = max_edge / float(max(h, w))
    pixel_scale = np.sqrt(max_pixels / float(max(1, h * w)))
    scale = min(1.0, edge_scale, float(pixel_scale))
    if scale >= 0.995:
        return image
    return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _resize_for_background(gray: np.ndarray, max_dim: int = 1200) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 1.0:
        return gray, 1.0
    resized = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _restore_background_size(background: np.ndarray, shape: tuple[int, int], scale: float) -> np.ndarray:
    if scale >= 1.0:
        return background
    return cv2.resize(background, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def estimate_luminance_background(gray: np.ndarray) -> np.ndarray:
    small, scale = _resize_for_background(gray)
    h, w = small.shape[:2]
    kernel = _odd_kernel(int(min(h, w) / 7), minimum=61, maximum=401)
    background = cv2.GaussianBlur(small, (kernel, kernel), 0)
    return _restore_background_size(background, gray.shape[:2], scale)


def preserve_high_frequency_detail(source: np.ndarray, corrected: np.ndarray, amount: float = 0.34) -> np.ndarray:
    detail = source.astype(np.float32) - cv2.GaussianBlur(source, (0, 0), 1.15).astype(np.float32)
    restored = corrected.astype(np.float32) + np.clip(detail, -80.0, 28.0) * amount
    return np.clip(restored, 0, 255).astype(np.uint8)


def estimate_shadow_illumination(gray: np.ndarray) -> np.ndarray:
    small, scale = _resize_for_background(gray)
    h, w = small.shape[:2]
    close_size = _odd_kernel(int(min(h, w) / 16), minimum=31, maximum=181)
    blur_size = _odd_kernel(int(min(h, w) / 6), minimum=75, maximum=451)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    background = cv2.GaussianBlur(background, (blur_size, blur_size), 0)
    return _restore_background_size(background, gray.shape[:2], scale)


def deshadow_luminance(gray: np.ndarray, scale: float = 252.0) -> np.ndarray:
    background = estimate_shadow_illumination(gray)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=scale)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return preserve_high_frequency_detail(gray, normalized, amount=0.28)


def should_use_frequency_deshadow(gray: np.ndarray) -> bool:
    background = estimate_shadow_illumination(gray)
    background_range = float(np.percentile(background, 95) - np.percentile(background, 5))
    return background_range > 24.0


def normalize_shadow_luminance(gray: np.ndarray, scale: float = 252.0) -> np.ndarray:
    background = estimate_luminance_background(gray)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=scale)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def estimate_gatos_background(gray: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    small_gray, scale = _resize_for_background(gray)
    if scale < 1.0:
        small_mask = cv2.resize(
            foreground_mask.astype(np.uint8),
            (small_gray.shape[1], small_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        small_mask = foreground_mask
    mask = small_mask.astype(np.uint8) * 255
    if float(np.mean(mask > 0)) < 0.002:
        filled = small_gray
    else:
        filled = cv2.inpaint(small_gray, mask, 3, cv2.INPAINT_TELEA)

    h, w = small_gray.shape[:2]
    kernel = _odd_kernel(int(min(h, w) / 8), minimum=51, maximum=351)
    background = cv2.GaussianBlur(filled, (kernel, kernel), 0)
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, np.ones((17, 17), dtype=np.uint8), iterations=1)
    background = _restore_background_size(background, gray.shape[:2], scale)
    return np.clip(background, 1, 255).astype(np.uint8)


def sauvola_threshold(gray_float: np.ndarray, window_size: int, k: float = 0.28, r: float = 128.0) -> np.ndarray:
    window_size = _odd_kernel(window_size, minimum=15, maximum=251)
    mean = cv2.boxFilter(gray_float, cv2.CV_32F, (window_size, window_size), normalize=True)
    sq_mean = cv2.boxFilter(gray_float * gray_float, cv2.CV_32F, (window_size, window_size), normalize=True)
    deviation = np.sqrt(np.maximum(0.0, sq_mean - mean * mean))
    return mean * (1.0 + k * (deviation / r - 1.0))


def estimate_textline_skew(image: np.ndarray, max_angle: float = 6.0, step: float = 0.5) -> tuple[float, float]:
    bgr = limit_image_resolution(ensure_bgr(image))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < 80:
        return 0.0, 0.0

    scale = min(1.0, 900.0 / float(max(width, height)))
    small = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    blur = cv2.GaussianBlur(small, (3, 3), 0)
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_threshold = max(35.0, min(185.0, min(float(otsu_threshold), float(np.percentile(blur, 35))) - 2.0))
    mask = blur <= dark_threshold
    ys, xs = np.nonzero(mask)
    if xs.size < small.size * 0.003:
        return 0.0, 0.0

    if xs.size > 90000:
        stride = max(1, xs.size // 90000)
        xs = xs[::stride]
        ys = ys[::stride]

    centered_x = xs.astype(np.float32) - small.shape[1] / 2.0
    centered_y = ys.astype(np.float32) - small.shape[0] / 2.0
    diagonal = int(np.ceil(np.hypot(*small.shape[:2]))) + 3
    offset = diagonal // 2
    angles = np.arange(-max_angle, max_angle + step * 0.5, step, dtype=np.float32)
    best_angle = 0.0
    best_score = -1.0
    zero_score = 0.0

    for angle in angles:
        radians = np.deg2rad(float(angle))
        projected_y = np.rint(centered_y * np.cos(radians) - centered_x * np.sin(radians)).astype(np.int32) + offset
        valid = (projected_y >= 0) & (projected_y < diagonal)
        bins = np.bincount(projected_y[valid], minlength=diagonal).astype(np.float32)
        score = float(np.sum(bins * bins) / max(1, xs.size))
        if abs(float(angle)) < step * 0.25:
            zero_score = score
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    if abs(best_angle) < 0.25:
        return 0.0, 0.0

    improvement = best_score / max(zero_score, 1.0)
    if improvement < 1.035:
        return 0.0, round(float(improvement), 3)
    return round(best_angle, 3), round(float(improvement), 3)


def rotate_image_keep_content(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.25:
        return image

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR, borderValue=border_value)


def deskew_by_text_lines(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    angle, confidence = estimate_textline_skew(image)
    if abs(angle) < 0.25:
        return image, {"angle": 0.0, "confidence": confidence}
    return rotate_image_keep_content(image, angle), {"angle": angle, "confidence": confidence}


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)

    if should_use_frequency_deshadow(lightness):
        normalized = deshadow_luminance(lightness, scale=252)
    else:
        normalized = normalize_shadow_luminance(lightness, scale=252)
    normalized = cv2.medianBlur(normalized, 3)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    corrected = clahe.apply(normalized)
    corrected = unsharp_mask(corrected, amount=0.55, radius=1.6)

    paper_mask = corrected > 218
    corrected = corrected.copy()
    corrected[paper_mask] = np.maximum(corrected[paper_mask], 248)

    merged = cv2.merge((corrected, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def to_clean_binary(image: np.ndarray) -> np.ndarray:
    bgr = ensure_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 24, 24)
    rough_block_size = _odd_kernel(int(min(gray.shape[:2]) / 18), minimum=41, maximum=161)
    rough_float = denoised.astype(np.float32)
    rough_mean = cv2.boxFilter(rough_float, cv2.CV_32F, (rough_block_size, rough_block_size), normalize=True)
    rough_sq_mean = cv2.boxFilter(rough_float * rough_float, cv2.CV_32F, (rough_block_size, rough_block_size), normalize=True)
    rough_std = np.sqrt(np.maximum(0.0, rough_sq_mean - rough_mean * rough_mean))
    otsu_seed_threshold, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rough_foreground = ((rough_mean - rough_float) > np.maximum(10.0, rough_std * 0.35 + 4.0)) | (
        rough_float < min(float(otsu_seed_threshold) - 8.0, 158.0)
    )
    gatos_background = estimate_gatos_background(denoised, rough_foreground)
    raw_range = float(np.percentile(denoised, 95) - np.percentile(denoised, 20))
    background_range = float(np.percentile(gatos_background, 95) - np.percentile(gatos_background, 5))
    foreground_ratio = float(np.mean(rough_foreground))
    use_gatos_background = background_range > 26.0 or raw_range > 45.0 or foreground_ratio > 0.14
    if use_gatos_background:
        normalized = cv2.divide(denoised, np.maximum(gatos_background, 1), scale=255)
        normalized = preserve_high_frequency_detail(denoised, np.clip(normalized, 0, 255).astype(np.uint8), amount=0.22)
    else:
        normalized = deshadow_luminance(denoised, scale=255) if should_use_frequency_deshadow(denoised) else normalize_shadow_luminance(denoised, scale=255)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=0.9, tileGridSize=(8, 8))
    gray = unsharp_mask(clahe.apply(normalized), amount=0.18, radius=1.0)

    h, w = gray.shape[:2]
    block_size = _odd_kernel(int(min(h, w) / 22), minimum=41, maximum=151)
    gray_float = gray.astype(np.float32)
    local_mean = cv2.boxFilter(gray_float, cv2.CV_32F, (block_size, block_size), normalize=True)
    local_sq_mean = cv2.boxFilter(gray_float * gray_float, cv2.CV_32F, (block_size, block_size), normalize=True)
    local_std = np.sqrt(np.maximum(0.0, local_sq_mean - local_mean * local_mean))
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.abs(sobel_x) + np.abs(sobel_y)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_delta = local_mean - gray_float

    if use_gatos_background:
        small_window = _odd_kernel(int(min(h, w) / 34), minimum=31, maximum=91)
        large_window = _odd_kernel(int(min(h, w) / 14), minimum=71, maximum=211)
        small_sauvola = sauvola_threshold(gray_float, small_window, k=0.18)
        large_sauvola = sauvola_threshold(gray_float, large_window, k=0.28)
        sauvola_fine = (gray_float < small_sauvola - 7.0) & (ink_delta > 13.0) & ((gradient > 18.0) | (gray_float < 156.0))
        sauvola_broad = (gray_float < large_sauvola - 12.0) & (ink_delta > 20.0) & (gray_float < 184.0)
    else:
        sauvola_fine = np.zeros_like(gray_float, dtype=bool)
        sauvola_broad = np.zeros_like(gray_float, dtype=bool)
    local_ink = (ink_delta > np.maximum(24.0, local_std * 0.58 + 10.0)) & (gray_float < 198) & (gradient > 15)
    text_body = (gray_float < min(float(otsu_threshold) - 12.0, 144.0)) & (ink_delta > 15)
    text_edge = (ink_delta > 18) & (gradient > 54) & (gray_float < 178)
    deep_ink = gray_float < 62
    paper_texture = (gray_float > 178) & (ink_delta < 42) & (gradient < 42)
    foreground_mask = (sauvola_fine | sauvola_broad | local_ink | text_body | text_edge | deep_ink) & ~paper_texture
    binary = np.where(foreground_mask, 0, 255).astype(np.uint8)

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
        density = area / max(1, width * height)
        aspect = max(width, height) / max(1, min(width, height))
        is_edge_stain = touches_edge and area > max(160, foreground.size * 0.0006) and (
            (width > foreground.shape[1] * 0.12 and height > 8) or (height > foreground.shape[0] * 0.12 and width > 8)
        )
        is_large_blob = area > foreground.size * 0.014 and min(width, height) > 16
        is_tiny_dust = area <= 4 or (area <= 12 and width <= 6 and height <= 6)
        is_sparse_texture = area < 30 and density < 0.16 and max(width, height) < 22
        is_small_text = area >= 5 and (aspect >= 2.0 or max(width, height) >= 8)
        if not (is_edge_stain or is_large_blob or is_tiny_dust or is_sparse_texture) and (area >= 14 or is_small_text):
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


def enhance_image(image: np.ndarray, mode: OutputMode = "color", auto_warp: bool = True, auto_dewarp: bool = True) -> EnhancementResult:
    if mode not in {"color", "gray", "binary"}:
        raise ValueError("mode must be one of: color, gray, binary")

    bgr = limit_image_resolution(ensure_bgr(image))
    detection = detect_document_corners(bgr)
    warped = four_point_transform(bgr, detection.corners) if auto_warp and detection.found else bgr.copy()
    dewarp_result = dewarp_by_textline_columns(warped) if auto_dewarp else None
    dewarped = dewarp_result.image if dewarp_result is not None else warped
    deskewed, deskew_report = deskew_by_text_lines(dewarped)
    enhanced = normalize_illumination(deskewed)

    if mode == "binary":
        output = to_clean_binary(deskewed)
    elif mode == "gray":
        output = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        output = enhanced

    quality_after = enhanced if mode == "binary" else output
    report = {
        "mode": mode,
        "auto_warp": auto_warp,
        "auto_dewarp": auto_dewarp,
        "document_detection": detection.to_dict(),
        "dewarp": dewarp_result.report if dewarp_result is not None else {"applied": False, "method": "disabled"},
        "deskew": deskew_report,
        "quality": compare_quality(deskewed, quality_after),
        "output_quality": assess_quality(output),
        "pipeline": ["document_detection", "perspective_correction", "textline_dewarp", "textline_deskew", "illumination_normalization", mode],
    }
    return EnhancementResult(image=output, report=report)


def process_file(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    mode: OutputMode = "color",
    auto_warp: bool = True,
    auto_dewarp: bool = True,
    side_by_side: bool = False,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.fromfile(str(input_path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    result = enhance_image(image, mode=mode, auto_warp=auto_warp, auto_dewarp=auto_dewarp)
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
