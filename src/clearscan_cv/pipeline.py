from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .corners import CornerPoints, manual_detection_from_corners, scale_corner_points
from .dewarp import dewarp_by_textline_columns
from .geometry import DocumentDetection, detect_document_corners, ensure_bgr, four_point_transform
from .image_io import decode_image_file
from .model_hooks import apply_external_corner_hook, apply_external_image_hook
from .quality import assess_quality, compare_quality, diagnose_scan_quality

OutputMode = Literal["auto", "color", "gray", "binary"]
CornerCoordinateSpace = Literal["input", "processed"]
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


def estimate_hough_textline_skew(image: np.ndarray, max_angle: float = 6.0) -> tuple[float, float]:
    bgr = limit_image_resolution(ensure_bgr(image), max_edge=1400, max_pixels=1_800_000)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if min(height, width) < 80:
        return 0.0, 0.0

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 42, 142)
    min_line_length = max(24, int(round(width * 0.035)))
    max_line_gap = max(6, int(round(width * 0.012)))
    threshold = max(36, int(round(width * 0.045)))
    raw_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw_lines is None:
        return 0.0, 0.0

    angles: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in raw_lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1.0:
            continue
        length = float(np.hypot(dx, dy))
        if length < min_line_length or length > width * 0.42:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle > 90.0:
            angle -= 180.0
        elif angle < -90.0:
            angle += 180.0
        if abs(angle) > max_angle:
            continue
        angles.append(angle)
        weights.append(length)

    if len(angles) < 12:
        return 0.0, 0.0

    angle_array = np.asarray(angles, dtype=np.float32)
    weight_array = np.asarray(weights, dtype=np.float32)
    low, high = np.percentile(angle_array, [12, 88])
    inlier_mask = (angle_array >= low) & (angle_array <= high)
    if int(np.sum(inlier_mask)) < 8:
        return 0.0, 0.0

    inlier_angles = angle_array[inlier_mask]
    inlier_weights = weight_array[inlier_mask]
    orientation = float(np.average(inlier_angles, weights=inlier_weights))
    spread = float(np.std(inlier_angles))
    confidence = 1.0 + min(0.24, len(inlier_angles) / 520.0) + min(
        0.16,
        max(0.0, max_angle - spread) / max_angle * 0.16,
    )
    if abs(orientation) < 0.25 or confidence < 1.045:
        return 0.0, round(confidence, 3)
    return round(orientation, 3), round(confidence, 3)


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
        return estimate_hough_textline_skew(image, max_angle=max_angle)

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
        hough_angle, hough_confidence = estimate_hough_textline_skew(image, max_angle=max_angle)
        if abs(hough_angle) >= 0.25 and hough_confidence >= 1.045:
            return hough_angle, hough_confidence
        return 0.0, round(hough_confidence, 3)

    improvement = best_score / max(zero_score, 1.0)
    if improvement < 1.035:
        hough_angle, hough_confidence = estimate_hough_textline_skew(image, max_angle=max_angle)
        if abs(hough_angle) >= 0.25 and hough_confidence >= 1.045:
            return hough_angle, hough_confidence
        return 0.0, round(float(max(improvement, hough_confidence)), 3)
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


def _quality_issue_codes(diagnostics: dict[str, object]) -> set[str]:
    issues = diagnostics.get("issues")
    if not isinstance(issues, list):
        return set()
    codes: set[str] = set()
    for issue in issues:
        if isinstance(issue, dict) and isinstance(issue.get("code"), str):
            codes.add(str(issue["code"]))
    return codes


def _auto_mode_choice(binary_quality: dict[str, object], gray_quality: dict[str, object], perspective_confidence: float) -> tuple[str, dict[str, object]]:
    binary_diagnostics = diagnose_scan_quality(binary_quality, perspective_confidence=perspective_confidence)
    gray_diagnostics = diagnose_scan_quality(gray_quality, perspective_confidence=perspective_confidence)
    binary_score = float(binary_quality["score"])
    gray_score = float(gray_quality["score"])
    binary_issue_codes = _quality_issue_codes(binary_diagnostics)
    fragile_binary_codes = {"shadow_residual", "bold_text", "low_quality"}
    choose_gray = bool(binary_issue_codes & fragile_binary_codes) and gray_score >= binary_score - 8.0
    if binary_diagnostics["status"] != "ready" and gray_diagnostics["status"] == "ready" and gray_score > binary_score + 4.0:
        choose_gray = True
    if binary_score < 55.0 and gray_score > binary_score + 4.0:
        choose_gray = True

    selected = "gray" if choose_gray else "binary"
    return selected, {
        "selected_mode": selected,
        "binary_score": round(binary_score, 2),
        "gray_score": round(gray_score, 2),
        "binary_status": binary_diagnostics["status"],
        "gray_status": gray_diagnostics["status"],
        "reason": "gray_preserves_fragile_text" if selected == "gray" else "binary_scan_is_readable",
    }


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


def template_guided_illumination(image: np.ndarray, template: np.ndarray | None) -> tuple[np.ndarray, dict[str, object]]:
    if template is None:
        return image, {"applied": False, "method": "disabled"}

    bgr = ensure_bgr(image)
    template_bgr = ensure_bgr(template)
    if min(template_bgr.shape[:2]) < 40:
        return image, {"applied": False, "method": "template_guided_illumination", "reason": "template_too_small"}

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    resized_template = cv2.resize(template_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
    template_lightness = cv2.cvtColor(resized_template, cv2.COLOR_BGR2LAB)[:, :, 0]
    source_background = estimate_shadow_illumination(lightness)
    template_background = estimate_shadow_illumination(template_lightness)
    source_range = float(np.percentile(source_background, 95) - np.percentile(source_background, 5))
    template_range = float(np.percentile(template_background, 95) - np.percentile(template_background, 5))

    target_background = np.clip(template_background.astype(np.float32), 180.0, 252.0)
    normalized = cv2.divide(lightness, np.maximum(source_background, 1), scale=1.0).astype(np.float32) * target_background
    corrected = np.clip(normalized, 0, 255).astype(np.uint8)
    corrected = preserve_high_frequency_detail(lightness, corrected, amount=0.2)
    corrected_background = estimate_shadow_illumination(corrected)
    corrected_range = float(np.percentile(corrected_background, 95) - np.percentile(corrected_background, 5))
    merged = cv2.merge((corrected, channel_a, channel_b))
    report = {
        "applied": True,
        "method": "template_guided_illumination",
        "template_size": {"width": int(template_bgr.shape[1]), "height": int(template_bgr.shape[0])},
        "source_background_range": round(source_range, 2),
        "template_background_range": round(template_range, 2),
        "corrected_background_range": round(corrected_range, 2),
    }
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), report


def _fragile_component_mask(foreground: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    preserved = np.zeros_like(foreground)
    small_area_limit = max(18, int(round(foreground.size * 0.000045)))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= small_area_limit or min(width, height) <= 2 or (area <= small_area_limit * 2 and max(width, height) <= 14):
            preserved[labels == label] = 255
    return preserved


def _reduce_bold_binary_strokes(binary: np.ndarray) -> np.ndarray:
    normalized = np.where(binary < 128, 0, 255).astype(np.uint8)
    quality_before = assess_quality(normalized)
    ink_density = float(quality_before["ink_density"])
    boldness_risk = float(quality_before["boldness_risk"])
    if ink_density < 0.105 and boldness_risk < 0.28:
        return normalized

    foreground = 255 - normalized
    thinned = cv2.erode(foreground, np.ones((3, 3), dtype=np.uint8), iterations=1)
    thinned = cv2.bitwise_or(thinned, _fragile_component_mask(foreground))
    candidate = 255 - thinned

    quality_after = assess_quality(candidate)
    candidate_ink = float(quality_after["ink_density"])
    candidate_edge_density = float(quality_after["edge_density"])
    original_edge_density = float(quality_before["edge_density"])
    if candidate_ink < max(0.012, ink_density * 0.42):
        return normalized
    if candidate_edge_density < original_edge_density * 0.52:
        return normalized
    if candidate_ink >= ink_density * 0.94:
        return normalized
    if float(quality_after["boldness_risk"]) <= max(0.2, boldness_risk - 0.08) or candidate_ink < 0.12:
        return candidate
    return normalized


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
    edge_margin_x = max(6, int(round(foreground.shape[1] * 0.025)))
    edge_margin_y = max(6, int(round(foreground.shape[0] * 0.025)))
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        touches_edge = (
            left <= 1
            or top <= 1
            or left + width >= foreground.shape[1] - 2
            or top + height >= foreground.shape[0] - 2
        )
        near_edge = (
            left <= edge_margin_x
            or top <= edge_margin_y
            or left + width >= foreground.shape[1] - edge_margin_x
            or top + height >= foreground.shape[0] - edge_margin_y
        )
        density = area / max(1, width * height)
        aspect = max(width, height) / max(1, min(width, height))
        is_edge_stain = (touches_edge or near_edge) and area > max(120, foreground.size * 0.00045) and (
            (width > foreground.shape[1] * 0.12 and height > 8) or (height > foreground.shape[0] * 0.12 and width > 8)
        )
        is_near_edge_blob = near_edge and area > max(240, foreground.size * 0.0012) and (density > 0.22 or aspect > 4.0)
        is_large_blob = area > foreground.size * 0.014 and min(width, height) > 16
        is_tiny_dust = area <= 4 or (area <= 12 and width <= 6 and height <= 6)
        is_sparse_texture = area < 30 and density < 0.16 and max(width, height) < 22
        is_small_text = area >= 5 and (aspect >= 2.0 or max(width, height) >= 8)
        if not (is_edge_stain or is_near_edge_blob or is_large_blob or is_tiny_dust or is_sparse_texture) and (area >= 14 or is_small_text):
            kept[labels == label] = 255
    return _reduce_bold_binary_strokes(255 - kept)


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


def _float_report_value(report: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(report.get(key, default))
    except (TypeError, ValueError):
        return default
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else default


def _detection_from_external_corners(corners: CornerPoints, width: int, height: int, report: dict[str, object]) -> DocumentDetection:
    validated = manual_detection_from_corners(corners, width=width, height=height)
    confidence = _float_report_value(report, "confidence", 0.86)
    return DocumentDetection(
        corners=validated.corners,
        confidence=round(confidence, 3),
        area_ratio=validated.area_ratio,
        method="external_detector",
        found=True,
    )


def _external_detector_pipeline_stage(report: dict[str, object], command: str | None) -> str:
    if not command:
        return "external_detector_disabled"
    return "external_detector" if report.get("applied") else "external_detector_fallback"


def enhance_image(
    image: np.ndarray,
    mode: OutputMode = "color",
    auto_warp: bool = True,
    auto_dewarp: bool = True,
    manual_corners: CornerPoints | None = None,
    manual_corners_space: CornerCoordinateSpace = "input",
    template_image: np.ndarray | None = None,
    external_detector_command: str | None = None,
    external_detector_timeout: float = 90.0,
    external_restorer_command: str | None = None,
    external_restorer_timeout: float = 180.0,
) -> EnhancementResult:
    if mode not in {"auto", "color", "gray", "binary"}:
        raise ValueError("mode must be one of: auto, color, gray, binary")
    if manual_corners_space not in {"input", "processed"}:
        raise ValueError("manual_corners_space must be input or processed")

    source_bgr = ensure_bgr(image)
    source_height, source_width = source_bgr.shape[:2]
    bgr = limit_image_resolution(source_bgr)
    height, width = bgr.shape[:2]
    external_detector_report: dict[str, object] = {
        "stage": "external_detector",
        "applied": False,
        "method": "disabled",
    }
    if manual_corners is not None:
        if manual_corners_space == "input":
            processed_corners = scale_corner_points(manual_corners, (source_width, source_height), (width, height))
        else:
            processed_corners = manual_corners
        detection = manual_detection_from_corners(processed_corners, width=width, height=height)
        if external_detector_command:
            external_detector_report = {
                "stage": "external_detector",
                "applied": False,
                "method": "manual_corners_override",
            }
    else:
        if external_detector_command:
            external_detector_result = apply_external_corner_hook(
                bgr,
                external_detector_command,
                stage="external_detector",
                timeout_seconds=external_detector_timeout,
            )
            external_detector_report = external_detector_result.report
            if external_detector_result.corners is not None:
                try:
                    detection = _detection_from_external_corners(external_detector_result.corners, width, height, external_detector_report)
                    external_detector_report.update({"applied": True, "document_detection": detection.to_dict()})
                except ValueError as exc:
                    external_detector_report.update({"applied": False, "reason": "invalid_detection_geometry", "error": str(exc)})
                    detection = detect_document_corners(bgr)
            else:
                detection = detect_document_corners(bgr)
        else:
            detection = detect_document_corners(bgr)
    use_perspective = manual_corners is not None or (auto_warp and detection.found)
    warped = four_point_transform(bgr, detection.corners) if use_perspective else bgr.copy()
    if mode == "color":
        output_quality = assess_quality(warped)
        color_stage_report = {"applied": False, "method": "color_geometry_only", "reason": "color_mode_bypasses_enhancement"}
        report = {
            "mode": mode,
            "selected_mode": "color",
            "auto_selection": None,
            "auto_warp": auto_warp,
            "auto_dewarp": auto_dewarp,
            "manual_corners": manual_corners is not None,
            "manual_corners_space": manual_corners_space if manual_corners is not None else None,
            "source_image_size": {"width": source_width, "height": source_height},
            "processing_image_size": {"width": width, "height": height},
            "document_detection": detection.to_dict(),
            "external_detector": external_detector_report,
            "dewarp": color_stage_report.copy(),
            "deskew": {"angle": 0.0, "confidence": 0.0, "method": "color_geometry_only"},
            "external_restorer": color_stage_report.copy(),
            "template_guided_illumination": color_stage_report.copy(),
            "quality": compare_quality(warped, warped),
            "output_quality": output_quality,
            "quality_diagnostics": diagnose_scan_quality(
                output_quality,
                perspective_confidence=float(detection.confidence) if detection.found else 0.0,
            ),
            "pipeline": [
                _external_detector_pipeline_stage(external_detector_report, external_detector_command),
                "document_detection",
                "perspective_correction",
                "color_geometry_only",
            ],
        }
        return EnhancementResult(image=warped, report=report)

    dewarp_result = dewarp_by_textline_columns(warped) if auto_dewarp else None
    dewarped = dewarp_result.image if dewarp_result is not None else warped
    deskewed, deskew_report = deskew_by_text_lines(dewarped)
    external_result = apply_external_image_hook(
        deskewed,
        external_restorer_command,
        stage="external_restorer",
        timeout_seconds=external_restorer_timeout,
    )
    restored = external_result.image
    templated, template_report = template_guided_illumination(restored, template_image)
    restored = templated
    enhanced = normalize_illumination(restored)

    selected_mode = mode
    auto_selection: dict[str, object] | None = None
    if mode == "auto":
        gray_output = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        binary_output = to_clean_binary(restored)
        perspective_confidence = float(detection.confidence) if detection.found else 0.0
        selected_mode, auto_selection = _auto_mode_choice(assess_quality(binary_output), assess_quality(gray_output), perspective_confidence)
        output = gray_output if selected_mode == "gray" else binary_output
    elif mode == "binary":
        output = to_clean_binary(restored)
    elif mode == "gray":
        output = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    else:
        output = enhanced

    quality_after = enhanced if selected_mode == "binary" else output
    output_quality = assess_quality(output)
    report = {
        "mode": mode,
        "selected_mode": selected_mode,
        "auto_selection": auto_selection,
        "auto_warp": auto_warp,
        "auto_dewarp": auto_dewarp,
        "manual_corners": manual_corners is not None,
        "manual_corners_space": manual_corners_space if manual_corners is not None else None,
        "source_image_size": {"width": source_width, "height": source_height},
        "processing_image_size": {"width": width, "height": height},
        "document_detection": detection.to_dict(),
        "external_detector": external_detector_report,
        "dewarp": dewarp_result.report if dewarp_result is not None else {"applied": False, "method": "disabled"},
        "deskew": deskew_report,
        "external_restorer": external_result.report,
        "template_guided_illumination": template_report,
        "quality": compare_quality(deskewed, quality_after),
        "output_quality": output_quality,
        "quality_diagnostics": diagnose_scan_quality(
            output_quality,
            perspective_confidence=float(detection.confidence) if detection.found else 0.0,
        ),
        "pipeline": [
            _external_detector_pipeline_stage(external_detector_report, external_detector_command),
            "document_detection",
            "perspective_correction",
            "textline_dewarp",
            "textline_deskew",
            "external_restorer" if external_restorer_command else "external_restorer_disabled",
            "template_guided_illumination" if template_image is not None else "template_guided_illumination_disabled",
            "illumination_normalization",
            selected_mode,
        ],
    }
    return EnhancementResult(image=output, report=report)


def process_file(
    input_path: str | Path,
    output_dir: str | Path = "outputs",
    mode: OutputMode = "color",
    auto_warp: bool = True,
    auto_dewarp: bool = True,
    side_by_side: bool = False,
    manual_corners: CornerPoints | None = None,
    manual_corners_space: CornerCoordinateSpace = "input",
    output_stem: str | None = None,
    template_path: str | Path | None = None,
    external_detector_command: str | None = None,
    external_detector_timeout: float = 90.0,
    external_restorer_command: str | None = None,
    external_restorer_timeout: float = 180.0,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        decoded_input = decode_image_file(input_path, flags=cv2.IMREAD_COLOR)
    except ValueError as exc:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    image = decoded_input.image
    template_image = None
    decoded_template = None
    if template_path is not None:
        try:
            decoded_template = decode_image_file(template_path, flags=cv2.IMREAD_COLOR)
        except ValueError as exc:
            raise FileNotFoundError(f"Could not read template image: {template_path}")
        template_image = decoded_template.image

    result = enhance_image(
        image,
        mode=mode,
        auto_warp=auto_warp,
        auto_dewarp=auto_dewarp,
        manual_corners=manual_corners,
        manual_corners_space=manual_corners_space,
        template_image=template_image,
        external_detector_command=external_detector_command,
        external_detector_timeout=external_detector_timeout,
        external_restorer_command=external_restorer_command,
        external_restorer_timeout=external_restorer_timeout,
    )
    stem = output_stem or input_path.stem
    output_path = output_dir / f"{stem}_clearscan.png"
    report_path = output_dir / f"{stem}_report.json"
    cv2.imwrite(str(output_path), result.image)

    report = {
        "input_path": str(input_path),
        "template_path": str(template_path) if template_path is not None else None,
        "input_decode": decoded_input.to_report(),
        "template_decode": decoded_template.to_report() if decoded_template is not None else None,
        "output_path": str(output_path),
        "report_path": str(report_path),
        **result.report,
    }

    if side_by_side:
        compare_path = output_dir / f"{stem}_comparison.png"
        cv2.imwrite(str(compare_path), build_side_by_side(image, result.image))
        report["comparison_path"] = str(compare_path)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
