from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.corners import parse_corner_points  # noqa: E402
from clearscan_cv.geometry import detect_bright_document_region, detect_connected_document_region, detect_document_corners, detect_hough_document_region  # noqa: E402
from clearscan_cv.dewarp import dewarp_by_textline_columns, estimate_textline_column_offsets  # noqa: E402
from clearscan_cv.pipeline import (  # noqa: E402
    MAX_PROCESS_IMAGE_EDGE,
    MAX_PROCESS_IMAGE_PIXELS,
    deskew_by_text_lines,
    enhance_image,
    estimate_hough_textline_skew,
    estimate_textline_skew,
    limit_image_resolution,
    process_file,
    rotate_image_keep_content,
)
from clearscan_cv.quality import assess_quality  # noqa: E402


def make_synthetic_document() -> np.ndarray:
    image = np.full((720, 960, 3), (36, 43, 50), dtype=np.uint8)
    document = np.array([[180, 90], [790, 130], [725, 630], [125, 580]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (238, 240, 234))
    for i in range(10):
        cv2.line(image, (240, 185 + i * 34), (665, 195 + i * 34), (55, 60, 70), 4, cv2.LINE_AA)
    return image


def make_low_contrast_document() -> np.ndarray:
    image = np.full((720, 960, 3), (105, 100, 88), dtype=np.uint8)
    document = np.array([[145, 110], [835, 150], [800, 640], [95, 610]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (172, 172, 164))
    for i in range(7):
        cv2.line(image, (230, 230 + i * 42), (670, 238 + i * 42), (118, 118, 112), 3, cv2.LINE_AA)
    image = cv2.GaussianBlur(image, (15, 15), 0)
    return image


def make_partial_bright_form() -> np.ndarray:
    image = np.full((720, 560, 3), (88, 88, 84), dtype=np.uint8)
    image[150:705, 0:550] = (214, 214, 207)
    cv2.putText(image, "HOSPITAL FORM", (130, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(image, "FORM TITLE", (160, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    for i in range(12):
        cv2.line(image, (45, 205 + i * 34), (505, 208 + i * 34), (70, 70, 70), 2, cv2.LINE_AA)
    return image


def make_connected_document_with_bright_distractor() -> np.ndarray:
    image = np.full((760, 620, 3), (54, 49, 42), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (86, 60), (190, 188, 174), -1)
    document = np.array([[86, 88], [550, 96], [610, 720], [18, 735]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (188, 188, 180))
    for i in range(13):
        cv2.line(image, (120, 190 + i * 34), (500, 186 + i * 34), (68, 68, 65), 2, cv2.LINE_AA)
    return image


def make_hough_line_document() -> np.ndarray:
    image = np.full((720, 960, 3), (112, 112, 108), dtype=np.uint8)
    corners = np.array([[168, 128], [792, 104], [838, 606], [126, 648]], dtype=np.int32)

    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        vector = end - start
        start_gap = start + vector * 0.06
        end_gap = end - vector * 0.06
        cv2.line(image, tuple(start_gap.astype(int)), tuple(end_gap.astype(int)), (236, 236, 230), 5, cv2.LINE_AA)

    for index in range(9):
        y = 190 + index * 34
        cv2.line(image, (245, y), (680, y + (index % 2)), (138, 138, 134), 2, cv2.LINE_AA)
    return image


def make_skewed_text_page(angle: float = 3.0) -> np.ndarray:
    image = np.full((680, 900, 3), 255, dtype=np.uint8)
    for i in range(14):
        y = 110 + i * 34
        cv2.line(image, (120, y), (760, y), (35, 35, 35), 3, cv2.LINE_AA)
        if i % 3 == 0:
            cv2.rectangle(image, (120, y + 10), (380, y + 16), (70, 70, 70), -1)
    center = (image.shape[1] / 2, image.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))


def make_curved_text_page(amplitude: float = 22.0) -> np.ndarray:
    image = np.full((520, 760, 3), 255, dtype=np.uint8)
    for index in range(11):
        y = 86 + index * 34
        cv2.line(image, (90, y), (660, y), (35, 35, 35), 3, cv2.LINE_AA)
        cv2.putText(image, f"CURVED LINE {index + 1}", (100, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 45), 1, cv2.LINE_AA)

    height, width = image.shape[:2]
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    shift = amplitude * np.sin((x_coords / max(1, width - 1)) * np.pi * 2.0)
    map_y = (grid_y - shift.reshape(1, -1)).astype(np.float32)
    return cv2.remap(image, grid_x.astype(np.float32), map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def make_complex_two_column_page() -> np.ndarray:
    image = np.full((520, 760, 3), 255, dtype=np.uint8)
    cv2.line(image, (380, 45), (380, 480), (35, 35, 35), 2, cv2.LINE_AA)
    for index in range(9):
        y = 80 + index * 38
        cv2.line(image, (70, y), (330, y), (35, 35, 35), 2, cv2.LINE_AA)
    for index in range(9):
        y = 118 + index * 38
        cv2.line(image, (430, y), (690, y), (35, 35, 35), 2, cv2.LINE_AA)
    return image


def make_shadowed_noisy_page() -> np.ndarray:
    rng = np.random.default_rng(7)
    height, width = 720, 920
    x_gradient = np.linspace(0, 45, width, dtype=np.float32)
    y_gradient = np.linspace(0, 24, height, dtype=np.float32)[:, None]
    paper = 235 - x_gradient - y_gradient
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, (width - 180, 180), (260, 210), -12, 0, 360, 46, -1, cv2.LINE_AA)
    gray = np.clip(paper - cv2.GaussianBlur(shadow, (151, 151), 0), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(10):
        y = 110 + i * 38
        cv2.line(image, (110, y), (770, y + (i % 2)), (38, 38, 38), 2, cv2.LINE_AA)
        cv2.putText(image, f"TEXT {i + 1:02d}", (120, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
    speckles = rng.choice(height * width, size=900, replace=False)
    flat = image.reshape(-1, 3)
    flat[speckles] = np.maximum(0, flat[speckles].astype(np.int16) - rng.integers(18, 45, size=(speckles.size, 1))).astype(np.uint8)
    return image


def make_degraded_low_contrast_page() -> np.ndarray:
    rng = np.random.default_rng(19)
    height, width = 640, 860
    x_gradient = np.linspace(0, 52, width, dtype=np.float32)
    paper = 211 - x_gradient
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, (width - 160, height // 2), (250, 280), 0, 0, 360, 42, -1, cv2.LINE_AA)
    gray = np.clip(paper - cv2.GaussianBlur(shadow, (181, 181), 0), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index in range(11):
        y = 105 + index * 38
        tone = 98 + (index % 3) * 11
        cv2.putText(image, f"LOW CONTRAST {index + 1}", (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (tone, tone, tone), 1, cv2.LINE_AA)
        cv2.line(image, (90, y + 13), (690, y + 14), (tone + 18, tone + 18, tone + 18), 1, cv2.LINE_AA)
    noise = rng.normal(0, 3.5, size=image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_heavy_shadow_page() -> np.ndarray:
    height, width = 700, 900
    x_gradient = np.linspace(0, 18, width, dtype=np.float32)
    paper = 232 - x_gradient
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, (650, 380), (320, 270), -18, 0, 360, 82, -1, cv2.LINE_AA)
    cv2.rectangle(shadow, (0, 0), (260, height), 28, -1)
    gray = np.clip(paper - shadow, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for index in range(10):
        y = 120 + index * 42
        cv2.putText(image, f"SHADOW TEXT {index + 1}", (110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (48, 48, 48), 2, cv2.LINE_AA)
        cv2.line(image, (110, y + 15), (720, y + 16), (72, 72, 72), 2, cv2.LINE_AA)
    return image


def make_soft_antialiased_text_page() -> np.ndarray:
    image = np.full((520, 760, 3), 246, dtype=np.uint8)
    for index in range(9):
        y = 80 + index * 45
        cv2.putText(image, f"Sample text line {index + 1}", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (92, 92, 92), 2, cv2.LINE_AA)
        cv2.line(image, (70, y + 15), (620, y + 17), (130, 130, 130), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_overbold_dense_text_page() -> np.ndarray:
    image = np.full((900, 760, 3), 255, dtype=np.uint8)
    for block in range(2):
        x = 20 + block * 360
        for index in range(27):
            y = 32 + index * 32
            cv2.putText(image, f"DENSE {index + 1:02d}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 4, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_near_edge_artifact_page() -> np.ndarray:
    image = make_soft_antialiased_text_page()
    cv2.rectangle(image, (8, 20), (280, 34), (0, 0, 0), -1)
    cv2.rectangle(image, (720, 150), (742, 450), (0, 0, 0), -1)
    cv2.rectangle(image, (230, 498), (620, 514), (0, 0, 0), -1)
    return image


def make_clean_form_template() -> np.ndarray:
    image = np.full((620, 820, 3), 248, dtype=np.uint8)
    cv2.rectangle(image, (54, 48), (766, 568), (230, 230, 230), 2)
    cv2.putText(image, "FORM TEMPLATE", (230, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (65, 65, 65), 2, cv2.LINE_AA)
    for row in range(6):
        y = 145 + row * 62
        cv2.line(image, (90, y), (720, y), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.rectangle(image, (90, y + 16), (210, y + 40), (120, 120, 120), 1)
        cv2.line(image, (240, y + 32), (700, y + 32), (150, 150, 150), 1, cv2.LINE_AA)
    return image


def make_shadowed_form_photo() -> np.ndarray:
    template = make_clean_form_template()
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape[:2]
    gradient = np.linspace(0, 58, width, dtype=np.float32)[None, :]
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, (width - 240, 240), (260, 190), -10, 0, 360, 52, -1, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow + gradient, (151, 151), 0)
    shaded = np.clip(gray - shadow, 0, 255).astype(np.uint8)
    return cv2.cvtColor(shaded, cv2.COLOR_GRAY2BGR)


def make_external_restorer_command(script_path: Path, *, should_fail: bool = False) -> str:
    if should_fail:
        script_path.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")
    else:
        script_path.write_text(
            "\n".join(
                [
                    "import sys",
                    "import cv2",
                    "import numpy as np",
                    "raw = np.fromfile(sys.argv[1], dtype=np.uint8)",
                    "image = cv2.imdecode(raw, cv2.IMREAD_COLOR)",
                    "if image is None:",
                    "    raise SystemExit(3)",
                    "gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                    "restored = cv2.cvtColor(np.full_like(gray, 246), cv2.COLOR_GRAY2BGR)",
                    "cv2.putText(restored, 'HOOK', (40, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (25, 25, 25), 3, cv2.LINE_AA)",
                    "ok, encoded = cv2.imencode('.png', restored)",
                    "if not ok:",
                    "    raise SystemExit(4)",
                    "encoded.tofile(sys.argv[2])",
                ]
            ),
            encoding="utf-8",
        )
    return f'"{sys.executable}" "{script_path}" {{input}} {{output}}'


def make_external_detector_command(script_path: Path, *, should_fail: bool = False) -> str:
    if should_fail:
        script_path.write_text("import sys\nsys.exit(7)\n", encoding="utf-8")
    else:
        script_path.write_text(
            "\n".join(
                [
                    "import json",
                    "import sys",
                    "payload = {",
                    "    'method': 'fake_segmentation',",
                    "    'confidence': 0.88,",
                    "    'corners': [[180, 90], [790, 130], [725, 630], [125, 580]],",
                    "}",
                    "with open(sys.argv[2], 'w', encoding='utf-8') as handle:",
                    "    json.dump(payload, handle)",
                ]
            ),
            encoding="utf-8",
        )
    return f'"{sys.executable}" "{script_path}" {{input}} {{output}}'


class PipelineTest(unittest.TestCase):
    def test_detects_document_quad(self) -> None:
        detection = detect_document_corners(make_synthetic_document())
        self.assertTrue(detection.found)
        self.assertGreater(detection.confidence, 0.45)
        self.assertEqual(len(detection.corners), 4)

    def test_brightness_fallback_detects_low_contrast_page(self) -> None:
        detection = detect_bright_document_region(make_low_contrast_document())

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "brightness_rect")
        self.assertGreater(detection.area_ratio, 0.25)

    def test_rejects_partial_bright_region_that_would_crop_header(self) -> None:
        detection = detect_document_corners(make_partial_bright_form())

        self.assertFalse(detection.found)
        self.assertEqual(detection.method, "image_border")

    def test_connected_detector_ignores_separate_bright_distractor(self) -> None:
        detection = detect_connected_document_region(make_connected_document_with_bright_distractor())

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "connected_paper")
        self.assertGreater(detection.corners[0][1], 65)

    def test_connected_detector_rejects_unsafe_top_crop(self) -> None:
        detection = detect_connected_document_region(make_partial_bright_form())

        self.assertIsNone(detection)

    def test_hough_fallback_detects_broken_document_edges(self) -> None:
        detection = detect_hough_document_region(make_hough_line_document())

        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertTrue(detection.found)
        self.assertEqual(detection.method, "hough_lines")
        self.assertGreater(detection.area_ratio, 0.42)

    def test_enhancement_binary_output(self) -> None:
        result = enhance_image(make_synthetic_document(), mode="binary")
        self.assertEqual(result.image.ndim, 2)
        self.assertIn("quality", result.report)
        self.assertGreater(result.image.shape[0], 100)
        self.assertGreater(result.image.shape[1], 100)

    def test_manual_corners_override_document_detection(self) -> None:
        manual_corners = [[180, 90], [790, 130], [725, 630], [125, 580]]
        result = enhance_image(make_synthetic_document(), mode="gray", manual_corners=manual_corners)

        self.assertTrue(result.report["manual_corners"])
        self.assertEqual(result.report["manual_corners_space"], "input")
        self.assertEqual(result.report["document_detection"]["method"], "manual_corners")  # type: ignore[index]
        self.assertEqual(result.report["source_image_size"], {"width": 960, "height": 720})
        self.assertEqual(result.report["processing_image_size"], {"width": 960, "height": 720})
        self.assertGreater(result.image.shape[0], 400)
        self.assertGreater(result.image.shape[1], 500)

    def test_parse_corner_points_accepts_text_and_json(self) -> None:
        text_points = parse_corner_points("180,90 790,130 725,630 125,580")
        json_points = parse_corner_points('[{"x": 180, "y": 90}, {"x": 790, "y": 130}, {"x": 725, "y": 630}, {"x": 125, "y": 580}]')

        self.assertEqual(text_points[0], [180.0, 90.0])
        self.assertEqual(json_points[2], [725.0, 630.0])

    def test_limits_large_python_inputs_like_web_pipeline(self) -> None:
        large = np.zeros((4200, 3100, 3), dtype=np.uint8)
        limited = limit_image_resolution(large)

        self.assertLessEqual(max(limited.shape[:2]), MAX_PROCESS_IMAGE_EDGE)
        self.assertLessEqual(limited.shape[0] * limited.shape[1], MAX_PROCESS_IMAGE_PIXELS)

    def test_estimates_textline_skew(self) -> None:
        page = make_skewed_text_page(3.0)
        angle, confidence = estimate_textline_skew(page)
        corrected, report = deskew_by_text_lines(page)
        corrected_angle, _ = estimate_textline_skew(corrected)

        self.assertAlmostEqual(abs(angle), 3.0, delta=0.75)
        self.assertGreater(confidence, 1.03)
        self.assertEqual(report["angle"], angle)
        self.assertLess(abs(corrected_angle), 1.0)

    def test_hough_textline_skew_ignores_vertical_rules(self) -> None:
        page = rotate_image_keep_content(make_complex_two_column_page(), 3.0)
        angle, confidence = estimate_hough_textline_skew(page)
        corrected = rotate_image_keep_content(page, angle)
        corrected_angle, _ = estimate_hough_textline_skew(corrected)

        self.assertAlmostEqual(angle, -3.0, delta=0.65)
        self.assertGreater(confidence, 1.04)
        self.assertLess(abs(corrected_angle), 0.75)

    def test_textline_dewarp_reduces_column_offsets(self) -> None:
        curved = make_curved_text_page()
        _, before_offsets, before_confidence = estimate_textline_column_offsets(curved)
        result = dewarp_by_textline_columns(curved)
        _, after_offsets, after_confidence = estimate_textline_column_offsets(result.image)

        self.assertTrue(result.report["applied"])
        self.assertGreater(before_confidence, 0.08)
        self.assertGreater(after_confidence, 0.08)
        self.assertGreater(float(np.max(np.abs(before_offsets))), 10.0)
        self.assertLess(float(np.max(np.abs(after_offsets))), float(np.max(np.abs(before_offsets))) * 0.72)

    def test_textline_dewarp_skips_complex_two_column_layout(self) -> None:
        result = dewarp_by_textline_columns(make_complex_two_column_page())

        self.assertFalse(result.report["applied"])
        self.assertEqual(result.report["reason"], "flat_or_low_confidence")

    def test_binary_enhancement_suppresses_shadow_noise(self) -> None:
        result = enhance_image(make_shadowed_noisy_page(), mode="binary", auto_warp=False)
        binary = result.image
        blank_region = binary[460:650, 110:810]
        text_region = binary[90:500, 100:790]

        self.assertGreater(float(np.mean(blank_region > 245)), 0.965)
        self.assertGreater(float(np.mean(text_region < 128)), 0.012)
        self.assertLess(float(np.mean(binary < 128)), 0.075)
        self.assertIn("textline_deskew", result.report["pipeline"])

    def test_gatos_sauvola_enhancement_keeps_low_contrast_text(self) -> None:
        result = enhance_image(make_degraded_low_contrast_page(), mode="binary", auto_warp=False)
        binary = result.image
        text_region = binary[80:535, 70:735]
        blank_region = binary[520:625, 95:760]

        self.assertGreater(float(np.mean(text_region < 128)), 0.008)
        self.assertGreater(float(np.mean(blank_region > 245)), 0.96)
        self.assertLess(float(np.mean(binary < 128)), 0.07)

    def test_frequency_deshadow_reduces_heavy_shadow_without_flattening_text(self) -> None:
        page = make_heavy_shadow_page()
        raw = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        corrected = enhance_image(page, mode="gray", auto_warp=False).image
        lit_region = (slice(70, 220), slice(310, 470))
        shadow_region = (slice(420, 620), slice(650, 835))
        text_region = corrected[90:550, 90:760]

        before_delta = abs(float(np.mean(raw[lit_region])) - float(np.mean(raw[shadow_region])))
        after_delta = abs(float(np.mean(corrected[lit_region])) - float(np.mean(corrected[shadow_region])))

        self.assertGreater(before_delta, 50)
        self.assertLess(after_delta, 18)
        self.assertGreater(float(np.std(text_region)), 35)

    def test_auto_mode_selects_more_readable_scan_variant(self) -> None:
        result = enhance_image(make_heavy_shadow_page(), mode="auto", auto_warp=False)

        self.assertEqual(result.report["mode"], "auto")
        self.assertEqual(result.report["selected_mode"], "gray")
        self.assertEqual(result.report["auto_selection"]["selected_mode"], "gray")  # type: ignore[index]
        self.assertEqual(result.report["quality_diagnostics"]["status"], "ready")  # type: ignore[index]
        self.assertEqual(result.image.ndim, 2)

    def test_external_restorer_hook_applies_between_geometry_and_enhancement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = make_external_restorer_command(Path(tmp) / "fake_restorer.py")
            result = enhance_image(make_synthetic_document(), mode="gray", auto_warp=False, external_restorer_command=command)

        report = result.report["external_restorer"]
        self.assertTrue(report["applied"])  # type: ignore[index]
        self.assertEqual(report["method"], "external_command")  # type: ignore[index]
        self.assertIn("external_restorer", result.report["pipeline"])
        self.assertIn("output_size", report)  # type: ignore[operator]

    def test_external_restorer_hook_falls_back_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = make_external_restorer_command(Path(tmp) / "failing_restorer.py", should_fail=True)
            result = enhance_image(make_synthetic_document(), mode="gray", auto_warp=False, external_restorer_command=command)

        report = result.report["external_restorer"]
        self.assertFalse(report["applied"])  # type: ignore[index]
        self.assertEqual(report["reason"], "nonzero_exit")  # type: ignore[index]
        self.assertEqual(report["returncode"], 7)  # type: ignore[index]
        self.assertGreater(result.image.shape[0], 100)

    def test_external_detector_hook_can_supply_document_corners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = make_external_detector_command(Path(tmp) / "fake_detector.py")
            result = enhance_image(make_synthetic_document(), mode="gray", external_detector_command=command)

        detector_report = result.report["external_detector"]
        document_detection = result.report["document_detection"]
        self.assertTrue(detector_report["applied"])  # type: ignore[index]
        self.assertTrue(detector_report["parsed"])  # type: ignore[index]
        self.assertEqual(detector_report["detector_method"], "fake_segmentation")  # type: ignore[index]
        self.assertEqual(document_detection["method"], "external_detector")  # type: ignore[index]
        self.assertEqual(document_detection["confidence"], 0.88)  # type: ignore[index]
        self.assertIn("external_detector", result.report["pipeline"])
        self.assertGreater(result.image.shape[0], 400)
        self.assertGreater(result.image.shape[1], 500)

    def test_external_detector_hook_falls_back_to_opencv_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = make_external_detector_command(Path(tmp) / "failing_detector.py", should_fail=True)
            result = enhance_image(make_synthetic_document(), mode="gray", external_detector_command=command)

        detector_report = result.report["external_detector"]
        document_detection = result.report["document_detection"]
        self.assertFalse(detector_report["applied"])  # type: ignore[index]
        self.assertEqual(detector_report["reason"], "nonzero_exit")  # type: ignore[index]
        self.assertEqual(detector_report["returncode"], 7)  # type: ignore[index]
        self.assertNotEqual(document_detection["method"], "external_detector")  # type: ignore[index]
        self.assertIn("external_detector_fallback", result.report["pipeline"])
        self.assertGreater(result.image.shape[0], 100)

    def test_template_guided_illumination_uses_form_template(self) -> None:
        template = make_clean_form_template()
        page = make_shadowed_form_photo()
        result = enhance_image(page, mode="gray", auto_warp=False, auto_dewarp=False, template_image=template)

        report = result.report["template_guided_illumination"]
        self.assertTrue(report["applied"])  # type: ignore[index]
        self.assertEqual(report["method"], "template_guided_illumination")  # type: ignore[index]
        self.assertIn("template_guided_illumination", result.report["pipeline"])
        self.assertLess(float(report["corrected_background_range"]), float(report["source_background_range"]) * 0.7)  # type: ignore[index]

    def test_process_file_accepts_template_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "form.jpg"
            template_path = tmp_path / "template.png"
            cv2.imwrite(str(input_path), make_shadowed_form_photo())
            cv2.imwrite(str(template_path), make_clean_form_template())

            report = process_file(input_path, tmp_path / "out", mode="gray", auto_dewarp=False, template_path=template_path)

            self.assertEqual(report["template_path"], str(template_path))
            self.assertTrue(report["template_guided_illumination"]["applied"])  # type: ignore[index]

    def test_quality_metrics_report_shadow_and_boldness_risk(self) -> None:
        page = make_heavy_shadow_page()
        raw = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        corrected = enhance_image(page, mode="gray", auto_warp=False).image
        before = assess_quality(raw)
        after = assess_quality(corrected)

        self.assertIn("shadow_residual", after)
        self.assertIn("shadow_score", after)
        self.assertIn("ink_density", after)
        self.assertIn("boldness_risk", after)
        self.assertGreater(float(before["shadow_residual"]), 50.0)
        self.assertLess(float(after["shadow_residual"]), float(before["shadow_residual"]) * 0.35)
        self.assertLess(float(after["boldness_risk"]), 0.2)

        report = enhance_image(page, mode="gray", auto_warp=False).report
        self.assertIn("quality_diagnostics", report)
        self.assertIn(report["quality_diagnostics"]["status"], {"ready", "review", "retake"})  # type: ignore[index]
        self.assertIn("issues", report["quality_diagnostics"])  # type: ignore[operator]

    def test_binary_enhancement_keeps_antialias_text_from_becoming_bold(self) -> None:
        result = enhance_image(make_soft_antialiased_text_page(), mode="binary", auto_warp=False)
        black_ratio = float(np.mean(result.image < 128))

        self.assertGreater(black_ratio, 0.01)
        self.assertLess(black_ratio, 0.055)
        self.assertLess(float(assess_quality(result.image)["boldness_risk"]), 0.2)

    def test_binary_enhancement_reduces_overbold_dense_text(self) -> None:
        page = make_overbold_dense_text_page()
        raw_gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        raw_quality = assess_quality(page)
        result = enhance_image(page, mode="binary", auto_warp=False)
        quality = assess_quality(result.image)
        black_ratio = float(np.mean(result.image < 128))

        self.assertGreater(float(raw_quality["boldness_risk"]), 0.5)
        self.assertLess(black_ratio, float(np.mean(raw_gray < 128)) * 0.7)
        self.assertGreater(black_ratio, 0.04)
        self.assertLess(float(quality["boldness_risk"]), 0.2)
        self.assertGreater(float(quality["edge_density"]), 0.03)

    def test_binary_enhancement_removes_near_edge_artifacts(self) -> None:
        result = enhance_image(make_near_edge_artifact_page(), mode="binary", auto_warp=False)
        binary = result.image

        self.assertGreater(float(np.mean(binary[15:42, 0:310] > 245)), 0.98)
        self.assertGreater(float(np.mean(binary[130:470, 705:755] > 245)), 0.98)
        self.assertGreater(float(np.mean(binary[488:519, 210:650] > 245)), 0.98)
        self.assertGreater(float(np.mean(binary[58:455, 55:650] < 128)), 0.01)

    def test_process_file_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jpg"
            cv2.imwrite(str(input_path), make_synthetic_document())

            report = process_file(input_path, tmp_path / "out", mode="gray", side_by_side=True)

            self.assertTrue(Path(str(report["output_path"])).exists())
            self.assertTrue(Path(str(report["report_path"])).exists())
            self.assertTrue(Path(str(report["comparison_path"])).exists())

    def test_process_file_reads_unicode_windows_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "病例截图.png"
            encoded = cv2.imencode(".png", make_synthetic_document())[1]
            encoded.tofile(str(input_path))

            report = process_file(input_path, tmp_path / "out", mode="binary")

            self.assertTrue(Path(str(report["output_path"])).exists())

    def test_cli_accepts_manual_corners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jpg"
            cv2.imwrite(str(input_path), make_synthetic_document())

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clearscan_cv.cli",
                    str(input_path),
                    "--out",
                    str(tmp_path / "out"),
                    "--mode",
                    "gray",
                    "--corners",
                    "180,90 790,130 725,630 125,580",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["manual_corners"])
            self.assertEqual(payload["manual_corners_space"], "input")
            self.assertEqual(payload["document_detection"]["method"], "manual_corners")
            self.assertEqual(payload["source_image_size"], {"width": 960, "height": 720})

    def test_cli_accepts_external_detector_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jpg"
            cv2.imwrite(str(input_path), make_synthetic_document())
            detector_command = make_external_detector_command(tmp_path / "fake_detector.py")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clearscan_cv.cli",
                    str(input_path),
                    "--out",
                    str(tmp_path / "out"),
                    "--mode",
                    "gray",
                    "--external-detector-command",
                    detector_command,
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["external_detector"]["applied"])
            self.assertEqual(payload["external_detector"]["detector_method"], "fake_segmentation")
            self.assertEqual(payload["document_detection"]["method"], "external_detector")

    def test_static_app_generates_pdf_on_device(self) -> None:
        html = (ROOT / "src" / "clearscan_cv" / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("createPdfBlob", html)
        self.assertIn("buildAnalysisCanvas", html)
        self.assertIn("warpPerspectiveCanvas", html)
        self.assertIn("edgeCanvas", html)
        self.assertIn('option value="auto"', html)
        self.assertIn("おすすめ自動", html)
        self.assertIn("chooseAutoScanCandidate", html)
        self.assertIn("gray_preserves_fragile_text", html)
        self.assertIn("gray_preserves_broken_strokes", html)
        self.assertIn("buildColorGeometryResult", html)
        self.assertIn('renderMode === "color"', html)
        self.assertIn("? buildColorGeometryResult(corrected.canvas, corrected.detection)", html)
        self.assertIn("dewarpCanvasByTextlineColumns", html)
        self.assertIn("estimateTextlineColumnOffsetsCanvas", html)
        self.assertIn("bestVerticalProjectionShift", html)
        self.assertIn("normalizeTextlineProjection", html)
        self.assertIn("formatDewarpOffset", html)
        self.assertIn("selectedMode", html)
        self.assertIn("qualityScore", html)
        self.assertIn("assessOutputScanMetrics", html)
        self.assertIn("scanQualityAdvice", html)
        self.assertIn("shadowResidual", html)
        self.assertIn("shadowScore", html)
        self.assertIn("inkDensity", html)
        self.assertIn("boldnessRisk", html)
        self.assertIn("brokenStrokeRisk", html)
        self.assertIn("estimateBinaryFragmentMetrics", html)
        self.assertIn("fragmentEdgeRatio", html)
        self.assertIn("文字欠けリスク", html)
        self.assertIn("文字欠け注意", html)
        self.assertIn("reduceOverboldBinaryStrokes", html)
        self.assertIn("buildFragileBinaryComponentMask", html)
        self.assertIn("binaryMaskEdgeDensity", html)
        self.assertIn("strokeThinApplied", html)
        self.assertIn("筆画細化", html)
        self.assertIn("paperGrayResidue", html)
        self.assertIn("MAX_SOURCE_IMAGE_EDGE = 3200", html)
        self.assertIn("MAX_SOURCE_IMAGE_PIXELS = 6500000", html)
        self.assertIn("PDF_MAX_IMAGE_EDGE = 3200", html)
        self.assertIn("PDF_MAX_IMAGE_PIXELS = 6500000", html)
        self.assertNotIn("MAX_IMAGE_DIMENSION", html)
        self.assertIn("formatResolution", html)
        self.assertIn("sourceOriginalWidth", html)
        self.assertIn("平均品質", html)
        self.assertIn("PDFファイル名", html)
        self.assertIn("スキャンPDFを生成", html)
        self.assertIn("画像保存", html)
        self.assertIn("解析レポート", html)
        self.assertIn("OCR実行", html)
        self.assertIn("TESSERACT_SCRIPT_URL", html)
        self.assertIn("runOcr", html)
        self.assertIn("buildOcrCanvas", html)
        self.assertIn("const dewarped = dewarpCanvasByTextlineColumns(corrected.canvas, corrected.detection);", html)
        self.assertIn("const deskewed = deskewCanvasByTextLines(dewarped.canvas, dewarped.detection);", html)
        self.assertIn("downloadOcrText", html)
        self.assertIn("文書復元", html)
        self.assertIn("recoverLayoutMarkdown", html)
        self.assertIn("detectColumnLayout", html)
        self.assertIn("downloadLayoutText", html)
        self.assertIn("downloadDocxDocument", html)
        self.assertIn("buildDocxBytes", html)
        self.assertIn("buildZipStore", html)
        self.assertIn("crc32", html)
        self.assertIn("復元Markdown", html)
        self.assertIn("DOCX保存", html)
        self.assertIn('docxDownloadButton.addEventListener("click", downloadDocxDocument)', html)
        self.assertIn("PDFを共有", html)
        self.assertIn("new File([pdfBlob], filename, { type: \"application/pdf\" })", html)
        self.assertIn("files: [pdfFile]", html)
        self.assertNotIn("url: location.href", html)
        self.assertNotIn("url: window.location.href", html)
        self.assertIn("アプリを追加", html)
        self.assertIn("白黒スキャン", html)
        self.assertIn("角調整", html)
        self.assertIn("pending-corner-canvas", html)
        self.assertIn("pendingProcessButton", html)
        self.assertIn("openPendingScan", html)
        self.assertIn("queueBlobForCornerEditing", html)
        self.assertIn("processPendingScan", html)
        self.assertIn("editPageInIntake", html)
        self.assertIn("四隅を調整してから生成してください", html)
        self.assertIn("この四隅で生成", html)
        self.assertNotIn("async function addBlob", html)
        self.assertIn("corner-editor-canvas", html)
        self.assertIn("manualCorners", html)
        self.assertIn("smoothTileValues", html)
        self.assertIn("interpolatedTileValue", html)
        self.assertIn("inkLum", html)
        self.assertIn("cleanupBinaryImageData", html)
        self.assertIn("shouldRejectPartialBrightQuad", html)
        self.assertIn("shouldRejectUnsafeFullFrameQuad", html)
        self.assertIn("detectConnectedPaperQuad", html)
        self.assertIn("detectHoughLineQuad", html)
        self.assertIn('method: "hough_lines"', html)
        self.assertIn("connected_paper", html)
        self.assertIn("orderQuadPoints", html)
        self.assertIn("* 0.006", html)
        self.assertIn("buildIntegralImage", html)
        self.assertIn("localStatsFromIntegrals", html)
        self.assertIn("formBinaryMode", html)
        self.assertIn("autoConfidence", html)
        self.assertIn("autoFound", html)
        self.assertIn("paperNoiseGuard", html)
        self.assertIn("strokeContrast", html)
        self.assertIn("sauvolaThresholdValue", html)
        self.assertIn("useGatosSauvola", html)
        self.assertIn("useFrequencyDeshadow", html)
        self.assertIn("rawHistogram", html)
        self.assertIn('const outputMode = renderMode === "report" ? "report" : enhanced.metrics.selectedMode || renderMode;', html)
        self.assertIn('const previewType = outputMode === "binary" || outputMode === "edges" ? "image/png" : "image/jpeg";', html)
        self.assertIn("pdfMode: outputMode", html)
        histogram_definition = html.find("const histogram = new Array(256).fill(0);")
        histogram_use = html.find(
            "const connectedCandidate = detectConnectedPaperQuad(sourceCanvas, image, step, histogram, samples)"
        )
        self.assertNotEqual(histogram_definition, -1)
        self.assertNotEqual(histogram_use, -1)
        self.assertLess(histogram_definition, histogram_use)
        normalized_histogram_definition = html.find("const normalizedHistogram = new Array(256).fill(0);")
        normalized_histogram_use = html.find("otsuThreshold(normalizedHistogram, pixelCount)")
        self.assertNotEqual(normalized_histogram_definition, -1)
        self.assertNotEqual(normalized_histogram_use, -1)
        self.assertLess(normalized_histogram_definition, normalized_histogram_use)
        self.assertIn("normalizedSquaredIntegral", html)
        self.assertIn("antialiasGuard", html)
        self.assertIn("textEdge = inkLum[pixel] > 118", html)
        self.assertIn("weakTextEdge", html)
        self.assertIn("strokeContrast > 30", html)
        self.assertIn("workspaceEl.classList.toggle(\"has-ocr\"", html)
        self.assertIn("grid-template-columns: minmax(520px, 1fr) minmax(360px, 440px)", html)
        self.assertIn("deskewCanvasByTextLines", html)
        self.assertIn("estimateTextLineSkew", html)
        self.assertIn("estimateHoughTextLineSkew", html)
        self.assertIn("dewarpApplied", html)
        self.assertIn("dewarpMaxOffset", html)
        self.assertIn("dewarpReason", html)
        self.assertIn("文字行カーブ補正", html)
        self.assertIn("カーブ補正", html)
        self.assertIn("nearOuterEdge", html)
        self.assertIn("removeNearEdgeBlob", html)
        cleanup_call = html.find("cleanupBinaryImageData(data, sourceCanvas.width, sourceCanvas.height);")
        thinning_call = html.find("strokeThinReport = reduceOverboldBinaryStrokes(data, sourceCanvas.width, sourceCanvas.height, tileSize);")
        scan_metrics_call = html.find("const scanMetrics = assessOutputScanMetrics(data, sourceCanvas.width, sourceCanvas.height, tileSize);")
        self.assertNotEqual(cleanup_call, -1)
        self.assertNotEqual(thinning_call, -1)
        self.assertNotEqual(scan_metrics_call, -1)
        self.assertLess(cleanup_call, thinning_call)
        self.assertLess(thinning_call, scan_metrics_call)
        self.assertIn("文字行傾き補正", html)
        self.assertIn("影ムラ残り", html)
        self.assertIn("文字の墨量", html)
        self.assertIn("加太りリスク", html)
        self.assertIn("診断", html)
        self.assertIn("推奨操作", html)
        self.assertIn("四隅を再調整", html)
        self.assertIn("文字太り注意", html)
        self.assertIn("件目を上へ移動", html)
        self.assertIn("端末内でPDFを生成中", html)
        self.assertIn("PocketCV 画像処理レポート", html)
        self.assertIn("encodePdfImageFromCanvas", html)
        self.assertIn("fitCanvasForPdf", html)
        self.assertIn("pdfHiddenTextCommands", html)
        self.assertIn("ocrPageForPdf", html)
        self.assertIn("3 Tr", html)
        self.assertIn("/Font << /F1", html)
        self.assertIn("createPdfBlob(state.pages, pdfTitle, state.ocrPages)", html)
        self.assertIn("OCR文字層付き", html)
        self.assertIn("CompressionStream", html)
        self.assertIn("/FlateDecode", html)
        self.assertIn("/Info", html)
        self.assertIn("navigator.canShare && navigator.canShare(shareData)", html)
        self.assertIn('new File([pdfBlob], filename, { type: "application/pdf" })', html)
        self.assertIn("PDFファイル共有に非対応", html)
        self.assertNotIn('text: "PocketCV PDFで生成したスキャンPDF"', html)
        self.assertIn('id="camera-file"', html)
        self.assertIn('capture="environment"', html)
        self.assertIn('<input id="file" class="file" type="file" accept="image/*" multiple />', html)
        self.assertIn("アルバムから選択", html)
        self.assertIn("openCameraCapture", html)
        self.assertNotIn("getUserMedia", html)
        self.assertNotIn("navigator.mediaDevices", html)
        self.assertNotIn("<video", html)
        self.assertNotIn("撮影して追加", html)
        self.assertIn("navigator.share", html)
        self.assertIn("serviceWorker", html)
        self.assertIn('navigator.serviceWorker.register("sw.js", { updateViaCache: "none" })', html)
        self.assertIn("registration.update()", html)
        self.assertIn("beforeinstallprompt", html)
        self.assertIn("canvasToBlob", html)
        self.assertIn("Tesseract.recognize", html)
        self.assertNotIn("/api/pdf", html)

    def test_static_app_has_pwa_worker(self) -> None:
        worker = (ROOT / "src" / "clearscan_cv" / "static" / "sw.js").read_text(encoding="utf-8")

        self.assertIn("CACHE_NAME", worker)
        self.assertIn("pocketcv-pdf-v22", worker)
        self.assertIn("install", worker)
        self.assertIn("fetch", worker)
        self.assertIn("event.request.mode === \"navigate\"", worker)
        self.assertIn("self.skipWaiting()", worker)
        self.assertIn("self.clients.claim()", worker)


if __name__ == "__main__":
    unittest.main()
