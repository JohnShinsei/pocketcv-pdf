from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.geometry import detect_bright_document_region, detect_connected_document_region, detect_document_corners, detect_hough_document_region  # noqa: E402
from clearscan_cv.pipeline import deskew_by_text_lines, enhance_image, estimate_textline_skew, process_file  # noqa: E402


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


def make_soft_antialiased_text_page() -> np.ndarray:
    image = np.full((520, 760, 3), 246, dtype=np.uint8)
    for index in range(9):
        y = 80 + index * 45
        cv2.putText(image, f"Sample text line {index + 1}", (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (92, 92, 92), 2, cv2.LINE_AA)
        cv2.line(image, (70, y + 15), (620, y + 17), (130, 130, 130), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0)


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

    def test_estimates_textline_skew(self) -> None:
        page = make_skewed_text_page(3.0)
        angle, confidence = estimate_textline_skew(page)
        corrected, report = deskew_by_text_lines(page)
        corrected_angle, _ = estimate_textline_skew(corrected)

        self.assertAlmostEqual(abs(angle), 3.0, delta=0.75)
        self.assertGreater(confidence, 1.03)
        self.assertEqual(report["angle"], angle)
        self.assertLess(abs(corrected_angle), 1.0)

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

    def test_binary_enhancement_keeps_antialias_text_from_becoming_bold(self) -> None:
        result = enhance_image(make_soft_antialiased_text_page(), mode="binary", auto_warp=False)
        black_ratio = float(np.mean(result.image < 128))

        self.assertGreater(black_ratio, 0.01)
        self.assertLess(black_ratio, 0.055)

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

    def test_static_app_generates_pdf_on_device(self) -> None:
        html = (ROOT / "src" / "clearscan_cv" / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("createPdfBlob", html)
        self.assertIn("buildAnalysisCanvas", html)
        self.assertIn("warpPerspectiveCanvas", html)
        self.assertIn("edgeCanvas", html)
        self.assertIn("qualityScore", html)
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
        self.assertIn("downloadOcrText", html)
        self.assertIn("文書復元", html)
        self.assertIn("recoverLayoutMarkdown", html)
        self.assertIn("detectColumnLayout", html)
        self.assertIn("downloadLayoutText", html)
        self.assertIn("復元Markdown", html)
        self.assertIn("PDFを共有", html)
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
        self.assertIn("normalizedSquaredIntegral", html)
        self.assertIn("textEdge = inkLum[pixel] > 92", html)
        self.assertIn("strokeContrast > 31", html)
        self.assertIn("workspaceEl.classList.toggle(\"has-ocr\"", html)
        self.assertIn("grid-template-columns: minmax(520px, 1fr) minmax(360px, 440px)", html)
        self.assertIn("deskewCanvasByTextLines", html)
        self.assertIn("estimateTextLineSkew", html)
        self.assertIn("文字行傾き補正", html)
        self.assertIn("件目を上へ移動", html)
        self.assertIn("端末内でPDFを生成中", html)
        self.assertIn("PocketCV 画像処理レポート", html)
        self.assertIn("encodePdfImageFromCanvas", html)
        self.assertIn("fitCanvasForPdf", html)
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
        self.assertIn("beforeinstallprompt", html)
        self.assertIn("canvasToBlob", html)
        self.assertIn("Tesseract.recognize", html)
        self.assertNotIn("/api/pdf", html)

    def test_static_app_has_pwa_worker(self) -> None:
        worker = (ROOT / "src" / "clearscan_cv" / "static" / "sw.js").read_text(encoding="utf-8")

        self.assertIn("CACHE_NAME", worker)
        self.assertIn("pocketcv-pdf-v9", worker)
        self.assertIn("install", worker)
        self.assertIn("fetch", worker)
        self.assertIn("event.request.mode === \"navigate\"", worker)


if __name__ == "__main__":
    unittest.main()
