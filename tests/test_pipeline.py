from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.geometry import detect_bright_document_region, detect_document_corners  # noqa: E402
from clearscan_cv.pipeline import enhance_image, process_file  # noqa: E402


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

    def test_enhancement_binary_output(self) -> None:
        result = enhance_image(make_synthetic_document(), mode="binary")
        self.assertEqual(result.image.ndim, 2)
        self.assertIn("quality", result.report)
        self.assertGreater(result.image.shape[0], 100)
        self.assertGreater(result.image.shape[1], 100)

    def test_process_file_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jpg"
            cv2.imwrite(str(input_path), make_synthetic_document())

            report = process_file(input_path, tmp_path / "out", mode="gray", side_by_side=True)

            self.assertTrue(Path(str(report["output_path"])).exists())
            self.assertTrue(Path(str(report["report_path"])).exists())
            self.assertTrue(Path(str(report["comparison_path"])).exists())

    def test_static_app_generates_pdf_on_device(self) -> None:
        html = (ROOT / "src" / "clearscan_cv" / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn("createPdfBlob", html)
        self.assertIn("buildAnalysisCanvas", html)
        self.assertIn("warpPerspectiveCanvas", html)
        self.assertIn("edgeCanvas", html)
        self.assertIn("qualityScore", html)
        self.assertIn("平均品質", html)
        self.assertIn("PDFファイル名", html)
        self.assertIn("PDFを共有", html)
        self.assertIn("アプリを追加", html)
        self.assertIn("白黒スキャン", html)
        self.assertIn("角調整", html)
        self.assertIn("corner-editor-canvas", html)
        self.assertIn("この四隅で生成", html)
        self.assertIn("manualCorners", html)
        self.assertIn("smoothTileValues", html)
        self.assertIn("interpolatedTileValue", html)
        self.assertIn("inkLum", html)
        self.assertIn("件目を上へ移動", html)
        self.assertIn("端末内でPDFを生成中", html)
        self.assertIn("PocketCV 画像処理レポート", html)
        self.assertIn("encodePdfImageFromCanvas", html)
        self.assertIn("fitCanvasForPdf", html)
        self.assertIn("CompressionStream", html)
        self.assertIn("/FlateDecode", html)
        self.assertIn("/Info", html)
        self.assertIn("navigator.mediaDevices", html)
        self.assertIn("navigator.share", html)
        self.assertIn("serviceWorker", html)
        self.assertIn("beforeinstallprompt", html)
        self.assertIn("canvasToBlob", html)
        self.assertNotIn("/api/pdf", html)

    def test_static_app_has_pwa_worker(self) -> None:
        worker = (ROOT / "src" / "clearscan_cv" / "static" / "sw.js").read_text(encoding="utf-8")

        self.assertIn("CACHE_NAME", worker)
        self.assertIn("install", worker)
        self.assertIn("fetch", worker)


if __name__ == "__main__":
    unittest.main()
