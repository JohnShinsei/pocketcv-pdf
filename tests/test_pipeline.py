from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.geometry import detect_document_corners  # noqa: E402
from clearscan_cv.pipeline import enhance_image, process_file  # noqa: E402


def make_synthetic_document() -> np.ndarray:
    image = np.full((720, 960, 3), (36, 43, 50), dtype=np.uint8)
    document = np.array([[180, 90], [790, 130], [725, 630], [125, 580]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (238, 240, 234))
    for i in range(10):
        cv2.line(image, (240, 185 + i * 34), (665, 195 + i * 34), (55, 60, 70), 4, cv2.LINE_AA)
    return image


class PipelineTest(unittest.TestCase):
    def test_detects_document_quad(self) -> None:
        detection = detect_document_corners(make_synthetic_document())
        self.assertTrue(detection.found)
        self.assertGreater(detection.confidence, 0.45)
        self.assertEqual(len(detection.corners), 4)

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
        self.assertIn("端末内でPDFを生成中", html)
        self.assertIn("PocketCV 画像処理レポート", html)
        self.assertIn("navigator.mediaDevices", html)
        self.assertIn("canvasToBlob", html)
        self.assertNotIn("/api/pdf", html)


if __name__ == "__main__":
    unittest.main()
