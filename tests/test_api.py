from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import sys
import unittest
import warnings

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HAS_API_DEPS = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
    and importlib.util.find_spec("multipart") is not None
)

if HAS_API_DEPS:
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient` is deprecated.*")
    from fastapi.testclient import TestClient  # type: ignore[import-not-found]

    import clearscan_cv.api as api_module  # noqa: E402
    from clearscan_cv.api import app  # noqa: E402
    from clearscan_cv.ocr import OcrLine, OcrResult  # noqa: E402


def make_api_document() -> bytes:
    image = np.full((420, 620, 3), (40, 46, 52), dtype=np.uint8)
    document = np.array([[74, 42], [538, 64], [568, 372], [52, 360]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (236, 238, 232))
    cv2.putText(image, "API SCAN", (150, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    for index in range(7):
        y = 166 + index * 28
        cv2.line(image, (126, y), (480, y + (index % 2)), (52, 52, 52), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode test image")
    return encoded.tobytes()


def make_api_template() -> bytes:
    image = np.full((420, 620, 3), 248, dtype=np.uint8)
    cv2.rectangle(image, (74, 42), (548, 374), (180, 180, 180), 2)
    cv2.putText(image, "API TEMPLATE", (145, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (70, 70, 70), 2, cv2.LINE_AA)
    for index in range(7):
        y = 166 + index * 28
        cv2.line(image, (126, y), (480, y), (120, 120, 120), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode test template")
    return encoded.tobytes()


@unittest.skipUnless(HAS_API_DEPS, "Install API dependencies with: pip install -e .[api]")
class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_local_backend_page_uses_process_api(self) -> None:
        response = self.client.get("/local")

        self.assertEqual(response.status_code, 200)
        self.assertIn("PocketCV PDF Local", response.text)
        self.assertIn("/api/process", response.text)
        self.assertIn("Python/OpenCV", response.text)

    def test_process_endpoint_returns_scan_pdf_and_manual_corner_report(self) -> None:
        response = self.client.post(
            "/api/process",
            files={"file": ("api-input.jpg", make_api_document(), "image/jpeg")},
            data={
                "mode": "gray",
                "pdf": "true",
                "readability": "true",
                "corners": "74,42 538,64 568,372 52,360",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        image_payload = base64.b64decode(payload["image_base64"])
        pdf_payload = base64.b64decode(payload["pdf_base64"])

        self.assertGreater(len(image_payload), 100)
        self.assertTrue(pdf_payload.startswith(b"%PDF-1.4"))
        self.assertFalse(payload["pdf_searchable"])
        self.assertIn("readability", payload)
        self.assertTrue(payload["report"]["manual_corners"])
        self.assertEqual(payload["report"]["manual_corners_space"], "input")
        self.assertEqual(payload["report"]["document_detection"]["method"], "manual_corners")

    def test_process_endpoint_rejects_invalid_corner_space(self) -> None:
        response = self.client.post(
            "/api/process",
            files={"file": ("api-input.jpg", make_api_document(), "image/jpeg")},
            data={
                "mode": "gray",
                "corners": "74,42 538,64 568,372 52,360",
                "corners_space": "screen",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("corners_space", response.json()["detail"])

    def test_process_endpoint_accepts_template_file(self) -> None:
        response = self.client.post(
            "/api/process",
            files={
                "file": ("api-input.jpg", make_api_document(), "image/jpeg"),
                "template_file": ("template.png", make_api_template(), "image/png"),
            },
            data={
                "mode": "gray",
                "readability": "true",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["report"]["template_guided_illumination"]["applied"])
        self.assertEqual(payload["report"]["template_guided_illumination"]["method"], "template_guided_illumination")

    def test_process_batch_endpoint_returns_multi_page_pdf(self) -> None:
        first = make_api_document()
        second = make_api_document()

        response = self.client.post(
            "/api/process-batch",
            files=[
                ("files", ("page-one.jpg", first, "image/jpeg")),
                ("files", ("page-two.jpg", second, "image/jpeg")),
            ],
            data={
                "mode": "gray",
                "pdf": "true",
                "readability": "true",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        pdf_payload = base64.b64decode(payload["pdf_base64"])

        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(len(payload["pages"]), 2)
        self.assertEqual(payload["pages"][0]["page_index"], 1)
        self.assertIn("readability", payload["pages"][0])
        self.assertTrue(base64.b64decode(payload["pages"][0]["image_base64"]).startswith(b"\x89PNG"))
        self.assertTrue(pdf_payload.startswith(b"%PDF-1.4"))
        self.assertIn(b"/Count 2", pdf_payload)
        self.assertEqual(pdf_payload.count(b"/Type /Page /Parent"), 2)

    def test_process_batch_endpoint_returns_searchable_pdf_and_layout_with_mock_ocr(self) -> None:
        original_recognize = api_module.recognize_image

        def fake_recognize(image: np.ndarray, language: str = "jpn+eng", engine: str = "auto") -> OcrResult:
            height, width = image.shape[:2]
            line = OcrLine("API OCR", 97.0, (max(1, width // 5), max(1, height // 5), max(1, width // 3), 24))
            return OcrResult("fake", language, "API OCR", 97.0, width, height, [line])

        api_module.recognize_image = fake_recognize
        try:
            response = self.client.post(
                "/api/process-batch",
                files=[
                    ("files", ("page-one.jpg", make_api_document(), "image/jpeg")),
                    ("files", ("page-two.jpg", make_api_document(), "image/jpeg")),
                ],
                data={
                    "mode": "gray",
                    "searchable_pdf": "true",
                    "ocr": "true",
                    "layout": "true",
                    "docx": "true",
                    "readability": "true",
                },
            )
        finally:
            api_module.recognize_image = original_recognize

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        pdf_payload = base64.b64decode(payload["pdf_base64"])
        docx_payload = base64.b64decode(payload["docx_base64"])

        self.assertTrue(payload["pdf_searchable"])
        self.assertEqual(payload["page_count"], 2)
        self.assertEqual(payload["pages"][0]["ocr"]["text"], "API OCR")
        self.assertIn("layout_markdown", payload["pages"][0])
        self.assertIn("Page 1", payload["layout_markdown"])
        self.assertIn("readability", payload["pages"][0])
        self.assertTrue(pdf_payload.startswith(b"%PDF-1.4"))
        self.assertEqual(pdf_payload.count(b"3 Tr"), 2)
        self.assertTrue(docx_payload.startswith(b"PK"))

    def test_process_endpoint_quality_diagnostics_include_low_ocr_confidence(self) -> None:
        original_recognize = api_module.recognize_image

        def fake_low_confidence(image: np.ndarray, language: str = "jpn+eng", engine: str = "auto") -> OcrResult:
            height, width = image.shape[:2]
            line = OcrLine("NOISY OCR", 31.0, (max(1, width // 5), max(1, height // 5), max(1, width // 3), 24))
            return OcrResult("fake", language, "NOISY OCR", 31.0, width, height, [line])

        api_module.recognize_image = fake_low_confidence
        try:
            response = self.client.post(
                "/api/process",
                files={"file": ("api-input.jpg", make_api_document(), "image/jpeg")},
                data={
                    "mode": "gray",
                    "ocr": "true",
                    "readability": "true",
                },
            )
        finally:
            api_module.recognize_image = original_recognize

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        issue_codes = {issue["code"] for issue in payload["report"]["quality_diagnostics"]["issues"]}
        self.assertIn("ocr_low_confidence", issue_codes)

    def test_ocr_status_endpoint_reports_backends_without_ocr_dependency(self) -> None:
        response = self.client.get("/api/ocr/status", params={"language": "jpn+eng"})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("engines", payload)
        self.assertIn("tesseract", payload["engines"])
        self.assertEqual(payload["requested_language"], "jpn+eng")


if __name__ == "__main__":
    unittest.main()
