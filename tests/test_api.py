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

    from clearscan_cv.api import app  # noqa: E402


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


@unittest.skipUnless(HAS_API_DEPS, "Install API dependencies with: pip install -e .[api]")
class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

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

    def test_ocr_status_endpoint_reports_backends_without_ocr_dependency(self) -> None:
        response = self.client.get("/api/ocr/status", params={"language": "jpn+eng"})

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("engines", payload)
        self.assertIn("tesseract", payload["engines"])
        self.assertEqual(payload["requested_language"], "jpn+eng")


if __name__ == "__main__":
    unittest.main()
