from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.export import build_docx_bytes, build_pdf_bytes, build_pdf_pages_bytes, write_docx, write_pdf, write_pdf_pages  # noqa: E402
from clearscan_cv.ocr import OcrLine, OcrResult  # noqa: E402


def make_scan_image() -> np.ndarray:
    image = np.full((260, 180), 255, dtype=np.uint8)
    cv2.putText(image, "PDF", (34, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.line(image, (24, 130), (152, 130), 0, 2, cv2.LINE_AA)
    return image


class ExportTest(unittest.TestCase):
    def test_builds_searchable_pdf_with_hidden_text_layer(self) -> None:
        image = make_scan_image()
        ocr = OcrResult(
            engine="fake",
            language="eng",
            text="PocketCV PDF",
            confidence=96.0,
            width=image.shape[1],
            height=image.shape[0],
            lines=[OcrLine("PocketCV PDF", 96.0, (28, 64, 128, 28))],
        )

        pdf = build_pdf_bytes(image, title="sample", ocr_result=ocr, searchable=True)

        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"/Subtype /Image", pdf)
        self.assertIn(b"/FlateDecode", pdf)
        self.assertIn(b"3 Tr", pdf)
        self.assertIn("PocketCV PDF".encode("utf-16-be").hex().upper().encode("ascii"), pdf)

    def test_write_pdf_reports_searchable_metadata(self) -> None:
        image = make_scan_image()
        ocr = OcrResult("fake", "eng", "PDF", 91.0, image.shape[1], image.shape[0], [OcrLine("PDF", 91.0, (34, 62, 62, 28))])

        with tempfile.TemporaryDirectory() as tmp:
            export = write_pdf(image, Path(tmp) / "scan.pdf", ocr_result=ocr, searchable=True)

            self.assertTrue(Path(export.path).exists())
            self.assertTrue(export.searchable)
            self.assertEqual(export.text_lines, 1)

    def test_builds_multi_page_image_pdf(self) -> None:
        first = make_scan_image()
        second = cv2.rotate(make_scan_image(), cv2.ROTATE_90_CLOCKWISE)

        pdf = build_pdf_pages_bytes([first, second], title="batch", searchable=False)

        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"/Count 2", pdf)
        self.assertEqual(pdf.count(b"/Type /Page /Parent"), 2)
        self.assertEqual(pdf.count(b"/Subtype /Image"), 2)

    def test_builds_multi_page_searchable_pdf(self) -> None:
        first = make_scan_image()
        second = cv2.rotate(make_scan_image(), cv2.ROTATE_90_CLOCKWISE)
        first_ocr = OcrResult(
            "fake",
            "eng",
            "First",
            95.0,
            first.shape[1],
            first.shape[0],
            [OcrLine("First", 95.0, (32, 60, 80, 28))],
        )
        second_ocr = OcrResult(
            "fake",
            "eng",
            "Second",
            93.0,
            second.shape[1],
            second.shape[0],
            [OcrLine("Second", 93.0, (28, 52, 96, 28))],
        )

        pdf = build_pdf_pages_bytes([first, second], title="batch", ocr_results=[first_ocr, second_ocr], searchable=True)

        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"/Count 2", pdf)
        self.assertEqual(pdf.count(b"/Type /Page /Parent"), 2)
        self.assertEqual(pdf.count(b"3 Tr"), 2)
        self.assertIn("First".encode("utf-16-be").hex().upper().encode("ascii"), pdf)
        self.assertIn("Second".encode("utf-16-be").hex().upper().encode("ascii"), pdf)

    def test_write_pdf_pages_reports_page_metadata(self) -> None:
        first = make_scan_image()
        second = cv2.rotate(make_scan_image(), cv2.ROTATE_90_CLOCKWISE)

        with tempfile.TemporaryDirectory() as tmp:
            export = write_pdf_pages([first, second], Path(tmp) / "batch.pdf", searchable=False)

            self.assertTrue(Path(export.path).exists())
            self.assertEqual(export.page_count, 2)
            self.assertFalse(export.searchable)
            self.assertEqual(export.page_sizes, [{"width": 180, "height": 260}, {"width": 260, "height": 180}])

    def test_cli_writes_image_pdf_without_ocr_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.png"
            cv2.imwrite(str(input_path), make_scan_image())

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
                    "--pdf",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            pdf_path = Path(payload["pdf_path"])
            self.assertTrue(pdf_path.exists())
            self.assertFalse(payload["pdf"]["searchable"])
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF-1.4"))

    def test_cli_writes_multi_page_pdf_from_multiple_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_one = tmp_path / "input-one.png"
            cv2.imwrite(str(input_one), make_scan_image())

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clearscan_cv.cli",
                    str(input_one),
                    str(input_one),
                    "--out",
                    str(tmp_path / "out"),
                    "--mode",
                    "gray",
                    "--pdf",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            pdf_path = Path(payload["batch_pdf_path"])
            self.assertEqual(payload["page_count"], 2)
            self.assertEqual(payload["pdf"]["page_count"], 2)
            self.assertNotEqual(payload["reports"][0]["output_path"], payload["reports"][1]["output_path"])
            self.assertTrue(pdf_path.exists())
            self.assertIn(b"/Count 2", pdf_path.read_bytes())

    def test_builds_docx_from_recovered_markdown(self) -> None:
        payload = build_docx_bytes("## 診断\n\n本文テキスト", title="OCR Result")

        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "result.docx"
            docx_path.write_bytes(payload)
            with zipfile.ZipFile(docx_path) as docx:
                names = set(docx.namelist())
                document_xml = docx.read("word/document.xml").decode("utf-8")

        self.assertIn("[Content_Types].xml", names)
        self.assertIn("word/document.xml", names)
        self.assertIn("word/styles.xml", names)
        self.assertIn("OCR Result", document_xml)
        self.assertIn("診断", document_xml)
        self.assertIn("本文テキスト", document_xml)
        self.assertIn('w:pStyle w:val="Heading2"', document_xml)

    def test_write_docx_reports_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export = write_docx("## Heading\n\nBody", Path(tmp) / "layout.docx", title="PocketCV")

            self.assertTrue(Path(export.path).exists())
            self.assertEqual(export.title, "PocketCV")
            self.assertEqual(export.paragraph_count, 3)


if __name__ == "__main__":
    unittest.main()
