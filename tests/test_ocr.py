from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.ocr import OcrLine, OcrResult, OcrWord, recover_layout_markdown  # noqa: E402


class OcrTest(unittest.TestCase):
    def test_ocr_result_serializes_line_boxes(self) -> None:
        word = OcrWord(text="PocketCV", confidence=91.5, bbox=(12, 20, 80, 18))
        line = OcrLine(text="PocketCV PDF", confidence=90.0, bbox=(12, 20, 140, 18), words=[word])
        result = OcrResult(engine="fake", language="eng", text="PocketCV PDF", confidence=90.0, width=600, height=800, lines=[line])

        payload = result.to_dict()

        self.assertEqual(payload["engine"], "fake")
        self.assertEqual(payload["lines"][0]["bbox"], [12, 20, 140, 18])  # type: ignore[index]
        self.assertEqual(payload["lines"][0]["words"][0]["text"], "PocketCV")  # type: ignore[index]

    def test_recover_layout_markdown_keeps_two_columns_readable(self) -> None:
        lines = [
            OcrLine("Title", 98.0, (230, 24, 140, 32)),
            OcrLine("Left first line", 90.0, (40, 100, 210, 20)),
            OcrLine("Left second line", 90.0, (42, 132, 230, 20)),
            OcrLine("Left third line", 90.0, (44, 164, 220, 20)),
            OcrLine("Right first line", 88.0, (360, 100, 230, 20)),
            OcrLine("Right second line", 88.0, (362, 132, 235, 20)),
            OcrLine("Right third line", 88.0, (364, 164, 220, 20)),
        ]
        result = OcrResult(engine="fake", language="eng", text="", confidence=91.0, width=640, height=900, lines=lines)

        markdown = recover_layout_markdown(result)

        self.assertIn("## Title", markdown)
        self.assertLess(markdown.index("Left first line"), markdown.index("Right first line"))
        self.assertIn("Left first line Left second line Left third line", markdown)
        self.assertIn("Right first line Right second line Right third line", markdown)

    def test_cli_reports_missing_requested_ocr_engine_without_breaking_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.png"
            image = np.full((160, 220, 3), 255, dtype=np.uint8)
            cv2.putText(image, "OCR", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.imwrite(str(input_path), image)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "clearscan_cv.cli",
                    str(input_path),
                    "--out",
                    str(tmp_path / "out"),
                    "--ocr",
                    "--ocr-engine",
                    "rapidocr",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("OCR engine unavailable", completed.stderr)
            self.assertTrue((tmp_path / "out" / "input_clearscan.png").exists())


if __name__ == "__main__":
    unittest.main()
