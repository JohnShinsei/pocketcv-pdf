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


def make_demo_input() -> np.ndarray:
    image = np.full((360, 520, 3), (45, 50, 55), dtype=np.uint8)
    document = np.array([[72, 46], [438, 68], [468, 314], [50, 300]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (235, 236, 230))
    for index in range(7):
        y = 112 + index * 28
        cv2.line(image, (118, y), (390, y + (index % 2)), (44, 44, 44), 2, cv2.LINE_AA)
    return image


class DemoPipelineTest(unittest.TestCase):
    def test_demo_script_writes_summary_pdf_and_diagnostics_without_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "demo_input.jpg"
            output_dir = tmp_path / "out"
            cv2.imwrite(str(input_path), make_demo_input())

            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_demo_pipeline.py",
                    str(input_path),
                    "--out",
                    str(output_dir),
                    "--mode",
                    "gray",
                    "--no-ocr",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            artifacts = payload["artifacts"]

            self.assertTrue(Path(artifacts["scan_image"]).exists())
            self.assertTrue(Path(artifacts["comparison_image"]).exists())
            self.assertTrue(Path(artifacts["scan_pdf"]).exists())
            self.assertTrue(Path(artifacts["pipeline_report"]).exists())
            self.assertTrue(Path(artifacts["demo_summary"]).exists())
            self.assertIn("ocr_status", payload)
            self.assertIn("readability", payload)
            self.assertFalse(payload["ocr"]["attempted"])
            self.assertFalse(payload["pdf"]["searchable"])


if __name__ == "__main__":
    unittest.main()
