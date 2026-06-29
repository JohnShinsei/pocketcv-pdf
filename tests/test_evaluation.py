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

from clearscan_cv.evaluation import character_error_rate, edit_distance, evaluate_ocr_result, evaluate_readability  # noqa: E402
from clearscan_cv.ocr import OcrLine, OcrResult, OcrWord  # noqa: E402
from clearscan_cv.quality import diagnose_scan_quality  # noqa: E402


def make_text_page() -> np.ndarray:
    image = np.full((300, 460, 3), 255, dtype=np.uint8)
    for index in range(6):
        y = 70 + index * 34
        cv2.line(image, (60, y), (390, y), (30, 30, 30), 2, cv2.LINE_AA)
    return image


class EvaluationTest(unittest.TestCase):
    def test_character_error_rate_uses_normalized_text(self) -> None:
        self.assertEqual(edit_distance("Pocket CV", "pocketcv"), 0)
        self.assertEqual(edit_distance("abcdef", "abcxef"), 1)
        self.assertAlmostEqual(character_error_rate("abcdef", "abcxef"), 1 / 6, places=4)

    def test_evaluate_ocr_result_reports_confidence_and_cer(self) -> None:
        result = OcrResult(
            engine="fake",
            language="eng",
            text="PocketCV PDE",
            confidence=90.0,
            width=600,
            height=800,
            lines=[
                OcrLine("PocketCV", 92.0, (40, 80, 120, 24), [OcrWord("PocketCV", 92.0, (40, 80, 120, 24))]),
                OcrLine("PDE", 48.0, (40, 120, 80, 24), [OcrWord("PDE", 48.0, (40, 120, 80, 24))]),
            ],
        )

        metrics = evaluate_ocr_result(result, expected_text="PocketCV PDF")

        self.assertEqual(metrics["line_count"], 2)
        self.assertEqual(metrics["word_count"], 2)
        self.assertEqual(metrics["edit_distance"], 1)
        self.assertGreater(metrics["character_error_rate"], 0)
        self.assertGreater(metrics["low_confidence_ratio"], 0)

    def test_evaluate_readability_includes_textline_score(self) -> None:
        metrics = evaluate_readability(make_text_page())

        self.assertIn("textline_angle", metrics)
        self.assertIn("textline_horizontal_score", metrics)
        self.assertGreaterEqual(metrics["readability_score"], 0)

    def test_quality_diagnostics_uses_ocr_confidence_and_cer(self) -> None:
        diagnostics = diagnose_scan_quality(
            {
                "score": 82.0,
                "shadow_residual": 0.0,
                "shadow_score": 1.0,
                "ink_density": 0.04,
                "edge_density": 0.04,
                "boldness_risk": 0.0,
            },
            perspective_confidence=0.9,
            readability={
                "textline_horizontal_score": 0.92,
                "ocr_quality": {
                    "line_count": 3,
                    "character_count": 24,
                    "mean_confidence": 42.0,
                    "low_confidence_ratio": 0.67,
                    "character_error_rate": 0.31,
                },
            },
        )

        issue_codes = {issue["code"] for issue in diagnostics["issues"]}  # type: ignore[index]
        self.assertIn("ocr_low_confidence", issue_codes)
        self.assertIn("ocr_high_cer", issue_codes)
        self.assertEqual(diagnostics["status"], "review")

    def test_cli_readability_report_without_ocr_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.png"
            cv2.imwrite(str(input_path), make_text_page())

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
                    "--readability",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertIn("readability", payload)
            self.assertIn("textline_horizontal_score", payload["readability"])
            self.assertIn("quality_diagnostics", payload)
            self.assertIn("issues", payload["quality_diagnostics"])


if __name__ == "__main__":
    unittest.main()
