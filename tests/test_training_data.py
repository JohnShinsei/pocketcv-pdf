from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from clearscan_cv.training_data import corners_to_mask, export_training_dataset, samples_from_annotations, samples_from_reports


ROOT = Path(__file__).resolve().parents[1]


def make_training_image(width: int = 120, height: int = 90) -> np.ndarray:
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (24, 18), (96, 72), (255, 255, 255), -1)
    cv2.putText(image, "DOC", (38, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA)
    return image


def test_corners_to_mask_fills_document_polygon() -> None:
    corners = [[20, 15], [100, 18], [96, 74], [18, 70]]
    mask = corners_to_mask(120, 90, corners)

    assert mask.shape == (90, 120)
    assert mask[40, 60] == 255
    assert mask[3, 3] == 0


def test_export_training_dataset_from_annotation_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "phone_photo.jpg"
        cv2.imwrite(str(image_path), make_training_image())
        annotations = tmp_path / "annotations.jsonl"
        annotations.write_text(
            json.dumps(
                {
                    "id": "sample_001",
                    "image": str(image_path),
                    "corners": [[20, 15], [100, 18], [96, 74], [18, 70]],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        samples = samples_from_annotations([annotations])
        summary = export_training_dataset(samples, tmp_path / "dataset", val_ratio=0.0)

        assert summary["sample_count"] == 1
        manifest = [json.loads(line) for line in (tmp_path / "dataset" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        assert manifest[0]["id"] == "sample_001"
        assert manifest[0]["split"] == "train"
        assert manifest[0]["normalized_corners"][0][0] > 0
        assert (tmp_path / "dataset" / manifest[0]["image"]).exists()
        mask = cv2.imread(str(tmp_path / "dataset" / manifest[0]["mask"]), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert mask[40, 60] == 255


def test_export_training_dataset_from_clearscan_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "input.jpg"
        cv2.imwrite(str(image_path), make_training_image())
        report = tmp_path / "input_report.json"
        report.write_text(
            json.dumps(
                {
                    "input_path": str(image_path),
                    "document_detection": {
                        "corners": [[20, 15], [100, 18], [96, 74], [18, 70]],
                        "confidence": 0.9,
                        "method": "manual_corners",
                    },
                }
            ),
            encoding="utf-8",
        )

        samples = samples_from_reports([report])
        summary = export_training_dataset(samples, tmp_path / "dataset", val_ratio=1.0)

        assert summary["val_count"] == 1
        assert (tmp_path / "dataset" / "val.jsonl").read_text(encoding="utf-8").strip()


def test_dataset_cli_exports_from_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "input.jpg"
        cv2.imwrite(str(image_path), make_training_image())
        report = tmp_path / "input_report.json"
        report.write_text(
            json.dumps(
                {
                    "input_path": str(image_path),
                    "document_detection": {"corners": [[20, 15], [100, 18], [96, 74], [18, 70]]},
                }
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "clearscan_cv.training_data",
                "--reports",
                str(report),
                "--out",
                str(tmp_path / "dataset"),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["task"] == "document_mask_and_corner_detection"
        assert payload["sample_count"] == 1
