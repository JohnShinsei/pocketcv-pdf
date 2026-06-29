from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from clearscan_cv.docnet import build_parser, mask_to_corners


ROOT = Path(__file__).resolve().parents[1]


def test_mask_to_corners_returns_external_detector_payload_shape() -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    points = np.asarray([[30, 18], [132, 28], [118, 104], [22, 92]], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    detection = mask_to_corners(mask)

    assert detection["found"] is True
    assert len(detection["corners"]) == 4
    assert detection["confidence"] > 0.1
    assert detection["area_ratio"] > 0.2


def test_mask_to_corners_marks_empty_mask_as_not_found() -> None:
    detection = mask_to_corners(np.zeros((20, 30), dtype=np.float32))

    assert detection["found"] is False
    assert detection["confidence"] == 0.0
    assert detection["area_ratio"] == 0.0
    assert detection["corners"] == [[0.0, 0.0], [29.0, 0.0], [29.0, 19.0], [0.0, 19.0]]


def test_docnet_cli_help_does_not_require_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "clearscan_cv.docnet", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Train or run a lightweight document mask detector" in completed.stdout
    assert "predict" in completed.stdout


def test_docnet_parser_registers_train_and_predict_commands() -> None:
    parser = build_parser()

    train_args = parser.parse_args(["train", "--dataset", "datasets/docnet", "--out", "models/docnet.pt"])
    predict_args = parser.parse_args(["predict", "--checkpoint", "models/docnet.pt", "--input", "photo.jpg"])

    assert train_args.command == "train"
    assert predict_args.command == "predict"


def test_docnet_scripts_and_training_extra_are_registered() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'clearscan-docnet = "clearscan_cv.docnet:main"' in pyproject
    assert "torch>=2.2" in pyproject
