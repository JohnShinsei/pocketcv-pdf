from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from clearscan_cv.geometry import polygon_area
from clearscan_cv.synthetic_data import generate_synthetic_dataset, generate_synthetic_document


ROOT = Path(__file__).resolve().parents[1]


def test_generate_synthetic_document_returns_mask_and_bounded_corners() -> None:
    sample = generate_synthetic_document(width=320, height=420, seed=123)

    assert sample.image.shape == (420, 320, 3)
    assert sample.mask.shape == (420, 320)
    assert sample.mask.dtype == np.uint8
    assert int(np.count_nonzero(sample.mask)) > 320 * 420 * 0.2
    assert len(sample.corners) == 4
    for x, y in sample.corners:
        assert 0 <= x < 320
        assert 0 <= y < 420
    assert polygon_area(np.asarray(sample.corners, dtype=np.float32)) > 320 * 420 * 0.2


def test_generate_synthetic_dataset_writes_training_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary = generate_synthetic_dataset(tmp_path / "dataset", count=6, width=240, height=320, seed=7, val_ratio=0.5)

        assert summary["sample_count"] == 6
        assert summary["train_count"] + summary["val_count"] == 6
        manifest = [
            json.loads(line)
            for line in (tmp_path / "dataset" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(manifest) == 6
        first = manifest[0]
        assert first["source"] == "synthetic"
        assert first["synthetic"]["seed"] == 7
        assert all(0.0 <= coord <= 1.0 for point in first["normalized_corners"] for coord in point)
        assert (tmp_path / "dataset" / first["image"]).exists()
        mask = cv2.imread(str(tmp_path / "dataset" / first["mask"]), cv2.IMREAD_GRAYSCALE)
        assert mask is not None
        assert int(np.count_nonzero(mask)) > 240 * 320 * 0.2
        assert (tmp_path / "dataset" / "train.jsonl").exists()
        assert (tmp_path / "dataset" / "val.jsonl").exists()


def test_synthetic_dataset_cli_exports_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "clearscan_cv.synthetic_data",
                "--out",
                str(tmp_path / "dataset"),
                "--count",
                "2",
                "--width",
                "220",
                "--height",
                "300",
                "--seed",
                "11",
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
        assert payload["sample_count"] == 2
        assert (tmp_path / "dataset" / "manifest.jsonl").exists()


def test_synthetic_script_is_registered() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'clearscan-synth = "clearscan_cv.synthetic_data:main"' in pyproject
