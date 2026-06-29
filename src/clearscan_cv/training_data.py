from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .corners import CornerPoints, manual_detection_from_corners, parse_corner_points


@dataclass(frozen=True)
class TrainingSample:
    image_path: Path
    corners: CornerPoints
    sample_id: str
    source: str


def _read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def corners_to_mask(width: int, height: int, corners: Sequence[Sequence[float]]) -> np.ndarray:
    detection = manual_detection_from_corners(corners, width=width, height=height)
    polygon = np.asarray(detection.corners, dtype=np.int32).reshape(1, 4, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask


def normalized_corners(width: int, height: int, corners: Sequence[Sequence[float]]) -> CornerPoints:
    detection = manual_detection_from_corners(corners, width=width, height=height)
    max_x = max(1.0, float(width - 1))
    max_y = max(1.0, float(height - 1))
    return [[round(point[0] / max_x, 6), round(point[1] / max_y, 6)] for point in detection.corners]


def _sample_id(image_path: Path, source: str) -> str:
    stem = image_path.stem.replace(" ", "_")
    digest = hashlib.sha1(f"{source}|{image_path}".encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def _resolve_image_path(value: object, image_root: Path | None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("training records require an image path")
    path = Path(value)
    if not path.is_absolute() and image_root is not None:
        path = image_root / path
    return path


def _load_json_or_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"annotation JSON must be a list: {path}")
        return [record for record in payload if isinstance(record, dict)]
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} is not an object: {path}")
        records.append(payload)
    return records


def samples_from_annotations(paths: Iterable[Path], image_root: Path | None = None) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for annotation_path in paths:
        for record in _load_json_or_jsonl(annotation_path):
            image_path = _resolve_image_path(record.get("image") or record.get("image_path"), image_root)
            corners = parse_corner_points(record.get("corners", []))
            sample_id = str(record.get("id") or _sample_id(image_path, str(annotation_path)))
            samples.append(TrainingSample(image_path=image_path, corners=corners, sample_id=sample_id, source=str(annotation_path)))
    return samples


def samples_from_reports(paths: Iterable[Path], image_root: Path | None = None) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for report_path in paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"report must be a JSON object: {report_path}")
        image_path = _resolve_image_path(report.get("input_path"), image_root)
        detection = report.get("document_detection")
        if not isinstance(detection, dict) or "corners" not in detection:
            raise ValueError(f"report has no document_detection.corners: {report_path}")
        corners = parse_corner_points(detection["corners"])
        samples.append(TrainingSample(image_path=image_path, corners=corners, sample_id=_sample_id(image_path, str(report_path)), source=str(report_path)))
    return samples


def _split_name(sample_id: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def export_training_dataset(samples: Sequence[TrainingSample], output_dir: Path, val_ratio: float = 0.2) -> dict[str, object]:
    if not samples:
        raise ValueError("no training samples were provided")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")

    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for sample in samples:
        image = _read_image(sample.image_path)
        height, width = image.shape[:2]
        detection = manual_detection_from_corners(sample.corners, width=width, height=height)
        mask = corners_to_mask(width, height, detection.corners)
        image_name = f"{sample.sample_id}.png"
        mask_name = f"{sample.sample_id}_mask.png"
        image_path = image_dir / image_name
        mask_path = mask_dir / mask_name
        _write_image(image_path, image)
        _write_image(mask_path, mask)

        manifest.append(
            {
                "id": sample.sample_id,
                "split": _split_name(sample.sample_id, val_ratio),
                "image": str(Path("images") / image_name).replace("\\", "/"),
                "mask": str(Path("masks") / mask_name).replace("\\", "/"),
                "width": width,
                "height": height,
                "corners": detection.corners,
                "normalized_corners": normalized_corners(width, height, detection.corners),
                "area_ratio": detection.area_ratio,
                "source_image": str(sample.image_path),
                "source": sample.source,
            }
        )

    _write_jsonl(output_dir / "manifest.jsonl", manifest)
    _write_jsonl(output_dir / "train.jsonl", [record for record in manifest if record["split"] == "train"])
    _write_jsonl(output_dir / "val.jsonl", [record for record in manifest if record["split"] == "val"])
    summary = {
        "sample_count": len(manifest),
        "train_count": sum(1 for record in manifest if record["split"] == "train"),
        "val_count": sum(1 for record in manifest if record["split"] == "val"),
        "task": "document_mask_and_corner_detection",
        "format": "image, binary mask, pixel corners, normalized corners",
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export document mask and corner labels for model training.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--annotations", nargs="*", default=[], help="JSON or JSONL files with image and corners fields.")
    parser.add_argument("--reports", nargs="*", default=[], help="ClearScan *_report.json files containing document_detection.corners.")
    parser.add_argument("--image-root", help="Base directory for relative image paths.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Deterministic validation split ratio.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image_root = Path(args.image_root) if args.image_root else None
    samples = samples_from_annotations([Path(path) for path in args.annotations], image_root=image_root)
    samples.extend(samples_from_reports([Path(path) for path in args.reports], image_root=image_root))
    if not samples:
        parser.error("provide at least one --annotations or --reports file")
    summary = export_training_dataset(samples, Path(args.out), val_ratio=args.val_ratio)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
