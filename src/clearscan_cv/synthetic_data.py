from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import order_points, polygon_area
from .training_data import normalized_corners


@dataclass(frozen=True)
class SyntheticDocument:
    image: np.ndarray
    mask: np.ndarray
    corners: list[list[float]]
    seed: int


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _split_name(sample_id: str, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    if val_ratio >= 1:
        return "val"
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def _make_background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    palettes = np.asarray(
        [
            [188, 180, 166],
            [154, 164, 168],
            [178, 168, 145],
            [120, 132, 128],
            [202, 198, 184],
        ],
        dtype=np.float32,
    )
    base = palettes[int(rng.integers(0, len(palettes)))]
    image = np.full((height, width, 3), base, dtype=np.float32)
    x_gradient = np.linspace(float(rng.uniform(-18, 18)), float(rng.uniform(-18, 18)), width, dtype=np.float32)
    y_gradient = np.linspace(float(rng.uniform(-14, 14)), float(rng.uniform(-14, 14)), height, dtype=np.float32)
    image += x_gradient[None, :, None] + y_gradient[:, None, None]
    noise = rng.normal(0.0, 4.0, size=(height, width, 1)).astype(np.float32)
    image += noise
    return np.clip(image, 0, 255).astype(np.uint8)


def _draw_text_like_content(page: np.ndarray, rng: np.random.Generator) -> None:
    height, width = page.shape[:2]
    margin_x = int(width * rng.uniform(0.08, 0.14))
    top = int(height * rng.uniform(0.06, 0.1))
    ink = int(rng.integers(35, 95))
    accent = (int(rng.integers(80, 140)), int(rng.integers(80, 140)), int(rng.integers(80, 140)))

    if rng.random() < 0.7:
        title_width = int(width * rng.uniform(0.22, 0.55))
        cv2.rectangle(page, (margin_x, top), (margin_x + title_width, top + int(height * 0.018)), (ink, ink, ink), -1)
        top += int(height * rng.uniform(0.045, 0.07))

    column_count = 2 if rng.random() < 0.25 else 1
    gap = int(width * 0.055)
    column_width = int((width - margin_x * 2 - gap * (column_count - 1)) / column_count)
    line_height = int(height * rng.uniform(0.018, 0.026))
    line_gap = int(height * rng.uniform(0.017, 0.026))

    for column in range(column_count):
        x0 = margin_x + column * (column_width + gap)
        y = top
        line_count = int(rng.integers(22, 48))
        for line_index in range(line_count):
            if y + line_height >= height - margin_x:
                break
            if rng.random() < 0.12:
                y += line_gap
                continue
            width_ratio = float(rng.uniform(0.38, 1.0))
            if line_index == line_count - 1:
                width_ratio *= float(rng.uniform(0.35, 0.75))
            x_indent = int(rng.uniform(0.0, 0.08) * column_width) if rng.random() < 0.24 else 0
            x1 = min(width - margin_x, x0 + x_indent + int(column_width * width_ratio))
            cv2.rectangle(page, (x0 + x_indent, y), (x1, y + max(2, line_height // 3)), (ink, ink, ink), -1)
            if rng.random() < 0.18:
                box_x = x0 + int(column_width * rng.uniform(0.05, 0.75))
                box_y = y - max(1, line_height // 4)
                cv2.rectangle(page, (box_x, box_y), (box_x + line_height, box_y + line_height), accent, 1)
            y += line_height + line_gap

    if rng.random() < 0.35:
        table_top = int(height * rng.uniform(0.55, 0.78))
        table_left = margin_x
        table_right = width - margin_x
        table_bottom = min(height - margin_x, table_top + int(height * rng.uniform(0.08, 0.18)))
        cv2.rectangle(page, (table_left, table_top), (table_right, table_bottom), (ink + 18, ink + 18, ink + 18), 2)
        for i in range(1, int(rng.integers(3, 6))):
            x = table_left + i * (table_right - table_left) // 5
            cv2.line(page, (x, table_top), (x, table_bottom), (ink + 25, ink + 25, ink + 25), 1)
        for i in range(1, int(rng.integers(2, 5))):
            y = table_top + i * (table_bottom - table_top) // 4
            cv2.line(page, (table_left, y), (table_right, y), (ink + 25, ink + 25, ink + 25), 1)


def _make_page(rng: np.random.Generator) -> np.ndarray:
    page_height = int(rng.integers(920, 1220))
    page_width = int(page_height * float(rng.uniform(0.66, 0.78)))
    paper_color = np.asarray(
        [
            int(rng.integers(224, 252)),
            int(rng.integers(224, 252)),
            int(rng.integers(218, 246)),
        ],
        dtype=np.float32,
    )
    page = np.full((page_height, page_width, 3), paper_color, dtype=np.float32)
    y_gradient = np.linspace(float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8)), page_height, dtype=np.float32)
    page += y_gradient[:, None, None]
    page = np.clip(page, 0, 255).astype(np.uint8)
    _draw_text_like_content(page, rng)
    return page


def _sample_corners(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    page_width = width * float(rng.uniform(0.58, 0.84))
    aspect = float(rng.uniform(1.24, 1.55))
    page_height = min(height * float(rng.uniform(0.62, 0.92)), page_width * aspect)
    if page_height < height * 0.58:
        page_height = height * 0.58
        page_width = min(width * 0.86, page_height / aspect)

    center_x = width * float(rng.uniform(0.47, 0.53))
    center_y = height * float(rng.uniform(0.48, 0.56))
    base = np.asarray(
        [
            [center_x - page_width / 2, center_y - page_height / 2],
            [center_x + page_width / 2, center_y - page_height / 2],
            [center_x + page_width / 2, center_y + page_height / 2],
            [center_x - page_width / 2, center_y + page_height / 2],
        ],
        dtype=np.float32,
    )
    jitter = np.asarray(
        [
            [rng.uniform(-0.06, 0.08) * page_width, rng.uniform(-0.04, 0.06) * page_height],
            [rng.uniform(-0.08, 0.06) * page_width, rng.uniform(-0.05, 0.07) * page_height],
            [rng.uniform(-0.07, 0.08) * page_width, rng.uniform(-0.08, 0.05) * page_height],
            [rng.uniform(-0.06, 0.08) * page_width, rng.uniform(-0.07, 0.06) * page_height],
        ],
        dtype=np.float32,
    )
    corners = base + jitter
    margin = max(4.0, min(width, height) * 0.015)
    corners[:, 0] = np.clip(corners[:, 0], margin, width - 1 - margin)
    corners[:, 1] = np.clip(corners[:, 1], margin, height - 1 - margin)
    return order_points(corners)


def _apply_lighting_and_capture_noise(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = mask.shape[:2]
    result = image.astype(np.float32)
    shade = np.zeros((height, width), dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(max(20, width // 5), max(21, width))), int(rng.integers(max(20, height // 7), max(21, height // 2))))
        angle = float(rng.uniform(-35, 35))
        strength = float(rng.uniform(12, 54))
        cv2.ellipse(shade, center, axes, angle, 0, 360, strength, -1, cv2.LINE_AA)
    shade = cv2.GaussianBlur(shade, (0, 0), sigmaX=max(18, width // 16), sigmaY=max(18, height // 16))
    result -= shade[:, :, None] * float(rng.uniform(0.35, 0.9))

    if rng.random() < 0.55:
        page_shadow = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(5, width // 85), sigmaY=max(5, height // 85)).astype(np.float32) / 255.0
        shifted = np.roll(page_shadow, int(rng.integers(4, 20)), axis=0)
        shifted = np.roll(shifted, int(rng.integers(-18, 18)), axis=1)
        result -= shifted[:, :, None] * float(rng.uniform(5, 22))

    noise = rng.normal(0.0, float(rng.uniform(1.0, 5.5)), size=result.shape).astype(np.float32)
    result += noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    if rng.random() < 0.42:
        ksize = int(rng.choice([3, 5]))
        result = cv2.GaussianBlur(result, (ksize, ksize), float(rng.uniform(0.2, 0.9)))
    return result


def generate_synthetic_document(width: int = 960, height: int = 1280, seed: int = 1) -> SyntheticDocument:
    if width < 160 or height < 160:
        raise ValueError("synthetic document size must be at least 160x160")
    rng = np.random.default_rng(seed)
    background = _make_background(width, height, rng)
    page = _make_page(rng)
    page_h, page_w = page.shape[:2]
    corners = _sample_corners(width, height, rng)
    source = np.asarray([[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped_page = cv2.warpPerspective(page, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warped_mask = cv2.warpPerspective(np.full((page_h, page_w), 255, dtype=np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST)

    alpha = cv2.GaussianBlur(warped_mask, (5, 5), 0).astype(np.float32) / 255.0
    image = background.astype(np.float32) * (1.0 - alpha[:, :, None]) + warped_page.astype(np.float32) * alpha[:, :, None]
    image = _apply_lighting_and_capture_noise(np.clip(image, 0, 255).astype(np.uint8), warped_mask, rng)
    mask = np.where(warped_mask > 0, 255, 0).astype(np.uint8)
    return SyntheticDocument(image=image, mask=mask, corners=np.round(corners, 2).tolist(), seed=seed)


def generate_synthetic_dataset(
    output_dir: Path,
    count: int = 100,
    width: int = 960,
    height: int = 1280,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    manifest: list[dict[str, Any]] = []

    for index in range(count):
        sample_seed = seed + index * 9973
        sample = generate_synthetic_document(width=width, height=height, seed=sample_seed)
        sample_id = f"synthetic_{index:05d}"
        image_name = f"{sample_id}.jpg"
        mask_name = f"{sample_id}_mask.png"
        _write_image(image_dir / image_name, sample.image)
        _write_image(mask_dir / mask_name, sample.mask)
        area_ratio = polygon_area(np.asarray(sample.corners, dtype=np.float32)) / max(1.0, float(width * height))
        manifest.append(
            {
                "id": sample_id,
                "split": _split_name(sample_id, val_ratio),
                "image": str(Path("images") / image_name).replace("\\", "/"),
                "mask": str(Path("masks") / mask_name).replace("\\", "/"),
                "width": width,
                "height": height,
                "corners": sample.corners,
                "normalized_corners": normalized_corners(width, height, sample.corners),
                "area_ratio": round(float(area_ratio), 6),
                "source_image": "",
                "source": "synthetic",
                "synthetic": {"seed": sample_seed},
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
        "format": "synthetic phone photo, binary mask, pixel corners, normalized corners",
        "width": width,
        "height": height,
        "seed": seed,
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic phone-photo document training samples.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument("--count", type=int, default=100, help="Number of synthetic samples to create.")
    parser.add_argument("--width", type=int, default=960, help="Output image width.")
    parser.add_argument("--height", type=int, default=1280, help="Output image height.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Deterministic validation split ratio.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = generate_synthetic_dataset(
        output_dir=Path(args.out),
        count=args.count,
        width=args.width,
        height=args.height,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
