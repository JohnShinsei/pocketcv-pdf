from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import order_points, polygon_area


class TorchUnavailableError(RuntimeError):
    pass


def _require_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
        import torch.nn.functional as functional  # type: ignore[import-not-found]
        from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TorchUnavailableError("PyTorch is not installed. Install training extras with: pip install -e .[train]") from exc
    return torch, nn, functional, (DataLoader, Dataset)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_number} in {path} is not a JSON object")
        records.append(payload)
    return records


def _read_color(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _read_gray(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return image


def mask_to_corners(mask: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype == np.uint8:
        mask_float = mask.astype(np.float32) / (255.0 if mask.max(initial=0) > 1 else 1.0)
    else:
        mask_float = mask.astype(np.float32)
    if mask.dtype != np.uint8:
        binary = (mask_float >= threshold).astype(np.uint8) * 255
    else:
        threshold_value = int(round(threshold * 255)) if mask.max(initial=0) > 1 else threshold
        binary = (mask >= threshold_value).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = binary.shape[:2]
        corners = [[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]]
        return {"corners": corners, "confidence": 0.0, "area_ratio": 0.0, "found": False}

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        points = approx.reshape(4, 2).astype(np.float32)
    else:
        points = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    ordered = order_points(points)
    height, width = binary.shape[:2]
    ordered[:, 0] = np.clip(ordered[:, 0], 0, max(0, width - 1))
    ordered[:, 1] = np.clip(ordered[:, 1], 0, max(0, height - 1))
    area_ratio = polygon_area(ordered) / max(1.0, float(width * height))
    fill = np.zeros(binary.shape, dtype=np.uint8)
    cv2.drawContours(fill, [contour], -1, 255, thickness=-1)
    confidence = float(np.mean(mask_float[fill > 0])) if np.any(fill) else 0.0
    return {
        "corners": np.round(ordered, 2).tolist(),
        "confidence": round(float(min(1.0, max(confidence, area_ratio))), 4),
        "area_ratio": round(float(min(1.0, area_ratio)), 4),
        "found": area_ratio > 0.01 and area > 32.0,
    }


def _build_model(base_channels: int = 16) -> Any:
    _, nn, functional, _ = _require_torch()

    class DoubleConv(nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)

    class TinyDocNet(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.down1 = DoubleConv(3, base_channels)
            self.down2 = DoubleConv(base_channels, base_channels * 2)
            self.down3 = DoubleConv(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
            self.conv2 = DoubleConv(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(base_channels * 2, base_channels)
            self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

        def forward(self, value: Any) -> Any:
            first = self.down1(value)
            second = self.down2(self.pool(first))
            third = self.down3(self.pool(second))
            value = self.up2(third)
            value = functional.interpolate(value, size=second.shape[-2:], mode="bilinear", align_corners=False)
            value = self.conv2(torch.cat([value, second], dim=1))
            value = self.up1(value)
            value = functional.interpolate(value, size=first.shape[-2:], mode="bilinear", align_corners=False)
            value = self.conv1(torch.cat([value, first], dim=1))
            return self.head(value)

    torch, _, _, _ = _require_torch()
    return TinyDocNet()


def _dataset_class(image_size: int) -> Any:
    torch, _, _, (_, Dataset) = _require_torch()

    class DocMaskDataset(Dataset):  # type: ignore[misc]
        def __init__(self, manifest_path: Path) -> None:
            self.manifest_path = manifest_path
            self.root = manifest_path.parent
            self.records = _read_jsonl(manifest_path)
            if not self.records:
                raise ValueError(f"manifest has no samples: {manifest_path}")

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            record = self.records[index]
            image_path = self.root / str(record["image"])
            mask_path = self.root / str(record["mask"])
            image = cv2.cvtColor(_read_color(image_path), cv2.COLOR_BGR2RGB)
            mask = _read_gray(mask_path)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            image_tensor = torch.from_numpy(np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)))
            mask_tensor = torch.from_numpy((mask.astype(np.float32) / 255.0)[None, :, :])
            return image_tensor, mask_tensor

    return DocMaskDataset


def _dice_loss(logits: Any, targets: Any) -> Any:
    torch, nn, _, _ = _require_torch()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
    return bce + dice


def _iou_from_logits(logits: Any, targets: Any) -> float:
    torch, _, _, _ = _require_torch()
    preds = torch.sigmoid(logits) > 0.5
    target_mask = targets > 0.5
    intersection = (preds & target_mask).sum().item()
    union = (preds | target_mask).sum().item()
    return float(intersection / max(1, union))


def train_detector(
    dataset_dir: Path,
    output_path: Path,
    epochs: int = 5,
    image_size: int = 256,
    batch_size: int = 4,
    lr: float = 1e-3,
    base_channels: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    torch, _, _, (DataLoader, _) = _require_torch()
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    train_manifest = dataset_dir / "train.jsonl"
    if not train_manifest.exists() or not train_manifest.read_text(encoding="utf-8").strip():
        train_manifest = dataset_dir / "manifest.jsonl"
    val_manifest = dataset_dir / "val.jsonl"
    DocMaskDataset = _dataset_class(image_size)
    train_loader = DataLoader(DocMaskDataset(train_manifest), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DocMaskDataset(val_manifest), batch_size=batch_size, shuffle=False) if val_manifest.exists() and val_manifest.read_text(encoding="utf-8").strip() else None

    model = _build_model(base_channels=base_channels).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for images, masks in train_loader:
            images = images.to(selected_device)
            masks = masks.to(selected_device)
            optimizer.zero_grad()
            logits = model(images)
            loss = _dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_iou = 0.0
        if val_loader is not None:
            model.eval()
            scores: list[float] = []
            with torch.no_grad():
                for images, masks in val_loader:
                    logits = model(images.to(selected_device))
                    scores.append(_iou_from_logits(logits.cpu(), masks))
            val_iou = float(np.mean(scores)) if scores else 0.0
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_iou": val_iou})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": {"image_size": image_size, "base_channels": base_channels, "task": "document_mask"},
            "history": history,
        },
        output_path,
    )
    return {"checkpoint": str(output_path), "epochs": epochs, "device": selected_device, "history": history}


def predict_detector(checkpoint_path: Path, image_path: Path, threshold: float = 0.5, output_path: Path | None = None) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 256))
    base_channels = int(config.get("base_channels", 16))
    model = _build_model(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = _read_color(image_path)
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None, :, :, :])
    with torch.no_grad():
        mask = torch.sigmoid(model(tensor))[0, 0].numpy()
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    detection = mask_to_corners(mask, threshold=threshold)
    payload = {
        "method": "docnet_mask",
        "model": str(checkpoint_path),
        "corners": detection["corners"],
        "confidence": detection["confidence"],
        "found": detection["found"],
        "mask_area_ratio": detection["area_ratio"],
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run a lightweight document mask detector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train Tiny DocNet on a clearscan-dataset export.")
    train.add_argument("--dataset", required=True, help="Dataset directory created by clearscan-dataset.")
    train.add_argument("--out", required=True, help="Output .pt checkpoint path.")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--image-size", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--base-channels", type=int, default=16)
    train.add_argument("--device", default="auto")

    predict = subparsers.add_parser("predict", help="Predict document corners as JSON for --external-detector-command.")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", help="Optional JSON output path. Prints JSON to stdout either way.")
    predict.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            payload = train_detector(
                dataset_dir=Path(args.dataset),
                output_path=Path(args.out),
                epochs=args.epochs,
                image_size=args.image_size,
                batch_size=args.batch_size,
                lr=args.lr,
                base_channels=args.base_channels,
                device=args.device,
            )
        else:
            payload = predict_detector(
                checkpoint_path=Path(args.checkpoint),
                image_path=Path(args.input),
                threshold=args.threshold,
                output_path=Path(args.output) if args.output else None,
            )
    except TorchUnavailableError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
