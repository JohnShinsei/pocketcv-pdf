from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import cv2
import numpy as np

from .corners import CornerPoints, parse_corner_points


@dataclass
class ExternalImageHookResult:
    image: np.ndarray
    report: dict[str, object]


@dataclass
class ExternalCornerHookResult:
    corners: CornerPoints | None
    report: dict[str, object]


def _quote_path(path: Path) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(path)])
    return shlex.quote(str(path))


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode hook input image.")
    encoded.tofile(str(path))


def _read_image(path: Path) -> np.ndarray | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    raw = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)


def _read_json(path: Path) -> object | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(value: str, limit: int = 500) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def _build_command(command: str, input_path: Path, output_path: Path) -> str:
    if "{input}" not in command or "{output}" not in command:
        raise ValueError("External hook command must include {input} and {output} placeholders.")
    return command.replace("{input}", _quote_path(input_path)).replace("{output}", _quote_path(output_path))


def _payload_corner_value(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    for key in ("corners", "quad", "points", "polygon"):
        if key in payload:
            return payload[key]
    return payload


def _payload_confidence(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    try:
        confidence = float(payload.get("confidence"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(confidence):
        return None
    return float(np.clip(confidence, 0.0, 1.0))


def _payload_model_name(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("method") or payload.get("model") or payload.get("detector")
    return str(value) if value else None


def _payload_uses_normalized_coordinates(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    space = str(payload.get("coordinate_space") or payload.get("space") or "").lower()
    return bool(payload.get("normalized")) or space in {"normalized", "relative", "ratio"}


def _scale_normalized_corners(corners: CornerPoints, width: int, height: int) -> CornerPoints:
    return [[point[0] * max(0, width - 1), point[1] * max(0, height - 1)] for point in corners]


def apply_external_image_hook(
    image: np.ndarray,
    command: str | None,
    *,
    stage: str,
    timeout_seconds: float = 180.0,
) -> ExternalImageHookResult:
    if not command:
        return ExternalImageHookResult(image=image, report={"stage": stage, "applied": False, "method": "disabled"})

    with tempfile.TemporaryDirectory(prefix="clearscan_hook_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        report: dict[str, object] = {
            "stage": stage,
            "applied": False,
            "method": "external_command",
            "timeout_seconds": timeout_seconds,
        }

        try:
            _write_png(input_path, image)
            command_line = _build_command(command, input_path, output_path)
            completed = subprocess.run(
                command_line,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            report.update(
                {
                    "returncode": completed.returncode,
                    "stdout_tail": _tail(completed.stdout),
                    "stderr_tail": _tail(completed.stderr),
                }
            )
            if completed.returncode != 0:
                report["reason"] = "nonzero_exit"
                return ExternalImageHookResult(image=image, report=report)

            restored = _read_image(output_path)
            if restored is None:
                report["reason"] = "missing_or_unreadable_output"
                return ExternalImageHookResult(image=image, report=report)

            report.update(
                {
                    "applied": True,
                    "output_size": {"width": int(restored.shape[1]), "height": int(restored.shape[0])},
                }
            )
            return ExternalImageHookResult(image=restored, report=report)
        except subprocess.TimeoutExpired as exc:
            report.update({"reason": "timeout", "stdout_tail": _tail(exc.stdout or ""), "stderr_tail": _tail(exc.stderr or "")})
            return ExternalImageHookResult(image=image, report=report)
        except Exception as exc:  # noqa: BLE001 - hook failures must not break the scanner.
            report.update({"reason": "exception", "error": str(exc)})
            return ExternalImageHookResult(image=image, report=report)


def apply_external_corner_hook(
    image: np.ndarray,
    command: str | None,
    *,
    stage: str = "external_detector",
    timeout_seconds: float = 90.0,
) -> ExternalCornerHookResult:
    if not command:
        return ExternalCornerHookResult(corners=None, report={"stage": stage, "applied": False, "method": "disabled"})

    height, width = image.shape[:2]
    with tempfile.TemporaryDirectory(prefix="clearscan_detector_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "corners.json"
        report: dict[str, object] = {
            "stage": stage,
            "applied": False,
            "parsed": False,
            "method": "external_command",
            "timeout_seconds": timeout_seconds,
        }

        try:
            _write_png(input_path, image)
            command_line = _build_command(command, input_path, output_path)
            completed = subprocess.run(
                command_line,
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            report.update(
                {
                    "returncode": completed.returncode,
                    "stdout_tail": _tail(completed.stdout),
                    "stderr_tail": _tail(completed.stderr),
                }
            )
            if completed.returncode != 0:
                report["reason"] = "nonzero_exit"
                return ExternalCornerHookResult(corners=None, report=report)

            payload = _read_json(output_path)
            if payload is None:
                report["reason"] = "missing_or_unreadable_output"
                return ExternalCornerHookResult(corners=None, report=report)

            corners = parse_corner_points(_payload_corner_value(payload))
            if _payload_uses_normalized_coordinates(payload):
                corners = _scale_normalized_corners(corners, width=width, height=height)

            report.update(
                {
                    "parsed": True,
                    "confidence": _payload_confidence(payload),
                    "detector_method": _payload_model_name(payload),
                    "corners": [[round(point[0], 2), round(point[1], 2)] for point in corners],
                }
            )
            return ExternalCornerHookResult(corners=corners, report=report)
        except subprocess.TimeoutExpired as exc:
            report.update({"reason": "timeout", "stdout_tail": _tail(exc.stdout or ""), "stderr_tail": _tail(exc.stderr or "")})
            return ExternalCornerHookResult(corners=None, report=report)
        except json.JSONDecodeError as exc:
            report.update({"reason": "invalid_json", "error": str(exc)})
            return ExternalCornerHookResult(corners=None, report=report)
        except ValueError as exc:
            report.update({"reason": "invalid_corners", "error": str(exc)})
            return ExternalCornerHookResult(corners=None, report=report)
        except Exception as exc:  # noqa: BLE001 - detector failures must not break OpenCV fallback.
            report.update({"reason": "exception", "error": str(exc)})
            return ExternalCornerHookResult(corners=None, report=report)
