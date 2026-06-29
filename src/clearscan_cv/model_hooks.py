from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import cv2
import numpy as np


@dataclass
class ExternalImageHookResult:
    image: np.ndarray
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


def _tail(value: str, limit: int = 500) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[-limit:]


def _build_command(command: str, input_path: Path, output_path: Path) -> str:
    if "{input}" not in command or "{output}" not in command:
        raise ValueError("External hook command must include {input} and {output} placeholders.")
    return command.replace("{input}", _quote_path(input_path)).replace("{output}", _quote_path(output_path))


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
