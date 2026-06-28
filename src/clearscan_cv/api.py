from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .pipeline import enhance_image

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, Response
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API dependencies with: pip install -e .[api]") from exc


app = FastAPI(title="PocketCV PDF", version="0.1.0")


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unsupported or unreadable image.")
    return image


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).with_name("static") / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/manifest.webmanifest")
def manifest() -> Response:
    manifest_path = Path(__file__).with_name("static") / "manifest.webmanifest"
    return Response(manifest_path.read_text(encoding="utf-8"), media_type="application/manifest+json")


@app.post("/api/process")
async def process_upload(file: UploadFile = File(...), mode: str = Form("color")) -> dict[str, object]:
    data = await file.read()
    image = _decode_image(data)

    try:
        result = enhance_image(image, mode=mode)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, encoded = cv2.imencode(".png", result.image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    return {
        "filename": file.filename,
        "mode": mode,
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "report": result.report,
    }
