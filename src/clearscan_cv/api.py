from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .ocr import OcrUnavailableError, recognize_image, recover_layout_markdown
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


@app.get("/sw.js")
def service_worker() -> Response:
    sw_path = Path(__file__).with_name("static") / "sw.js"
    return Response(
        sw_path.read_text(encoding="utf-8"),
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


@app.post("/api/process")
async def process_upload(
    file: UploadFile = File(...),
    mode: str = Form("color"),
    ocr: bool = Form(False),
    ocr_lang: str = Form("jpn+eng"),
    ocr_engine: str = Form("auto"),
) -> dict[str, object]:
    data = await file.read()
    image = _decode_image(data)

    try:
        result = enhance_image(image, mode=mode)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, encoded = cv2.imencode(".png", result.image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")

    response: dict[str, object] = {
        "filename": file.filename,
        "mode": mode,
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "report": result.report,
    }
    if ocr:
        try:
            ocr_result = recognize_image(result.image, language=ocr_lang, engine=ocr_engine)  # type: ignore[arg-type]
        except OcrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response["ocr"] = ocr_result.to_dict()
        response["layout_markdown"] = recover_layout_markdown(ocr_result)
    return response
