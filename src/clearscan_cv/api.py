from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .corners import parse_corner_points
from .evaluation import evaluate_readability
from .export import build_docx_bytes, build_pdf_bytes, build_pdf_pages_bytes
from .ocr import OcrUnavailableError, ocr_engine_status, recognize_image, recover_layout_markdown
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


def _encode_png_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode processed image.")
    return base64.b64encode(encoded).decode("ascii")


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


@app.get("/api/ocr/status")
def ocr_status(language: str = "jpn+eng") -> dict[str, object]:
    return ocr_engine_status(language=language)


@app.post("/api/process")
async def process_upload(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    auto_warp: bool = Form(True),
    auto_dewarp: bool = Form(True),
    corners: str | None = Form(None),
    corners_space: str = Form("input"),
    ocr: bool = Form(False),
    ocr_lang: str = Form("jpn+eng"),
    ocr_engine: str = Form("auto"),
    pdf: bool = Form(False),
    searchable_pdf: bool = Form(False),
    docx: bool = Form(False),
    readability: bool = Form(False),
    expected_text: str | None = Form(None),
) -> dict[str, object]:
    data = await file.read()
    image = _decode_image(data)
    if corners and not auto_warp:
        raise HTTPException(status_code=400, detail="corners cannot be combined with auto_warp=false")
    if corners_space not in {"input", "processed"}:
        raise HTTPException(status_code=400, detail="corners_space must be input or processed")
    try:
        manual_corners = parse_corner_points(corners) if corners else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = enhance_image(
            image,
            mode=mode,
            auto_warp=auto_warp,
            auto_dewarp=auto_dewarp,
            manual_corners=manual_corners,
            manual_corners_space=corners_space,  # type: ignore[arg-type]
        )  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response: dict[str, object] = {
        "filename": file.filename,
        "mode": mode,
        "image_base64": _encode_png_base64(result.image),
        "report": result.report,
    }
    ocr_result = None
    if ocr or searchable_pdf or docx:
        try:
            ocr_result = recognize_image(result.image, language=ocr_lang, engine=ocr_engine)  # type: ignore[arg-type]
        except OcrUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response["ocr"] = ocr_result.to_dict()
        response["layout_markdown"] = recover_layout_markdown(ocr_result)
    if docx:
        markdown = str(response.get("layout_markdown") or "")
        docx_bytes = build_docx_bytes(markdown or (ocr_result.text if ocr_result else ""), title=Path(file.filename or "pocketcv-scan").stem)
        response["docx_base64"] = base64.b64encode(docx_bytes).decode("ascii")
    if pdf or searchable_pdf:
        pdf_bytes = build_pdf_bytes(
            result.image,
            title=Path(file.filename or "pocketcv-scan").stem,
            ocr_result=ocr_result,
            searchable=searchable_pdf,
        )
        response["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
        response["pdf_searchable"] = bool(searchable_pdf and ocr_result is not None and ocr_result.lines)
    if readability or ocr_result is not None:
        response["readability"] = evaluate_readability(result.image, ocr_result=ocr_result, expected_text=expected_text)
    return response


@app.post("/api/process-batch")
async def process_batch_upload(
    files: list[UploadFile] = File(...),
    mode: str = Form("auto"),
    auto_warp: bool = Form(True),
    auto_dewarp: bool = Form(True),
    ocr: bool = Form(False),
    ocr_lang: str = Form("jpn+eng"),
    ocr_engine: str = Form("auto"),
    layout: bool = Form(False),
    pdf: bool = Form(True),
    searchable_pdf: bool = Form(False),
    docx: bool = Form(False),
    readability: bool = Form(False),
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")

    pages: list[dict[str, object]] = []
    processed_images: list[np.ndarray] = []
    ocr_results = []
    layout_pages: list[str] = []
    for index, file in enumerate(files, start=1):
        image = _decode_image(await file.read())
        try:
            result = enhance_image(
                image,
                mode=mode,
                auto_warp=auto_warp,
                auto_dewarp=auto_dewarp,
            )  # type: ignore[arg-type]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{file.filename or f'page-{index}'}: {exc}") from exc

        page_payload: dict[str, object] = {
            "page_index": index,
            "filename": file.filename,
            "image_base64": _encode_png_base64(result.image),
            "report": result.report,
        }
        ocr_result = None
        if ocr or searchable_pdf or layout or docx:
            try:
                ocr_result = recognize_image(result.image, language=ocr_lang, engine=ocr_engine)  # type: ignore[arg-type]
            except OcrUnavailableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            page_payload["ocr"] = ocr_result.to_dict()
            if layout or docx:
                page_layout = recover_layout_markdown(ocr_result)
                page_payload["layout_markdown"] = page_layout
                layout_pages.append(
                    f"## Page {index}: {file.filename or f'page-{index}'}\n\n{page_layout or ocr_result.text}".strip()
                )
        if readability:
            page_payload["readability"] = evaluate_readability(result.image, ocr_result=ocr_result)
        pages.append(page_payload)
        processed_images.append(result.image)
        ocr_results.append(ocr_result)

    response: dict[str, object] = {
        "mode": mode,
        "page_count": len(pages),
        "pages": pages,
    }
    if layout or docx:
        response["layout_markdown"] = "\n\n".join(layout_pages)
    if docx:
        docx_bytes = build_docx_bytes(str(response.get("layout_markdown") or ""), title="pocketcv-batch")
        response["docx_base64"] = base64.b64encode(docx_bytes).decode("ascii")
    if pdf or searchable_pdf:
        pdf_bytes = build_pdf_pages_bytes(
            processed_images,
            title="pocketcv-batch",
            ocr_results=ocr_results if searchable_pdf else None,
            searchable=searchable_pdf,
        )
        response["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
        response["pdf_searchable"] = bool(searchable_pdf and any(result and result.lines for result in ocr_results))
    return response
