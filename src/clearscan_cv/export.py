from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import zlib

import cv2
import numpy as np

from .geometry import ensure_bgr
from .ocr import OcrResult

A4_WIDTH_PT = 595.28
A4_HEIGHT_PT = 841.89


@dataclass(frozen=True)
class PdfExport:
    path: str
    width: int
    height: int
    searchable: bool
    text_lines: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "searchable": self.searchable,
            "text_lines": self.text_lines,
        }


class _PdfBuilder:
    def __init__(self) -> None:
        self._objects: list[bytes] = []

    def add(self, body: bytes | str) -> int:
        if isinstance(body, str):
            body = body.encode("latin-1")
        self._objects.append(body)
        return len(self._objects)

    def build(self, root_object: int, info_object: int | None = None) -> bytes:
        chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets: list[int] = []
        current = len(chunks[0])
        for index, body in enumerate(self._objects, start=1):
            object_bytes = f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
            offsets.append(current)
            chunks.append(object_bytes)
            current += len(object_bytes)

        xref_offset = current
        xref = [f"xref\n0 {len(self._objects) + 1}\n0000000000 65535 f \n".encode("ascii")]
        xref.extend(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
        info_ref = f" /Info {info_object} 0 R" if info_object is not None else ""
        trailer = (
            f"trailer\n<< /Size {len(self._objects) + 1} /Root {root_object} 0 R{info_ref} >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
        return b"".join(chunks + xref + [trailer])


def _pdf_literal(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})"


def _pdf_utf16_hex(value: str) -> str:
    data = b"\xfe\xff" + value.encode("utf-16-be", errors="ignore")
    return "<" + data.hex().upper() + ">"


def _pdf_date(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.strftime("D:%Y%m%d%H%M%SZ")


def _image_stream(image: np.ndarray) -> tuple[bytes, str, int, int]:
    if image.ndim == 2:
        payload = zlib.compress(np.ascontiguousarray(image).tobytes(), level=6)
        return payload, "/DeviceGray", image.shape[1], image.shape[0]

    rgb = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2RGB)
    payload = zlib.compress(np.ascontiguousarray(rgb).tobytes(), level=6)
    return payload, "/DeviceRGB", rgb.shape[1], rgb.shape[0]


def _line_text_commands(result: OcrResult, draw_width: float, draw_height: float, origin_x: float, origin_y: float) -> list[str]:
    commands: list[str] = []
    if result.width <= 0 or result.height <= 0:
        return commands

    scale_x = draw_width / float(result.width)
    scale_y = draw_height / float(result.height)
    for line in result.lines:
        text = line.text.strip()
        if not text:
            continue
        x, y, _width, height = line.bbox
        pdf_x = origin_x + x * scale_x
        pdf_y = origin_y + (result.height - y - height) * scale_y
        font_size = max(4.0, min(32.0, height * scale_y * 0.92))
        # Rendering mode 3 makes OCR text invisible while preserving selectable/searchable text.
        commands.append(f"BT /F1 {font_size:.2f} Tf 3 Tr 1 0 0 1 {pdf_x:.2f} {pdf_y:.2f} Tm {_pdf_utf16_hex(text)} Tj ET\n")
    return commands


def build_pdf_bytes(image: np.ndarray, title: str = "PocketCV PDF", ocr_result: OcrResult | None = None, searchable: bool = True) -> bytes:
    image_payload, color_space, image_width, image_height = _image_stream(image)
    page_width = A4_WIDTH_PT
    page_height = A4_HEIGHT_PT
    scale = min(page_width / float(image_width), page_height / float(image_height))
    draw_width = image_width * scale
    draw_height = image_height * scale
    origin_x = (page_width - draw_width) / 2.0
    origin_y = (page_height - draw_height) / 2.0

    builder = _PdfBuilder()
    catalog_object = builder.add("<< /Type /Catalog /Pages 2 0 R >>")
    pages_object = builder.add("<< /Type /Pages /Kids [6 0 R] /Count 1 >>")
    image_object = builder.add(
        (
            f"<< /Type /XObject /Subtype /Image /Width {image_width} /Height {image_height} "
            f"/ColorSpace {color_space} /BitsPerComponent 8 /Filter /FlateDecode /Length {len(image_payload)} >>\n"
            "stream\n"
        ).encode("latin-1")
        + image_payload
        + b"\nendstream"
    )
    font_object = builder.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    commands = [
        "q\n",
        f"{draw_width:.2f} 0 0 {draw_height:.2f} {origin_x:.2f} {origin_y:.2f} cm\n",
        "/Im0 Do\n",
        "Q\n",
    ]
    if searchable and ocr_result is not None:
        text_commands = _line_text_commands(ocr_result, draw_width, draw_height, origin_x, origin_y)
        commands.extend(text_commands)

    content_payload = "".join(commands).encode("latin-1")
    content_object = builder.add(
        f"<< /Length {len(content_payload)} >>\nstream\n".encode("latin-1") + content_payload + b"\nendstream"
    )
    page_object = builder.add(
        (
            f"<< /Type /Page /Parent {pages_object} 0 R /MediaBox [0 0 {page_width:.2f} {page_height:.2f}] "
            f"/Resources << /XObject << /Im0 {image_object} 0 R >> /Font << /F1 {font_object} 0 R >> >> "
            f"/Contents {content_object} 0 R >>"
        )
    )
    if page_object != 6:
        raise RuntimeError("Unexpected PDF object layout.")
    info_object = builder.add(
        (
            "<< "
            f"/Title {_pdf_utf16_hex(title)} "
            f"/Author {_pdf_utf16_hex('PocketCV PDF')} "
            f"/Creator {_pdf_utf16_hex('ClearScan CV Python exporter')} "
            f"/Producer {_pdf_utf16_hex('PocketCV PDF')} "
            f"/CreationDate {_pdf_literal(_pdf_date())} "
            ">>"
        )
    )
    return builder.build(catalog_object, info_object=info_object)


def write_pdf(
    image: np.ndarray,
    output_path: str | Path,
    title: str = "PocketCV PDF",
    ocr_result: OcrResult | None = None,
    searchable: bool = True,
) -> PdfExport:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_pdf_bytes(image, title=title, ocr_result=ocr_result, searchable=searchable)
    output_path.write_bytes(payload)
    image_height, image_width = image.shape[:2]
    text_lines = len([line for line in (ocr_result.lines if ocr_result else []) if line.text.strip()]) if searchable else 0
    return PdfExport(path=str(output_path), width=image_width, height=image_height, searchable=searchable and text_lines > 0, text_lines=text_lines)
