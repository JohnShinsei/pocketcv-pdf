from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
import re
from xml.sax.saxutils import escape
import zipfile
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


@dataclass(frozen=True)
class DocxExport:
    path: str
    paragraph_count: int
    title: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "paragraph_count": self.paragraph_count, "title": self.title}


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


def _content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def _root_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def _document_relationships_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:after="240"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="34"/><w:szCs w:val="34"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="180" w:after="80"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>
  </w:style>
</w:styles>
"""


def _core_properties_xml(title: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>PocketCV PDF</dc:creator>
  <cp:lastModifiedBy>PocketCV PDF</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""


def _app_properties_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>PocketCV PDF</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company>PocketCV PDF</Company>
</Properties>
"""


def _paragraph_xml(text: str, style: str | None = None) -> str:
    escaped = escape(text)
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def _markdown_paragraphs(markdown: str, title: str) -> tuple[list[str], int]:
    paragraphs = [_paragraph_xml(title, style="Title")]
    count = 1
    for block in re.split(r"\n\s*\n", markdown.strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        if cleaned.startswith("## "):
            paragraphs.append(_paragraph_xml(cleaned[3:].strip(), style="Heading2"))
        else:
            paragraphs.append(_paragraph_xml(re.sub(r"\s*\n\s*", " ", cleaned)))
        count += 1
    return paragraphs, count


def build_docx_bytes(markdown: str, title: str = "PocketCV OCR") -> bytes:
    paragraphs, _count = _markdown_paragraphs(markdown, title)
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(paragraphs)}
    <w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>
  </w:body>
</w:document>
"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", _content_types_xml())
        docx.writestr("_rels/.rels", _root_relationships_xml())
        docx.writestr("word/_rels/document.xml.rels", _document_relationships_xml())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", _styles_xml())
        docx.writestr("docProps/core.xml", _core_properties_xml(title))
        docx.writestr("docProps/app.xml", _app_properties_xml())
    return buffer.getvalue()


def write_docx(markdown: str, output_path: str | Path, title: str = "PocketCV OCR") -> DocxExport:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_docx_bytes(markdown, title=title)
    output_path.write_bytes(payload)
    _paragraphs, paragraph_count = _markdown_paragraphs(markdown, title)
    return DocxExport(path=str(output_path), paragraph_count=paragraph_count, title=title)
