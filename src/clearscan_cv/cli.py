from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .evaluation import evaluate_readability
from .export import write_docx, write_pdf
from .ocr import OcrUnavailableError, ocr_engine_status, recognize_image, recover_layout_markdown
from .pipeline import process_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance document photos and generate an image quality report.")
    parser.add_argument("input", nargs="?", help="Path to an input image.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--mode", choices=["color", "gray", "binary"], default="color", help="Output style.")
    parser.add_argument("--no-warp", action="store_true", help="Disable automatic perspective correction.")
    parser.add_argument("--no-dewarp", action="store_true", help="Disable lightweight textline dewarping.")
    parser.add_argument("--compare", action="store_true", help="Write a side-by-side comparison image.")
    parser.add_argument("--pdf", action="store_true", help="Write an image-only PDF from the processed scan.")
    parser.add_argument("--searchable-pdf", action="store_true", help="Run OCR and write a searchable PDF text layer.")
    parser.add_argument("--ocr", action="store_true", help="Run optional OCR on the processed scan and write a TXT file.")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="OCR language code, for example jpn+eng, eng, chi_sim+eng.")
    parser.add_argument("--ocr-status", action="store_true", help="Print OCR backend availability and installation hints.")
    parser.add_argument("--expected-text", help="Path to reference text for OCR edit distance and CER evaluation.")
    parser.add_argument("--readability", action="store_true", help="Add OCR/readability metrics to the report.")
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "rapidocr", "tesseract", "paddleocr"],
        default="auto",
        help="Optional OCR backend. Auto prefers Tesseract for Japanese and RapidOCR for Chinese/English.",
    )
    parser.add_argument("--layout", action="store_true", help="Recover a simple Markdown layout from OCR line boxes.")
    parser.add_argument("--docx", action="store_true", help="Write recovered OCR layout as a DOCX document.")
    return parser


def _read_image(path: str | Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ocr_status:
        print(json.dumps(ocr_engine_status(language=args.ocr_lang), indent=2))
        return 0
    if not args.input:
        parser.error("input is required unless --ocr-status is used")

    report = process_file(
        input_path=args.input,
        output_dir=args.out,
        mode=args.mode,
        auto_warp=not args.no_warp,
        auto_dewarp=not args.no_dewarp,
        side_by_side=args.compare,
    )
    output_path = Path(str(report["output_path"]))
    output_image: np.ndarray | None = None
    ocr_result = None
    expected_text = Path(args.expected_text).read_text(encoding="utf-8") if args.expected_text else None
    layout_markdown: str | None = None
    if args.ocr or args.layout or args.searchable_pdf or args.docx:
        output_image = _read_image(output_path)
        try:
            ocr_result = recognize_image(output_image, language=args.ocr_lang, engine=args.ocr_engine)
        except OcrUnavailableError as exc:
            parser.exit(2, f"OCR engine unavailable: {exc}\n")

        text_path = output_path.with_name(f"{output_path.stem}_ocr.txt")
        text_path.write_text(ocr_result.text, encoding="utf-8")
        report["ocr"] = ocr_result.to_dict()
        report["ocr_text_path"] = str(text_path)

        if args.layout:
            layout_path = output_path.with_name(f"{output_path.stem}_layout.md")
            layout_markdown = recover_layout_markdown(ocr_result)
            layout_path.write_text(layout_markdown, encoding="utf-8")
            report["ocr_layout_path"] = str(layout_path)

        if args.docx:
            if layout_markdown is None:
                layout_markdown = recover_layout_markdown(ocr_result)
            docx_path = output_path.with_name(f"{output_path.stem}_layout.docx")
            docx_export = write_docx(layout_markdown or ocr_result.text, docx_path, title=Path(str(args.input)).stem)
            report["docx"] = docx_export.to_dict()
            report["docx_path"] = str(docx_path)

    if args.readability or ocr_result is not None:
        if output_image is None:
            output_image = _read_image(output_path)
        report["readability"] = evaluate_readability(output_image, ocr_result=ocr_result, expected_text=expected_text)

    if args.pdf or args.searchable_pdf:
        if output_image is None:
            output_image = _read_image(output_path)
        pdf_path = output_path.with_name(f"{output_path.stem}_{'searchable' if args.searchable_pdf else 'scan'}.pdf")
        pdf_export = write_pdf(
            output_image,
            pdf_path,
            title=Path(str(args.input)).stem,
            ocr_result=ocr_result,
            searchable=args.searchable_pdf,
        )
        report["pdf"] = pdf_export.to_dict()
        report["pdf_path"] = str(pdf_path)

    if args.ocr or args.layout or args.pdf or args.searchable_pdf or args.readability or args.docx:
        Path(str(report["report_path"])).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
