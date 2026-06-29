from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .corners import parse_corner_points
from .evaluation import evaluate_readability
from .export import write_docx, write_pdf, write_pdf_pages
from .ocr import OcrUnavailableError, ocr_engine_status, recognize_image, recover_layout_markdown
from .pipeline import process_file
from .quality import diagnose_scan_quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance document photos and generate an image quality report.")
    parser.add_argument("inputs", nargs="*", help="Path to one or more input images.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--mode", choices=["auto", "color", "gray", "binary"], default="auto", help="Output style.")
    parser.add_argument("--no-warp", action="store_true", help="Disable automatic perspective correction.")
    parser.add_argument("--no-dewarp", action="store_true", help="Disable lightweight textline dewarping.")
    parser.add_argument("--template-image", help="Optional ideal form/template image for template-guided illumination correction.")
    parser.add_argument(
        "--external-restorer-command",
        help="Trusted local command for an optional deep restoration stage. Use {input} and {output} placeholders.",
    )
    parser.add_argument("--external-restorer-timeout", type=float, default=180.0, help="Timeout in seconds for the external restorer command.")
    parser.add_argument(
        "--corners",
        help='Manual document corners in input-image coordinates, for example "10,20 300,18 310,420 8,430" or JSON.',
    )
    parser.add_argument(
        "--corners-space",
        choices=["input", "processed"],
        default="input",
        help="Coordinate space for --corners. Use processed when copying corners from a ClearScan report.",
    )
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


def _perspective_confidence(report: dict[str, object]) -> float:
    detection = report.get("document_detection")
    if isinstance(detection, dict):
        try:
            return float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _refresh_quality_diagnostics(report: dict[str, object], readability: dict[str, object]) -> None:
    output_quality = report.get("output_quality")
    if isinstance(output_quality, dict):
        report["quality_diagnostics"] = diagnose_scan_quality(
            output_quality,
            perspective_confidence=_perspective_confidence(report),
            readability=readability,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ocr_status:
        print(json.dumps(ocr_engine_status(language=args.ocr_lang), indent=2))
        return 0
    if not args.inputs:
        parser.error("input is required unless --ocr-status is used")
    if len(args.inputs) > 1 and args.corners:
        parser.error("--corners can only be used with one input image")
    if len(args.inputs) > 1 and args.expected_text:
        parser.error("--expected-text can only be used with one input image")
    if args.no_warp and args.corners:
        parser.error("--corners cannot be combined with --no-warp")
    try:
        manual_corners = parse_corner_points(args.corners) if args.corners else None
    except ValueError as exc:
        parser.error(str(exc))

    if len(args.inputs) > 1:
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports: list[dict[str, object]] = []
        output_images: list[np.ndarray] = []
        ocr_results = []
        layout_pages: list[str] = []
        for page_index, input_path in enumerate(args.inputs, start=1):
            output_stem = f"{page_index:03d}_{Path(input_path).stem}"
            page_report = process_file(
                input_path=input_path,
                output_dir=output_dir,
                mode=args.mode,
                auto_warp=not args.no_warp,
                auto_dewarp=not args.no_dewarp,
                side_by_side=args.compare,
                output_stem=output_stem,
                template_path=args.template_image,
                external_restorer_command=args.external_restorer_command,
                external_restorer_timeout=args.external_restorer_timeout,
            )
            page_report["page_index"] = page_index
            page_output_path = Path(str(page_report["output_path"]))
            output_image = _read_image(page_output_path)
            output_images.append(output_image)
            ocr_result = None
            if args.ocr or args.layout or args.searchable_pdf or args.docx:
                try:
                    ocr_result = recognize_image(output_image, language=args.ocr_lang, engine=args.ocr_engine)
                except OcrUnavailableError as exc:
                    parser.exit(2, f"OCR engine unavailable: {exc}\n")
                text_path = page_output_path.with_name(f"{page_output_path.stem}_ocr.txt")
                text_path.write_text(ocr_result.text, encoding="utf-8")
                page_report["ocr"] = ocr_result.to_dict()
                page_report["ocr_text_path"] = str(text_path)

                if args.layout or args.docx:
                    layout_markdown = recover_layout_markdown(ocr_result)
                    layout_path = page_output_path.with_name(f"{page_output_path.stem}_layout.md")
                    layout_path.write_text(layout_markdown, encoding="utf-8")
                    page_report["ocr_layout_path"] = str(layout_path)
                    layout_pages.append(
                        f"## Page {page_index}: {Path(input_path).name}\n\n{layout_markdown or ocr_result.text}".strip()
                    )
            ocr_results.append(ocr_result)
            if args.readability:
                page_report["readability"] = evaluate_readability(output_image, ocr_result=ocr_result)
                _refresh_quality_diagnostics(page_report, page_report["readability"])  # type: ignore[arg-type]
                Path(str(page_report["report_path"])).write_text(json.dumps(page_report, indent=2), encoding="utf-8")
            reports.append(page_report)

        batch_report: dict[str, object] = {
            "inputs": args.inputs,
            "output_dir": str(output_dir),
            "mode": args.mode,
            "page_count": len(reports),
            "reports": reports,
        }
        if args.ocr:
            ocr_text_path = output_dir / "clearscan_batch_ocr.txt"
            ocr_text_path.write_text(
                "\n\n".join(
                    f"--- Page {index + 1}: {Path(args.inputs[index]).name} ---\n{result.text if result else ''}"
                    for index, result in enumerate(ocr_results)
                ),
                encoding="utf-8",
            )
            batch_report["batch_ocr_text_path"] = str(ocr_text_path)
        if args.layout or args.docx:
            layout_path = output_dir / "clearscan_batch_layout.md"
            layout_path.write_text("\n\n".join(layout_pages), encoding="utf-8")
            batch_report["batch_layout_path"] = str(layout_path)
        if args.docx:
            docx_path = output_dir / "clearscan_batch_layout.docx"
            docx_export = write_docx("\n\n".join(layout_pages), docx_path, title="clearscan-batch")
            batch_report["docx"] = docx_export.to_dict()
            batch_report["batch_docx_path"] = str(docx_path)
        if args.pdf or args.searchable_pdf:
            pdf_path = output_dir / f"clearscan_batch_{'searchable' if args.searchable_pdf else 'scan'}.pdf"
            pdf_export = write_pdf_pages(
                output_images,
                pdf_path,
                title="clearscan-batch",
                ocr_results=ocr_results if args.searchable_pdf else None,
                searchable=args.searchable_pdf,
            )
            batch_report["pdf"] = pdf_export.to_dict()
            batch_report["batch_pdf_path"] = str(pdf_path)

        batch_report_path = output_dir / "clearscan_batch_report.json"
        batch_report["batch_report_path"] = str(batch_report_path)
        batch_report_path.write_text(json.dumps(batch_report, indent=2), encoding="utf-8")
        print(json.dumps(batch_report, indent=2))
        return 0

    report = process_file(
        input_path=args.inputs[0],
        output_dir=args.out,
        mode=args.mode,
        auto_warp=not args.no_warp,
        auto_dewarp=not args.no_dewarp,
        side_by_side=args.compare,
        manual_corners=manual_corners,
        manual_corners_space=args.corners_space,
        template_path=args.template_image,
        external_restorer_command=args.external_restorer_command,
        external_restorer_timeout=args.external_restorer_timeout,
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
            docx_export = write_docx(layout_markdown or ocr_result.text, docx_path, title=Path(str(args.inputs[0])).stem)
            report["docx"] = docx_export.to_dict()
            report["docx_path"] = str(docx_path)

    if args.readability or ocr_result is not None:
        if output_image is None:
            output_image = _read_image(output_path)
        report["readability"] = evaluate_readability(output_image, ocr_result=ocr_result, expected_text=expected_text)
        _refresh_quality_diagnostics(report, report["readability"])  # type: ignore[arg-type]

    if args.pdf or args.searchable_pdf:
        if output_image is None:
            output_image = _read_image(output_path)
        pdf_path = output_path.with_name(f"{output_path.stem}_{'searchable' if args.searchable_pdf else 'scan'}.pdf")
        pdf_export = write_pdf(
            output_image,
            pdf_path,
            title=Path(str(args.inputs[0])).stem,
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
