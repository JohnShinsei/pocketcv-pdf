from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.evaluation import evaluate_readability  # noqa: E402
from clearscan_cv.export import write_docx, write_pdf  # noqa: E402
from clearscan_cv.ocr import OcrUnavailableError, ocr_engine_status, recognize_image, recover_layout_markdown  # noqa: E402
from clearscan_cv.pipeline import process_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full PocketCV demo pipeline and write scan, PDF, OCR diagnostics, and a summary report."
    )
    parser.add_argument("input", nargs="?", help="Path to a document photo. If omitted, a synthetic demo photo is generated.")
    parser.add_argument("--out", default="outputs/demo", help="Output directory for demo artifacts.")
    parser.add_argument("--mode", choices=["color", "gray", "binary"], default="binary", help="Scan output style.")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="OCR language code, for example jpn+eng, eng, chi_sim+eng.")
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "rapidocr", "tesseract", "paddleocr"],
        default="auto",
        help="Optional OCR backend. Auto uses the first installed backend suitable for the requested language.",
    )
    parser.add_argument("--expected-text", help="Optional reference text for OCR edit distance and CER.")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR extraction but still write OCR backend diagnostics.")
    parser.add_argument("--require-ocr", action="store_true", help="Return a non-zero exit code if OCR cannot run.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip image-only PDF export.")
    parser.add_argument("--no-dewarp", action="store_true", help="Disable lightweight textline dewarping.")
    parser.add_argument("--no-warp", action="store_true", help="Disable automatic perspective correction.")
    return parser


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def _default_sample_path(output_dir: Path) -> Path:
    return output_dir / "sample_document.jpg"


def ensure_input_image(input_path: str | None, output_dir: Path) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image does not exist: {path}")
        return path

    from generate_sample import synthetic_document  # noqa: WPS433

    sample_path = _default_sample_path(output_dir)
    if not sample_path.exists():
        _write_image(sample_path, synthetic_document())
    return sample_path


def read_image(path: str | Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _read_expected_text(path: str | None) -> str | None:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _first_install_hint(status: dict[str, Any]) -> str | None:
    engines = status.get("engines", {})
    if not isinstance(engines, dict):
        return None
    for payload in engines.values():
        if isinstance(payload, dict) and payload.get("install"):
            return str(payload["install"])
    return None


def run_demo(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = ensure_input_image(args.input, output_dir)

    scan_report = process_file(
        input_path=input_path,
        output_dir=output_dir,
        mode=args.mode,
        auto_warp=not args.no_warp,
        auto_dewarp=not args.no_dewarp,
        side_by_side=True,
    )
    output_path = Path(str(scan_report["output_path"]))
    output_image = read_image(output_path)
    expected_text = _read_expected_text(args.expected_text)

    artifacts: dict[str, Any] = {
        "scan_image": str(output_path),
        "comparison_image": scan_report.get("comparison_path"),
        "pipeline_report": scan_report.get("report_path"),
    }
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "artifacts": artifacts,
        "document_detection": scan_report.get("document_detection"),
        "dewarp": scan_report.get("dewarp"),
        "deskew": scan_report.get("deskew"),
        "quality": scan_report.get("quality"),
        "output_quality": scan_report.get("output_quality"),
        "ocr_status": ocr_engine_status(language=args.ocr_lang),
    }

    if not args.no_pdf:
        pdf_path = output_path.with_name(f"{output_path.stem}_scan.pdf")
        pdf_export = write_pdf(output_image, pdf_path, title=input_path.stem, searchable=False)
        artifacts["scan_pdf"] = pdf_export.path
        summary["pdf"] = pdf_export.to_dict()

    ocr_result = None
    exit_code = 0
    if args.no_ocr:
        summary["ocr"] = {"attempted": False, "reason": "disabled_by_user"}
    else:
        try:
            ocr_result = recognize_image(output_image, language=args.ocr_lang, engine=args.ocr_engine)
            text_path = output_path.with_name(f"{output_path.stem}_ocr.txt")
            text_path.write_text(ocr_result.text, encoding="utf-8")
            layout_markdown = recover_layout_markdown(ocr_result)
            layout_path = output_path.with_name(f"{output_path.stem}_layout.md")
            layout_path.write_text(layout_markdown, encoding="utf-8")
            docx_path = output_path.with_name(f"{output_path.stem}_layout.docx")
            docx_export = write_docx(layout_markdown or ocr_result.text, docx_path, title=input_path.stem)
            searchable_path = output_path.with_name(f"{output_path.stem}_searchable.pdf")
            searchable_export = write_pdf(
                output_image,
                searchable_path,
                title=input_path.stem,
                ocr_result=ocr_result,
                searchable=True,
            )

            artifacts["ocr_text"] = str(text_path)
            artifacts["layout_markdown"] = str(layout_path)
            artifacts["layout_docx"] = docx_export.path
            artifacts["searchable_pdf"] = searchable_export.path
            summary["ocr"] = {
                "attempted": True,
                "engine": ocr_result.engine,
                "language": ocr_result.language,
                "line_count": len(ocr_result.lines),
                "confidence": ocr_result.confidence,
                "text_path": str(text_path),
                "layout_path": str(layout_path),
            }
            summary["docx"] = docx_export.to_dict()
            summary["searchable_pdf"] = searchable_export.to_dict()
        except OcrUnavailableError as exc:
            hint = _first_install_hint(summary["ocr_status"])
            summary["ocr"] = {
                "attempted": True,
                "available": False,
                "error": str(exc),
                "install_hint": hint,
            }
            exit_code = 2 if args.require_ocr else 0

    summary["readability"] = evaluate_readability(output_image, ocr_result=ocr_result, expected_text=expected_text)
    summary_path = output_dir / "demo_summary.json"
    summary["artifacts"]["demo_summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary, exit_code = run_demo(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
