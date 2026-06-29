"""ClearScan CV package."""

from .evaluation import character_error_rate, edit_distance, evaluate_ocr_result, evaluate_readability
from .export import build_docx_bytes, build_pdf_bytes, write_docx, write_pdf
from .ocr import recognize_image, recover_layout_markdown
from .pipeline import enhance_image, process_file

__all__ = [
    "build_pdf_bytes",
    "build_docx_bytes",
    "character_error_rate",
    "edit_distance",
    "enhance_image",
    "evaluate_ocr_result",
    "evaluate_readability",
    "process_file",
    "recognize_image",
    "recover_layout_markdown",
    "write_pdf",
    "write_docx",
]
