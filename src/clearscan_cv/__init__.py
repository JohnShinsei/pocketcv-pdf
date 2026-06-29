"""ClearScan CV package."""

from .corners import parse_corner_points
from .dewarp import dewarp_by_textline_columns, estimate_textline_column_offsets
from .evaluation import character_error_rate, edit_distance, evaluate_ocr_result, evaluate_readability
from .export import build_docx_bytes, build_pdf_bytes, build_pdf_pages_bytes, write_docx, write_pdf, write_pdf_pages
from .ocr import ocr_engine_status, recognize_image, recover_layout_markdown
from .pipeline import enhance_image, process_file

__all__ = [
    "build_pdf_bytes",
    "build_pdf_pages_bytes",
    "build_docx_bytes",
    "character_error_rate",
    "dewarp_by_textline_columns",
    "edit_distance",
    "enhance_image",
    "evaluate_ocr_result",
    "evaluate_readability",
    "estimate_textline_column_offsets",
    "ocr_engine_status",
    "parse_corner_points",
    "process_file",
    "recognize_image",
    "recover_layout_markdown",
    "write_pdf",
    "write_pdf_pages",
    "write_docx",
]
