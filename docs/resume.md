# Resume Notes

## One-line project description

PocketCV PDF is a privacy-first mobile document scanner that captures or imports phone photos, enhances them into standalone scanned images/PDFs, runs OCR, and restores OCR output into searchable PDF, Markdown, and DOCX document formats.

## Resume bullets

- Built a mobile-first image processing web app that performs local Canvas-based perspective correction, illumination normalization, sharpening, grayscale/binary conversion, artifact cleanup, per-page image export, OCR, and PDF generation without uploading photos.
- Built an image processing pipeline with OpenCV for document boundary detection, perspective correction, textline deskew, illumination normalization, stroke-aware binarization, and before/after quality scoring.
- Added a Hough-based near-horizontal textline fallback for deskewing pages where projection-based skew estimation is confused by vertical rules, borders, or low-contrast text.
- Added a lightweight textline-projection dewarping stage that estimates per-column vertical offsets and remaps mildly curved document photos before OCR.
- Added connected-component cleanup for near-edge stains and black border artifacts so cropped scans export as standalone document pages.
- Designed a public GitHub-ready project with a Python package, CLI, optional FastAPI dev server, synthetic image tests, and reproducible sample generation.
- Added a one-command demo runner that produces a processed scan image, comparison image, PDF, OCR backend diagnostics, readability metrics, and a consolidated JSON report for portfolio review.
- Added multi-page CLI/API export that processes several document photos through the OpenCV pipeline and combines them into A4 scan PDFs, searchable OCR PDFs, Markdown, DOCX, and batch JSON reports.
- Implemented quality metrics and diagnostics such as Laplacian sharpness, contrast, edge density, exposure balance, shadow residual, ink density, boldness risk, score deltas, and retake/review recommendations to make image enhancement measurable.
- Added browser-side Tesseract.js OCR over the enhanced scan output with copy, TXT/DOCX export, and hidden-text searchable PDF generation, structured as the second stage before document layout recovery.
- Implemented Python-side optional OCR adapters for RapidOCR, Tesseract, and PaddleOCR, normalizing output into line/word bounding boxes and confidence scores.
- Added manual corner overrides across the Python pipeline, CLI, and API so difficult phone photos can be reproducibly rectified when automatic page detection is imperfect.
- Added OCR backend diagnostics that report missing Python packages, Tesseract binaries, language data, and install hints before users run extraction.
- Implemented layout recovery heuristics that use OCR line bounding boxes to detect columns, headings, and paragraphs, then export Markdown and DOCX.
- Added searchable PDF export with hidden OCR text layers plus OCR readability metrics such as confidence, edit distance, CER, and textline horizontal score.
- Added an auto scan mode that compares binary and grayscale candidates using artifact diagnostics, then selects the more reliable output for OCR/PDF export.

## Interview talking points

- Why contour detection plus perspective transform is a strong baseline for document scanning.
- Why on-device processing is useful for privacy-sensitive document workflows.
- How the browser PDF writer embeds processed scan pages into a valid A4 PDF without a server.
- How edge maps, contrast metrics, and perspective confidence make the project demonstrably about image processing rather than simple file conversion.
- How textline geometry can act as a practical constraint for dewarping without deploying a heavyweight deep model.
- How the project handles failure cases by falling back to the original image border with low confidence.
- Why synthetic test images are useful for public computer vision repositories when real user documents cannot be committed.
- Why OCR should consume the cleaned scan output instead of raw phone photos, and how OCR positions can be converted into readable Markdown structure.
- How OCR confidence, CER, edit distance, and textline horizontality create a feedback loop for tuning image-processing parameters.
