# Resume Notes

## One-line project description

PocketCV PDF is a privacy-first mobile document scanner that captures or imports phone photos, enhances them into standalone scanned images/PDFs, and keeps optional analysis reports for debugging image quality.

## Resume bullets

- Built a mobile-first image processing web app that performs local Canvas-based perspective correction, illumination normalization, sharpening, grayscale/binary conversion, artifact cleanup, per-page image export, OCR, and PDF generation without uploading photos.
- Built an image processing pipeline with OpenCV for document boundary detection, perspective correction, illumination normalization, adaptive binarization, and before/after quality scoring.
- Designed a public GitHub-ready project with a Python package, CLI, optional FastAPI dev server, synthetic image tests, and reproducible sample generation.
- Implemented quality metrics such as Laplacian sharpness, contrast, edge density, exposure balance, and score deltas to make image enhancement measurable.
- Added browser-side Tesseract.js OCR over the enhanced scan output with copy and TXT export, structured as the second stage before document layout recovery.

## Interview talking points

- Why contour detection plus perspective transform is a strong baseline for document scanning.
- Why on-device processing is useful for privacy-sensitive document workflows.
- How the browser PDF writer embeds processed scan pages into a valid A4 PDF without a server.
- How edge maps, contrast metrics, and perspective confidence make the project demonstrably about image processing rather than simple file conversion.
- How the project handles failure cases by falling back to the original image border with low confidence.
- Why synthetic test images are useful for public computer vision repositories when real user documents cannot be committed.
- Why OCR should consume the cleaned scan output instead of raw phone photos, and how layout recovery can use OCR positions in the next stage.
