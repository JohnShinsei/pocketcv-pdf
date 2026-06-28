# Resume Notes

## One-line project description

PocketCV PDF is a privacy-first mobile image processing app that captures or imports photos, runs browser-side CV operations, and generates a local PDF report containing the original image, enhanced result, edge map, and objective processing metrics.

## Resume bullets

- Built a mobile-first image processing web app that performs local Canvas-based perspective correction, illumination normalization, sharpening, edge-map rendering, grayscale/binary conversion, metric extraction, and PDF report generation without uploading photos.
- Built an image processing pipeline with OpenCV for document boundary detection, perspective correction, illumination normalization, adaptive binarization, and before/after quality scoring.
- Designed a public GitHub-ready project with a Python package, CLI, optional FastAPI dev server, synthetic image tests, and reproducible sample generation.
- Implemented quality metrics such as Laplacian sharpness, contrast, edge density, exposure balance, and score deltas to make image enhancement measurable.

## Interview talking points

- Why contour detection plus perspective transform is a strong baseline for document scanning.
- Why on-device processing is useful for privacy-sensitive document workflows.
- How the browser PDF writer embeds processed JPEG report pages into a valid A4 PDF without a server.
- How edge maps, contrast metrics, and perspective confidence make the project demonstrably about image processing rather than simple file conversion.
- How the project handles failure cases by falling back to the original image border with low confidence.
- Why synthetic test images are useful for public computer vision repositories when real user documents cannot be committed.
- How the system could be extended with manual corner adjustment, OCR evaluation, OpenCV.js, or WebAssembly acceleration.
