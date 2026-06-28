from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clearscan_cv.pipeline import process_file  # noqa: E402


def synthetic_document() -> np.ndarray:
    image = np.full((720, 960, 3), (42, 48, 54), dtype=np.uint8)
    document = np.array([[185, 78], [795, 138], [730, 642], [118, 575]], dtype=np.int32)
    cv2.fillConvexPoly(image, document, (232, 235, 229))

    for i in range(12):
        y = 170 + i * 34
        start = (235, y)
        end = (690 if i % 3 else 620, y + 8)
        cv2.line(image, start, end, (62, 70, 82), 5, cv2.LINE_AA)

    cv2.putText(image, "CLEARSCAN", (250, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 38, 48), 3, cv2.LINE_AA)
    gradient = np.tile(np.linspace(0, 58, image.shape[1], dtype=np.uint8), (image.shape[0], 1))
    shadow = cv2.merge([gradient, gradient, gradient])
    return cv2.subtract(image, shadow)


def main() -> int:
    sample_dir = ROOT / "examples" / "generated"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_path = sample_dir / "sample_document.jpg"
    cv2.imwrite(str(input_path), synthetic_document())
    report = process_file(input_path, sample_dir, mode="color", side_by_side=True)
    print(f"Wrote {report['output_path']}")
    print(f"Wrote {report['report_path']}")
    print(f"Wrote {report['comparison_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

