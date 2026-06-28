from __future__ import annotations

import argparse
import json

from .pipeline import process_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance document photos and generate an image quality report.")
    parser.add_argument("input", help="Path to an input image.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--mode", choices=["color", "gray", "binary"], default="color", help="Output style.")
    parser.add_argument("--no-warp", action="store_true", help="Disable automatic perspective correction.")
    parser.add_argument("--compare", action="store_true", help="Write a side-by-side comparison image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = process_file(
        input_path=args.input,
        output_dir=args.out,
        mode=args.mode,
        auto_warp=not args.no_warp,
        side_by_side=args.compare,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

