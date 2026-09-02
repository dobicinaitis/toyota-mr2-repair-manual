#!/usr/bin/env python
# Renders a region of a page at high resolution for close reading, optionally
# with OCR. Used to verify specification values, tables and charts that are
# hard to read on the 120 dpi page preview.
#
# Usage:
#     python utils/zoom.py 44 --box 60,50,96,80                 # writes .staging/pages/0044/zoom-60-50-96-80.png
#     python utils/zoom.py 44 --box 60,50,96,80 --ocr --psm 6   # also prints OCR text of the region
#
# The box is x0,y0,x1,y1 in percent of the page (as in the manifests).

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helper_functions as hf  # noqa: E402
import manual_map as mm  # noqa: E402
from crop_figure import load_deskewed_page, parse_box  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="View or OCR a page region at high dpi.")
    parser.add_argument("page", type=int, help="PDF page number")
    parser.add_argument("--box", type=parse_box, default=[0, 0, 100, 100], help="x0,y0,x1,y1 in percent")
    parser.add_argument("--dpi", type=int, default=250, help="resolution of the written PNG")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--ocr", action="store_true", help="print OCR text of the region (at 300 dpi)")
    parser.add_argument("--psm", type=int, default=6, help="tesseract page segmentation mode for --ocr")
    parser.add_argument("--staging", default=".staging")
    parser.add_argument("--out", help="output PNG path (default: inside the page's staging directory)")
    parser.add_argument("--pdf")
    args = parser.parse_args()

    doc = mm.open_manual(args.pdf)
    bitmap, _ = load_deskewed_page(doc, args.page, args.staging)
    x0, y0, x1, y1 = hf.pct_to_px(args.box, bitmap.shape)
    crop = bitmap[y0:y1, x0:x1]
    if args.rotate:
        crop = hf.rotate_multiple_of_90(crop, args.rotate)

    if args.out:
        out = Path(args.out)
    else:
        page_dir = Path(args.staging) / "pages" / f"{args.page:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        out = page_dir / ("zoom-" + "-".join(str(int(v)) for v in args.box) + ".png")
    image = hf.scale_to_dpi(crop, args.dpi)
    cv2.imwrite(str(out), image)
    print(f"{out}: {image.shape[1]}x{image.shape[0]} px at {args.dpi} dpi")

    if args.ocr:
        print()
        print(hf.ocr_region(hf.scale_to_dpi(crop, 300), psm=args.psm))


if __name__ == "__main__":
    main()
