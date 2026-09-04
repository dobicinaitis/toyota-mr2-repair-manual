#!/usr/bin/env python
# Crops an unframed figure, chart or table from a page and saves it as a
# lossless transparent WebP, the same way prepare_pages.py saves framed
# illustrations.
#
# Usage:
#     python utils/crop_figure.py 44 --box 6,9.5,96,84 --id shim-chart-intake --images-dir docs/engine-mechanical/images
#
# The box is x0,y0,x1,y1 in percent of the page (same coordinate system as
# bbox_pct in the staging manifests and the page.png preview). The page is
# deskewed with the angle recorded in the manifest (or estimated) before
# cropping, the crop is trimmed to its ink and a PNG preview is written next
# to the page's other figures in .staging.
#
# Pass --framed when the box takes in a printed frame line, as when one detected
# frame holds two diagrams that belong in separate files: the line is removed the
# way prepare_pages.py removes it from the frames it extracts itself.

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helper_functions as hf  # noqa: E402
import manual_map as mm  # noqa: E402

FULL_WIDTH_PX = 3000  # figures wider than this are shown without a width attribute


def parse_box(text):
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("box must be x0,y0,x1,y1 in percent")
    return values


def load_deskewed_page(doc, page_number, staging):
    """Page bitmap deskewed with the manifest angle when available (keeps coordinates consistent)."""
    bitmap = hf.load_page_bitmap(doc, page_number)
    manifest = Path(staging) / "pages" / f"{page_number:04d}" / "manifest.json"
    if manifest.exists():
        skew = json.loads(manifest.read_text()).get("skew_deg", 0.0)
    else:
        skew = hf.estimate_skew(bitmap)
    if abs(skew) >= 0.15:
        bitmap = hf.deskew(bitmap, skew)
    return bitmap, skew


def figure_snippet(rel_path, width_px, indent=0):
    pad = " " * indent
    attrs = "" if width_px > FULL_WIDTH_PX else '{ width="80%" }'
    return f'{pad}<figure markdown="span">\n{pad}  ![](images/{rel_path}#illustration){attrs}\n{pad}</figure>'


def main():
    parser = argparse.ArgumentParser(description="Crop an unframed figure from a manual page.")
    parser.add_argument("page", type=int, help="PDF page number")
    parser.add_argument("--box", type=parse_box, required=True, help="x0,y0,x1,y1 in percent of the page")
    parser.add_argument("--id", required=True, help="file name without extension, e.g. shim-chart-intake")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--staging", default=".staging")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="rotate the crop clockwise (for landscape pages)")
    parser.add_argument("--framed", action="store_true",
                        help="the box includes a printed frame line; remove it before trimming to ink")
    parser.add_argument("--no-trim", action="store_true", help="keep the exact box instead of trimming to ink")
    parser.add_argument("--pad", type=int, default=20, help="padding kept around the ink when trimming")
    parser.add_argument("--force", action="store_true", help="overwrite an existing image")
    parser.add_argument("--pdf")
    args = parser.parse_args()

    doc = mm.open_manual(args.pdf)
    bitmap, skew = load_deskewed_page(doc, args.page, args.staging)
    x0, y0, x1, y1 = hf.pct_to_px(args.box, bitmap.shape)
    crop = bitmap[y0:y1, x0:x1]
    if args.rotate:
        crop = hf.rotate_multiple_of_90(crop, args.rotate)
    if args.framed:
        crop = hf.trim_border(crop)
    if not args.no_trim:
        crop = hf.trim_to_ink(crop, args.pad)

    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    webp = images_dir / f"{args.id}.webp"
    if webp.exists() and not args.force:
        sys.exit(f"Error: {webp} exists, use --force to overwrite")
    hf.save_webp(webp, hf.white_to_alpha(crop))

    figures_dir = Path(args.staging) / "pages" / f"{args.page:04d}" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    preview = figures_dir / f"{args.id}.png"
    hf.save_preview_png(preview, crop, max_width=1000)

    height, width = crop.shape[:2]
    print(f"{webp}: {width}x{height} px, {webp.stat().st_size / 1024:.1f} KB (skew {skew:+.2f} deg)")
    print(f"preview: {preview}")
    print()
    print(figure_snippet(f"{args.id}.webp", width))


if __name__ == "__main__":
    main()
