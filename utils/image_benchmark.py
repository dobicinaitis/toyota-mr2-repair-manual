#!/usr/bin/env python
# Development helper: compares encodings for the extracted illustrations so
# the choice of format (lossless WebP with alpha) can be re-checked.
#
# Usage:
#     python utils/image_benchmark.py docs/introduction/images/*.webp

import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helper_functions as hf  # noqa: E402


def encode_size(image, ext, params):
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
    try:
        cv2.imwrite(path, image, params)
        return os.path.getsize(path)
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description="Compare image encodings for illustrations.")
    parser.add_argument("images", nargs="+", help="WebP or PNG illustrations (alpha = ink)")
    args = parser.parse_args()

    print(f"{'image':36} {'px':>11} {'webp lossy':>11} {'webp lossless':>14} {'png':>9} {'png 1-bit':>10}")
    totals = np.zeros(4, dtype=np.int64)
    for path in args.images:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 3 and image.shape[2] == 4:
            gray = 255 - image[:, :, 3]
        elif image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        bgra = hf.white_to_alpha(gray)
        sizes = [
            encode_size(bgra, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 100]),
            encode_size(bgra, ".webp", [cv2.IMWRITE_WEBP_QUALITY, 101]),
            encode_size(bgra, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 9]),
            encode_size(np.where(gray < 128, 0, 255).astype(np.uint8), ".png",
                        [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_BILEVEL, 1]),
        ]
        totals += sizes
        print(f"{os.path.basename(path):36} {gray.shape[1]}x{gray.shape[0]:<5} "
              f"{sizes[0] / 1024:9.1f} KB {sizes[1] / 1024:11.1f} KB {sizes[2] / 1024:6.1f} KB {sizes[3] / 1024:7.1f} KB")
    print(f"{'total':36} {'':>11} {totals[0] / 1024:9.1f} KB {totals[1] / 1024:11.1f} KB "
          f"{totals[2] / 1024:6.1f} KB {totals[3] / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
