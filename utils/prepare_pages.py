#!/usr/bin/env python
# Stages PDF pages for digitization: renders previews, extracts framed
# illustrations as lossless transparent WebP, runs OCR with the illustrations
# masked out and writes a manifest per page.
#
# Usage:
#     python utils/prepare_pages.py --pages 40-51 --images-dir docs/engine-mechanical/images
#     python utils/prepare_pages.py --topic "Engine tune-up" --images-dir docs/engine-mechanical/images
#
# Output per page in .staging/pages/NNNN/:
#     page.png       grayscale preview of the (deskewed) page
#     overlay.png    preview with detected frames, illustration IDs and step labels
#     ocr.txt        OCR text with the frames masked out
#     ocr.tsv        word level OCR data (positions, confidences)
#     figures/       PNG previews of the extracted illustrations (+ WebP for unnamed frames)
#     manifest.json  page code, header, figures, steps and warnings
# and .staging/run.json summarising the run.

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helper_functions as hf  # noqa: E402
import manual_map as mm  # noqa: E402

STEP_LABEL = re.compile(r"^(\d{1,2}\.|\([a-z]\))$")
TEXT_COLUMN_PCT = (35, 60)  # step labels start in this horizontal band of the page
DESKEW_MIN_DEG = 0.15


def parse_pages(spec):
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        elif part:
            pages.append(int(part))
    return pages


def find_steps(tsv, shape):
    """Step and sub-step labels ("8.", "(a)") in the text column, from the OCR TSV."""
    height, width = shape[:2]
    steps = []
    for line in tsv.splitlines()[1:]:
        columns = line.split("\t")
        if len(columns) < 12:
            continue
        word = columns[11].strip()
        if not STEP_LABEL.match(word):
            continue
        x_pct = int(columns[6]) / width * 100
        y_pct = int(columns[7]) / height * 100
        if TEXT_COLUMN_PCT[0] <= x_pct <= TEXT_COLUMN_PCT[1]:
            steps.append({"label": word, "x_pct": round(x_pct, 1), "y_pct": round(y_pct, 1)})
    return steps


def edge_ink(image, gutter=12):
    """
    Sides of an extracted illustration that still carry ink hard against the edge with a
    blank gutter behind it — what a frame line looks like when trimming missed it.
    :return: names of the offending sides
    """
    ink = image < hf.INK_THRESHOLD
    sides = {"left": ink.mean(axis=0), "right": ink.mean(axis=0)[::-1],
             "top": ink.mean(axis=1), "bottom": ink.mean(axis=1)[::-1]}
    found = []
    for name, profile in sides.items():
        n = 0
        while n < len(profile) and n < 25 and profile[n] > 0:
            n += 1
        if 0 < n <= 8 and profile[n:n + gutter].max() == 0:
            found.append(name)
    return found


def nearest_step(frame_box, steps):
    """
    The step label printed beside a frame: figures are aligned with the top of the step
    they illustrate, so pick the label closest to the frame's top edge (above its bottom).
    """
    _, y0, _, y1 = frame_box
    candidates = [s for s in steps if s["y_pct"] <= y1]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s["y_pct"] - y0))["label"]


def steps_beside(frame_box, steps):
    """All step labels whose line starts within the frame's vertical span."""
    _, y0, _, y1 = frame_box
    return [s["label"] for s in steps if y0 - 1 <= s["y_pct"] <= y1]


def draw_overlay(preview, frames, figures, steps):
    overlay = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    height, width = preview.shape[:2]
    for frame, figure in zip(frames, figures):
        x0, y0, x1, y1 = hf.pct_to_px(frame.bbox_pct, preview.shape)
        colour = (0, 160, 0) if figure["ids"] else (0, 0, 220)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), colour, 2)
        label = figure["name"]
        cv2.putText(overlay, label, (x0 + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    for step in steps:
        x = int(step["x_pct"] / 100 * width)
        y = int(step["y_pct"] / 100 * height)
        cv2.line(overlay, (x - 12, y + 4), (x - 3, y + 4), (220, 0, 0), 2)
        cv2.putText(overlay, step["label"], (int(width * 0.93), y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 0, 0), 1)
    return overlay


def infer_code(previous_code, section):
    if previous_code:
        match = re.match(r"([A-Z]+)-(\d+)", previous_code)
        if match and match.group(1) == section:
            return f"{section}-{int(match.group(2)) + 1}"
    return None


def process_page(doc, entries, page_number, args, previous_code):
    page_dir = Path(args.staging) / "pages" / f"{page_number:04d}"
    manifest_path = page_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())
        manifest["skipped"] = True
        return manifest
    figures_dir = page_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    entry = mm.find_page(entries, page_number)
    section = entry.code if entry else None

    # page bitmap, deskewed
    bitmap = hf.load_page_bitmap(doc, page_number)
    skew = hf.estimate_skew(bitmap)
    if abs(skew) >= DESKEW_MIN_DEG:
        bitmap = hf.deskew(bitmap, skew)

    # framed illustrations
    frames = hf.find_frames(bitmap)
    figures = []
    unnamed = 0
    for frame in frames:
        cut = hf.trim_border(hf.extract_frame(bitmap, frame))
        edges = edge_ink(cut)
        if edges:
            warnings.append(f"frame at {frame.bbox_pct} still has ink on its {', '.join(edges)} edge "
                            f"(leftover border line?)")
        ids = hf.read_frame_ids(cut)
        if ids:
            name = "_".join(ids)
            webp = images_dir / f"{name}.webp"
            # the same illustration is often printed on several pages: keep the first copy
            if not webp.exists():
                hf.save_webp(webp, hf.white_to_alpha(cut))
        else:
            unnamed += 1
            name = f"unnamed-{unnamed}"
            webp = figures_dir / f"{name}.webp"
            hf.save_webp(webp, hf.white_to_alpha(cut))
            # frames without an ID are usually tables or charts: keep their text
            (figures_dir / f"{name}.txt").write_text(hf.ocr_region(hf.scale_to_dpi(cut, 300)))
        preview = figures_dir / f"{name}.png"
        hf.save_preview_png(preview, cut)
        figures.append({
            "name": name,
            "ids": ids,
            "file": str(webp),
            "preview": str(preview.relative_to(page_dir)),
            "bbox_pct": frame.bbox_pct,
            "size_px": [cut.shape[1], cut.shape[0]],
            "angle_deg": frame.angle_deg,
            "full_width": frame.full_width,
        })

    # OCR with frames masked
    low = hf.scale_to_dpi(bitmap, args.ocr_dpi)
    code, header = hf.read_header(low, section)
    code_source = "ocr"
    rotation = 0
    if code is None:
        rotation, confidence = hf.detect_rotation(hf.scale_to_dpi(bitmap, 150))
        if rotation:
            warnings.append(f"page appears rotated by {rotation} deg (confidence {confidence:.1f}); "
                            f"use crop_figure.py --rotate for its content")
        code = infer_code(previous_code, section)
        code_source = "inferred" if code else "unknown"
        warnings.append(f"page code not read from header, {code_source}: {code}")
    text, tsv = hf.ocr_page(low, [f["bbox_pct"] for f in figures])
    (page_dir / "ocr.txt").write_text(text)
    (page_dir / "ocr.tsv").write_text(tsv)
    steps = find_steps(tsv, low.shape)
    for figure in figures:
        figure["near_step"] = nearest_step(figure["bbox_pct"], steps)
        figure["steps_beside"] = steps_beside(figure["bbox_pct"], steps)

    # previews
    preview = hf.scale_to_dpi(bitmap, args.preview_dpi)
    cv2.imwrite(str(page_dir / "page.png"), preview)
    cv2.imwrite(str(page_dir / "overlay.png"), draw_overlay(preview, frames, figures, steps))

    manifest = {
        "pdf_page": page_number,
        "code": code,
        "code_source": code_source,
        "header": header,
        "section": section,
        "topic": entry.title if entry else None,
        "rotation": rotation,
        "skew_deg": round(skew, 2),
        "figures": [f for f in figures if f["ids"]],
        "unnamed_frames": [f for f in figures if not f["ids"]],
        "steps": [{"label": s["label"], "y_pct": s["y_pct"]} for s in steps],
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Stage manual pages for digitization.")
    parser.add_argument("--pdf", help="path to the manual PDF (default: $MR2_DOCS_MANUAL_PATH)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", help="PDF page range, e.g. 40-51 or 40,42,45-47")
    group.add_argument("--topic", help="outline topic title or docs/ path (see manual_map.py --list)")
    parser.add_argument("--chapter", help="chapter of --topic, where the title repeats across chapters")
    parser.add_argument("--images-dir", required=True, help="where named illustration WebPs are written")
    parser.add_argument("--staging", default=".staging", help="staging directory (default: .staging)")
    parser.add_argument("--preview-dpi", type=int, default=120)
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="re-process pages that already have a manifest")
    args = parser.parse_args()

    doc = mm.open_manual(args.pdf)
    entries = mm.load_outline(doc)
    slug = None
    if args.topic:
        entry = mm.find_topic(entries, args.topic, args.chapter)
        pages = list(range(entry.page, entry.end_page + 1))
        slug = mm.slugify(entry.title)
        print(f"{entry.title}: PDF pages {entry.page}-{entry.end_page} -> {mm.doc_path(entry, entries)}")
    else:
        pages = parse_pages(args.pages)

    run = {"pages": [], "warnings": []}
    previous_code = None
    for page_number in pages:
        manifest = process_page(doc, entries, page_number, args, previous_code)
        previous_code = manifest["code"] or previous_code
        figure_names = [f["name"] for f in manifest["figures"]]
        unnamed = len(manifest["unnamed_frames"])
        flag = " (cached)" if manifest.get("skipped") else ""
        print(f"page {page_number:4d}  {manifest['code'] or '?':7} {manifest['code_source']:8} "
              f"figures={figure_names} unnamed={unnamed} skew={manifest['skew_deg']:+.2f}{flag}")
        for warning in manifest["warnings"]:
            print(f"           ! {warning}")
            run["warnings"].append({"page": page_number, "warning": warning})
        run["pages"].append({
            "pdf_page": page_number, "code": manifest["code"], "code_source": manifest["code_source"],
            "figures": figure_names, "unnamed_frames": unnamed, "dir": f"pages/{page_number:04d}",
        })

    # page codes should increase by one per page
    codes = [(p["pdf_page"], p["code"]) for p in run["pages"] if p["code"]]
    for (page_a, code_a), (page_b, code_b) in zip(codes, codes[1:]):
        section_a, number_a = code_a.rsplit("-", 1)
        section_b, number_b = code_b.rsplit("-", 1)
        if section_a == section_b and int(number_b) - int(number_a) != page_b - page_a:
            warning = f"non-monotonic page codes: {code_a} (p{page_a}) -> {code_b} (p{page_b})"
            print(f"! {warning}")
            run["warnings"].append({"page": page_b, "warning": warning})

    Path(args.staging).mkdir(parents=True, exist_ok=True)
    (Path(args.staging) / "run.json").write_text(json.dumps(run, indent=2))
    summary = f"{args.staging}/run.json"
    if slug:
        # run.json is overwritten by every run; a per-topic copy survives staging
        # several topics up front, so each one can still be read back afterwards
        runs_dir = Path(args.staging) / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{slug}.json").write_text(json.dumps(run, indent=2))
        summary += f" and {args.staging}/runs/{slug}.json"
    print(f"\n{len(pages)} pages staged in {args.staging}/pages, summary in {summary}")


if __name__ == "__main__":
    main()
