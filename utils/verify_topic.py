#!/usr/bin/env python
# Checks a digitized page for content that was left out.
#
# lint_docs.py --ocr-audit checks the forward direction: every number written for
# a page appears in that page's OCR text. This checks the reverse -- pages that
# were staged but never written, figures that were extracted but never placed,
# steps the manual numbers but the markdown skips, and numbers the OCR has that
# the markdown does not. That is the failure an unattended agent produces, and
# nothing else catches it.
#
# Usage:
#     python utils/verify_topic.py docs/engine-mechanical/timing-belt-3s-gte.md
#     python utils/verify_topic.py --all --staging .staging
#
# Findings are reported, not enforced: an omission is often correct (a repeated
# illustration, a figure that belongs to no step). Treat the output as the review
# queue for a topic, not as a build failure.

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docs_index import DocsIndex  # noqa: E402

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
OCR_STEP = re.compile(r"^\s*(\d{1,2})\.\s+\S")
# a numbered operation is a list item, or a heading that keeps the manual's
# number ("### 5. Replace spark plugs"), which the schedule tables link to
MD_STEP = re.compile(r"^\s*(?:#{1,6}\s+)?(\d{1,2})\.\s+\S")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)")
# tokens that are page furniture rather than content
NOISE = re.compile(r"^(19\d\d|20\d\d)$")
# the manual's own illustration IDs (EM8548, AB0014_AB0257); anything else is a
# figure cropped by hand with crop_figure.py
MANUAL_ID = re.compile(r"^[A-Z]{1,3}\d{3,5}(_[A-Z]{1,3}\d{3,5})*$")


def staged_pages(staging):
    """Page code -> (manifest, directory) for everything staged."""
    pages = {}
    for manifest_path in Path(staging).glob("pages/*/manifest.json"):
        data = json.loads(manifest_path.read_text())
        if data.get("code"):
            pages[data["code"]] = (data, manifest_path.parent)
    return pages


def page_slices(doc):
    """Page code -> the markdown lines written for that page."""
    slices = {}
    boundaries = [(a.code, a.line) for a in doc.anchors]
    for i, (code, start) in enumerate(boundaries):
        end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(doc.lines) + 1
        slices[code] = doc.lines[start - 1:end - 1]
    return slices


def numbers(text):
    return {n.rstrip(".,") for n in NUMBER.findall(text) if not NOISE.match(n)}


def verify(doc, pages, chapter_images, shared_codes=frozenset()):
    """
    Report (severity, message) findings for one page.

    Scope matters here. A section's pages are split across several files
    (MA-1..3 in maintenance-schedule.md, MA-4..9 in maintenance-operations.md),
    so only the pages inside this file's own anchor range are its responsibility.
    Illustrations, on the other hand, are deduplicated per chapter directory: a
    figure printed twice is stored once and may be placed by a sibling file, so
    figure use is checked against the whole chapter.
    """
    findings = []
    slices = page_slices(doc)
    codes = [a.code for a in doc.anchors]
    if not codes:
        return findings

    section = codes[0].split("-")[0]
    seen = sorted(int(c.split("-")[1]) for c in codes)
    for n in range(seen[0], seen[-1] + 1):
        if n not in seen:
            findings.append(("error", f"no page anchor for {section}-{n}, "
                                      f"inside this file's range {section}-{seen[0]} to {section}-{seen[-1]}"))

    for code, (manifest, directory) in sorted(pages.items()):
        parts = code.split("-")
        if parts[0] != section or not (seen[0] <= int(parts[1]) <= seen[-1]):
            continue
        if code not in slices:
            continue
        body = "\n".join(slices[code])
        # ~24 pairs of outline topics share pages (Identification information and
        # General repair instructions are both 6-8), so one page's OCR is split
        # across two files and neither can be measured against the whole of it
        split_page = code in shared_codes

        for figure in manifest.get("figures", []):
            name = Path(figure.get("file", figure.get("name", ""))).stem
            if not name or f"{name}.webp" in chapter_images:
                continue
            if Path(figure.get("file", "")).exists():
                continue  # still on disk: lint_docs.py reports it as unreferenced
            # removed rather than placed, which is right for a framed table that
            # was transcribed instead -- worth confirming the content survived
            findings.append(("info", f"page {code}: figure {name} was extracted and then removed; "
                                     f"check its content was transcribed"))

        ocr = (directory / "ocr.txt").read_text()
        ocr_steps = {int(m.group(1)) for m in (OCR_STEP.match(line) for line in ocr.splitlines()) if m}
        md_steps = {int(m.group(1)) for m in (MD_STEP.match(line) for line in slices[code]) if m}
        # a step continued from the previous page is written there, not here, so
        # only flag numbers the page starts that the markdown never picks up at all
        skipped = sorted(ocr_steps - md_steps - all_steps(doc))
        if skipped and not split_page:
            findings.append(("info", f"page {code}: step numbers in the OCR that the markdown never uses: "
                                     f"{', '.join(str(s) for s in skipped)}"))

        # under-transcription shows up as volume, not as individual values: the
        # OCR text and the markdown for one page should be broadly comparable
        # a chart cropped by hand was unframed, so prepare_pages.py could not mask
        # it out of ocr.txt: the whole chart is still in the OCR while the markdown
        # rightly shows it as one image (the shim selection charts, EM-14 and EM-15)
        cropped = any(not MANUAL_ID.match(Path(m.split("#")[0]).stem) for m in IMAGE.findall(body))

        ocr_words = len(ocr.split())
        body_words = len(re.sub(r"[|*#!\[\]()<>-]", " ", body).split())
        if ocr_words >= 120 and body_words < ocr_words * 0.35 and not split_page and not cropped:
            findings.append(("warning", f"page {code}: {body_words} words written for {ocr_words} words of OCR; "
                                        f"likely under-transcribed"))
    return findings


def all_steps(doc):
    """Every step number the file uses anywhere (steps run across page breaks)."""
    return {int(m.group(1)) for m in (MD_STEP.match(line) for line in doc.lines) if m}


def main():
    parser = argparse.ArgumentParser(description="Report content a digitized page left out.")
    parser.add_argument("paths", nargs="*", help="markdown files to verify (default: --all)")
    parser.add_argument("--all", action="store_true", help="verify every page with anchors")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--config", default="zensical.toml")
    parser.add_argument("--staging", default=".staging")
    parser.add_argument("--strict", action="store_true", help="exit 1 when anything is reported")
    args = parser.parse_args()

    if not args.paths and not args.all:
        parser.error("name at least one markdown file, or pass --all")

    index = DocsIndex(args.docs, args.config)
    pages = staged_pages(args.staging)
    selected = {os.path.relpath(p, args.docs) for p in args.paths} if args.paths else None

    # illustrations are deduplicated per chapter directory, so a figure may be
    # placed by any file of the chapter
    chapter_images = {}
    for rel, doc in index.files.items():
        chapter = str(Path(rel).parent)
        chapter_images.setdefault(chapter, set()).update(
            # illustrations carry a "#illustration" fragment that drives the
            # dark mode inversion; it is not part of the file name
            Path(m.split("#")[0]).name for m in IMAGE.findall("\n".join(doc.lines)))

    # page codes written by more than one file: those pages are split between
    # two topics whose outline ranges overlap
    claims = {}
    for rel, doc in index.files.items():
        for anchor in doc.anchors:
            claims.setdefault(anchor.code, set()).add(rel)
    shared = {code for code, files in claims.items() if len(files) > 1}

    total = 0
    for rel, doc in index.files.items():
        if selected is not None and rel not in selected:
            continue
        if not doc.anchors:
            continue
        findings = verify(doc, pages, chapter_images.get(str(Path(rel).parent), set()), shared)
        if findings:
            print(f"\ndocs/{rel}")
            for severity, message in findings:
                print(f"  {severity}: {message}")
        total += len(findings)

    print(f"\n{total} finding(s)")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
