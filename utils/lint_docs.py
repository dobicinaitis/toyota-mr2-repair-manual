#!/usr/bin/env python
# Style and consistency checks for the markdown documentation. Runs in CI
# (standard library + the markdown package only).
#
# Usage:
#     python utils/lint_docs.py                          # lint all of docs/
#     python utils/lint_docs.py docs/engine-mechanical/engine-tune-up.md
#     python utils/lint_docs.py --ocr-audit .staging     # also cross-check numbers against the OCR text
#     python utils/lint_docs.py --fix                    # regenerate generated files (glossary)
#
# Checks:
#   - referenced images exist, every image in an images/ directory is used
#   - illustrations use the #illustration fragment and a width for images narrower than 3000 px
#   - page anchors are well formed, strictly increasing per file and not inside tables/admonitions
#   - no legacy markers (:warning:, :material-lightbulb:, bold CAUTION/HINT/NOTICE), no "to be continued"
#   - no ALL-CAPS headings, no headings inside content tabs
#   - a blank line closes every admonition (without it the following blocks are swallowed)
#   - every docs/**/*.md is in the nav and vice versa
#   - --ocr-audit: every number in a page's text appears in that page's OCR text (warnings only)

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docs_index import ANCHOR, DocsIndex  # noqa: E402

IMAGE = re.compile(r"!\[[^\]]*\]\((?P<src>[^)\s]+)\)(?P<attrs>\{[^}]*\})?")
LEGACY = re.compile(r":warning:|:material-lightbulb:|\*\*(CAUTION|HINT|NOTICE|EXAMPLE)\*\*|"
                    r"^\s*(?:\d+\.\s+|\*\s+)?\*?\*?(CAUTION|HINT|NOTICE|EXAMPLE):", re.M)
FULL_WIDTH_PX = 3000
NUMBER = re.compile(r"\d+(?:\.\d+)?")
# already a link, in backticks, or a page anchor: not a loose reference
PROTECTED = re.compile(r"`[^`]*`|!?\[[^\]]*\]\([^)]*\)|\[\]\(\)\{[^}]*\}")
# looser than resolve_refs.REF on purpose, to catch wordings it does not know
LOOSE_REF = re.compile(r"(?:see|refer to)[^.\n]{0,40}?\b([A-Za-z]{1,3})-(\d{1,3})\b", re.I)
ACRONYM_HEADING_OK = re.compile(r"^[A-Z0-9/\- ()]+$")


def webp_size(path):
    """(width, height) of a WebP file from its header."""
    with open(path, "rb") as f:
        head = f.read(30)
    if head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        return None
    chunk = head[12:16]
    if chunk == b"VP8X":
        w = 1 + int.from_bytes(head[24:27], "little")
        h = 1 + int.from_bytes(head[27:30], "little")
    elif chunk == b"VP8L":
        bits = int.from_bytes(head[21:25], "little")
        w = (bits & 0x3FFF) + 1
        h = ((bits >> 14) & 0x3FFF) + 1
    elif chunk == b"VP8 ":
        w = struct.unpack("<H", head[26:28])[0] & 0x3FFF
        h = struct.unpack("<H", head[28:30])[0] & 0x3FFF
    else:
        return None
    return w, h


class Linter:
    def __init__(self, index):
        self.index = index
        self.errors = []
        self.warnings = []
        self.used_images = set()

    def error(self, path, line, message):
        self.errors.append(f"{path}:{line}: {message}")

    def warn(self, path, line, message):
        self.warnings.append(f"{path}:{line}: {message}")

    def lint_file(self, doc):
        path = f"docs/{doc.path}"
        in_fence = False
        in_table = False
        admonition_indent = None
        admonition_body_seen = False
        tab_indent = None
        last_anchor = None
        previous_line = ""
        for number, line in enumerate(doc.lines, 1):
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))

            # block context
            in_table = stripped.startswith("|")
            # an admonition can also be the content of a list item ("1. !!! warning ...");
            # what matters is the column the "!!!" starts at
            admonition_start = re.match(r"^(\s*(?:(?:\d+\.|[-*+])\s+)?)(?:!!!|\?\?\?)\s", line)
            if admonition_start:
                admonition_indent = len(admonition_start.group(1))
                admonition_body_seen = False
            elif admonition_indent is not None and stripped and indent <= admonition_indent:
                # a line that closes an admonition must be preceded by a blank line, or
                # markdown folds it into the admonition's last paragraph
                if admonition_body_seen and previous_line.strip():
                    self.error(path, number,
                               f"blank line missing before this line; it is swallowed by the admonition above: "
                               f"{stripped[:50]}")
                admonition_indent = None
            elif admonition_indent is not None and stripped:
                admonition_body_seen = True
            if re.match(r'^\s*===\s+"', line):
                tab_indent = indent
            elif tab_indent is not None and stripped and indent <= tab_indent:
                tab_indent = None

            # headings
            if stripped.startswith("#"):
                text = re.sub(r"\s*\{[^}]*\}\s*$", "", stripped.lstrip("#").strip())
                words = re.findall(r"[A-Za-z]{4,}", text)
                if len(words) >= 2 and all(w.isupper() for w in words):
                    self.error(path, number, f"ALL-CAPS heading: {text}")
                if tab_indent is not None:
                    self.error(path, number, "heading inside a content tab (breaks anchors)")

            # page anchors
            for match in ANCHOR.finditer(line):
                code = f"{match.group(1).upper()}-{int(match.group(2))}"
                if in_table:
                    self.error(path, number, f"page anchor {code} inside a table")
                if admonition_indent is not None:
                    self.error(path, number, f"page anchor {code} inside an admonition")
                if last_anchor:
                    last_sec, last_num = last_anchor.split("-")
                    sec, num = code.split("-")
                    if sec != last_sec:
                        self.error(path, number, f"page anchor {code} changes section (previous {last_anchor})")
                    elif int(num) <= int(last_num):
                        self.error(path, number, f"page anchor {code} not increasing (previous {last_anchor})")
                last_anchor = code
            if re.search(r"\[\]\(\)\{\s*#p-", line) and not ANCHOR.search(line):
                self.error(path, number, f"malformed page anchor: {stripped}")

            # images
            for match in IMAGE.finditer(line):
                src, attrs = match.group("src"), match.group("attrs") or ""
                if src.startswith(("http:", "https:")):
                    continue
                file_part = src.split("#")[0]
                target = (self.index.docs_dir / doc.path).parent / file_part
                if not target.exists():
                    self.error(path, number, f"image not found: {src}")
                    continue
                self.used_images.add(target.resolve())
                if target.suffix == ".webp":
                    if not src.endswith("#illustration"):
                        self.warn(path, number, f"illustration without #illustration fragment: {src}")
                    size = webp_size(target)
                    if size and size[0] <= FULL_WIDTH_PX and "width=" not in attrs:
                        self.error(path, number, f"illustration {size[0]} px wide needs a width attribute: {src}")
                    if size and size[0] > FULL_WIDTH_PX and "width=" in attrs:
                        self.warn(path, number, f"full width illustration ({size[0]} px) has a width attribute: {src}")

            # legacy markers and leftovers
            if LEGACY.search(line):
                self.error(path, number, f"legacy marker, use an admonition: {stripped[:60]}")
            if "to be continued" in line.lower():
                self.error(path, number, "unfinished page marker")
            if re.search(r"\bIb\b|\bIbf\b", line):
                self.error(path, number, "OCR typo 'Ib' for 'lb'")

            previous_line = line

    def lint_images(self):
        for images_dir in self.index.docs_dir.rglob("images"):
            for image in images_dir.iterdir():
                if image.is_file() and image.resolve() not in self.used_images:
                    self.warn(str(image), 0, "image not referenced by any page")

    def lint_unlinked_refs(self, doc):
        """
        A "see page XX-n" whose target exists but which is still plain text.

        resolve_refs.py leaves a reference alone when its target is not digitized
        yet, which is normal. What is not normal is a reference whose anchor does
        exist: that means its wording did not match the resolver's pattern, and
        neither the resolver nor its --check can see the problem. This looks for
        them with a deliberately looser pattern than the resolver's own.
        """
        path = f"docs/{doc.path}"
        in_fence = False
        for number, line in enumerate(doc.lines, 1):
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            # drop anything already linked, or in backticks, or an anchor
            bare = PROTECTED.sub(" ", line)
            for match in LOOSE_REF.finditer(bare):
                code = f"{match.group(1).upper()}-{int(match.group(2))}"
                if self.index.resolve(code, doc.volume):
                    self.error(path, number,
                               f"reference to {code} is not linked although the page exists; "
                               f"resolve_refs.py did not recognise the wording: {match.group(0).strip()}")

    def lint_chapter_indexes(self):
        """
        Every digitized topic is linked from its chapter's contents page. That page
        is transcribed from the printed one, so it cannot be generated (its wording
        is the manual's, not the outline's) — but it does drift as topics land.
        """
        for rel, doc in self.index.files.items():
            if not rel.endswith("/index.md"):
                continue
            chapter = str(Path(rel).parent)
            linked = set()
            for line in doc.lines:
                for match in re.finditer(r"\[[^\]]*\]\(([^)#]+\.md)[^)]*\)", line):
                    linked.add((Path(rel).parent / match.group(1)).as_posix())
            for other in self.index.files:
                if other == rel or not other.startswith(chapter + "/") or other.endswith("/index.md"):
                    continue
                if other not in linked:
                    self.warn(f"docs/{rel}", 0, f"chapter contents page does not link {other}")

    def lint_nav(self):
        nav = set(self.index.nav_paths)
        for rel in self.index.files:
            # the site home page is reached through the logo, not through the nav
            if rel not in nav and rel != "index.md":
                self.error(f"docs/{rel}", 0, "page not in nav (zensical.toml)")
        for rel in nav:
            if rel not in self.index.files:
                self.error("zensical.toml", 0, f"nav entry without a file: {rel}")

    def ocr_audit(self, staging):
        """Every number written for a page must appear in that page's OCR text."""
        ocr_by_code = {}
        for manifest in Path(staging).glob("pages/*/manifest.json"):
            data = json.loads(manifest.read_text())
            if data.get("code"):
                text = (manifest.parent / "ocr.txt").read_text()
                for extra in manifest.parent.glob("figures/*.txt"):
                    text += "\n" + extra.read_text()
                ocr_by_code[data["code"]] = self._normalise(text)
        for doc in self.index.files.values():
            if not doc.anchors:
                continue
            boundaries = [(a.code, a.line) for a in doc.anchors]
            for i, (code, start) in enumerate(boundaries):
                end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(doc.lines) + 1
                if code not in ocr_by_code:
                    continue
                segment = "\n".join(doc.lines[start - 1:end - 1])
                segment = self._strip_markup(segment)
                missing = sorted({n for n in NUMBER.findall(segment) if n not in ocr_by_code[code]})
                if missing:
                    self.warn(f"docs/{doc.path}", start, f"{code}: numbers not found in OCR: {', '.join(missing)}")

    @staticmethod
    def _normalise(text):
        return re.sub(r"[–—]", "-", text)

    @staticmethod
    def _strip_markup(text):
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)\{[^}]*\}|!\[[^\]]*\]\([^)]*\)", " ", text)  # images
        text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
        text = re.sub(r"\[\]\(\)\{[^}]*\}", " ", text)  # anchors
        text = re.sub(r"<[^>]+>", " ", text)  # html
        text = re.sub(r"\{[^}]*\}", " ", text)  # attr lists
        text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)  # ordered list markers
        text = re.sub(r"^#{1,6}\s+\d+\.\s", " ", text, flags=re.M)  # numbered headings
        return text


def main():
    parser = argparse.ArgumentParser(description="Lint the markdown documentation.")
    parser.add_argument("paths", nargs="*", help="files to lint (default: all of docs/)")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--config", default="zensical.toml")
    parser.add_argument("--ocr-audit", metavar="STAGING", help="cross-check numbers against staged OCR text")
    parser.add_argument("--fix", action="store_true", help="regenerate generated files")
    args = parser.parse_args()

    if args.fix:
        here = os.path.dirname(__file__)
        subprocess.run([sys.executable, os.path.join(here, "build_glossary.py")], check=True)
        # the nav and the checklist are derived from the outline, so this needs
        # the PDF; skip it (with a warning) where it is not available, as in CI
        if os.environ.get("MR2_DOCS_MANUAL_PATH"):
            subprocess.run([sys.executable, os.path.join(here, "sync_nav.py"),
                            "--docs", args.docs, "--config", args.config], check=True)
        else:
            print("warning: MR2_DOCS_MANUAL_PATH is not set, skipping the nav and checklist")

    index = DocsIndex(args.docs, args.config)
    linter = Linter(index)
    selected = {os.path.relpath(p, args.docs) for p in args.paths} if args.paths else None
    for rel, doc in index.files.items():
        if selected is None or rel in selected:
            linter.lint_file(doc)
            linter.lint_unlinked_refs(doc)
    if selected is None:
        linter.lint_images()
        linter.lint_chapter_indexes()
        linter.lint_nav()
    if args.ocr_audit:
        linter.ocr_audit(args.ocr_audit)

    for warning in linter.warnings:
        print(f"warning: {warning}")
    for error in linter.errors:
        print(f"error: {error}")
    print(f"{len(linter.errors)} error(s), {len(linter.warnings)} warning(s)")
    if linter.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
