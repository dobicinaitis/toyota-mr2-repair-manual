#!/usr/bin/env python
# Maps the PDF outline (bookmarks) of the manual to PDF page ranges, section
# codes and target markdown paths under docs/.
#
# Usage:
#     python utils/manual_map.py --list [--chapter "Engine Mechanical"]
#     python utils/manual_map.py --topic "Engine tune-up" [--json]
#     python utils/manual_map.py --page 44 [--json]
#
# The manual path is taken from the MR2_DOCS_MANUAL_PATH environment variable
# or from --pdf.

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

import pymupdf

# Section code printed in the page header of every chapter, per volume.
# Codes A-D repeat in both volumes, hence the volume-aware lookup.
CHAPTER_CODES = {
    (1, "Introduction"): "IN",
    (1, "Maintenance"): "MA",
    (1, "Engine Mechanical"): "EM",
    (1, "Exhaust System"): "EX",
    (1, "Turbocharger System"): "TC",
    (1, "Emission Control Systems"): "EC",
    (1, "EFI System"): "FI",
    (1, "Cooling System"): "CO",
    (1, "Lubrication System"): "LU",
    (1, "Ignition System"): "IG",
    (1, "Starting System"): "ST",
    (1, "Charging System"): "CH",
    (1, "Service Specification"): "A",
    (1, "Standard Bolt Torque Specifications"): "B",
    (2, "Clutch"): "CL",
    (2, "Manual Transaxle"): "MT",
    (2, "Automatic Transaxle (A241E)"): "AT",
    (2, "Suspension and Axle"): "SA",
    (2, "Brake System"): "BR",
    (2, "Steering"): "SR",
    (2, "SRS Airbag"): "AB",
    (2, "Body Electrical System"): "BE",
    (2, "Body"): "BO",
    (2, "Air Conditioning System"): "AC",
    (2, "Service Specifications"): "A",
    (2, "Standard Bolt Torque Specifications"): "B",
    (2, "Special Service Tools and Materials"): "C",
    (2, "Electrical Wiring Diagrams"): "D",
}

# Directory names for chapters whose outline title would give an awkward slug
# or that exist in both volumes.
CHAPTER_DIRS = {
    (1, "Service Specification"): "service-specifications",
    (2, "Service Specifications"): "service-specifications",
    (1, "Standard Bolt Torque Specifications"): "standard-bolt-torque-specifications",
    (2, "Standard Bolt Torque Specifications"): "standard-bolt-torque-specifications",
    (2, "Automatic Transaxle (A241E)"): "automatic-transaxle",
    (2, "Special Service Tools and Materials"): "special-service-tools-and-materials",
}

# Typos in the PDF bookmarks
TITLE_FIXES = {
    "Descritption (3S-GTE)": "Description (3S-GTE)",
    "Genaral troubleshooting": "General troubleshooting",
    "Electronic controlled unit (ECU)": "Electronic control unit (ECU)",
}


@dataclass
class Entry:
    level: int
    title: str
    page: int  # first PDF page (1-based)
    volume: int
    index: int  # position in the outline
    children: list = field(default_factory=list)
    parent: "Entry | None" = None
    end_page: int = 0

    @property
    def chapter(self):
        entry = self
        while entry.level > 2:
            entry = entry.parent
        return entry

    @property
    def code(self):
        return CHAPTER_CODES.get((self.volume, self.chapter.title))


def slugify(text):
    """Convert a title to a lowercase, hyphen separated slug (matches the existing docs/ naming)."""
    text = text.lower().replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def open_manual(pdf_path=None):
    pdf_path = pdf_path or os.environ.get("MR2_DOCS_MANUAL_PATH")
    if not pdf_path:
        sys.exit("Error: pass --pdf or set MR2_DOCS_MANUAL_PATH")
    pdf_path = os.path.expanduser(pdf_path)
    if not os.path.isfile(pdf_path):
        sys.exit(f"Error: manual not found: {pdf_path}")
    return pymupdf.open(pdf_path)


def load_outline(doc):
    """
    Build the outline tree and compute page ranges.
    :return: flat list of entries in document order (with parent/children links)
    """
    entries = []
    stack = []
    volume = 0
    for index, (level, title, page) in enumerate(doc.get_toc()):
        title = TITLE_FIXES.get(title, title)
        if level == 1:
            volume += 1
        entry = Entry(level, title, page, volume, index)
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            entry.parent = stack[-1]
            stack[-1].children.append(entry)
        stack.append(entry)
        entries.append(entry)

    # An entry ends right before the next entry (in document order) that is not
    # one of its descendants and starts on a later page. Some sibling bookmarks
    # share the same start page, so a plain "next entry" is not enough.
    for i, entry in enumerate(entries):
        end = doc.page_count
        for later in entries[i + 1:]:
            if later.level > entry.level and _is_descendant(later, entry):
                continue
            if later.page > entry.page:
                end = later.page - 1
                break
        entry.end_page = end
    return entries


def _is_descendant(entry, ancestor):
    parent = entry.parent
    while parent:
        if parent is ancestor:
            return True
        parent = parent.parent
    return False


def chapter_dir(entry):
    chapter = entry.chapter
    return CHAPTER_DIRS.get((chapter.volume, chapter.title), slugify(chapter.title))


def doc_path(entry, entries):
    """
    Suggested markdown path for an outline entry:
    docs/<chapter>/[<group>/]<topic>.md, with chapter index pages as index.md.
    """
    if entry.level == 1 or entry.title == "Contents":
        return None
    parts = [chapter_dir(entry)]
    if entry.level == 2:
        # chapters present in both volumes share one directory and index page
        return "/".join(parts) + "/index.md"
    ancestors = []
    parent = entry.parent
    while parent and parent.level > 2:
        ancestors.append(slugify(parent.title))
        parent = parent.parent
    parts.extend(reversed(ancestors))
    if entry.children:
        parts.append(slugify(entry.title))
        path = "/".join(parts) + "/index.md"
    else:
        path = "/".join(parts) + "/" + slugify(entry.title) + ".md"

    # the appendix chapters (service specifications, bolt torques) exist in both
    # volumes; keep volume 1 paths clean and suffix the volume 2 twins
    if entry.volume == 2 and not entry.children:
        twins = [e for e in entries if e.volume == 1 and e.level == entry.level and e.title == entry.title
                 and chapter_dir(e) == chapter_dir(entry)]
        if twins:
            path = path[:-3] + "-volume-2.md"
    return path


def describe(entry, entries):
    return {
        "title": entry.title,
        "level": entry.level,
        "volume": entry.volume,
        "chapter": entry.chapter.title if entry.level >= 2 else None,
        "code": entry.code,
        "pages": [entry.page, entry.end_page],
        "page_count": entry.end_page - entry.page + 1,
        "path": doc_path(entry, entries),
        "group": entry.parent.title if entry.parent and entry.parent.level > 2 else None,
        "is_leaf": not entry.children,
    }


def find_topic(entries, query, chapter=None):
    """
    Find an outline entry by title (case-insensitive; exact, then substring match),
    or by its docs/ path, which is unique — titles like "Description" and
    "Troubleshooting" repeat in a dozen chapters, so scripts should address a topic
    by path or pass a chapter to disambiguate.
    """
    q = query.strip().lower()
    if "/" in q or q.endswith(".md"):
        path = q[5:] if q.startswith("docs/") else q
        for entry in entries:
            if (doc_path(entry, entries) or "").lower() == path:
                return entry
        raise SystemExit(f"No outline entry maps to '{query}'. Try --list.")
    if chapter:
        pool = [e for e in entries if e.level >= 2
                and (chapter_dir(e) == chapter or e.chapter.title.lower() == chapter.lower())]
        if not pool:
            raise SystemExit(f"Chapter '{chapter}' not found. Try --list.")
        entries = pool
    exact = [e for e in entries if e.title.lower() == q]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # prefer leaves, then the earliest
        leaves = [e for e in exact if not e.children]
        if len(leaves) == 1:
            return leaves[0]
        raise SystemExit(f"Ambiguous topic '{query}':\n" + "\n".join(
            f"  {e.chapter.title} > {e.title} (pages {e.page}-{e.end_page})" for e in exact))
    partial = [e for e in entries if q in e.title.lower()]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise SystemExit(f"Ambiguous topic '{query}', candidates:\n" + "\n".join(
            f"  {e.chapter.title} > {e.title} (pages {e.page}-{e.end_page})" for e in partial))
    raise SystemExit(f"Topic '{query}' not found in the outline. Try --list.")


def find_page(entries, page):
    """Deepest outline entry containing the given PDF page."""
    best = None
    for entry in entries:
        if entry.page <= page <= entry.end_page and (best is None or entry.level >= best.level):
            best = entry
    return best


def remaining(entries, docs_dir="docs", chapter_filter=None):
    """Leaf topics that have no markdown file yet, in outline order."""
    todo = []
    for entry in entries:
        if entry.children or entry.level < 2 or entry.title == "Contents":
            continue
        if chapter_filter and chapter_dir(entry) != chapter_filter and entry.chapter.title.lower() != chapter_filter.lower():
            continue
        path = doc_path(entry, entries)
        if path and not os.path.exists(os.path.join(docs_dir, path)):
            todo.append(entry)
    return todo


def print_tree(entries, chapter_filter=None):
    for entry in entries:
        if chapter_filter and entry.level >= 2 and entry.chapter.title.lower() != chapter_filter.lower():
            continue
        if chapter_filter and entry.level == 1:
            continue
        indent = "  " * (entry.level - 1)
        path = doc_path(entry, entries) or ""
        code = f"[{entry.code}] " if entry.level == 2 and entry.code else ""
        print(f"{indent}{code}{entry.title}  ({entry.page}-{entry.end_page})  {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="path to the manual PDF (default: $MR2_DOCS_MANUAL_PATH)")
    parser.add_argument("--list", action="store_true", help="print the outline tree with page ranges")
    parser.add_argument("--list-remaining", action="store_true",
                        help="print the leaf topics that have no markdown file yet, one title per line")
    parser.add_argument("--chapter", help="restrict --list/--list-remaining to one chapter (title or directory)")
    parser.add_argument("--docs", default="docs", help="documentation root, for --list-remaining")
    parser.add_argument("--topic", help="resolve a topic title to pages/path")
    parser.add_argument("--page", type=int, help="resolve a PDF page to its chapter/topic")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    args = parser.parse_args()

    doc = open_manual(args.pdf)
    entries = load_outline(doc)

    if args.list:
        print_tree(entries, args.chapter)
        return
    if args.list_remaining:
        todo = remaining(entries, args.docs, args.chapter)
        if args.json:
            print(json.dumps([describe(e, entries) for e in todo], indent=2))
        else:
            for entry in todo:
                print(entry.title)
        return
    if args.topic:
        entry = find_topic(entries, args.topic, args.chapter)
    elif args.page:
        entry = find_page(entries, args.page)
    else:
        parser.print_help()
        return

    info = describe(entry, entries)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
