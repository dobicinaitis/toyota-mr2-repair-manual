#!/usr/bin/env python
# Regenerates the two files every finished topic used to touch by hand:
# the nav in zensical.toml and the completion checklist in readme.md.
#
# Both are derived from the PDF outline plus the files that exist under docs/,
# so two branches that digitized different topics produce the same result and
# merge without conflicts.
#
# Usage:
#     python utils/sync_nav.py                 # rewrite zensical.toml and readme.md
#     python utils/sync_nav.py --check         # do not write; exit 1 if anything is stale
#
# The nav is generated in full. The checklist is only ticked in place: its tree
# is hand-maintained (completed chapters get their topics collapsed away), so
# this only flips [ ] to [x] and back on the lines that are already there.

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manual_map as mm  # noqa: E402

CHECKBOX = re.compile(r"^(?P<indent>\s*)\* \[(?P<state>[ x])\] (?P<title>.+?)\s*$")
NAV_START = re.compile(r"^nav = \[\s*$")

# Chapters printed in both volumes share one directory, and their outline titles
# differ between the volumes; pick the nav label explicitly.
CHAPTER_NAV_TITLES = {
    "service-specifications": "Service specifications",
}


def nav_title(title):
    """
    Chapter title as the nav spells it: sentence case, but acronyms and model
    codes keep their capitals ("Engine Mechanical" -> "Engine mechanical",
    "EFI System" -> "EFI system", "SRS Airbag" -> "SRS airbag").
    """
    words = title.split(" ")
    out = [words[0]]
    for word in words[1:]:
        # only plain Titlecase words are lowered; EFI, SRS, (A241E), 3S-GTE stay
        out.append(word.lower() if word.isalpha() and word.istitle() else word)
    return " ".join(out)


def chapters_with_pages(entries, docs_dir):
    """
    Nav structure: [(chapter label, [doc paths in outline order])], covering only
    the pages that actually exist. Chapters printed in both volumes are merged
    into the single directory they share.
    """
    docs_dir = Path(docs_dir)
    chapters = {}  # chapter dir -> [paths]
    labels = {}
    order = []
    for entry in entries:
        if entry.level == 1 or entry.title == "Contents":
            continue
        path = mm.doc_path(entry, entries)
        if not path or not (docs_dir / path).exists():
            continue
        directory = mm.chapter_dir(entry)
        if directory not in chapters:
            chapters[directory] = []
            labels[directory] = CHAPTER_NAV_TITLES.get(directory, nav_title(entry.chapter.title))
            order.append(directory)
        if path not in chapters[directory]:
            chapters[directory].append(path)
    return [(labels[d], chapters[d]) for d in order]


def render_nav(chapters):
    lines = ["nav = ["]
    for label, paths in chapters:
        lines.append(f'  {{ "{label}" = [')
        lines.extend(f'      "{path}",' for path in paths)
        lines.append("  ] },")
    lines.append("]")
    return "\n".join(lines)


def replace_nav(text, rendered):
    """Swap the nav = [ … ] block in zensical.toml, leaving the rest of the file untouched."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if NAV_START.match(line)), None)
    if start is None:
        sys.exit("Error: no 'nav = [' block in the config")
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("[") - lines[end].count("]")
        if depth == 0:
            break
    else:
        sys.exit("Error: unterminated nav block in the config")
    return "\n".join(lines[:start] + rendered.splitlines() + lines[end + 1:]) + "\n"


def normalize(title):
    return title.rstrip(".").strip().casefold()


class Item:
    """One checklist line in readme.md."""

    def __init__(self, line, indent, title, state):
        self.line = line
        self.indent = indent
        self.title = title
        self.state = state
        self.children = []
        self.entry = None


def parse_checklist(lines):
    """Build the checklist tree from readme.md, in file order."""
    items = []
    stack = []
    for number, line in enumerate(lines):
        match = CHECKBOX.match(line)
        if not match:
            continue
        item = Item(number, len(match.group("indent")), match.group("title"), match.group("state"))
        while stack and stack[-1].indent >= item.indent:
            stack.pop()
        if stack:
            stack[-1].children.append(item)
        stack.append(item)
        items.append(item)
    return items


def match_outline(items, entries):
    """
    Attach each checklist item to its outline entry. Titles come from the outline,
    but the tree is hand-collapsed, so match by title within the parent's subtree.
    """
    by_parent = {}
    for entry in entries:
        by_parent.setdefault(id(entry.parent) if entry.parent else None, []).append(entry)

    def walk(nodes, candidates):
        for node in nodes:
            match = next((e for e in candidates if normalize(e.title) == normalize(node.title)), None)
            if match is None:
                continue
            node.entry = match
            walk(node.children, by_parent.get(id(match), []))

    roots = [item for item in items if item.indent == min(i.indent for i in items)]
    walk(roots, by_parent.get(None, []))


def leaf_paths(entry, entries):
    """Every leaf doc path under an outline entry (the entry itself when it is a leaf)."""
    if not entry.children:
        path = mm.doc_path(entry, entries)
        return [path] if path else []
    paths = []
    for child in entry.children:
        paths.extend(leaf_paths(child, entries))
    return paths


def tick(item, entries, docs_dir):
    """
    Ticked when the work behind the line is done: all of its children for a line
    with children, otherwise every leaf page of its outline subtree. A line that
    stands for no page at all (Contents, which has no markdown of its own) keeps
    whatever it says.
    """
    for child in item.children:
        tick(child, entries, docs_dir)
    if item.children:
        item.state = "x" if all(child.state == "x" for child in item.children) else " "
    elif item.entry:
        paths = leaf_paths(item.entry, entries)
        if paths:
            item.state = "x" if all((Path(docs_dir) / p).exists() for p in paths) else " "


def update_readme(text, entries, docs_dir):
    lines = text.splitlines()
    items = parse_checklist(lines)
    if not items:
        return text
    before = {item.line: item.state for item in items}
    match_outline(items, entries)
    top = min(item.indent for item in items)
    for item in items:
        if item.indent == top:
            tick(item, entries, docs_dir)
    for item in items:
        # rewrite only the lines whose state changed, so the rest keep their
        # exact text (some carry trailing whitespace)
        if item.state != before[item.line]:
            lines[item.line] = f"{' ' * item.indent}* [{item.state}] {item.title}"
    ending = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + ending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="path to the manual PDF (default: $MR2_DOCS_MANUAL_PATH)")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--config", default="zensical.toml")
    parser.add_argument("--readme", default="readme.md")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if anything is stale")
    args = parser.parse_args()

    entries = mm.load_outline(mm.open_manual(args.pdf))

    stale = []
    for path, new in (
        (Path(args.config), None),
        (Path(args.readme), None),
    ):
        old = path.read_text()
        if path == Path(args.config):
            new = replace_nav(old, render_nav(chapters_with_pages(entries, args.docs)))
        else:
            new = update_readme(old, entries, args.docs)
        if new == old:
            continue
        stale.append(str(path))
        if not args.check:
            path.write_text(new)

    if args.check:
        if stale:
            print("stale, run utils/sync_nav.py: " + ", ".join(stale))
            return 1
        print("nav and checklist are up to date")
        return 0
    print("updated: " + ", ".join(stale) if stale else "nav and checklist already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
