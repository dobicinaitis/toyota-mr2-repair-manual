#!/usr/bin/env python
# Rewrites references to printed manual pages into markdown links.
#
#     (See step 4 on page TC-20)      -> (See [Turbocharger › Removal, step 4](../turbocharger-system/turbocharger.md#removal))
#     (See pages EM-26 to 31)          -> (See [Timing belt (3S-GTE) › Removal](../engine-mechanical/timing-belt-3s-gte.md#removal))
#     Refer to page AB-16 of ...       -> Refer to [Steering wheel pad](...) of ...
#     | AB-16 |  (in "Page" columns)   -> | [Steering wheel pad](...) |
#
# Targets are the [](){ #p-xx-nn } page anchors in docs/ (see docs_index.py).
# References whose target page is not digitized yet are left as they are and
# listed in the report. Run site-wide after every new section so that older
# pages gain links.
#
# Usage:
#     python utils/resolve_refs.py                 # rewrite docs/, write .staging/refs-report.md
#     python utils/resolve_refs.py --check         # CI: fail on resolvable-but-unlinked refs or broken links
#     python utils/resolve_refs.py --report -      # print the report to stdout

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docs_index import DocsIndex  # noqa: E402

CODE = r"(?P<sec>[A-Z]{1,3})-(?P<num>\d{1,3})"
SEP = r"\s*(?:,|and|or|to|–|-)\s*"
NUMS = rf"\d+(?:{SEP}\d+)*"
# "page" and "step" are matched case-insensitively: the manual is not consistent
# about it ("Check turbocharging pressure. (See Page TC-7)" on TC-5). The section
# code stays upper case on purpose, so ordinary prose is not mistaken for a code.
STEPS = rf"(?:[Pp]rocedure\s+)?[Ss]teps?\s+(?P<steps>{NUMS})"
PAGES = rf"[Pp]ages?\s+{CODE}(?P<more>(?:{SEP}(?:[A-Z]{{1,3}}-)?\d{{1,3}})*)"
# parenthesised form first so an unresolved "(See page X)" is not matched again by the bare form
REF = re.compile(rf"(?P<paren>\((?P<verb>See|Refer to)\s+(?:{STEPS}\s+on\s+)?{PAGES}\))"
                 rf"|(?P<bare>(?<![\[(])(?:{STEPS.replace('?P<steps>', '?P<steps2>')}\s+on\s+)?"
                 rf"{PAGES.replace('?P<sec>', '?P<sec2>').replace('?P<num>', '?P<num2>').replace('?P<more>', '?P<more2>')})")
# A "Page" cell holds one code, or several separated by <br> when the row's cause
# column is a <ul> of sub-causes with a page each — so <br> delimits a code like | does.
TABLE_CELL = re.compile(rf"(?:(?<=\|)|(?<=<br>))\s*{CODE}(?P<more>(?:\s*,\s*\d{{1,3}})*)\s*(?=\||<br>)")
PROTECTED = re.compile(r"`[^`]*`|!?\[[^\]]*\]\([^)]*\)|\[\]\(\)\{[^}]*\}")
MD_LINK_TO_DOC = re.compile(r"\]\((?P<path>[^)#\s]+\.md)?(?:#(?P<fragment>[^)\s]+))?\)")


def normalise_steps(text):
    """"3, 5 to 8, 10 and 11" -> "3, 5–8, 10, 11"."""
    text = re.sub(r"\s*(?:to|–|-)\s*", "–", text)
    text = re.sub(r"\s*(?:,|and|or)\s*", ", ", text)
    return text


def relative_href(from_path, to_path, fragment):
    if from_path == to_path:
        return f"#{fragment}"
    rel = os.path.relpath(to_path, os.path.dirname(from_path))
    return f"{rel}#{fragment}"


def link_text(doc, anchor, same_file, steps=None, heading_only=False):
    heading = anchor.heading
    if heading is None or heading.level == 1:
        text = doc.title
    elif same_file or heading_only:
        text = heading.text
    else:
        text = f"{doc.title} › {heading.text}"
    if steps:
        steps = normalise_steps(steps)
        text += (", steps " if re.search(r"[,–]", steps) else ", step ") + steps
    return text


class Resolver:
    def __init__(self, index):
        self.index = index
        self.unresolved = defaultdict(list)  # code -> [(path, line, original)]
        self.changes = []  # (path, line, before, after)

    def target(self, code, doc):
        return self.index.resolve(code, doc.volume)

    def make_link(self, doc, code, steps=None, heading_only=False):
        found = self.target(code, doc)
        if not found:
            return None
        target_doc, anchor = found
        same_file = target_doc.path == doc.path
        href = relative_href(doc.path, target_doc.path, anchor.fragment)
        return f"[{link_text(target_doc, anchor, same_file, steps, heading_only)}]({href})"

    def rewrite_line(self, doc, line_no, line, in_page_table):
        protected = [(m.start(), m.end()) for m in PROTECTED.finditer(line)]

        def free(start, end):
            return not any(s < end and start < e for s, e in protected)

        def replace_ref(match):
            if not free(match.start(), match.end()):
                return match.group(0)
            paren = match.group("paren") is not None
            sec, num, steps = (match.group("sec"), match.group("num"), match.group("steps")) if paren else (
                match.group("sec2"), match.group("num2"), match.group("steps2"))
            code = f"{sec}-{int(num)}"
            link = self.make_link(doc, code, steps)
            if not link:
                self.unresolved[code].append((doc.path, line_no, match.group(0)))
                return match.group(0)
            return f"({match.group('verb')} {link})" if paren else link

        # chapter index pages list the topic and its page code side by side; the row
        # already links to the topic, so linking the code again would only duplicate it
        earlier_links = [m.start() for m in MD_LINK_TO_DOC.finditer(line) if m.group("path")]

        def replace_cell(match):
            if any(start < match.start() for start in earlier_links):
                return match.group(0)
            code = f"{match.group('sec')}-{int(match.group('num'))}"
            link = self.make_link(doc, code, heading_only=True)
            if not link:
                self.unresolved[code].append((doc.path, line_no, match.group(0).strip()))
                return match.group(0)
            # pad against a | cell wall, but not against a <br>, which needs no spacing
            text = match.string
            lead = "" if text[max(0, match.start() - 4):match.start()] == "<br>" else " "
            trail = "" if text[match.end():match.end() + 4] == "<br>" else " "
            return f"{lead}{link}{trail}"

        new = REF.sub(replace_ref, line)
        if in_page_table:
            new = TABLE_CELL.sub(replace_cell, new)
        return new

    def process(self, doc, write):
        lines = list(doc.lines)
        in_fence = False
        in_page_table = False
        for i, line in enumerate(lines):
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            stripped = line.strip()
            if stripped.startswith("|"):
                if not in_page_table and re.search(r"\|\s*Page\s*\|", line, re.IGNORECASE):
                    in_page_table = True
            else:
                in_page_table = False
            new = self.rewrite_line(doc, i + 1, line, in_page_table)
            if new != line:
                self.changes.append((doc.path, i + 1, line.strip(), new.strip()))
                lines[i] = new
        if write and lines != doc.lines:
            (self.index.docs_dir / doc.path).write_text("\n".join(lines) + "\n")

    def broken_links(self):
        """Links to .md files or fragments that do not exist."""
        broken = []
        for doc in self.index.files.values():
            for line_no, line in enumerate(doc.lines, 1):
                for match in MD_LINK_TO_DOC.finditer(line):
                    path, fragment = match.group("path"), match.group("fragment")
                    if path and path.startswith(("http:", "https:")):
                        continue
                    target = doc.path if not path else os.path.normpath(
                        os.path.join(os.path.dirname(doc.path), path))
                    if target not in self.index.files:
                        broken.append((doc.path, line_no, f"missing file: {match.group(0)}"))
                    elif fragment and not self.index.has_fragment(target, fragment):
                        broken.append((doc.path, line_no, f"missing anchor: {match.group(0)}"))
        return broken

    def report(self):
        out = ["# Reference report", ""]
        if self.changes:
            out.append(f"## Rewritten ({len(self.changes)})")
            out.append("")
            for path, line_no, before, after in self.changes:
                out.append(f"- `{path}:{line_no}`  \n  `{before}`  \n  → `{after}`")
            out.append("")
        if self.unresolved:
            total = sum(len(v) for v in self.unresolved.values())
            out.append(f"## Unresolved ({total} references to {len(self.unresolved)} pages)")
            out.append("")
            for code in sorted(self.unresolved, key=lambda c: (c.split('-')[0], int(c.split('-')[1]))):
                refs = self.unresolved[code]
                out.append(f"- **{code}** ({len(refs)})")
                for path, line_no, original in refs:
                    out.append(f"    - `{path}:{line_no}` {original}")
            out.append("")
        return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Link manual page references to their markdown targets.")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--config", default="zensical.toml")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if there is anything to fix")
    parser.add_argument("--report", default=".staging/refs-report.md", help="report path, or - for stdout")
    parser.add_argument("paths", nargs="*", help="restrict rewriting to these files (relative to docs/)")
    args = parser.parse_args()

    index = DocsIndex(args.docs, args.config)
    resolver = Resolver(index)
    for rel, doc in index.files.items():
        if args.paths and rel not in args.paths:
            continue
        resolver.process(doc, write=not args.check)
    broken = resolver.broken_links()

    report = resolver.report()
    if broken:
        report += "\n## Broken links\n\n" + "\n".join(f"- `{p}:{l}` {msg}" for p, l, msg in broken) + "\n"
    if args.report == "-":
        print(report)
    else:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report)

    unresolved = sum(len(v) for v in resolver.unresolved.values())
    print(f"{'would rewrite' if args.check else 'rewrote'} {len(resolver.changes)} reference(s), "
          f"{unresolved} unresolved, {len(broken)} broken link(s)"
          + (f"; report: {args.report}" if args.report != "-" else ""))
    for path, line_no, msg in broken:
        print(f"  broken: {path}:{line_no} {msg}")
    if args.check and (resolver.changes or broken):
        for path, line_no, before, after in resolver.changes:
            print(f"  unlinked: {path}:{line_no} {before}")
        sys.exit(1)


if __name__ == "__main__":
    main()
