# Index of the markdown documentation: headings, page anchors and nav
# structure. Shared by resolve_refs.py and lint_docs.py; depends only on the
# standard library and the markdown package that ships with zensical, so it
# can run in CI.
#
# Page anchors mark where a page of the printed manual starts:
#     [](){ #p-em-11 }
# They are written on their own line right before the block (usually a
# heading) that starts on that page, or as the first thing in a list item.

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from markdown.extensions.toc import slugify, unique

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR = re.compile(r"\[\]\(\)\{\s*#p-([a-z]{1,3})-(\d{1,3})\s*\}")
ATTR_LIST = re.compile(r"\s*\{[^}]*\}\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
MARKDOWN_MARKUP = re.compile(r"[*_`]|\[([^\]]*)\]\([^)]*\)")


@dataclass
class Heading:
    line: int
    level: int
    text: str
    slug: str


@dataclass
class PageAnchor:
    code: str  # e.g. "EM-11"
    line: int
    heading: "Heading | None"  # heading in effect on that page
    fragment: str  # URL fragment to link to: the heading slug when the anchor introduces it, else the anchor id


@dataclass
class DocFile:
    path: str  # relative to docs/, e.g. "engine-mechanical/engine-tune-up.md"
    title: str
    headings: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    volume: int = 1
    lines: list = field(default_factory=list)

    @property
    def section(self):
        return self.anchors[0].code.split("-")[0] if self.anchors else None


def heading_text(raw):
    """Plain heading text: attr_list and inline markup removed (as the toc extension sees it)."""
    text = ATTR_LIST.sub("", raw)
    text = MARKDOWN_MARKUP.sub(lambda m: m.group(1) or "", text)
    return text.strip()


def parse_file(path, rel_path, volume=1):
    lines = Path(path).read_text().splitlines()
    doc = DocFile(path=rel_path, title=Path(rel_path).stem.replace("-", " ").capitalize(), volume=volume, lines=lines)
    ids = set()
    in_fence = False
    heading_lines = {}
    for number, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING.match(line)
        if match:
            level = len(match.group(1))
            text = heading_text(match.group(2))
            # an explicit id in the attr_list wins over the generated slug
            explicit = re.search(r"\{[^}]*#([\w-]+)[^}]*\}\s*$", match.group(2))
            slug = explicit.group(1) if explicit else unique(slugify(text, "-"), ids)
            ids.add(slug)
            heading = Heading(number, level, text, slug)
            doc.headings.append(heading)
            heading_lines[number] = heading
            if level == 1 and doc.title == Path(rel_path).stem.replace("-", " ").capitalize():
                doc.title = text
    # page anchors: heading in effect is the next heading if it follows immediately, else the previous one
    in_fence = False
    for number, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in ANCHOR.finditer(line):
            code = f"{match.group(1).upper()}-{int(match.group(2))}"
            following = _next_heading(lines, number, heading_lines)
            if following:
                doc.anchors.append(PageAnchor(code, number, following, following.slug))
            else:
                previous = [h for h in doc.headings if h.line < number]
                heading = previous[-1] if previous else None
                doc.anchors.append(PageAnchor(code, number, heading, f"p-{match.group(1)}-{int(match.group(2))}"))
    return doc


def _next_heading(lines, anchor_line, heading_lines):
    """Heading that directly follows an anchor (skipping blank lines), if any."""
    for number in range(anchor_line + 1, min(anchor_line + 4, len(lines) + 1)):
        if number in heading_lines:
            return heading_lines[number]
        if lines[number - 1].strip():
            return None
    return None


def load_nav(config_path="zensical.toml"):
    """
    Flatten the nav from zensical.toml.
    :return: ordered list of doc paths
    """
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    paths = []

    def walk(items):
        for item in items:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        paths.append(value)
                    else:
                        walk(value)

    walk(config["project"].get("nav", []))
    return paths


def volume_of(rel_path):
    """
    Which volume of the printed manual a page came from. The site itself is not split by
    volume; this only disambiguates the page codes that repeat in both (A-x, B-x, C-x, D-x),
    whose volume 2 pages are suffixed by manual_map.doc_path.
    """
    return 2 if rel_path.endswith("-volume-2.md") else 1


class DocsIndex:
    def __init__(self, docs_dir="docs", config_path="zensical.toml"):
        self.docs_dir = Path(docs_dir)
        self.nav_paths = load_nav(config_path)
        self.files = {}
        for path in sorted(self.docs_dir.rglob("*.md")):
            rel = path.relative_to(self.docs_dir).as_posix()
            self.files[rel] = parse_file(path, rel, volume_of(rel))
        self.anchors = {}  # code -> [(DocFile, PageAnchor)]
        for doc in self.files.values():
            for anchor in doc.anchors:
                self.anchors.setdefault(anchor.code, []).append((doc, anchor))

    def resolve(self, code, from_volume=None):
        """
        Find the file and anchor for a manual page code, preferring the referring volume
        when the code exists in both (A-x, B-x, C-x, D-x).
        :return: (DocFile, PageAnchor) or None
        """
        candidates = self.anchors.get(code.upper())
        if not candidates:
            return None
        if from_volume is not None:
            same = [c for c in candidates if c[0].volume == from_volume]
            if same:
                return same[0]
        return candidates[0]

    def has_fragment(self, rel_path, fragment):
        doc = self.files.get(rel_path)
        if not doc:
            return False
        if any(h.slug == fragment for h in doc.headings):
            return True
        return any(f"p-{a.code.lower()}" == fragment for a in doc.anchors)
