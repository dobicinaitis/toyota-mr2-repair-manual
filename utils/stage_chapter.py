#!/usr/bin/env python
# Stages every not-yet-digitized topic of one or more chapters ahead of time,
# so the digitizing agent never waits for rendering and OCR.
#
# Usage:
#     python utils/stage_chapter.py turbocharger-system exhaust-system
#     python utils/stage_chapter.py --all --jobs 8
#     python utils/stage_chapter.py cooling-system --dry-run
#
# Chapters run in parallel, topics within a chapter run one after another: the
# outline has ~24 pairs of topics that share pages (FI "Precautions" and
# "Inspection precautions" are both 322-327), and those pairs are always inside
# one chapter, so a serial stream per chapter is what keeps two processes from
# writing the same .staging/pages/NNNN/ at once.
#
# Staging is ~0.75 MB and ~3 s per page, so stage the wave you are about to work
# on rather than the whole manual (~1.2 GB).

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manual_map as mm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def chapters_of(entries, docs_dir):
    """Chapter directory -> remaining leaf entries, in outline order."""
    todo = {}
    for entry in mm.remaining(entries, docs_dir):
        todo.setdefault(mm.chapter_dir(entry), []).append(entry)
    return todo


def stage_chapter(chapter, topics, entries, args):
    """Stage one chapter's topics in order. Returns (chapter, staged pages, failures)."""
    staged, failures = 0, []
    for entry in topics:
        path = mm.doc_path(entry, entries)
        command = [
            sys.executable, os.path.join(HERE, "prepare_pages.py"),
            "--topic", path,
            "--images-dir", str(Path(args.docs) / chapter / "images"),
            "--staging", args.staging,
        ]
        if args.pdf:
            command += ["--pdf", args.pdf]
        if args.dry_run:
            print(f"[{chapter}] would stage {entry.title} ({entry.page}-{entry.end_page})")
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            failures.append(entry.title)
            print(f"[{chapter}] FAILED {entry.title}\n{result.stderr.strip()}", file=sys.stderr)
        else:
            staged += entry.end_page - entry.page + 1
            print(f"[{chapter}] staged {entry.title} ({entry.end_page - entry.page + 1} pages)")
    return chapter, staged, failures


def main():
    parser = argparse.ArgumentParser(description="Batch stage the remaining topics of whole chapters.")
    parser.add_argument("chapters", nargs="*", help="chapter directories, e.g. turbocharger-system")
    parser.add_argument("--all", action="store_true", help="every chapter with remaining topics")
    parser.add_argument("--pdf", help="path to the manual PDF (default: $MR2_DOCS_MANUAL_PATH)")
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--staging", default=".staging")
    parser.add_argument("--jobs", type=int, default=4, help="chapters to stage at once (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="list the work without staging it")
    args = parser.parse_args()

    if not args.chapters and not args.all:
        parser.error("name at least one chapter, or pass --all")

    entries = mm.load_outline(mm.open_manual(args.pdf))
    todo = chapters_of(entries, args.docs)
    if not args.all:
        unknown = [c for c in args.chapters if c not in todo]
        if unknown:
            parser.error(f"no remaining topics for: {', '.join(unknown)}\n"
                         f"known chapters: {', '.join(sorted(todo))}")
        todo = {c: todo[c] for c in args.chapters}

    pages = sum(e.end_page - e.page + 1 for topics in todo.values() for e in topics)
    print(f"{len(todo)} chapter(s), {sum(len(t) for t in todo.values())} topic(s), "
          f"{pages} pages, ~{pages * 0.75 / 1024:.1f} GB\n")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda item: stage_chapter(item[0], item[1], entries, args), todo.items()))

    failures = [(chapter, title) for chapter, _, titles in results for title in titles]
    total = sum(staged for _, staged, _ in results)
    print(f"\n{total} pages staged in {args.staging}/pages")
    for chapter, title in failures:
        print(f"failed: {chapter} / {title}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
