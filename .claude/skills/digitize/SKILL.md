---
name: digitize
description: Convert a section of the scanned Toyota MR2 repair manual into markdown. Use when the user asks to digitize, transcribe or convert a topic, chapter or page range of the manual.
---

# Digitize a section of the manual

Converts one outline topic (or page range) of the scanned manual into a markdown page under `docs/`, with its
illustrations extracted and its cross-references linked.

Arguments: a topic title (`/digitize Engine tune-up`), a page range (`/digitize --pages 40-51`), or `--continue` to
resume the topic whose file already has some page anchors.

## 1. Locate the work

```bash
.venv/bin/python utils/manual_map.py --topic "<topic>"
```

Gives the PDF page range, the section code (EM, FI, …), the volume and the suggested `docs/…` path. With `--continue`,
read the target file first and resume after its last `[](){ #p-… }` anchor.

Titles like "Description", "Precautions" and "Troubleshooting" occur in a dozen chapters, so `--topic` alone is
ambiguous for them. Add `--chapter <chapter-slug>`, or address the topic by its path
(`--topic efi-system/troubleshooting.md`), which is always unique.

If the topic spans more than ~40 pages, split it: do the first chunk, then continue.

## 2. Stage the pages

```bash
.venv/bin/python utils/prepare_pages.py --topic "<topic>" --images-dir docs/<chapter>/images
```

Read `.staging/run.json` — or `.staging/runs/<topic-slug>.json`, the copy that survives a later run, when the pages
were staged ahead of time by `stage_chapter.py`. **Stop and ask the user** if it reports rotated pages,
`code_source: "unknown"`, or non-monotonic page codes — those mean the page structure is not what the script expects.

Per page, `.staging/pages/NNNN/` holds:

| File            | Use                                                                          |
|-----------------|------------------------------------------------------------------------------|
| `overlay.png`   | Read this: the page with figure boxes, illustration IDs and step ticks drawn |
| `page.png`      | The clean preview, when the overlay's boxes get in the way                   |
| `ocr.txt`       | The page text with the illustrations masked out — **the source for digits**  |
| `ocr.tsv`       | Word positions, if you need to work out a column layout                      |
| `figures/*.png` | Previews of the extracted illustrations (`Read` cannot open WebP)            |
| `manifest.json` | Page code, figures with `bbox_pct`/`near_step`, detected steps, warnings      |

## 3. Write the markdown

Work **sequentially, about 6 pages at a time**, appending to the target file. Do not delegate this to subagents:
heading continuity and figure placement need the previous page in context.

Per page:

1. `Read` the page's `overlay.png` and `ocr.txt`.
2. Put `[](){ #p-<code> }` where that printed page starts — on its own line before a heading, or inline at the start of
   the list item or paragraph that begins the page.
3. Write the content following `style-guide.md` (worked example in `example-page.md`).
4. Place every figure from `manifest.json` **under** the step it sits beside (`near_step` / `steps_beside`), not to the
   side. Tick each one off; if a figure genuinely belongs to no step, say so in the summary.

    Framed tables are detected as figures too, and a page code in the bottom right corner of one (a troubleshooting
    table's "Page" column, say) can be read as an illustration ID — `IG-17, 22` became `IG1722.webp`. Transcribe the
    table and delete the extracted image; `lint_docs.py` reports it as unreferenced if you forget.
5. For a chart or table with no printed frame, estimate its box from `page.png` (percent of page, same system as
   `bbox_pct`), then:
   ```bash
   .venv/bin/python utils/crop_figure.py <page> --box x0,y0,x1,y1 --id <kebab-name> --images-dir docs/<chapter>/images
   ```
   `Read` the printed preview path and adjust the box once if needed. The script prints the `<figure>` snippet to use.
6. For dense numeric content (spec tables, charts), confirm the values first:
   ```bash
   .venv/bin/python utils/zoom.py <page> --box x0,y0,x1,y1 --ocr
   ```

After each chunk:

```bash
.venv/bin/python utils/lint_docs.py docs/<chapter>/<topic>.md --ocr-audit .staging
```

Fix what it reports before moving on.

## 4. Finish the topic

1. Link the topic from `docs/<chapter>/index.md`, creating that contents page if the chapter is new — a bulleted list
   of links to its topics, plus any general notes printed on it. Transcribe the titles as the printed contents page
   spells them, not as the outline does. Do not carry over the printed page-code column: the codes are meaningless on
   the web, and the link already goes to the right place. `lint_docs.py` reports a topic the page does not link.
2. Regenerate what is derived from the outline — the `nav` in `zensical.toml`, the `readme.md` checklist and the
   glossary. Never edit the first two by hand; they are generated so that parallel branches do not conflict.
   ```bash
   .venv/bin/python utils/lint_docs.py --fix
   ```
3. Check the topic:
   ```bash
   .venv/bin/python utils/lint_docs.py --ocr-audit .staging
   .venv/bin/python utils/verify_topic.py docs/<chapter>/<topic>.md
   ```
   `verify_topic.py` reports what was left out — a page anchor missing from the sequence, a figure extracted and
   never placed, steps the manual numbers that the markdown never uses. Not every finding is a fault, but account
   for each one.
4. Run the site-wide steps — **unless you are working in a chapter worktree**, where they belong to the merge pass
   (see "Batch mode" below):
   ```bash
   .venv/bin/python utils/resolve_refs.py
   .venv/bin/zensical build --clean --strict
   ```
   `resolve_refs.py` runs site-wide on purpose: earlier pages gain links as their targets appear.
5. Report: pages written, figures used and skipped, unresolved references (from `.staging/refs-report.md`), any
   OCR-audit mismatches you checked by hand, and anything you had to guess.
6. **Stop.** The user reviews `zensical serve` before anything is committed. Never commit unasked, never stage
   `.staging/`.

## Batch mode

When several chapters are digitized in parallel worktrees (`utils/worktree.sh`, driven by
`utils/digitize_chapter.sh`), the rules change slightly:

* **You are given exactly one topic. Digitize that topic and stop.** Do not carry on into the rest of the chapter,
  however small the remaining topics look. `digitize_chapter.sh` invokes you once per topic, in outline order, and
  commits each one on its own; a run that writes several topics collapses them into a single commit under the
  first topic's name and loses the fresh context per topic that keeps long chapters accurate.
* **Do not commit.** The loop commits for you once the page exists.
* **Do not run `resolve_refs.py`, `build_glossary.py`, `sync_nav.py` or the zensical build.** They are site-wide.
  `resolve_refs.py` in particular rewrites already-committed pages across the whole site as new anchors appear, so
  two worktrees running it produce conflicting edits to the same files. They run once, on the integration branch,
  after the merge.
* `lint_docs.py <file> --ocr-audit .staging` is the per-chunk check: given explicit paths it skips the whole-tree nav
  and image checks, which are incomplete inside a worktree by design.
* Where step 2 above says to stop and ask the user, there is no user to ask: leave the topic undone, say so, and let
  a human pick it up. Never guess at a page whose structure the staging flagged.
