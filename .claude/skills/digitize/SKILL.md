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

If the topic spans more than ~40 pages, split it: do the first chunk, then continue.

## 2. Stage the pages

```bash
.venv/bin/python utils/prepare_pages.py --topic "<topic>" --images-dir docs/<chapter>/images
```

Read `.staging/run.json`. **Stop and ask the user** if it reports rotated pages, `code_source: "unknown"`, or
non-monotonic page codes — those mean the page structure is not what the script expects.

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

1. Add the page to `nav` in `zensical.toml` (inside its chapter, in manual order). The nav is flat — chapters at the
   top level, no volume grouping; the printed volumes are not part of the site.
2. Create `docs/<chapter>/index.md` if the chapter is new — the chapter's contents page as a bulleted list of links to
   its topics, plus any general notes printed on it. Do not carry over the printed page-code column: the codes are
   meaningless on the web, and the link already goes to the right place.
3. Tick the topic in the `readme.md` checklist.
4. Run the full check:
   ```bash
   .venv/bin/python utils/lint_docs.py --ocr-audit .staging
   .venv/bin/python utils/resolve_refs.py
   .venv/bin/python utils/build_glossary.py
   .venv/bin/zensical build --clean --strict
   ```
   `resolve_refs.py` runs site-wide on purpose: earlier pages gain links as their targets appear.
5. Report: pages written, figures used and skipped, unresolved references (from `.staging/refs-report.md`), any
   OCR-audit mismatches you checked by hand, and anything you had to guess.
6. **Stop.** The user reviews `zensical serve` before anything is committed. Never commit unasked, never stage
   `.staging/`.
