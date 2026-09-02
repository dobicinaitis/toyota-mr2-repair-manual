# The Big Green Book

Converts the scanned 1991 Toyota MR2 repair manual (2 volumes, 1644 pages, 600 dpi bitmaps, no text layer) into a
Zensical static site under `docs/`. The PDF outline is the unit of work: one leaf topic per markdown file.

## Environment

```bash
source .venv/bin/activate                    # Python 3.14, see requirements.txt
export MR2_DOCS_MANUAL_PATH="<path-to-manual>.pdf"
```

The scripts read the path to the scanned manual from `MR2_DOCS_MANUAL_PATH`, which
`.claude/settings.local.json` sets (`.claude/settings.json` holds the placeholder). Run every script with
`.venv/bin/python`.

## Digitizing a section

Use the `/digitize` skill (`.claude/skills/digitize/SKILL.md`). In short:

```bash
.venv/bin/python utils/manual_map.py --topic "Engine tune-up"                 # pages, target file, section code
.venv/bin/python utils/prepare_pages.py --topic "Engine tune-up" --images-dir docs/engine-mechanical/images
```

`prepare_pages.py` writes `.staging/pages/NNNN/` per page: `page.png` and `overlay.png` (read these), `ocr.txt`,
`ocr.tsv`, `figures/*.png` and `manifest.json`; named illustrations go straight to `--images-dir` as lossless WebP.

Helpers: `crop_figure.py` (unframed figures and charts), `zoom.py` (read a region at high dpi, `--ocr` to OCR it),
`resolve_refs.py` (link "See page EM-11" phrases), `lint_docs.py`, `build_glossary.py`.

## Hard rules

* **Digits never come from the page preview.** Take every number from `ocr.txt`, a `zoom.py --ocr` read, or a figure's
  own `.txt`; verify with `lint_docs.py --ocr-audit .staging`.
* **Write references verbatim** ("(See page EM-11)"). `resolve_refs.py` turns them into links; never hand-write one.
* **Every page gets an anchor**: `[](){ #p-em-11 }` where that printed page starts. Never inside a table or admonition.
* **Every detected figure is used or explicitly skipped.** `lint_docs.py` reports unused images.
* Wording, numbering and values stay faithful to the manual; fix only obvious scan/print typos.
* Never stage `.staging/`. Commit only when the user asks.

## Before every commit

```bash
.venv/bin/python utils/lint_docs.py --ocr-audit .staging
.venv/bin/python utils/resolve_refs.py
.venv/bin/python utils/build_glossary.py
.venv/bin/zensical build --clean --strict
```

Then add the new pages to `nav` in `zensical.toml` and tick the checklist in `readme.md`.

Markdown conventions live in `.claude/skills/digitize/style-guide.md`, with a worked example in
`.claude/skills/digitize/example-page.md`.
