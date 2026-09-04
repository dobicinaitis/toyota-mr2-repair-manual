# Markdown style guide

The printed manual puts illustrations in a left column and text in a right column, uses ALL-CAPS task headings and
numbered steps with `(a)`, `(b)`, `(c)` sub-steps. The web version keeps the wording and the numbering, but reflows it
into a single column.

## Structure

* `# H1` = the outline topic title, sentence case. One per file.
* ALL-CAPS procedure headings in the manual → `##`, sentence case ("REPLACE SPARK PLUGS" → `## Replace spark plugs`).
* Sub-procedures (Removal, Inspection, Installation, Disassembly) → `###`.
* Where the manual numbers its operations (as in Maintenance operations) keep the number in the heading:
  `### 5. Replace spark plugs` — the schedule tables reference them by number.
* Drop "(Cont'd)" from headings continued across pages; continue the previous section instead.
* Never put a heading inside a content tab (`=== "…"`) — it breaks anchors.

## Steps

* Numbered task steps → ordered list items, sentence case.
* `(a)`, `(b)`, `(c)` sub-steps → a nested ordered list; the site's CSS renders them as letters. Never type the letters.
* `•` bullets → `*`.
* Indent everything belonging to a step by 4 spaces (figures, admonitions, spec blocks, nested lists).
* Use `1.  ` (two spaces) for items whose body has several blocks, so continuation lines align.

## Admonitions

| Manual     | Markdown                |
|------------|-------------------------|
| CAUTION    | `!!! warning "Caution"` |
| NOTICE     | `!!! note "Notice"`     |
| HINT       | `!!! tip "Hint"`        |
| EXAMPLE:   | `!!! example`           |

`EXAMPLE:` takes no title — the admonition renders its own — and the printed colon is dropped.

Body indented 4 spaces below the `!!!` line. Inside a list item the admonition can be the item's content — the manual
often prints a whole sub-step as a CAUTION, and this keeps the lettering intact:

```markdown
    1.  !!! warning "Caution"

            Work must be started after approx. 20 seconds or longer …

    2.  Before performing electrical work, disconnect the negative cable from the battery terminal.
```

**Always leave a blank line after an admonition's body.** Without it, everything that follows is folded into the
admonition's last paragraph — the following list items disappear into the caution box. `lint_docs.py` fails on this.

## Specifications

* Label in bold, value after it: `**Correct electrode gap:** 0.8 mm (0.031 in.)`
* Several values → bold label then a bullet list.
* Torque: `**Torque:** 180 kg-cm (13 ft-lb, 18 N·m)` — keep every unit the manual prints, in its order.
* The manual's narrow columns wrap a spec over two or three printed lines (label, then the metric value, then the
  alternate units). That is wrapping, not structure: reflow it onto one line. This holds inside a table cell too —
  `**Turbocharging pressure:** 0.50 – 0.83 kg/cm² (7.1 – 11.8 psi, 49 – 81 kPa)` between `<br>`s, not three
  `<br>`-separated lines.
* SST numbers, part numbers and literal switch positions in backticks: `` `09350-30020` ``, `` `PK20R8` ``.
* A terminal's polarity sign goes in backticks too, the negative one as an en dash: positive (`` `+` ``),
  negative (`` `–` ``). `lint_docs.py --fix` rewrites the plain forms.
* Ranges use a spaced en dash: `0.15 – 0.25 mm`. Tolerances use `±`. Watch for the OCR reading `±` as `+` and `lb` as
  `Ib`.

## Tables

Use a markdown table for spec lists of three or more rows, troubleshooting tables
(`Problem | Possible cause | Remedy | Page`), and service specification pages. Merge tables that continue across pages.
A framed table detected as a figure comes with its own OCR text in `figures/<name>.txt` — transcribe it, don't embed
the image.

A column of printed page codes only earns its place when it is the reader's way to the procedure — a troubleshooting
table's "Page" column, for instance, which `resolve_refs.py` turns into links. Where the row already links to the
topic, drop the codes; the printed pages do not exist on the site.

## Figures

```markdown
<figure markdown="span">
  ![](images/EM8548.webp#illustration){ width="80%" }
</figure>
```

* Indent by 4 spaces per list level when the figure belongs to a step.
* Drop `{ width="80%" }` for images wider than 3000 px (full-width diagrams) — `lint_docs.py` enforces this.
* The `#illustration` fragment drives the dark-mode inversion; it is required.
* File names are the manual's own illustration IDs (`EM8548.webp`, `AB0014_AB0257.webp`). Hand-cropped figures get a
  descriptive kebab-case name (`shim-chart-intake.webp`).
* A figure printed on several pages is stored once; reference it again where it reappears, or skip the repeat when the
  text does not need it.

## Page anchors

`[](){ #p-em-11 }` marks where printed page EM-11 starts. On its own line directly before a heading (then references
land on that heading), or inline at the start of the paragraph or list item where the page begins:

```markdown
4. [](){ #p-in-5 } Check hose and wiring connectors …
```

Codes increase by one per page within a file. Never inside a table or an admonition.

## References

Write them exactly as printed — `(See page EM-11)`, `(See steps 3, 5 to 8, 10 and 11 on pages FI-135 and 136)`,
`Refer to page AB-16 of this manual`. `resolve_refs.py` rewrites them into links whose text is the target heading, and
leaves the ones whose target is not digitized yet for a later run.

## Fidelity

Keep the manual's wording, numbering and values. Fix only obvious scan or print errors: OCR confusions (`Ib`→`lb`,
`0`↔`O`), hyphenated line breaks, and misprints that the surrounding text makes unambiguous. Keep the manual's own
quirks (for example "excessively, damaged or oily") as printed.
