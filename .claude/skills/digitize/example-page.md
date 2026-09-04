# Worked example: PDF page 41 (EM-11)

The page is `.staging/pages/0041/`. Its manifest:

```json
{
  "pdf_page": 41, "code": "EM-11", "code_source": "ocr",
  "header": "ENGINE MECHANICAL — Engine Tune-Up EM-11",
  "figures": [
    {"name": "EM8548", "bbox_pct": [6.7, 56.2, 39.2, 71.7], "near_step": "8.",  "steps_beside": ["8.", "(a)", "(b)"]},
    {"name": "EM7889", "bbox_pct": [6.8, 73.3, 39.2, 88.8], "near_step": "9.",  "steps_beside": ["9.", "(a)"]}
  ],
  "steps": [{"label": "1.", "y_pct": 12.7}, ..., {"label": "9.", "y_pct": 73.4}]
}
```

`ocr.txt` (abridged):

```
ENGINE MECHANICAL — Engine Tune-Up                           EM-11

INSPECTION AND ADJUSTMENT OF VALVE
CLEARANCE (3S-GTE)

HINT: Inspect and adjust the valve clearance when the
engine is cold.

1. DISCONNECT CABLE FROM NEGATIVE TERMINAL
OF BATTERY
CAUTION: Work must be started after approx. 20
seconds or longer from the time the ignition switch is
turned to the "LOCK" position and the negative (-) terminal cable is disconnected from the battery.
...
5. REMOVE THROTTLE BODY
(See steps 3, 5 to 8, 10 and 11 on pages FI-135 and 136)
...
8. SET NO.1 CYLINDER TO TDC/COMPRESSION
(a) Turn the crankshaft pulley and align its groove with
timing mark "0" of the No.1 timing belt cover.
(b) Check that the valve lifters on the No.1 cylinder are
loose and valve lifters on No.4 are tight.
If not, turn the crankshaft one revolution (360°) and align
the mark as above.

9. INSPECT VALVE CLEARANCE
(a) Check only those valves indicated.
* Using a feeler gauge, measure the clearance between the valve lifter and camshaft.
* Record the specifications of the valve clearance measurements. They will be used later to determine the
  required replacement adjusting shim.
Valve clearance (Cold):
Intake     0.15 – 0.25 mm (0.006 – 0.010 in.)
Exhaust    0.20 – 0.30 mm (0.008 – 0.012 in.)
```

Converted:

```markdown
[](){ #p-em-11 }
## Inspection and adjustment of valve clearance (3S-GTE)

!!! tip "Hint"

    Inspect and adjust the valve clearance when the engine is cold.

1.  Disconnect cable from negative terminal of battery.

    !!! warning "Caution"

        Work must be started after approx. 20 seconds or longer from the time the ignition switch is turned to the
        `LOCK` position and the negative (`–`) terminal cable is disconnected from the battery.

2.  Remove No.1 air intake connector. (See step 4 on page TC-20)
3.  Disconnect high-tension cords from spark plugs. (See page IG-6)
4.  Remove EGR vacuum modulator and VSV. (See step 22 on page EM-66)
5.  Remove throttle body. (See steps 3, 5 to 8, 10 and 11 on pages FI-135 and 136)
6.  Remove hose clamp and VTV clamp of air by-pass valve. (See steps 15 and 16 on page TC-9)
7.  Remove cylinder head cover. (See step 35 on page EM-70)
8.  Set No.1 cylinder to TDC/compression.

    1. Turn the crankshaft pulley and align its groove with timing mark "0" of the No.1 timing belt cover.
    2. Check that the valve lifters on the No.1 cylinder are loose and valve lifters on No.4 are tight.

    If not, turn the crankshaft one revolution (360°) and align the mark as above.

    <figure markdown="span">
      ![](images/EM8548.webp#illustration){ width="80%" }
    </figure>

9.  Inspect valve clearance.

    1.  Check only those valves indicated.

        * Using a feeler gauge, measure the clearance between the valve lifter and camshaft.
        * Record the specifications of the valve clearance measurements. They will be used later to determine the
          required replacement adjusting shim.

        **Valve clearance (cold):**

        * Intake – 0.15 – 0.25 mm (0.006 – 0.010 in.)
        * Exhaust – 0.20 – 0.30 mm (0.008 – 0.012 in.)

    <figure markdown="span">
      ![](images/EM7889.webp#illustration){ width="80%" }
    </figure>
```

Points to note:

* The ALL-CAPS task headings became ordered list items in sentence case; `(a)`/`(b)` became a nested ordered list with
  no typed letters.
* Both figures were placed under the step they sit beside, not before it.
* The references were copied verbatim; `resolve_refs.py` links them once the targets exist.
* The clearance values came from `ocr.txt`, never from reading `page.png`.
* The page anchor sits before the `##` heading, so references to EM-11 land on that heading.
