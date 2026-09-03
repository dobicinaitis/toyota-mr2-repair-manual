[](){ #p-ch-3 }
# On-vehicle inspection

1.  **Inspect battery specific gravity and electrolyte level**

    <figure markdown="span">
      ![](images/CH0609.webp#illustration){ width="400px" }
    </figure>

    1.  Check the specific gravity of each cell.

        **Standard specific gravity:** 1.25 – 1.27 when fully charged at 20°C (68°F)

        If not within specification, charge the battery.

    2.  Check the electrolyte quantity of each cell.

        If insufficient, refill with distilled (or purified) water.

2.  **Check battery terminals, fusible links and fuses**

    <figure markdown="span">
      ![](images/CH0793.webp#illustration){ width="400px" }
    </figure>

    1.  Check that the battery terminals are not loose or corroded.
    2.  Check the fusible links and fuses for continuity.

        **Fusible link:**

        * MAIN 2.0L
        * ALT 120A
        * AM1 50A
        * AM2 40A

        **Fuse:**

        * ECU-IG 15A
        * ALT SENCING 7.5A
        * AM2 7.5A

3.  **Inspect alternator drive belt**

    1.  Visually check the drive belt for excessive wear, frayed cords etc.

        <figure markdown="span">
          ![](images/CH0004_CH0752.webp#illustration){ width="400px" }
        </figure>

        If necessary, replace the drive belt.

        !!! tip "Hint"

            Cracks on rib side of a drive belt are considered acceptable. If the drive belt has chunks missing from the
            ribs, it should be replaced.

    2.  Using a belt tension gauge, measure the drive belt tension.

        <figure markdown="span">
          ![](images/EC0003_EC0004_EC0001.webp#illustration){ width="400px" }
        </figure>

        **Belt tension gauge:**

        * Nippondenso BTG-20 (95506-00020)
        * Borroughs No. BT-33-73F

        **Drive belt tension:**

        * New belt 120 ± 20 lb
        * Used belt 104 ± 20 lb

        If the belt tension is not as specified, adjust it.

        <figure markdown="span">
          ![](images/CH0087.webp#illustration){ width="400px" }
        </figure>

        !!! tip "Hint"

            * "New belt" refer to a belt which has been used 5 minutes or less on a running engine.
            * "Used belt" refers to a belt which has been used on a running engine for 5 minutes or more.
            * After installing a belt, check that it fits properly in the ribbed grooves.
            * Check by hand to confirm that the belt has not slipped out of the groove on the bottom of the pulley.
            * After installing a new belt, run the engine for about 5 minutes and recheck the belt tension.

4.  [](){ #p-ch-4 } **Visually check alternator wiring and listen for abnormal noises**

    <figure markdown="span">
      ![](images/CH0889.webp#illustration){ width="400px" }
    </figure>

    1.  Check that the wiring is in good condition.
    2.  Check that there is no abnormal noise from the alternator while the engine is running.

5.  **Inspect charge warning light circuit**

    1.  Warm up the engine and then turn it off.
    2.  Turn off all accessories.
    3.  Turn the ignition switch to "ON". Check that the charge warning light is lit.
    4.  Start the engine. Check that the light goes out.

        If the light does not go off as specified, troubleshoot the charge light circuit.

6.  **Inspect charging circuit without load**

    <figure markdown="span">
      ![](images/CH0732.webp#illustration){ width="400px" }
    </figure>

    !!! tip "Hint"

        If a battery/alternator tester is available, connect the tester to the charging circuit as per manufacturer's
        instructions.

    1.  If a tester is not available, connect a voltmeter and ammeter to the charging circuit as follows:

        * Disconnect the wire from terminal B of the alternator and connect it to the negative (–) probe of the
          ammeter.
        * Connect the positive (+) probe of the ammeter to terminal B of the alternator.
        * Connect the positive (+) probe of the voltmeter to terminal B of the alternator.
        * Ground the negative (–) probe of the voltmeter.

    2.  Check the charging circuit as follows:

        <figure markdown="span">
          ![](images/CH0205_CH0010.webp#illustration){ width="400px" }
        </figure>

        With the engine running from idling to 2,000 rpm, check the reading on the ammeter and voltmeter.

        **Standard amperage:** 10 A or less

        **Standard voltage:**

        * 13.9 – 15.1 V at 25°C (77°F)
        * 13.5 – 14.3 V at 115°C (239°F)

        If the voltmeter reading is greater than standard voltage, replace the IC regulator.

        [](){ #p-ch-5 } If the voltmeter reading is less than standard voltage, check the IC regulator and alternator
        as follows:

        <figure markdown="span">
          ![](images/CH0914.webp#illustration){ width="400px" }
        </figure>

        * With terminal F grounded, start the engine and check the voltmeter reading of terminal B.

        <figure markdown="span">
          ![](images/CH0067.webp#illustration){ width="400px" }
        </figure>

        * If the voltmeter reading is greater than standard voltage, replace the IC regulator.
        * If the voltmeter reading is less than standard voltage, check the alternator.

7.  **Inspect charging circuit with load**

    <figure markdown="span">
      ![](images/CH0068.webp#illustration){ width="400px" }
    </figure>

    1.  With the engine running at 2,000 rpm, turn on the high beam headlights and place the heater blower switch at
        "HI".
    2.  Check the reading on the ammeter.

        **Standard amperage:** 30 A or more

        <figure markdown="span">
          ![](images/CH0069.webp#illustration){ width="400px" }
        </figure>

        If the ammeter reading is less than standard amperage, repair the alternator. (See [Alternator › Disassembly of alternator](alternator.md#disassembly-of-alternator))

        !!! tip "Hint"

            With the battery fully charged, the indication will sometimes be less than standard amperage.

