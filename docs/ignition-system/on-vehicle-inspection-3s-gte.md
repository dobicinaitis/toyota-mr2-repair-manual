[](){ #p-ig-5 }
# On-vehicle inspection (3S-GTE)

## Spark test

### Check that spark occurs

1.  Disconnect the high-tension cord from the distributor.
2.  Hold the end about 12.5 mm (0.50 in.) from the body of car.
3.  Check if spark occurs while engine is being cranked.

!!! tip "Hint"

    To prevent gasoline from being injected from injectors during this test, crank the engine for no more than
    1 – 2 seconds at a time.

If the spark does not occurs, perform the test as follows:

1.  **Check connection of ignition coil. Igniter and distributor connector**

    **BAD:** Connect securely.

2.  **Check resistance of high-tension cord** (See [Inspection of high-tension cords](#inspection-of-high-tension-cords))

    **Maximum resistance:** 25 KΩ per cord

    **BAD:** Replace the cord(s).

3.  **Check power supply to ignition coil and igniter**

    1.  Ignition switch turn to ON.
    2.  Check that there is battery voltage at Ignition coil positive (`+`) terminal.

    **BAD:** Check wiring between ignition switch to ignition coil and igniter.

4.  **Check resistance of ignition coil** (See [Inspection of spark plugs](#p-ig-8))

    **Resistance (Cold):**

    * Primary 0.40 – 0.50 Ω
    * Secondary 10.0 – 14.0 kΩ

    **BAD:** Replace the ignition coil.

5.  **Check resistance of signal generator (pickup coil)** (See [Inspection of distributor](#inspection-of-distributor))

    **Resistance (Cold):**

    * G1 and G⊖ 140 – 180 Ω
    * G2 and G⊖ 140 – 180 Ω
    * NE and G⊖ 180 – 220 Ω

    **BAD:** Replace the distributor housing assembly.

6.  **Check air gap of distributor** (See [Inspection of distributor](#inspection-of-distributor))

    **Air gap:** 0.2 – 0.4 mm (0.008 – 0.016 in.)

    **BAD:** Replace the distributor housing assembly.

7.  **Check IGT signal from ECU** (See page FI-45)

    **BAD:** Check wiring between ECU, distributor and igniter, only then try another ECU.

8.  **Try another igniter**

[](){ #p-ig-6 }
## Inspection of high-tension cords

1.  **Disconnect high-tension cords from spark plugs**

    <figure markdown="span">
      ![](images/IG1390.webp#illustration){ width="400px" }
    </figure>

    Disconnect the high-tension cords at rubber boot. DO NOT pull on the cords.

    !!! note "Notice"

        Pulling on or bending the cords may damage the conductor inside.

2.  **Disconnect high-tension cord from ignition coil** (See [Distributor (3S-GTE), step 3](distributor-3s-gte.md#distributor-3s-gte))
3.  **Remove distributor cap without disconnecting high-tension cords**

    <figure markdown="span">
      ![](images/IG1318.webp#illustration){ width="400px" }
    </figure>

4.  **Inspect high-tension cord resistance**

    Using an ohmmeter, measure the resistance without disconnecting the distributor cap.

    **Maximum resistance:** 25 kΩ per cord

    If the resistance is greater than maximum, check the terminals. If necessary, replace the high-tension cord and/or
    distributor cap.

5.  **Reinstall distributor cap**
6.  **Reconnect high-tension cord to ignition coil** (See [Distributor (3S-GTE) › Installation of distributor, step 5](distributor-3s-gte.md#installation-of-distributor))
7.  **Reconnect high-tension cords to spark plugs**

## Inspection of spark plugs

!!! note "Notice"

    * Never use a wire brush for cleaning.
    * Never attempt to adjust the electrode gap on used spark plug.
    * Spark plug should be replaced every 100,000 km (60,000 miles).

1.  **Disconnect high-tension cords from spark plugs**
2.  **Inspect electrode**

    <figure markdown="span">
      ![](images/IG0147.webp#illustration){ width="400px" }
    </figure>

    Using a megger (insulation resistance meter), measure the insulation resistance.

    **Correct insulation resistance:** 10 MΩ or more

    If the resistance is less than specified, proceed to step 4.

    !!! tip "Hint"

        If a megger is not available, the following simple method of inspection provides fairly accurate results.

    [](){ #p-ig-7 } **(Simple method)**

    <figure markdown="span">
      ![](images/IG0148.webp#illustration){ width="400px" }
    </figure>

    1.  Quickly race the engine to 4,000 rpm five times.
    2.  Remove the spark plug. (See step 3)
    3.  Visually check the spark plug.

        If the electrode is dry ... Okey

        If the electrode is wet ... Proceed to step 3

    4.  Reinstall the spark plug. (See [Inspection of spark plugs, step 7](#p-ig-8))

3.  **Remove spark plugs**

    <figure markdown="span">
      ![](images/IG1361.webp#illustration){ width="400px" }
    </figure>

    Using a 16 mm plug wrench, remove the spark plug.

4.  **Visually inspect spark plugs**

    <figure markdown="span">
      ![](images/IG0316.webp#illustration){ width="400px" }
    </figure>

    Check the spark plug for thread damage and insulator damage.

    If abnormal, replace the spark plug.

    **Recommended spark plug:**

    * ND `PK20R8`
    * NGK `BKR6EP8`

5.  **Inspect electrode cap**

    <figure markdown="span">
      ![](images/IG0317.webp#illustration){ width="400px" }
    </figure>

    **Maximum electrode gap:** 1.0 mm (0.039 in.)

    If the gap is greater than maximum, replace the spark plug.

    **Correct electrode gap of new spark plug:** 0.8 mm (0.031 in.)

    !!! note "Notice"

        If adjusting the gap of a new spark plug, bent only the base of the ground electrode. Do not touch the tip.
        Never attempt to adjust the gap on the used plug.

6.  **Clean spark plugs**

    <figure markdown="span">
      ![](images/IG0152.webp#illustration){ width="400px" }
    </figure>

    If the electrode has traces of wet carbon, allow it to dry and then clean with a spark plug cleaner.

    **Air pressure:** Below 6 kg/cm² (85 psi, 588 kPa)

    **Duration:** 20 seconds or less

    !!! tip "Hint"

        If there are traces of oil, remove it with gasoline before using the spark plug cleaner.

7.  [](){ #p-ig-8 } **Install spark plugs**

    <figure markdown="span">
      ![](images/IG1361.webp#illustration){ width="400px" }
    </figure>

    Using a 16 mm plug wrench, install the spark plug.

    **Torque:** 180 kg-cm (13 ft-lb, 18 N·m)

8.  **Reconnect high-tension cords to spark plugs**

## Inspection of ignition coil

1.  **Disconnect ignition coil connector**
2.  **Disconnect high-tension cord from ignition coil** (See [Distributor (3S-GTE), step 3](distributor-3s-gte.md#distributor-3s-gte))
3.  **Inspect primary coil resistance**

    <figure markdown="span">
      ![](images/IG1392.webp#illustration){ width="400px" }
    </figure>

    Using an ohmmeter, measure the resistance between positive (`+`) and negative (`–`) terminals.

    **Primary coil resistance (Cold):** 0.41 – 0.50 Ω

    If the resistance is not as specified, replace the ignition coil.

4.  **Inspect secondary coil resistance**

    <figure markdown="span">
      ![](images/IG1393.webp#illustration){ width="400px" }
    </figure>

    Using an ohmmeter, measure the resistance between positive (`+`) and high-tension terminals

    **Secondary coil resistance (Cold):** 10.0 – 14.0 kΩ

    If the resistance is not as specified, replace the ignition coil.

5.  **Reconnect high-tension cord to ignition coil** (See [Distributor (3S-GTE) › Installation of distributor, step 5](distributor-3s-gte.md#installation-of-distributor))
6.  **Reconnect ignition coil connector**

[](){ #p-ig-9 }
## Inspection of distributor

1.  **Disconnect distributor connector**
2.  **Remove distributor cap**
3.  **Remove rotor**
4.  **Inspect air gap**

    <figure markdown="span">
      ![](images/IG1330.webp#illustration){ width="400px" }
    </figure>

    <figure markdown="span">
      ![](images/IG1329.webp#illustration){ width="400px" }
    </figure>

    Using SST (G1 and G2 pickups) and a feeler gauge (NE pickup), measure the air gap between the signal rotor and
    pickup coil projection.

    **SST** `09240-00020` for G1 and G2 pickups

    **Air gap:** 0.2 – 0.4 mm (0.008 – 0.016 in.)

    If the air gap is not as specified, replace the distributor housing assembly.

5.  **Inspect signal generator (pickup coil) resistance**

    <figure markdown="span">
      ![](images/IG1331.webp#illustration){ width="400px" }
    </figure>

    Using an ohmmeter, measure the resistance between terminals.

    **Pickup coil resistance (Cold):**

    * G1 and G⊖ 140 – 180 Ω
    * G2 and G⊖ 140 – 180 Ω
    * NE and G⊖ 180 – 220 Ω

    If the resistance is not as specified, replace the distributor housing assembly.

6.  **Reinstall rotor**
7.  **Reinstall distributor cap**
8.  **Reconnect distributor connector**

## Inspection of igniter

(See procedure Spark Test on [On-vehicle inspection (3S-GTE)](#on-vehicle-inspection-3s-gte))
