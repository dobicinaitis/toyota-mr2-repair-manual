[](){ #p-ig-10 }
# On-vehicle inspection (5S-FE)

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

    **BAD:** Check wiring between ignition switch to ignition coil.

4.  **Check resistance of ignition coil** (See [Inspection of spark plugs](#p-ig-12))

    **Resistance (Cold):**

    * Primary 0.40 – 0.50 Ω
    * Secondary 10.0 – 14.0 kΩ

    **BAD:** Replace the ignition coil.

5.  **Check resistance of signal generator (pickup coil)** (See [Inspection of ignition coil](#p-ig-13))

    **Resistance (Cold):** 170 – 210 Ω

    **BAD:** Replace the distributor housing assembly.

6.  **Check air gap of distributor** (See [Inspection of ignition coil](#p-ig-13))

    **Air gap:** 0.2 – 0.4 mm (0.008 – 0.016 in.)

    **BAD:** Replace the distributor housing assembly.

7.  **Check IGT signal from ECU** (See page FI-62 or 79)

    **BAD:** Check wiring between ECU, distributor and igniter, only then try another ECU.

8.  **Try another igniter**

[](){ #p-ig-11 }
## Inspection of high-tension cords

1.  **Disconnect high-tension cords from spark plugs**

    <figure markdown="span">
      ![](images/IG0863.webp#illustration){ width="80%" }
    </figure>

    Disconnect the high-tension cords at rubber boot. DO NOT pull on the cords.

    !!! note "Notice"

        Pulling on or bending the cords may damage the conductor inside.

2.  **Disconnect high-tension cord from ignition coil** (See [Distributor (3S-GTE), step 3](distributor-3s-gte.md#distributor-3s-gte))
3.  **Remove distributor cap without disconnecting high-tension cords**

    <figure markdown="span">
      ![](images/IG1244.webp#illustration){ width="80%" }
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

1.  **Disconnect high-tension cords from spark plugs**
2.  **Remove spark plugs**

    <figure markdown="span">
      ![](images/IG0864.webp#illustration){ width="80%" }
    </figure>

    Using a 16 mm plug wrench, remove the spark plug.

3.  **Clean spark plugs**

    <figure markdown="span">
      ![](images/IG0152.webp#illustration){ width="80%" }
    </figure>

    Using a spark plug cleaner or wire brush, clean the spark plug.

4.  [](){ #p-ig-12 } **Visually inspect spark plugs**

    <figure markdown="span">
      ![](images/IG0148.webp#illustration){ width="80%" }
    </figure>

    Check the spark plug for electrode wear, thread damage and insulator damage.

    If abnormal, replace the spark plug.

    **Recommended spark plug:**

    * ND `K16R-U11`
    * NGK `BKR5EYA11`

5.  **Adjust electrode cap**

    <figure markdown="span">
      ![](images/IG0657_IG0658.webp#illustration){ width="80%" }
    </figure>

    Carefully bent the outer electrode to obtain the correct electrode gap.

    **Correct electrode gap:** 1.1 mm (0.043 in.)

6.  **Install spark plugs**

    <figure markdown="span">
      ![](images/IG0864.webp#illustration){ width="80%" }
    </figure>

    Using a 16 mm plug wrench, install the spark plug.

    **Torque:** 180 kg-cm (13 ft-lb, 18 N·m)

7.  **Reconnect high-tension cords to spark plugs**

## Inspection of ignition coil

1.  **Disconnect ignition coil connector**
2.  **Disconnect high-tension cord from ignition coil** (See [Distributor (3S-GTE) › Installation of distributor, step 3](distributor-3s-gte.md#installation-of-distributor))
3.  **Inspect primary coil resistance**

    <figure markdown="span">
      ![](images/IG1371.webp#illustration){ width="80%" }
    </figure>

    Using an ohmmeter, measure the resistance between positive (`+`) and negative (`–`) terminals.

    **Primary coil resistance (Cold):** 0.40 – 0.50 Ω

    If the resistance is not as specified, replace the ignition coil.

4.  [](){ #p-ig-13 } **Inspect secondary coil resistance**

    <figure markdown="span">
      ![](images/IG1372.webp#illustration){ width="80%" }
    </figure>

    Using an ohmmeter, measure the resistance between positive (`+`) and high-tension terminals

    **Secondary coil resistance (Cold):** 10.0 – 14.0 kΩ

    If the resistance is not as specified, replace the ignition coil.

5.  **Reconnect high-tension cord to ignition coil** (See [Distributor (3S-GTE) › Installation of distributor, step 5](distributor-3s-gte.md#installation-of-distributor))
6.  **Reconnect ignition coil connector**

## Inspection of distributor

1.  **Disconnect distributor connector**
2.  **Remove distributor cap**
3.  **Remove rotor**
4.  **Inspect air gap**

    <figure markdown="span">
      ![](images/IG1172.webp#illustration){ width="80%" }
    </figure>

    <figure markdown="span">
      ![](images/IG1171.webp#illustration){ width="80%" }
    </figure>

    Using SST (G1 pickup) and a feeler gauge (NE pickup), measure the air gap between the signal rotor and pickup coil
    projection.

    **SST** `09240-00020` for G1 pickup

    **Air gap:** 0.2 mm – 0.4 mm (0.008 – 0.016 in.)

    If the air gap is not as specified, replace the distributor housing assembly.

5.  **Inspect signal generator (pickup coil) resistance**

    <figure markdown="span">
      ![](images/IG1173.webp#illustration){ width="80%" }
    </figure>

    Using an ohmmeter, measure the resistance between terminals (G1 and G⊖, NE and G⊖).

    **Pickup coil resistance (Cold):** 170 – 210 Ω

    If the resistance is not as specified, replace the distributor housing assembly.

6.  **Reinstall rotor**
7.  **Reinstall distributor cap**
8.  **Reconnect distributor connector**

## Inspection of igniter

(See procedure Spark Test on [On-vehicle inspection (5S-FE)](#on-vehicle-inspection-5s-fe))
