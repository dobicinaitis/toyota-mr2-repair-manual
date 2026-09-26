[](){ #p-co-21 }
# Radiator cooling fans (w/ A/C)

## System circuit

<figure markdown="span">
  ![](images/CO1142.webp#illustration)
</figure>

## Location of radiator cooling fan components

<figure markdown="span">
  ![](images/CO1151.webp#illustration)
</figure>

[](){ #p-co-22 }
## On-vehicle inspection

### Low temperature (below 85°C (185°F))

1.  Turn ignition switch "ON".

    Check that the cooling fans stops.

    If not, check the cooling fan relays and water temperature sensor, and check for a separated connector or
    severed wire between the cooling fan relay and water temperature sensor.

    <figure markdown="span">
      ![](images/CO1122_CO1131.webp#illustration){ width="80%" }
    </figure>

2.  Disconnect radiator water temperature sensor connector.

    Check that the cooling fans rotates.

    If not, check the fan main relay, cooling fan relays, A/C amplifier, cooling fan and fuses, and check for a
    short circuit between the cooling fan relay and water temperature sensor.

    <figure markdown="span">
      ![](images/CO1130.webp#illustration){ width="80%" }
    </figure>

3.  Connect radiator water temperature sensor connector.

### High temperature (85 – 90°C (185 – 194°F))

4.  Start engine.

    1.  Raise coolant temperature to 85 – 90°C (185 – 194°F).
    2.  Check that the cooling fans rotates (at low speed).

        <figure markdown="span">
          ![](images/CO1108_CO1137_low-speed.webp#illustration){ width="80%" }
        </figure>

    If not, replace the water temperature sensor.

### High temperature (above 90°C (194°F))

5.  Start engine.

    1.  Raise coolant temperature to above 90°C (194°F).
    2.  Check that the cooling fans rotates (at high speed).

        <figure markdown="span">
          ![](images/CO1108_CO1137_high-speed.webp#illustration){ width="80%" }
        </figure>

    If not, replace the water temperature sensor.

## Inspection of radiator cooling fan components

1.  Inspect A/C amplifier for circuit.

    Disconnect the A/C amplifier connector, and check the connector on the wiring harness side as shown in the
    chart on the next page.

    <figure markdown="span">
      ![](images/a-c-amplifier-connector-wiring-harness-side.webp#illustration){ width="80%" }
    </figure>

    [](){ #p-co-23 }

    | Check for   | Tester connection | Condition                   | Specified value  |
    |-------------|--------------------|------------------------------|-------------------|
    | Continuity  | 3 – Ground         | –                            | Continuity        |
    | Voltage     | 4 – Ground         | Ignition switch ON           | Battery voltage   |
    | Resistance  | 9 – 15             | Coolant temp. 85°C (185°F)   | Approx. 1.35 kΩ   |
    |             |                    | Coolant temp. 90°C (194°F)   | Approx. 1.19 kΩ   |
    |             |                    | Coolant temp. 95°C (203°F)   | Approx. 1.05 kΩ   |
    | Voltage     | 10 – Ground        | Ignition switch ON           | Battery voltage   |
    | Continuity  | 13 – Ground        | –                            | Continuity        |

2.  Inspect fan main relay ("FAN MAIN"). (See [Ignition main relay "IGN"](../../charging-system/ignition-main-relay.md#ignition-main-relay-ign))

    Check the relay in the same way as the Ignition Main Relay.

3.  Inspect No.1 cooling fan relay ("FAN NO.1").

    **A. Inspect relay continuity**

    1.  Using an ohmmeter, check that there is continuity between terminals 1 and 2.
    2.  Check that there is continuity between terminals 3 and 4.

    If continuity is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO0103.webp#illustration){ width="80%" }
    </figure>

    **B. Inspect relay operation**

    1.  Apply battery voltage across terminal 1 and 2.
    2.  Using an ohmmeter, check that there is no continuity between terminals 3 and 4.

    If operation is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO0104.webp#illustration){ width="80%" }
    </figure>

4.  Inspect No.2 cooling fan relay ("FAN NO.2").

    **A. Inspect relay continuity**

    1.  Using an ohmmeter, check that there is continuity between terminals 1 and 2.
    2.  Check that there is continuity between terminals 3 and 4.
    3.  Check that there is no continuity between terminals 3 and 5.

    If continuity is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO1123.webp#illustration){ width="80%" }
    </figure>

    [](){ #p-co-24 }
    **B. Inspect relay operation**

    1.  Apply battery voltage across terminal 1 and 2.
    2.  Using an ohmmeter, check that there is no continuity between terminals 3 and 4.
    3.  Check that there is continuity between terminals 3 and 5.

    If operation is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO1124.webp#illustration){ width="80%" }
    </figure>

5.  Inspect No.3 cooling fan relay ("FAN NO.3").

    **A. Inspect relay continuity**

    1.  Using an ohmmeter, check that there is continuity between terminals 1 and 2.
    2.  Check that there is no continuity between terminals 3 and 5.

    If continuity is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO1125.webp#illustration){ width="80%" }
    </figure>

    **B. Inspect relay operation**

    1.  Apply battery voltage across terminal 1 and 2.
    2.  Using an ohmmeter, check that there is continuity between terminals 3 and 5.

    If operation is not as specified, replace the relay.

    <figure markdown="span">
      ![](images/CO1126.webp#illustration){ width="80%" }
    </figure>

6.  Inspect radiator water temperature sensor.

    Using an ohmmeter, measure the resistance between the terminals.

    **Resistance:**

    * Approx. 1.35 kΩ at 85°C (185°F)
    * Approx. 1.19 kΩ at 90°C (194°F)
    * Approx. 1.05 kΩ at 95°C (203°F)

    If resistance is not as specified, replace the sensor.

    <figure markdown="span">
      ![](images/AC0536.webp#illustration){ width="80%" }
    </figure>

7.  Inspect No.1 and No.2 radiator cooling fans.

    1.  Connect battery and ammeter to the cooling fan connector.
    2.  Check that the cooling fan rotates smoothly, and check the reading on the ammeter.

    **Standard amperage:**

    * M/T – 5.8 – 7.4 A
    * A/T – 8.8 – 10.8 A

    <figure markdown="span">
      ![](images/CO1092.webp#illustration){ width="80%" }
    </figure>

[](){ #p-co-25 }
## Removal of radiator cooling fans

<figure markdown="span">
  ![](images/CO1016.webp#illustration)
</figure>

1.  Disconnect cable from negative terminal of battery.

    !!! warning "Caution"

        Work must be started after approx. 20 seconds or longer from the time the ignition switch is turned to
        the "LOCK" position and the negative (`–`) terminal cable is disconnected from the battery.

2.  Disconnect front luggage under covers.
3.  Disconnect upper radiator support seal. (See [Radiator › Removal of radiator, step 4](../radiator.md#removal-of-radiator))
4.  Disconnect radiator cooling fan connectors.
5.  Remove radiator cooling fans.

    Remove the three bolts and cooling fan. Remove the two cooling fans.

    <figure markdown="span">
      ![](images/CO0974.webp#illustration){ width="80%" }
    </figure>

[](){ #p-co-26 }
## Components

<figure markdown="span">
  ![](images/CO1002.webp#illustration)
</figure>

## Disassembly of radiator cooling fans

1.  Remove fan.

    Remove the nut and fan.

    <figure markdown="span">
      ![](images/CO1001.webp#illustration){ width="80%" }
    </figure>

2.  Remove fan motor.

    Remove the three screws and fan motor.

    <figure markdown="span">
      ![](images/CO1000.webp#illustration){ width="80%" }
    </figure>

## Assembly of radiator cooling fans

1.  Install fan motor.
2.  Install fan.

[](){ #p-co-27 }
## Installation of radiator cooling fans

(See [Removal of radiator cooling fans](#removal-of-radiator-cooling-fans))

1.  Install radiator cooling fans.

    Install the cooling fan with the three bolts. Install the two cooling fans.

    <figure markdown="span">
      ![](images/CO0974.webp#illustration){ width="80%" }
    </figure>

2.  Connect radiator cooling fan connectors.
3.  Connect upper radiator support seal. (See [Radiator › Installation of radiator, step 10](../radiator.md#p-co-20))
4.  Connect front luggage under covers.
5.  Connect cable to negative terminal of battery.
