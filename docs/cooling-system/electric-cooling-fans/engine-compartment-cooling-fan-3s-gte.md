[](){ #p-co-31 }
# Engine compartment cooling fan (3S-GTE)

## System circuit

<figure markdown="span">
  ![](images/CO1144.webp#illustration)
</figure>

## Location of engine compartment cooling fan components

<figure markdown="span">
  ![](images/CO1153.webp#illustration)
</figure>

[](){ #p-co-32 }
## On-vehicle inspection

### Low temperature (below 45.5°C (113.9°F))

1.  Turn ignition switch "ON".

    Check that the cooling fan stops.

    If not, check the cooling fan relays and engine compartment temperature sensor, and check for a separated
    connector or severed wire between the cooling fan relay and engine compartment temperature sensor.

    <figure markdown="span">
      ![](images/CO1122_CO1135.webp#illustration){ width="80%" }
    </figure>

2.  Disconnect engine compartment temperature sensor connector.

    Check that the cooling fan rotates.

    If not, check the ignition main relay, cooling fan relays, cooling fan ECU, cooling fan and fuses, and check
    for a short circuit between the cooling fan relay and engine compartment temperature sensor.

    <figure markdown="span">
      ![](images/CO1109_CO1136.webp#illustration){ width="80%" }
    </figure>

3.  Connect engine compartment temperature sensor connector.

### High temperature (above 62.5°C (144.5°F))

4.  Start engine.

    1.  Raise coolant temperature to above 62.5°C (144.5°F).
    2.  Check that the cooling fan rotates.

        <figure markdown="span">
          ![](images/CO1157_CO1136.webp#illustration){ width="80%" }
        </figure>

    If not, replace the engine compartment temperature sensor.

[](){ #p-co-33 }
## Inspection of engine compartment cooling fan components

1.  Inspect cooling fan ECU for circuit.

    Disconnect the cooling fan ECU connector, and check the connector on the wiring harness side as shown in the
    chart.

    <figure markdown="span">
      ![](images/cooling-fan-ecu-connector-wiring-harness-side.webp#illustration){ width="80%" }
    </figure>

    | Check for   | Tester connection | Condition                    | Specified value  |
    |-------------|--------------------|-------------------------------|-------------------|
    | Continuity  | 1 – Ground         | –                             | Continuity        |
    | Voltage     | 2 – Ground         | Ignition switch ON            | Battery voltage   |
    | Voltage     | 3 – Ground         | Ignition switch ON            | Battery voltage   |
    | Resistance  | 5 – 6              | Coolant temp. 20°C (68°F)     | Approx. 2.45 kΩ   |
    |             |                    | Coolant temp. 57.5°C (135.5°F) | Approx. 0.63 kΩ  |
    |             |                    | Coolant temp. 80°C (176°F)    | Approx. 0.32 kΩ   |
    | Voltage     | 7 – Ground         | Ignition switch ON            | Battery voltage   |
    | Continuity  | 9 – Ground         | Ignition switch ON            | Battery voltage   |

2.  Inspect ignition main relay ("IGN"). (See [Ignition main relay "IGN"](../../charging-system/ignition-main-relay.md#ignition-main-relay-ign))
3.  Inspect cooling fan main relay ("VENT"). (See [Radiator cooling fans (w/ A/C) › Inspection of radiator cooling fan components](radiator-cooling-fans-w-a-c.md#p-co-23))

    Check the relay the same way as for the No.1 Cooling Fan Relay.

4.  Inspect engine compartment temperature sensor.

    Using an ohmmeter, measure the resistance between the terminals.

    **Resistance:**

    * Approx. 2.45 kΩ at 20°C (68°F)
    * Approx. 0.63 kΩ at 57.5°C (135.5°F)
    * Approx. 0.32 kΩ at 80°C (176°F)

    If resistance is not as specified, replace the sensor.

    <figure markdown="span">
      ![](images/CO1134.webp#illustration){ width="80%" }
    </figure>

5.  Inspect engine compartment cooling fan.

    1.  Connect battery and ammeter to the cooling fan connector.
    2.  Check that the cooling fan rotates smoothly, and check the reading on the ammeter.

    **Standard amperage:** 3.1 – 4.3 A

    <figure markdown="span">
      ![](images/CO1111.webp#illustration){ width="80%" }
    </figure>

[](){ #p-co-34 }
## Removal of engine compartment cooling fan

<figure markdown="span">
  ![](images/CO1051.webp#illustration)
</figure>

1.  Disconnect cable from negative terminal of battery.

    !!! warning "Caution"

        Work must be started after approx. 20 seconds or longer from the time the ignition switch is turned to
        the "LOCK" position and the negative (`–`) terminal cable is disconnected from the battery.

2.  Remove RH engine hood side panel.
3.  Remove No.1 and No.2 air intake connectors. (See [Intercooler, steps 4, 5](../../turbocharger-system/intercooler.md#intercooler))
4.  Disconnect engine compartment cooling fan connector.
5.  Remove engine compartment cooling fan.

    Loosen the three bolts, and remove the cooling fan.

    <figure markdown="span">
      ![](images/CO0967.webp#illustration){ width="80%" }
    </figure>

[](){ #p-co-35 }
## Components

<figure markdown="span">
  ![](images/CO0990.webp#illustration)
</figure>

## Disassembly of engine compartment cooling fan

1.  Remove fan.

    Remove the nut and fan.

    <figure markdown="span">
      ![](images/CO0991.webp#illustration){ width="80%" }
    </figure>

2.  Remove fan motor.

    Remove the three screws and fan motor.

    <figure markdown="span">
      ![](images/CO0992.webp#illustration){ width="80%" }
    </figure>

## Assembly of engine compartment cooling fan

1.  Install fan motor.
2.  Install fan.

[](){ #p-co-36 }
## Installation of engine compartment cooling fan

(See [Removal of engine compartment cooling fan](#removal-of-engine-compartment-cooling-fan))

1.  Install engine compartment cooling fan.

    Install the cooling fan with the three bolts.

    <figure markdown="span">
      ![](images/CO0967.webp#illustration){ width="80%" }
    </figure>

2.  Connect engine compartment cooling fan connector.
3.  Install No.1 and No.2 air intake connectors. (See [Intercooler › Installation of intercooler, steps 10, 11](../../turbocharger-system/intercooler.md#p-tc-25))
4.  Install RH engine hood side panel.
5.  Connect cable to negative terminal of battery.
