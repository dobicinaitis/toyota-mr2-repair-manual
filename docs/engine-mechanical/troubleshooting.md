[](){ #p-em-6 }
# Troubleshooting

## Engine overheating

| Problem          | Possible cause          | Remedy                      | Page      |
|------------------|-------------------------|-----------------------------|-----------|
| Engine overheats | Cooling system faulty   | Troubleshoot cooling system | CO-4      |
|                  | Incorrect ignition timing | Reset timing              | IG-17, 22 |

## Hard starting

| Problem                                        | Possible cause                                                                                                    | Remedy                     | Page      |
|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------------------------|-----------|
| Engine will not crank or cranks slowly         | Starting system faulty                                                                                              | Troubleshoot starting system | [Troubleshooting](../starting-system/troubleshooting.md#troubleshooting) |
| Engine will not start / hard to start (cranks OK) | No fuel supply to injector<ul><li>No fuel in tank</li><li>Fuel pump no working</li><li>Fuel filter clogged</li><li>Fuel line clogged or leaking</li></ul> | Troubleshoot EFI system | FI-11 |
|                                                | EFI system problems                                                                                                 | Repair as necessary        |           |
|                                                | Ignition problems<ul><li>Ignition coil</li><li>Igniter</li><li>Distributor</li></ul>                                | Perform spark test         | IG-5, 10  |
|                                                | Spark plug faulty                                                                                                   | Inspect plugs              | IG-6, 11  |
|                                                | High-tension cords disconnected or broken                                                                           | Inspect cords              | IG-6, 11  |
|                                                | Vacuum leaks<ul><li>PCV line</li><li>EGR line</li><li>Intake manifold</li><li>T-VIS valve (3S-GTE)</li><li>Throttle body</li><li>ISC valve</li><li>Brake booster line</li></ul> | Repair as necessary |  |
|                                                | Air suction between air flow meter and throttle body                                                                | Repair as necessary        |           |
|                                                | Low compression                                                                                                     | Check compression          | [Compression check](compression-check.md#compression-check) |

## Rough idling

| Problem                       | Possible cause                                                                                                    | Remedy                  | Page          |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------|---------------|
| Rough idle, stalls or misses  | Spark plug faulty                                                                                                     | Inspect plugs           | IG-6, 11      |
|                               | High-tension cord faulty                                                                                              | Inspect cords           | IG-6, 11      |
|                               | Ignition problems<ul><li>Ignition coil</li><li>Igniter</li><li>Distributor</li></ul>                                  | Inspect coil<br>Inspect igniter<br>Inspect distributor | IG-8, 12<br>IG-9, 13<br>IG-9, 13 |
|                               | Incorrect ignition timing                                                                                             | Reset timing            | IG-17, 22     |
|                               | Vacuum leaks<ul><li>PCV line</li><li>EGR line</li><li>Intake manifold</li><li>T-VIS valve (3S-GTE)</li><li>Throttle body</li><li>ISC valve</li><li>Brake booster line</li></ul> | Repair as necessary |  |
|                               | Air suction between air flow meter and throttle body                                                                  | Repair as necessary     |               |
|                               | Incorrect idle speed                                                                                                  | Check ISC system        | FI-148, 151   |
|                               | Incorrect valve clearance                                                                                             | Adjust valve clearance  | [Inspection and adjustment of valve clearance (3S-GTE)](engine-tune-up.md#inspection-and-adjustment-of-valve-clearance-3s-gte) |
|                               | EFI system problems                                                                                                   | Repair as necessary     |               |
|                               | Engine overheats                                                                                                      | Check cooling system    | CO-4          |
|                               | Low compression                                                                                                       | Check compression       | [Compression check](compression-check.md#compression-check) |

[](){ #p-em-7 }
## Engine hesitates / poor acceleration

| Problem                                | Possible cause                                                                                                    | Remedy                 | Page      |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------|-----------|
| Engine hesitates / poor acceleration   | Spark plug faulty                                                                                                     | Inspect plug           | IG-6, 11  |
|                                        | High-tension cord faulty                                                                                              | Inspect cords          | IG-6, 11  |
|                                        | Vacuum leaks<ul><li>PCV line</li><li>EGR line</li><li>Intake manifold</li><li>T-VIS valve</li><li>Throttle body</li><li>ISC valve</li><li>Brake booster line</li></ul> | Repair as necessary |  |
|                                        | Air suction between air flow meter and throttle body                                                                  | Repair as necessary    |           |
|                                        | Incorrect ignition timing                                                                                             | Reset timing           | IG-17, 22 |
|                                        | Incorrect valve clearance                                                                                             | Adjust valve clearance | [Inspection and adjustment of valve clearance (3S-GTE)](engine-tune-up.md#inspection-and-adjustment-of-valve-clearance-3s-gte) |
|                                        | Fuel system clogged                                                                                                   | Check fuel system      |           |
|                                        | Air cleaner clogged                                                                                                   | Check air cleaner      | [Maintenance operations](../maintenance/maintenance-operations.md#maintenance-operations) |
|                                        | EFI system problems                                                                                                   | Repair as necessary    |           |
|                                        | Emission control system problem (cold engine)<ul><li>EGR system always on</li></ul>                                   | Check EGR system       | EC-8, 24  |
|                                        | Engine overheats                                                                                                      | Check cooling system   | CO-4      |
|                                        | Low compression                                                                                                       | Check compression      | [Compression check](compression-check.md#compression-check) |

[](){ #p-em-8 }
## Engine dieseling

| Problem                                                   | Possible cause            | Remedy              | Page      |
|-----------------------------------------------------------|---------------------------|---------------------|-----------|
| Engine diesels (runs after ignition switch is turned off) | EFI system problems       | Repair as necessary |           |
|                                                           | Incorrect ignition timing | Reset timing        | IG-17, 22 |
|                                                           | EGR system faulty         | Check EGR system    | EC-8, 24  |

## After fire, backfire

| Problem                                             | Possible cause                                                                                                    | Remedy                       | Page        |
|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------|-------------|
| Muffler explosion (after fire) on deceleration only | Deceleration fuel cut system always off                                                                               | Check EFI (fuel cut) system  |             |
| Muffler explosion (after fire) all the time         | Air cleaner clogged                                                                                                   | Check air cleaner            | [Maintenance operations](../maintenance/maintenance-operations.md#maintenance-operations) |
|                                                     | EFI system problem                                                                                                    | Repair as necessary          |             |
|                                                     | Incorrect ignition timing                                                                                             | Reset timing                 | IG-17, 22   |
| Engine backfires                                    | EFI system problem                                                                                                    | Repair as necessary          |             |
|                                                     | Vacuum leak<ul><li>PCV line</li><li>EGR line</li><li>Intake manifold</li><li>T-VIS valve</li><li>Throttle body</li><li>ISC valve</li><li>Brake booster line</li></ul> | Check hoses and repair as necessary | |
|                                                     | Air suction between air flow meter and throttle body                                                                  | Repair as necessary          |             |
|                                                     | Insufficient fuel flow                                                                                                | Troubleshoot fuel system     | FI-11       |
|                                                     | Incorrect ignition timing                                                                                             | Reset timing                 | IG-17, 22   |
|                                                     | Incorrect valve clearance                                                                                             | Adjust valve clearance       | [Inspection and adjustment of valve clearance (3S-GTE)](engine-tune-up.md#inspection-and-adjustment-of-valve-clearance-3s-gte) |
|                                                     | Carbon deposits in combustion chambers                                                                                | Inspect cylinder head        | EM-74, 110  |

## Excessive oil consumption

| Problem                 | Possible cause                     | Remedy                          | Page         |
|-------------------------|------------------------------------|---------------------------------|--------------|
| Excessive oil consumption | Oil leak                         | Repair as necessary             |              |
|                         | PCV line clogged                   | Check PCV system                |              |
|                         | Piston ring worn or damaged        | Check rings                     | EM-185, 203  |
|                         | Valve stem and guide bushing worn  | Check valves and guide bushing  | EM-75, 111   |
|                         | Valve stem oil seal worn           | Check seals                     |              |

[](){ #p-em-9 }
## Excessive fuel consumption

| Problem               | Possible cause                                                                          | Remedy                        | Page        |
|-----------------------|-------------------------------------------------------------------------------------------|-------------------------------|-------------|
| Poor gasoline mileage | Fuel leak                                                                                   | Repair as necessary           |             |
|                       | Air cleaner clogged                                                                         | Check air cleaner             | [Maintenance operations](../maintenance/maintenance-operations.md#maintenance-operations) |
|                       | Incorrect ignition timing                                                                   | Reset timing                  | IG-17, 22   |
|                       | EFI system problems<ul><li>Injector faulty</li><li>Deceleration fuel cut system faulty</li></ul> | Repair as necessary      |             |
|                       | Idle speed too high                                                                         | Check ISC system              | FI-148, 151 |
|                       | Spark plug faulty                                                                           | Inspect plugs                 | IG-6, 11    |
|                       | EGR system always on                                                                        | Check EGR system              | EC-8, 24    |
|                       | Low compression                                                                             | Check compression             | [Compression check](compression-check.md#compression-check) |
|                       | Tires improperly inflated                                                                   | Inflate tire to proper pressure |           |
|                       | Clutch slips                                                                                | Troubleshoot clutch           |             |
|                       | Brakes drag                                                                                 | Troubleshoot brakes           |             |

## Unpleasant odor

| Problem         | Possible cause                                                                                                    | Remedy              | Page        |
|-----------------|---------------------------------------------------------------------------------------------------------------------|---------------------|-------------|
| Unpleasant odor | Incorrect idle speed                                                                                                  | Check ISC system    | FI-148, 151 |
|                 | Incorrect ignition timing                                                                                             | Reset timing        | IG-17, 22   |
|                 | Vacuum leaks<ul><li>PCV line</li><li>EGR line</li><li>Intake manifold</li><li>T-VIS valve</li><li>Throttle body</li><li>ISC valve</li><li>Brake booster line</li></ul> | Repair as necessary | |
|                 | EFI system problems                                                                                                   | Repair as necessary |             |
