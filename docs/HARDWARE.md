# TARS hardware: the 50 cm build

Parts chosen 2026-11-05. Every price was read on the product page itself; a screenshot of each
is in `runs/parts-proof/` (not in git). Nothing is ordered yet.

## Decisions so far

- **Size:** 50 cm tall. The printer (Bambu Lab X2D, ordered 2026-11-04) has a 256 mm build
  volume, so each slab prints in two ~250 mm halves.
- **Material:** PETG (4 kg ordered with the printer).
- **CAD:** Fusion (student licence). One design file with components: `central`, one `arm`
  component placed twice (turned 180°), bought parts inserted as STEP components with their real
  masses, printed parts with an effective density for their infill. Hinges and slides as joints.
  Export to MuJoCo via the ACDC4Robot add-in, then check its masses and inertias against Fusion's.
- **Simulation order:** no half-made 50 cm model. CAD first, then masses, inertias and centres
  of gravity from it, then full training on that body with the recipe chosen on the 36 cm TARS.
- **Actuators:** Feetech bus servos (position, speed and load feedback; 360° magnetic encoder;
  free rotation in motor mode, so reusable in other projects).
- **Hinge range (2026-11-06):** each arm must turn past 180° both clockwise and
  counterclockwise, so either end of an arm can be the foot: `hinge_range` = ±210° in
  `hardware/tars_params.json` (the 36 cm simulator body has ±150°). About 8 conductors cross each
  hinge (slide servo bus + corner switches), so they need a twist-tolerant route (see Things to know). Beyond
  ±180° the hinge servos must run in multi-turn mode; single-turn position mode covers one turn.
- **Hinges (2026-11-06): bearings on a stiff pipe.** A 12 mm steel/stainless tube runs along y
  through the central body and both carriers, clamped in the central body's side walls. Each
  carrier has a hub turning on it on two 6801-2RS ball bearings spaced apart, so the axle carries
  the arm's weight and impacts and the servo only gives torque. Each STS3250 sits off-axis in the
  central body (below the axle) and drives a pulley on its hub with a closed GT2 belt, 1:1 (no
  backlash, unlike printed gears). The 8 wires per arm run inside the tube and only twist where
  they leave it in the carrier. The slide servo sits beside the axle in the carrier.
- **Carrier = one printed piece (2026-11-06):** carrier box + hub + hinge pulley, no coupling (a
  toothed face coupling was considered and dropped as more complex). Assembly: push the hub and
  pulley in from outside through a ~30 mm hole in the central body's side wall; put the closed belt
  on the hub pulley and the servo pulley (servo in slotted holes to tension it); slide the tube in
  and lock it with a 12 mm shaft collar at each end; close the two sleeve halves around the carrier.
  The sleeve's inner wall needs a ~300 mm slot (2 × slide_travel + hub) for the hub, so stiffen
  the other walls and the slot edges.
- **Slide surfaces:** PTFE on PETG, never PETG on PETG. Removable printed corner pads on the
  carrier, wrapped in 0.3 mm PTFE film clamped on their non-sliding faces, screwed back with
  adjustment; PETG corner rails in the sleeve, printed lying down so the layer lines run along
  the slide.
- **Slides:** servo-driven GT2 belt along each arm. It can be pushed back when unpowered, like
  the simulator's slide; powered, the servo holds position.

## Parts list (verified prices)

### Actuators: Alibaba, official Feetech store (Shenzhen Feite Model Co.), free shipping

| part | qty | each | link |
|---|---|---|---|
| STS3250 12 V, 50 kg·cm stall, 74.5 g (hinges), without bracket | 2 | 443.84 SEK (474.85 with bracket) | [Alibaba](https://www.alibaba.com/product-detail/Feetech-STS3250-12V-50kg-Servo-High_1601711893549.html) |
| STS3215-C046 7.4 V, 1:147, 14.4 kg·cm stall, 4.8 rated, ~106 rpm (slides) | 2 | 137.59 SEK (min. 2) | [Alibaba](https://www.alibaba.com/product-detail/Low-Cost-Feetech-STS3215-Servo-7_1600999461525.html) |
| FE-URT-2 USB bus board (one for the 12 V bus, one for the 7.4 V bus) | 2 | 71.32 SEK | [Alibaba](https://www.alibaba.com/product-detail/URT-1-Updated-FE-URT-2_1601733887787.html) |

Alibaba: "customs processing fee may apply for EU orders", so up to 25 % Swedish VAT may come
on top. Delivery quoted 17 Nov – 26 Dec.

### Electronics

| part | price | where |
|---|---|---|
| 5 V 5 A converter for the Pi 5, 8–40 V in (Zorblix XY40V5A4LKT), wire output | 175 kr | [Amazon.se B0FNWMCZKM](https://www.amazon.se/dp/B0FNWMCZKM) |
| 2 × 300 W 20 A adjustable step-down, 6–40 V in (12 V rail, 7.4 V rail) | 2 × 128.99 kr | [Amazon.se B083PZDF17](https://www.amazon.se/dp/B083PZDF17) (4 left 2026-11-05) |
| 5" HDMI 800×480 touchscreen (TFT, not IPS) for the face | 371.99 kr | [AliExpress](https://www.aliexpress.com/item/1005007080913479.html) |
| ReSpeaker Lite USB 2-mic board (echo cancelling, speaker output) | 500.04 kr | [AliExpress](https://www.aliexpress.com/item/1005010073062099.html), variant "2Mic Array ReSpeaker" |
| 5 W speaker for it | 63.30 kr | same listing, variant "Only 5W Speaker"; shipping ~59 kr |
| GT2 belt 5 m + 8 × 20T pulleys (5 mm bore) | 149.99 kr | [Amazon.se B07QNWJ1XC](https://www.amazon.se/dp/B07QNWJ1XC) |
| 2 × Pi 5 camera cable, 22 to 15 pin, 30 cm | 139.28 kr | [Amazon.se B0DQBHNT9C](https://www.amazon.se/dp/B0DQBHNT9C) |
| 6801-2RS ball bearings 12 × 21 × 5 mm, pack of 10 (4 for the hinge hubs) | 114.69 kr | [Amazon.se B09FSQ466J](https://www.amazon.se/dp/B09FSQ466J) |

Already owned: Raspberry Pi 5, 2 × IMX219 camera, 2 × DXF 4S 5200 mAh 60C LiPo (138 × 47 × 48.5 mm, ~480 g each, XT90 plug; Amazon.se B0B9K69TMQ), an MPU-6050 6-axis IMU
(accelerometer + gyro).

Not yet priced: XT60 connectors, main switch, fuse, LiPo voltage alarm, 12 mm axle tube (Amazon.se lists 304 stainless 12 × 1 mm in 250/300 mm and 12 × 10 mm carbon tubes; buy 300 mm and cut; prices not yet checked on the pages), 12 mm shaft collars, PTFE film (0.3 mm sheet seen at 137 kr, unchecked) for the slide pads, smooth GT2 idlers (5 for 118.43 kr seen on Amazon.se, bore not yet checked).

### Screws, nuts and sheet for the arm mechanism (from the CAD, not yet priced)

Counts are for the whole robot: two hinge drives (one per arm) and one centre axle clamp (two mirrored halves).
An M2 + M3 socket-head assortment with nuts covers all of it.

| part | qty | used for |
|---|---|---|
| M2×10 socket head | 8 | servo sleeve to the servo's back holes (4 per sleeve) |
| M3×16 socket head | 4 | axle clamp top to bottom (2 per half), into a pilot hole |
| M3×20 socket head | 4 | bearing cap to the sleeve (2 per cap) |
| M3×30 socket head | 4 | belt tension jack screws (2 per side); 30 mm gives ~6 mm of tension travel |
| M3 hex nut, DIN 934 (or nylon lock nut DIN 985, 4 mm tall: set `m3_nut_h` to 4 mm) | 4 | sunk into the clamp bottom under the jack screws |
| M3 screws, pulley to servo horn | 2 × the pulley's holes | length not yet checked |
| 1 mm steel sheet, cut by hand | 2 plates of 28 × 10 mm | tension plates on the sleeve tops, under the jack screw tips |
| Turmberg3D GT2 closed belt, 10 mm wide, 218 mm (109 teeth), fibreglass cord, pack of 2: 108.85 kr, [Amazon.se B09B2FGTN7](https://www.amazon.se/dp/B09B2FGTN7) (checked 2026-11-08) | **ordered 2026-11-08**: 2 packs = 4 belts (2 spares), delivery est. 17 Nov | hinge belts; 49.0 mm centre distance, inside the 47.4–56 mm the rails and jack screws allow. No published load rating (Gates 9 mm: 6.8 N·m at 60T) |

**Total ≈ 3,000 kr** before possible VAT on the Alibaba order (≈ 330 kr).

## Rejected after checking prices

- SteadyWin GIM6010-8 brushless with driver: €111.35 (OpenELab), 1,223.58 kr (AliExpress);
  €124.58 / 1,352.43 kr with dual encoder. About 3× an STS3250. The "€58" and "412 kr" prices
  seen in search results were the motor or the driver board alone.
- SteadyWin GIM3505-8 with driver: €127.89 (OpenELab).
- STS3250 elsewhere: AliExpress 694.17 kr; OpenELab €76.05 (pre-order, 10–20 days).
- ELECROW 5" HDMI IPS: 630.37 kr on Amazon.se.
- USB audio board with speakers (Amazon.se B0CXJ2RTWJ): unavailable.
- The phone-screen spare part as a face: phone panels need their own driver, timing and
  connector, and the Pi 5's two MIPI ports are taken by the two cameras.

## Things to know

- **Two voltages:** STS3250 runs at 12 V, STS3215-C046 at 7.4 V (5–8.4 V), so two buses, two
  bus boards and two converters, all from the 4S battery (14–16.8 V).
- **Slide speed:** C046 ~106 rpm unloaded at 7.4 V. On a 40-tooth pulley (80 mm per turn) that's
  ~0.14 m/s and ~37 N at rated torque; on 20T, ~0.07 m/s and ~74 N. The 36 cm policy used
  0.14 m/s median, 0.4 m/s at the 90th percentile, so the real slides are slower; the simulator
  gets the real limits.
- **What the 36 cm policy asks of its actuators** (CPU MuJoCo, flat, 8 episodes): hinge typical
  0.55 N·m, at its 1.5 N·m limit ~10 % of the time, 46 rpm typical, 63 rpm p90; slide at its 10 N
  limit ~78 % of the time.
- **Pi 5 power:** a plain 5 V supply (no USB-PD) limits USB current unless
  `usb_max_current_enable=1` is set.
- **Corner contact switches (2026-11-06):** each arm has 4 corner detectors: a printed springy
  corner that closes a circuit when pressed. They sit on the sleeve, so their wires cross the slide
  as well as the hinge.
- **Cables per arm:** the slide servo's bus (V+, GND, data) and the 4 corner switches (4 signals,
  shared GND): about 8 conductors, read by Pi GPIO with pull-ups. Planned routes:
  - sleeve to carrier (slide travel): a flat ribbon in a rolling loop along the arm, as on a
    printer's moving head;
  - carrier to central (±hinge_range): a clock spring, i.e. the cable wound 2-3 loops in a flat
    ring-shaped pocket around the hinge axis, so it coils and uncoils like a car steering wheel's.
    A slip ring would allow endless rotation but has to sit on the axis, where the hinge servo is.
- **Stereo cameras:** the two IMX219s are not hardware-synchronised; frames are matched in
  software for mapping.
