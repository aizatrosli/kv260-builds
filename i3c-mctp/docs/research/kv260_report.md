# KV260: I3C/I2C master on PMOD — grounded facts

Research folder: `<local download>/`
(`docs/` = AMD doc text dumps + UG571 PDF + translator datasheets, `files/` = XDC/board files, `dl/` = git clones)

Abbreviations for local copies:
- `base.xdc` = files/KriaPYNQ_kv260_base.xdc (Xilinx/Kria-PYNQ @ 8b975f4, `kv260/base/vivado/constraints/base.xdc`)
- `bist.xdc` = files/kria-vitis-platforms_kv260_bist_pin.xdc (Xilinx/kria-vitis-platforms @ 60c3f54, `kv260/platforms/kv260_bist/xdc/pin.xdc`; github.com/Xilinx/kv260-vitis is the same repo/commit)
- `board.xml` = XilinxBoardStore @ 9c23992 (branch 2022.2) `boards/Xilinx/kv260_carrier/1.3/board.xml`
- `part0_pins.xml` = XilinxBoardStore `boards/Xilinx/kv260_som/1.4/part0_pins.xml`

## 1. PMOD header J2

- J2 = "Digilent Pmod 2x6 expansion header", on the carrier card (UG1089, Table 1, "Interfaces").
- The PMOD supply pin is 3.3 V, 100 mA (UG1089, "Powering the Starter Kit and Power Budgets", "Pmod Connector").
- All 8 signals are on SOM240_1, signals HDA11 to HDA18, in **HDIO bank 45** (bank HDA). Sources: the bist.xdc comment at line 81 ("HDA11 ... HDA18"); DS987 "SOM240_1 Signal Names and Descriptions" Table 1 ("HDIO on bank 45"); DS987 "HDIO: HDA, HDB, and HDC Banks" Table 1 (HDA = bank 45, VCCO = VCCO_HDA).

| PMOD bit (official XDC port) | SOM240_1 pin / signal (DS987) | Package LOC | Pin name (base.xdc comment) | Cross-checks |
|---|---|---|---|---|
| pmod[0] / PMOD_GPIO_tri_io[0] | A17 / HDA11 | **H12** | IO_L4N_AD12N_45 | base.xdc L26, bist.xdc L82 |
| pmod[1] | D20 / HDA12 (HDGC) | **E10** | IO_L7P_HDGC_45 | base.xdc L25, bist.xdc L83, part0_pins.xml L62 |
| pmod[2] | D21 / HDA13 (HDGC) | **D10** | IO_L7N_HDGC_45 | base.xdc L24, bist.xdc L84, part0_pins.xml L63 |
| pmod[3] | D22 / HDA14 | **C11** | IO_L9P_AD11P_45 | base.xdc L23, bist.xdc L85, part0_pins.xml L64 |
| pmod[4] | B20 / HDA15 | **B10** | IO_L9N_AD11N_45 | base.xdc L22, bist.xdc L86 |
| pmod[5] | B21 / HDA16_CC (HDGC) | **E12** | IO_L8P_HDGC_45 | base.xdc L21, bist.xdc L87, part0_pins.xml L58 |
| pmod[6] | B22 / HDA17 (HDGC) | **D11** | IO_L8N_HDGC_45 | base.xdc L20, bist.xdc L88, part0_pins.xml L59 |
| pmod[7] | C22 / HDA18 | **B11** | IO_L10P_AD10P_45 | base.xdc L19, bist.xdc L89, part0_pins.xml L60 |

- Both official XDCs give the same 8 LOCs. part0_pins.xml confirms 6 of them independently. H12 and B10 are not in part0_pins.xml, because board.xml L43 and L107 define som240_1_a17 and b20 with no interface. The HDA-number-to-SOM-pin mapping comes from DS987 SOM240_1 Table 1.
- **J2 physical pin number for each pmod[i]: UNVERIFIED.** The carrier schematic is login-gated, so I could not check it. By Digilent convention, [0..3] would be pins 1–4 and [4..7] pins 7–10, but that is an assumption.
- **VCCO_HDA voltage.** The DS987 SOM240_1 Table 1 rows B13/B14 say "VCCO_HDA — HDA I/O voltage rail, 1.2V to 3.3V", supplied by the carrier. The KV260 value is **3.3 V by inference, not seen in a schematic**. All official constraints use LVCMOS33 on bank 45: base.xdc L27; bist.xdc L90; board.xml L229–231 (som240_1_d20/d21/d22: LVCMOS33, drive 4, SLOW) and L108–109, L170 (b21, b22, c22: LVCMOS33). The base.xdc comments read "Bank 45 VCCO - som240_1_b13". **Exact KV260 VCCO_HDA rail: UNVERIFIED (schematic).**
- **IOSTANDARD in the official XDCs:** LVCMOS33.
  - bist.xdc L90–92 also sets `SLEW SLOW` and `DRIVE 4`.
  - base.xdc L28–31 sets `PULLUP true` on pmod[2], [3], [6] and [7] (the PYNQ IOP I2C pin positions).
  - UG1091 "SOM I/O Drive Strength Definition" recommends 4 mA and slow slew for HDIO when carrier traces are 8 inches or less.
- **Series resistors, ESD or level translators between the FPGA and J2: UNVERIFIED.** I found no statement in UG1089, DS987 or UG1091 (I searched for "series", "ESD", "level shift" and "translator"). The schematic (XTP682), BOM (XTP713) and carrier XDC (XTP686) sit behind an AMD account license form: the docs.amd.com content endpoint returned an empty zip, and the originUrl is account.amd.com/.../license-form-tip.html. You need to download XTP682 while logged in to settle this.
- In the official designs, the PMOD pins are already driven through tri-state IOBUFs: Kria-PYNQ base.tcl L814–819 (`util_ds_buf` with `C_BUF_TYPE {IOBUF}`, `C_SIZE {8}`) and L1520–1533 (T and I/O wiring).
- The board files also reuse these pins for the Digilent I2S2 audio PMOD: board.xml L583–591, i2s_in on b21/b22/c22 and i2s_out on d20/d21/d22. UG1089 "Supported Peripherals" lists "Audio Codec I2S PMOD (J2), Digilent PMOD SKU 410-379".

## 2. Raspberry Pi header

- **The KV260 has no Raspberry Pi 40-pin GPIO header.** UG1089 Table 1 lists none.
- **J21 on the KV260 is the "PMU reset" jumper**, present on Rev 2.0 only, not populated by default (UG1089 Table 1).
- The only RPi item is **J9, the "RPi camera" connector**: a 15-pin connector with 2 MIPI lanes "directly connected to the Zynq UltraScale+ MPSoC HPA bank" (UG1089 Table 1). HPA is HPIO bank 66, which supports 1.0–1.8 V rails (DS987 "Programmable Logic" Table 1 and Functional Interfaces table). Its supply pin is 3.3 V (UG1089 power section).
  - MIPI LOCs: bist.xdc L38–43 (D7/D6 clock; E5/D5 and G6/F6 data).
  - Camera enable: HDA09 = F11 (bist.xdc L57–58; base.xdc L34).
  - The camera I2C is not on J9 as a user bus. It goes from HDA00/HDA01 = G11/F10 (LVCMOS33) through the "HDA IIC switch": base.xdc L11–16, bist.xdc L70–77, board.xml L281 and L500–528. It is not exposed as a header.
- There is no evidence of buffers or level shifters on the J9 path. **UNVERIFIED (schematic).**
- Of the Kria boards, the KR260 is the one with an RPi header. It was not investigated.

## 3. HD bank (bank 45) I/O capability

- **Device:** XCK26-SFVC784-2LV, -2 speed grade, VCCINT = 0.72 V (DS987 "Supported I/O Standards").
- **Supported standards:** the SOM supports all standards of the bank except VREF-based ones (DS987 "Supported I/O Standards"; UG1091 "Supported I/O Standards").
- **3.3 V standards in HD banks: LVTTL and LVCMOS** (UG571 v1.16 Table 3-1, p.339). The DC levels are in DS925 "PL I/O Levels" Table 1 (HD banks):
  - LVCMOS33: VIL max 0.8 V, VIH min 2.0 V, VOL max 0.4 V, VOH min VCCO – 0.4 V.
  - HD VCCO range is 1.14–3.4 V (DS925 "Recommended Operating Conditions").
- **DRIVE for LVCMOS33:** 4, 8, 12 or 16 mA (UG571 Table 3-2, p.343; DS925 PL I/O Levels Table 1 note 4).
- **SLEW:** SLOW (default) or FAST (UG571 p.340, "Output Slew Rate Control"; Table 3-2).
- **Toggle rate:**
  - HD I/O "Maximum data rate 250 Mb/s DDR" (UG571 Table 3-1 p.339, and Table 3-2 note 1 p.344).
  - "For 4 mA drive strength, the maximum frequency of operation is limited to 125 Mb/s" (UG571 Table 3-2 note 2, p.344).
  - HDIO maximum 250 Mb/s (DS987 HDIO section; UG1091 "HDIO Signals").
  - **12.5 MHz SCL (about 25 Mb/s equivalent) is well inside even the 4 mA / 125 Mb/s limit.**
- **Timing at 0.72 V, -2** (DS925 "IOB High Density (HD) Switching Characteristics" Table 1):
  - LVCMOS33_S_4: TOUTBUF_DELAY_O_PAD 3.097 ns.
  - LVCMOS33_F_8: 2.995 ns.
  - LVCMOS33 input TINBUF: 1.213 ns.
  - 3-state enable TOUTBUF_DELAY_TE_PAD (HD) is 6.699 ns (DS925 "IOB 3-state Output Switching Characteristics" Table 1). Budget for this at SDA handoff.
- **Weak pull-up/down:**
  - PULLTYPE NONE / PULLUP / PULLDOWN / KEEPER is supported in HD (UG571 p.342; Table 3-1).
  - DS925 "DC Characteristics Over Recommended Operating Conditions" Table 1, IRPU at VIN = 0 V, VCCO = 3.3 V: **75–190 µA**. At VCCO = 1.8 V: 60–120 µA. IRPD at VIN = 3.3 V: 60–200 µA.
  - DS925 gives currents, not ohms. Derived: 3.3 V / 190–75 µA ≈ **17–44 kΩ equivalent**. My calculation, not an AMD number.
  - This is far too weak for an I2C or I3C open-drain pull-up, so use an external resistor.
  - Pull-ups are disabled and the pins are 3-stated during power-up and configuration (DS987 and UG1091 "Supported I/O Standards").
  - HD pin capacitance CIN max 4.75 pF; leakage max 15 µA (DS925 DC table).
- **Tri-state for open-drain emulation:** supported.
  - UG571 p.340: HD single-ended pins support "input, output, and bidirectional" modes.
  - UG571 p.39 lists OBUFT and IOBUF; p.47 describes IOBUF with an active-High T pin.
  - The Kria-PYNQ base design already uses IOBUF on all 8 PMOD pins.
  - Standard I2C open-drain emulation is I = 0, T = ~drive_low.
  - The I3C push-pull phase is T = 0 with I = data.
  - **HD banks have no DCI** (UG571 Table 3-1 cont., p.340). Uncalibrated ODT applies only to SSTL/HSTL, not LVCMOS.

## 4. Other low-speed PL-connected items on the KV260 (bank 45)

- **J13 fan power (12 V).** The fan gating signal is on an HD I/O pin (UG1089 "Fan and Heat Sink"). It is `fan_en_b` = **A12** (HDA20, SOM240_1 C24), LVCMOS33 (bist.xdc L63–67; part0_pins.xml L56; board.xml bank45_gpio connection_4, L579). This is an output to the fan switch, not a usable signal header.
- **Bank-45 GPIOs already committed to on-board functions:**
  - J11 (HDA02): AP1302 reset.
  - J10: AP1302 standby.
  - K13: IAS shutdown.
  - F11 (HDA09): RPi camera enable.
  - G11/F10 (HDA00/01): I2C to the camera mux.
  - Sources: bist.xdc L50–77; board.xml L423 and L575–579.
  - The bist.xdc comments label J10 and K13 as HDA03 and HDA04, but by DS987 plus part0_pins.xml they are SOM240_1 B17 and B18 = HDA04 and HDA05. The comment numbering looks off by one; the LOCs are what count.
- **Other headers (UG1089 Table 1):** J3 JTAG, J20 JTAG boot-mode jumper (Rev 2), J18/J19 IAS AF jumpers, J21 PMU reset jumper (Rev 2). None of these is a user PL GPIO header.
- Only SOM240_1 is used on the KV260 carrier (board.xml defines only `som240_1_connector`), so HDB/HDC (banks 43/44 on SOM240_2) are not reachable.
- **Conclusion: J2 is the only general-purpose PL I/O header on the KV260.**

## 5. Consequences for I3C

- I3C device voltages (1.2 V / 1.8 V typical) are **not verified here**. I3C spec research is a separate task.
- The PMOD is a 3.3 V LVCMOS33 interface, so a 1.2 V or 1.8 V target needs a translator that does both 12.5 MHz push-pull and open-drain. Note also that the HD bank could only drop to LVCMOS18 or LVCMOS12 if the carrier's VCCO_HDA were changed, which is not possible on the KV260 without hardware rework. That inference is UNVERIFIED (schematic).

Candidates (vendor datasheets saved in `docs/translators/`):

| Part | Supplies | Speed / I3C claim | Notes |
|---|---|---|---|
| **NXP P3A1604UK** (Rev 1.1, 27 Jun 2025) | VCCA 0.72–1.98 V, VCCB 1.62–3.63 V (§1, §2) | "Support I3C/I2C/SMBus/SPI/UART"; push-pull 40 Mbit/s DDR (20 MHz); open-drain 6.8 Mbit/s (3.4 MHz) (§2) | Fits 3.3 V FPGA side ↔ 1.2/1.8 V target. Internal 10 kΩ switched pull-ups on both sides (§7.6). Open-drain needs at least 1 external pull-up (§7.6, §8). 4-bit, WLCSP12 only. |
| **TI TCA39306** (SCPS274B, Mar 2023) | VREF1 0.9–3.3 V, VREF2 1.8–5.5 V; 1.2 V↔3.3 V listed (§1) | "I3C compatible (12.5 MHz supported)" (§1) | Passive FET translator. §9.3 warns that "it may not be possible to support all I3C applications with a FET-based translator" and that pull-ups are needed on both sides for translation (§9.3.2). Push-pull highs on the high side rise only through the pull-up. |
| NXP P3A9606JK (Rev 2.2, 10 Jul 2025) | VCCA and VCCB both 0.72–1.98 V (§1, §2) | "12.5 MHz I3C-bus applications", open-drain and push-pull (§1) | **Not usable directly on a 3.3 V PMOD** (1.98 V max). |
| TI TXS0102 (SCES640L, Jan 2026) | A 1.65–3.6 V, B 2.3–5.5 V | 24 Mbps push-pull, 2 Mbps open-drain (§1); no I3C claim | Cannot reach 1.2 V. 24 Mbps is marginal for a 12.5 MHz clock. Internal 10 kΩ pull-ups (§7.3.5). Not recommended. |
| TI LSF0102 (SDLS972B, Apr 2024) | 0.95/1.2/1.8 V ↔ up to 5 V | ≤100 MHz at ≤30 pF (§1); no I3C claim | Passive FET; needs pull-ups, with the same caveat as TCA39306. |
| NXP NCA9306 | — | — | **UNVERIFIED.** Datasheet not retrieved (nxp.com returned 404 to curl). |

- **External SDA pull-up:** the FPGA's internal pull-up (about 17–44 kΩ equivalent, derived from 75–190 µA) is not adequate. Fit an external pull-up on the target-voltage side. With P3A1604UK, the internal 10 kΩ pull-ups are already present (§7.6); the rise-time estimate is Rup_ext × (CLA + CLB) + about 10 ns (§8).
- For I3C, the controller must be able to switch the SDA pull-up for open-drain vs push-pull phases. A fixed pull-up through a translator is a design tradeoff, and is my inference.
- **SCL:** push-pull is sufficient on the FPGA side for I3C. Use IOBUF with T = 0 (always driving), or OBUF.

## URLs used

- UG1089 (rev 1.4, 2025-06-25): https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit — fetched via https://docs.amd.com/api/khub/maps/v2mvjU92OO9xGlXK~wDWtg/toc. Topics used: Interfaces …/topics/Tif7w5CWe1dvKZ5C8dum3w/content; Power …/GSIWVBXyN~R_H0necbRq6Q; Fan …/Ewf1QCnK7yEHhtTh2fpIVg; Supported Peripherals …/2DiFH8m9pNGT4RuOEkME2w.
- DS987: https://docs.amd.com/r/en-US/ds987-k26-som (map gX6THsxXsWUzU8xcmObJnQ). Topics used: SOM240_1 Signal Names …/n~NtROA_ktDB9AASGhAWFQ; HDIO …/AWqx4K07fIjyWStkxKfzOw; Supported I/O Standards …/WBRD02B~5jHNWAv1VXY_vg; Functional Interfaces …/i0qbb~p7_ydjcEUNrZzzew; PL …/MX5P5OkogY6Pkd8~AvnflA.
- UG1091: https://docs.amd.com/r/en-US/ug1091-carrier-card-design (map slb28oMha7QNvMoAOY5cwg). Topics used: Supported I/O Standards, HDIO Signals, SOM I/O Drive Strength Definition.
- DS925: https://docs.amd.com/r/en-US/ds925-zynq-ultrascale-plus (map sTfrXTPA6iXT4rkhrRi7qg). Topics used: DC Characteristics …/qKLZq6vYwd1esHPEsSAJTA; PL I/O Levels …/aQWGk5ZRufQxcQiFXKR6uQ; HD Switching …/rZvMyQVXiWqnG~PaZBdBSQ; 3-state …/skjhOEYb1tE1Xmqays~6EA; Recommended Operating Conditions.
- UG571 v1.16 PDF: https://docs.amd.com/v/u/en-US/ug571-ultrascale-selectio (downloaded via https://docs.amd.com/api/khub/documents/kFbaUC5HGcXyGNauhgU6Gw/content).
- KV260 carrier schematic XTP682 / XDC XTP686 / BOM XTP713 (login-gated, NOT obtained): https://docs.amd.com/v/u/en-US/XTP682-en-us-shadow, …/XTP686-en-us-shadow, …/XTP713-en-us-shadow.
- https://github.com/Xilinx/Kria-PYNQ (kv260/base/vivado/constraints/base.xdc, kv260/base/base.tcl)
- https://github.com/Xilinx/kria-vitis-platforms (= Xilinx/kv260-vitis) kv260/platforms/kv260_bist/xdc/pin.xdc
- https://github.com/Xilinx/XilinxBoardStore (branch 2022.2) boards/Xilinx/kv260_carrier/1.3/board.xml, boards/Xilinx/kv260_som/1.4/part0_pins.xml
- https://www.nxp.com/docs/en/data-sheet/P3A1604UK.pdf ; https://www.nxp.jp/docs/en/data-sheet/P3A9606JK.pdf
- https://www.ti.com/lit/ds/symlink/tca39306.pdf ; https://www.ti.com/lit/ds/symlink/txs0102.pdf ; https://www.ti.com/lit/ds/symlink/lsf0102.pdf
