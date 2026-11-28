# I3C Basic: facts for an SDR controller-side signal generator

## 0. Source document

| Item | Value |
|---|---|
| Document | *MIPI Alliance Specification for I3C Basic℠*, **Version 1.1.1, 9 June 2021**, MIPI Board Adopted 23 July 2021, **Public Release Edition** (cover page; PDF metadata title "MIPI Alliance Specification for I3C Basic, Version 1.1.1 (Public Release Edition)", created 2021-07-27, 436 pages) |
| Local file | `<local download>/pdf/mipi_I3C-Basic_specification_v1-1-1.pdf` |
| SHA-256 | `a0fad6834c2b66901f3dc7fed1dac9edade32cdd75c5df769363721863265753` (4,700,742 bytes) |
| Downloaded from | `https://raw.githubusercontent.com/HateHanzo/i3c/master/doc/mipi_I3C-Basic_specification_v1-1-1.pdf`, a third-party GitHub mirror (repo page: https://github.com/HateHanzo/i3c/blob/master/doc/mipi_I3C-Basic_specification_v1-1-1.pdf) |
| Text extraction | `spec.txt` and `spec_pp.txt` (page-tagged) in the same folder |
| Page numbering | "p.N" below is the **printed** page number. The PDF page index is printed page + 18. |

Why a mirror was used:
- The official MIPI pages only offer the spec through a HubSpot form:
  - https://www.mipi.org/mipi-i3c-basic-download (latest, v1.2)
  - https://www.mipi.org/mipi-i3c-basic-download-previous-versions (v1.0 and v1.1.1)
- After the form, the thank-you page (https://www.mipi.org/thank-you-for-downloading-mipi-i3c-basic-v1-2) says the link is **sent by email**. No direct PDF URL appears on those pages. Guessed `hubfs` URLs returned 404.
- I did not submit the form, because that would need personal contact details.
- The mirrored file is the unmodified MIPI "Public Release Edition": the MIPI copyright is on every page, and the metadata comes from Acrobat PDFMaker. It is still a copy from a third-party host, not from mipi.org.

What could not be verified:
- **I3C Basic v1.2** (April 2025) is email-gated. Differences between v1.1.1 and v1.2 are UNVERIFIED.
- The **v1.1.1 Errata 01** (2022) was not obtained, so errata changes are UNVERIFIED.
- The **full (member-only) MIPI I3C v1.1.1** spec was not used. It adds HDR-TSP/TSL and other features.

Every fact below comes from the v1.1.1 Public Release Edition.

Other URLs fetched (HTML only, no spec content): https://www.mipi.org/specifications/i3c-sensor-specification

---

## 1. SDR bus framing

### 1.1 START, Repeated START, STOP and messages

| Fact | Citation |
|---|---|
| I3C START and STOP are signalled exactly like I2C. Only the timing differs (compare Table 86 with Table 85). | §5.1.2, p.37, l.756-758 |
| In SDR (not HDR), a STOP or Sr is tolerated whenever SCL is High while the Controller controls SDA or SDA is Open Drain. A STOP or Sr in the middle of an Address or data byte cancels it. | §5.1.2 Note, p.37, l.759-766 |
| A Message runs from the initial S (or Sr) to the next Sr or P. | §5.1.2, p.37, l.776-777 |
| SCL is driven only by the Controller. It is normally Push-Pull, "but it can also be Open Drain". (Table 82 Note 6 says "Open Drain never occurs on SCL".) | §5.1.2, p.37, l.771-772; Table 82 Note 6, p.361 |
| A Message is I3C SDR if its address is 7'h7E or the Target's Dynamic Address. All I3C Targets shall match 7'h7E, and no I2C target matches 7'h7E. | §5.1.2, p.37, l.778-784 |
| Figure 12 compares the header forms: `S/Sr, 7'h7E, W(0), ACK, DATA, T, P` versus the I2C form `..., DATA, ACK/NACK, P`. | Fig 12, p.38 |

### 1.2 Address header, the 7'h7E broadcast and arbitration

| Fact | Citation |
|---|---|
| Header format: 7 address bits, RnW, ACK/NACK. A header after **S** is *Arbitrable*: S, at least the first address bit, and the ACK/NACK are Open Drain. | §5.1.2.2, p.44, l.957-962 |
| A header after **Sr** is always Push-Pull, except the ACK/NACK, which is Open Drain. | §5.1.2.2, p.44, l.963-964; §5.1.2.2.4, p.49, l.1087-1092 |
| Targets can request three things in an arbitrable header:<br>• IBI: own Dynamic Address with RnW=1<br>• Controller Role Request: own Dynamic Address with RnW=0<br>• Hot-Join: 7'h02 | §5.1.2.2, p.44, l.965-978 |
| Targets make these requests only after a START that follows Bus Free, or during Bus Available by pulling SDA Low themselves. In the second case:<br>• the Controller shall pull SCL Low within **tCAS**<br>• the Controller also pulls SDA Low<br>• the Target then drives SDA as Open Drain | §5.1.2.2, p.44, l.979-990 |
| Arbitration rule: to send a 0, drive SDA Low after SCL falls and hold it until the next SCL fall. To send a 1, release SDA (High-Z) on SCL fall and sample SDA on SCL rise. Seeing Low means arbitration is lost, and the device stops transmitting. | §5.1.2.2.1, p.45, l.991-1006 |
| Optimization: this is **not allowed** on the first `S + 7'h7E + W` after bus initialization, so Targets can disable their I2C spike filters. If the Controller sends A6=1 and samples SDA=1 on the SCL rise, it may send the rest of the header (up to ACK) in Push-Pull. If A6=0 or a Target pulled SDA Low, the header stays Open Drain. | §5.1.2.2.2, p.46, l.1008-1035; Fig 13, p.47 |
| Address allocation hint: IBI-capable Targets go in 7'h03-7'h3F (A6=0); the others go in 7'h40-7'h7B. | §5.1.2.2.2, p.46, l.1016-1021 |
| Collisions when the Controller starts a frame with a Dynamic Address:<br>• Private Write vs IBI: the Controller wins.<br>• Private Read vs Controller Role Request: the Target wins.<br>• Same RnW on both sides: nobody ACKs, and the Controller should retry after an Sr, which is not arbitrated. | §5.1.2.2.3, p.48, l.1049-1086 |
| Reserved or forbidden Dynamic Addresses:<br>• 7'h00, 01, 02, 7E, 7F are reserved.<br>• 7'h3E, 5E, 6E, 76, 7A, 7C, 7F are single-bit-error aliases of 7E.<br>• 7'h03 is optional.<br>• 7'h04-07, 78-7B, 7C-7D are conditional. | §5.1.2.2.5, p.49; **Table 8**, p.50 |
| Hot-Join address is 7'h02. 7'h01 is used for SETDASA point-to-point. | Table 8, p.50 |

### 1.3 ACK/NACK and the switch from Open Drain to Push-Pull

| Event | SDA drive and handoff | Citation |
|---|---|---|
| ACK of the address header | Always Open Drain, driven by the Target(s). For 7'h7E all I3C Targets ACK. | §5.1.2.3.1, p.51, l.1130-1134 |
| ACK, then Controller write data | 1. Target holds SDA Low during the ACK (Open Drain SCL Low period).<br>2. Target releases SDA as soon as it sees SCL rise.<br>3. Controller drives SDA Low after SCL rises (a short overlap is safe). The SCL High may be as short as tDIG_H min.<br>4. Controller drives data **Push-Pull from the falling edge of SCL**. | §5.1.2.3.1, p.51, l.1135-1148; Fig 142, p.371 |
| NACK of the address | Controller issues Sr or P. | §5.1.2.3.1, p.51, l.1149-1153; Fig 143, p.372 |
| IBI ACK, then the MDB | 1. Controller holds SDA Low during the ACK (Open Drain).<br>2. Target drives Low after SCL rises.<br>3. Controller releases SDA after tSCO, a safe time, or the next SCL fall.<br>4. Target drives the MDB Push-Pull from the SCL fall. | §5.1.2.3.2, p.52, l.1154-1174; Fig 14, p.53 |
| IBI NACK | Controller issues Sr or P. | §5.1.2.3.2, p.52, l.1175-1180; Fig 15, p.54 |

Phase-by-phase drive (Tables 88-90, p.383):

| Frame | S | A6 | A5:A0 | RnW | ACK | N×(Data+T) | P |
|---|---|---|---|---|---|---|---|
| Table 88: new frame, no contention on A6 | OD | OD | PP (optional) | PP (optional) | OD | PP | PP |
| Table 89: new frame, contention on A6 | OD | OD | OD | OD | OD | PP | PP |
| Table 90: after Sr | Sr is PP | header (8 clocks) is PP | | | OD | PP | PP |

### 1.4 Ninth bit (T-bit)

**Controller write: odd parity**

| Fact | Citation |
|---|---|
| The ninth bit of each SDR data word the Controller writes is odd parity: `T = XOR(Data[7:0], 1)`. Examples: 0xFF gives 1, 0x00 gives 1, 0xFE or 0x01 gives 0. | §5.1.2.3.3, p.55, l.1185-1192 |
| The Target shall not drive SDA on Controller-written data. | §5.1.2.3, p.51, l.1117-1121 |
| The T bit must stay valid through SCL High. The next SCL Low may then change SDA (or not) to set up Sr or P. | §5.1.2.3.3, p.55, l.1193-1196 |

**Target read: End-of-Data**

The rules below are all from §5.1.2.3.4, p.56, l.1197-1230. Figures 150-154 are on p.376-380.

- **T = 0, the Target ends the read:**
  - The Target drives SDA Low on the SCL fall and releases it (High-Z) on the next SCL rise.
  - The Controller drives SDA Low on that SCL rise, overlapping the Target.
  - The Controller then issues P (Fig 150) or Sr (Fig 151).
- **T = 1, the Target offers more data:**
  - The Target drives SDA High on the SCL fall and goes High-Z on the SCL rise. This "parks" the bus, and the weak pull-up holds SDA High.
  - **To continue:** the Controller does nothing (Fig 152).
  - **To abort:** the Controller drives SDA Low *after* the SCL rise, which creates an Sr (it may stretch SCL High to avoid contention). It then takes control from the SCL fall and sends a new address (Fig 154) or a P (Fig 153).
  - The Target samples SDA on the SCL fall. Low means the read was aborted.
- **Target timeout:** the Target should abandon a read if SCL does not change for 100 µs or more. (§5.1.2.3 Note, p.51, l.1126-1129)

### 1.5 Controller clock stalling

Source: §5.1.2.5, p.59; **Table 10**.

| Stall point (SCL held Low) | Max |
|---|---|
| ACK/NACK phase | 100 µs |
| Write parity bit | 100 µs |
| Read T-bit | 100 µs |
| ENTDAA, first bit of the assigned address | 15 ms |

The absolute maximum stall is 15 ms.

---

## 2. Timing

### Table 86: I3C Open Drain timing (§6.2, p.366; notes on p.367)

| Symbol | Min | Max | Notes |
|---|---|---|---|
| tLOW_OD (SCL Low) | 200 ns | – | Notes 1, 2 (a shorter Low is allowed if SDA is known to be above VIH) |
| tDIG_OD_L | tLOW_ODmin + tfDA_ODmin | – | |
| tHIGH_INIT (SCL High for the first Broadcast Address after init) | 200 ns | – | Note 9 |
| tHIGH (Mixed Bus) | – | **41 ns** | Notes 3, 4 |
| tDIG_H (Mixed Bus) | – | tHIGH + tCF | |
| tHIGH (Pure Bus) | 24 ns (tHIGH min from Push-Pull) | – | |
| tDIG_H (Pure Bus) | 32 ns (tDIG_H min from Push-Pull) | – | |
| tfDA_OD (SDA fall) | – | 12 ns | |
| tSU_OD (SDA setup, OD) | 3 ns | – | |
| **tCAS** (clock after S) | **38.4 ns** | ENTAS0: 1 µs; ENTAS1: 100 µs; ENTAS2: 2 ms; ENTAS3: 50 ms | Notes 5, 6 (Targets without ENTASx use 50 ms) |
| **tCBP** (clock before P) | tCASmin / 2 | – | |
| tCRHPOverlap | tDIG_OD_Lmin | – | |
| tAVAL (Bus Available) | 1 µs | – | Note 7 |
| tIDLE (Bus Idle) | 200 µs | – | |
| tNEWCRLock | tAVALmin | – | |

### Table 87: I3C Push-Pull timing for SDR, ML, HDR-DDR, HDR-BT (p.368; notes on p.369)

| Symbol | Min | Typ | Max | Notes |
|---|---|---|---|---|
| **fSCL** | 0.01 MHz | **12.5 MHz** | **12.9 MHz** | Note 1: fSCL = 1/(tDIG_L + tDIG_H) |
| tLOW | 24 ns | | | |
| tDIG_L | 32 ns | | | Notes 2, 4 |
| tHIGH_MIXED | 24 ns | | | |
| tDIG_H_MIXED | 32 ns | | **45 ns** | Notes 2, 3 |
| tHIGH (Pure) | 24 ns | | | |
| tDIG_H (Pure) | 32 ns | | | Note 2 (32 ns min allows a 40/60 duty cycle at 12.5 MHz) |
| tSCO (Target clock-to-data out) | | | 12 ns | Notes 7, 8 (>12 ns requires BCR[0]=1 and GETMXDS) |
| tCR / tCF (SCL rise/fall) | | | 150e06 × 1/fSCL, capped at 60 ns | Note 5 |
| tHD_PP, Controller | tCR + 3 and tCF + 3 | | | Notes 4, 6 |
| tHD_PP, Target | 0 | | | |
| **tSU_PP** | **3 ns** | | N/A | |
| tCASr (clock after Sr) | tCASmin / 2 | | N/A | Note 9 |
| tCBSr (clock before Sr) | tCASmin / 2 | | N/A | |
| Cb per line | | | 50 pF | |
| tBT_FREQ (HDR-BT) | 0.1 MHz | 12.5 MHz | 12.9 MHz | |

Notes on parameters the request named:
- **tSU_STA, tHD_STA, tSU_STO and tBUF** appear **only in Table 85** (the I2C legacy table). For I3C-only framing, Tables 86 and 87 define tCAS, tCBP, tCASr and tCBSr instead.
- Figure 144 (p.373) shows the I3C START timing: tfDA_OD, tCAS, tCF, tLOW_OD, tCR, tDS_OD, tSU_OD, trDA_OD.
- Figure 145 (p.373) shows the I3C STOP timing: tCR, tCBP.
- **tDS_OD** appears in Figure 144 but has no row in Table 86, so it is UNVERIFIED as a numeric spec.
- The text says Bus Free on a Pure Bus is at least tCAS (§5.1.3.2.1, p.65).

### Table 85: I3C timing when talking to I2C legacy devices (p.365)

| Symbol | Fm (400 kHz) min / max | Fm+ (1 MHz) min / max |
|---|---|---|
| fSCL | 0 / 0.4 MHz | 0 / 1.0 MHz |
| tSU_STA | 600 ns / – | 260 ns / – |
| tHD_STA | 600 ns / – | 260 ns / – |
| tLOW | 1300 ns / – | 500 ns / – |
| tDIG_L | tLOW + trCL / – | tLOW + trCL / – |
| tHIGH | 600 ns / – | 260 ns / – |
| tDIG_H | tHIGH − trCL / – (as printed) | tHIGH + trCL / – (as printed; the sign differs from the Fm column in the source) |
| tSU_DAT | 100 ns / – | 50 ns / – |
| tHD_DAT | – / – | – / – |
| trCL | 20 / 300 ns | – / 120 ns |
| tfCL | 20×(VDD/5.5V) / 300 ns | 20×(VDD/5.5V) / 120 ns |
| trDA, trDA_OD | 20 / 300 ns | – / 120 ns |
| tfDA | 20×(VDD/5.5V) / 300 ns | 20×(VDD/5.5V) / 120 ns |
| tSU_STO | 600 ns / – | 260 ns / – |
| **tBUF** | **1.3 µs** / – | **0.5 µs** / – |
| tSPIKE | 0 / 50 ns | 0 / 50 ns |

Related rules for Mixed Buses:
- Bus Free on a Mixed Bus is at least tBUF (§5.1.3.2.1, p.65; Fig 24).
- On a Mixed Fast Bus, I3C traffic keeps SCL High at or below tDIG_H_MIXED so that I2C 50 ns spike filters do not see it. The Low period can be stretched to slow the effective rate. (§5.1.2.4.1, p.58; Table 9, p.57)

---

## 3. CCC format

| Fact | Citation |
|---|---|
| In SDR, every CCC starts with S/Sr + **7'h7E + W(0)** + ACK, then the Command Code + T. | §5.1.9.1, p.110, l.2755-2757; Table 15, p.112 |
| Broadcast CCCs are always writes. Direct CCCs go to one Target (or a Group, for some writes). They can be Read, Write or Read/Write. | §5.1.9, p.109 |
| **Code ranges:**<br>• Broadcast: **0x00-0x7F**<br>• Direct: **0x80-0xFE**<br>• **0xFF** reserved<br>Bit 7 of the code tells broadcast from direct. | §5.1.9.2, p.113, l.2804-2810 |
| Broadcast frame (Fig 30, p.111): `S/Sr, 7E/W/ACK, CCC/T, [Defining Byte/T], [Data/T]...`, then P or Sr. | Fig 30 |
| Direct frame (Fig 31, p.111): `S/Sr, 7E/W/ACK, CCC/T, [DefByte/T], then per Target: Sr + Addr/RnW/ACK + [SubCmd/T] + Data/T...`, ended by P or Sr + 7'h7E. | Fig 31; §5.1.9.2.2, p.114 |
| A **Defining Byte** follows the CCC code and applies to all later Direct segments until a new CCC. Table 16 marks it R, O or N per CCC. | Table 15, p.112 |
| End of a CCC:<br>• P<br>• for a Broadcast CCC, Sr with any address<br>• for a Direct CCC, Sr + 7'h7E | §5.1.9.2.1, p.113 |
| Direct GET reads use the normal SDR T-bit read model. | Table 15, p.112; §5.1.9.2.2 |
| An unsupported Direct CCC is NACKed at the Target's address. | §5.1.9.2.2, p.114 |
| Direct GET retry: the Target NACKs, the Controller sends Sr + the same address **once more** (it may stall first). If NACKed again, there are no further retries. | §5.1.9.2.3, p.116 |

### Table 16: I3C Common Command Codes (p.118-124)

Column codes:
- "Req" is the CCC requirement level: R = Required, C = Conditional, O = Optional, N = not in I3C Basic.
- "DB" and "SB" say whether the Defining Byte and Sub-Command Byte are R (required), O (optional) or N (not permitted).

**Broadcast CCCs**

| Code | Name | Req | DB | SB | Section |
|---|---|---|---|---|---|
| 0x00 | ENEC | R | N | N | 5.1.9.3.1 |
| 0x01 | DISEC | R | N | N | 5.1.9.3.1 |
| 0x02 | ENTAS0 | C | N | N | 5.1.9.3.2 |
| 0x03 | ENTAS1 | O | N | N | 5.1.9.3.2 |
| 0x04 | ENTAS2 | O | N | N | 5.1.9.3.2 |
| 0x05 | ENTAS3 | O | N | N | 5.1.9.3.2 |
| 0x06 | RSTDAA | R | N | N | 5.1.9.3.3 |
| 0x07 | ENTDAA | R (note 9) | N | N | 5.1.9.3.4 |
| 0x08 | DEFTGTS | C | N | N | 5.1.9.3.7 |
| 0x09 | SETMWL | R | N | N | 5.1.9.3.5 |
| 0x0A | SETMRL | R | N | N | 5.1.9.3.6 |
| 0x0B | ENTTM | O | N | N | 5.1.9.3.8 |
| 0x0C | SETBUSCON | O | N | N | 5.1.9.3.31 |
| 0x0D-0x11 | MIPI reserved | | | | |
| 0x12 | ENDXFER | C | R | N | 5.1.9.3.25 |
| 0x13-0x1E | MIPI reserved | | | | |
| 0x1F | Reserved; dummy code for HDR CCC flows | C | – | – | 5.2.1.2.1 |
| 0x20 | ENTHDR0 (HDR-DDR) | C | N | N | 5.1.9.3.9 |
| 0x21 | ENTHDR1 (TSP) | N | N | N | |
| 0x22 | ENTHDR2 (TSL) | N | N | N | |
| 0x23 | ENTHDR3 (HDR-BT) | C | N | N | 5.1.9.3.9 |
| 0x24-0x27 | ENTHDR4-7 | N | N | N | |
| 0x28 | SETXTIME | C | N | R | 5.1.9.3.21 |
| 0x29 | SETAASA | O | N | N | 5.1.9.3.23 |
| 0x2A | RSTACT | R | R | N | 5.1.9.3.26 |
| 0x2B | DEFGRPA | C | N | N | 5.1.9.3.29 |
| 0x2C | RSTGRPA | C | N | N | 5.1.9.3.28 |
| 0x2D | MLANE | C | R | N | 5.1.9.3.30 |
| 0x2E-0x48 | I3C WG reserved | | | | |
| 0x49-0x4C | Camera WG reserved | | | | |
| 0x4D-0x57 | MIPI reserved | | | | |
| 0x58-0x5B | Debug WG reserved | | | | |
| 0x5C-0x60 | RIO WG reserved | | | | |
| 0x61-0x7F | Vendor / standards extension | | | | |

**Direct CCCs**

| Code | Name | Req | DB | SB | Section |
|---|---|---|---|---|---|
| 0x80 | ENEC | R | N | N | 5.1.9.3.1 |
| 0x81 | DISEC | R | N | N | 5.1.9.3.1 |
| 0x82 | ENTAS0 | C | N | N | 5.1.9.3.2 |
| 0x83 | ENTAS1 | O | N | N | 5.1.9.3.2 |
| 0x84 | ENTAS2 | O | N | N | 5.1.9.3.2 |
| 0x85 | ENTAS3 | O | N | N | 5.1.9.3.2 |
| 0x86 | RSTDAA Direct: **DEPRECATED** (v1.1 Controllers shall not use it; v1.1 Targets NACK it) | | | | |
| 0x87 | SETDASA (Direct Set) | O | N | N | 5.1.9.3.10 |
| 0x88 | SETNEWDA (Direct Set) | C | N | N | 5.1.9.3.11 |
| 0x89 | SETMWL (Direct Set) | R | N | N | 5.1.9.3.5 |
| 0x8A | SETMRL (Direct Set) | R | N | N | 5.1.9.3.6 |
| 0x8B | GETMWL | R | N | N | 5.1.9.3.5 |
| 0x8C | GETMRL | R | N | N | 5.1.9.3.6 |
| 0x8D | GETPID | C | N | N | 5.1.9.3.12 |
| 0x8E | GETBCR | C | N | N | 5.1.9.3.13 |
| 0x8F | GETDCR | C | N | N | 5.1.9.3.14 |
| 0x90 | GETSTATUS | R | O | N | 5.1.9.3.15 |
| 0x91 | GETACCCR | C | N | N | 5.1.9.3.16 |
| 0x92 | ENDXFER | C | R | N | 5.1.9.3.25 |
| 0x93 | SETBRGTGT | O | N | N | 5.1.9.3.17 |
| 0x94 | GETMXDS | C | O | N | 5.1.9.3.18 |
| 0x95 | GETCAPS (formerly GETHDRCAPS) | C | O | N | 5.1.9.3.19 |
| 0x96 | SETROUTE | O | N | N | 5.1.9.3.20 |
| 0x97 | D2DXFER | N | | | |
| 0x98 | SETXTIME | C | N | R | 5.1.9.3.21 |
| 0x99 | GETXTIME | C | N | N | 5.1.9.3.22 |
| 0x9A | RSTACT | R | R | N | 5.1.9.3.26 |
| 0x9B | SETGRPA | C | N | N | 5.1.9.3.27 (the Table 16 row lists 5.1.9.3.29) |
| 0x9C | RSTGRPA | C | N | N | 5.1.9.3.28 |
| 0x9D | MLANE | C | R | N | 5.1.9.3.30 |
| 0x9E-0xDF | Reserved (I3C WG, Camera, MIPI, Debug, RIO) | | | | |
| 0xE0-0xFE | Vendor / standards extension | | | | |
| 0xFF | Reserved | | | | |

Payload details for selected CCCs:

| CCC | Payload | Citation |
|---|---|---|
| ENEC / DISEC event byte | bit 0 = ENINT/DISINT; bit 1 = ENCR/DISCR; bit 3 = ENHJ/DISHJ; other bits reserved | Tables 18-19, p.126 |
| SETMWL / GETMWL | 2 bytes, MSB first; minimum 16 | §5.1.9.3.5, p.130 |
| SETMRL / GETMRL | 2 bytes, MSB first, plus an optional 3rd byte = max IBI payload size (0 = unlimited) when BCR[2]=1 | §5.1.9.3.6, p.132 |
| SETDASA (0x87), SETNEWDA (0x88) | Data byte = {DA[6:0], 1'b0} | p.138, p.140 |
| SETDASA point-to-point form | Static address 7'h01 and Dynamic Address 7'h01 | §5.1.9.3.10, p.139 |
| GETPID | 6 bytes, MSB first | p.141 |
| GETBCR, GETDCR | 1 byte each, MSb first | p.142-143 |
| GETSTATUS Format 1 | 2 bytes:<br>• [15:8] vendor<br>• [7:6] Activity Mode<br>• [5] Protocol Error<br>• [4] reserved<br>• [3:0] Pending Interrupt | Table 27, p.145 |
| RSTACT Defining Byte | • 0x00 = no reset<br>• 0x01 = reset I3C peripheral (default)<br>• 0x02 = reset whole target<br>• 0x03 = debug adaptor reset<br>• 0x04 = Virtual Target detect<br>• 0x81 = GET peripheral reset time (default 1 ms)<br>• 0x82 = GET whole-target reset time (default 1 s)<br>• 0x83 = GET debug network adaptor reset time (100 ms)<br>• 0x84 = GET virtual target indication<br>DB 0x00 and 0x01 are required. | §5.1.9.3.26, p.189; Table 53, p.191 |

---

## 4. ENTDAA procedure

Source: §5.1.4.2, p.69-72; Fig 25, p.71; Figure 170 (DAA FSM).

The whole DAA process is **Open Drain**. Only the Sr and the 7'h7E/R header may optionally be Push-Pull (p.69, l.1501-1505).

1. **Pre-assignment (Primary Controller only).** SETDASA or SETAASA may be used first for Targets with a known Static Address.
2. **Enter DAA.** Send `S/Sr, 7'h7E/W, ACK, 0x07 (ENTDAA), T`. RSTDAA is 0x06.
3. **Address the unassigned Targets.** Send **Sr + 7'h7E + RnW=1**. Every unassigned, DAA-capable Target ACKs (Open Drain). (l.1538-1545)
4. **Release SDA.** The Controller drives only SCL and releases SDA to High-Z, so the pull-up holds it High. (l.1557-1558)
5. **Read the 64 ID bits.** Each Target shifts out its **48-bit Provisioned ID, MSB (bit 47) first, with no delimiters or ACK bits**, then **BCR**, then **DCR**. It drops out when it loses arbitration. This is 8 bytes, i.e. 64 bits. (l.1559-1570; Fig 25 "Read Data: 8 Bytes")
   - The lowest concatenated {PID, BCR, DCR} wins. (l.1571-1572)
   - The Controller may stall SCL on the first bit of the assigned address, up to 15 ms. (§5.1.2.5.4, p.63)
6. **Assign the address.** The Controller drives the 7-bit Dynamic Address, then **PAR = odd parity = ~XOR(DA[6:0])**, Open Drain. The spec writes it as "~XOR(dynamic_address[7:1]) is placed in position 0", so the byte is **{DA[6:0], PAR}**. (p.71, l.1579-1586)
7. **Target response.** If the parity is valid, the Target **ACKs** on the next SCL. If invalid, it NACKs passively. (l.1587-1589)
8. **Repeat** from step 3 (Sr + 7'h7E/R) **until no Device ACKs**, i.e. 7'h7E/R is NACKed. Then send **P**. (l.1590-1593)
   - If a Target NACKs its address twice, the Controller exits and runs error handling. (p.72, l.1605-1610)

48-bit PID layout (§5.1.4.1.1, p.68):

| Bits | Content |
|---|---|
| [47:33] | MIPI Manufacturer ID (15 LSBs) |
| [32] | Type: 1 = random, 0 = vendor-fixed |
| [31:16] | Part ID (vendor-fixed type) |
| [15:12] | Instance ID (vendor-fixed type) |
| [11:0] | Additional meaning (vendor-fixed type) |

When bit 32 = 1, bits [31:0] are a 32-bit random value instead.

Collision handling: if fewer addresses were assigned than expected, send RSTDAA and retry. Three attempts are recommended. (§5.1.4.3, p.73)

---

## 5. IBI, HDR Exit Pattern, Target Reset Pattern

### 5.1 How a Target requests an IBI

Source: §5.1.6.2, p.82-83; §5.1.2.2, p.44.

1. The Target places its **Dynamic Address + RnW=1** in the arbitrable header after an S (not after an Sr).
2. If no START comes, then after **Bus Available** (tAVAL ≥ 1 µs; §5.1.3.2.2, p.65) the Target pulls SDA Low.
3. The Controller must pull SCL Low within **tCAS** (38.4 ns min; max 1 µs to 50 ms depending on ENTASx; Table 86).
4. The Controller also pulls SDA Low, then clocks the header. The Target drives its address Open Drain.
5. Arbitration uses the Open Drain rules, so the lower address wins (priority). (§5.1.6.1, p.82)

The Controller then does one of three things:

| Choice | What happens | Citation |
|---|---|---|
| **ACK (Open Drain), BCR[2]=1** | The Controller must read the **Mandatory Data Byte** (Push-Pull, so it cannot be declined). Extra bytes may follow, up to the SETMRL IBI size, with the usual T-bit continuation. Then P or Sr. | §5.1.6.2, p.82-83; Fig 26, p.83 |
| **ACK, BCR[2]=0** | No payload follows. P or Sr. | §5.1.6.2, p.83 |
| **NACK** | The IBI is refused. The Target retries later. | §5.1.6.2, p.83 |
| **NACK + Sr + DISEC(DISINT)** | The IBI is refused and interrupts are disabled. | §5.1.6.2, p.83 |

The Controller may stall SCL in the ACK phase while it decides (§5.1.2.5.1, p.60).

### 5.2 MDB and IBI-related BCR bits

**MDB** layout (Table 12, p.84):
- [7:5] = Interrupt Group Identifier
- [4:0] = Specific Interrupt Identifier

Group values (Table 13, p.85):

| MDB[7:5] | Group |
|---|---|
| 3'b000 | User / vendor defined |
| 3'b001 | I3C WG (5'h0D = error with data; 5'h0E = generic error) |
| 3'b010 | MIPI groups |
| 3'b011 | Non-MIPI |
| 3'b100 | Timing information |
| 3'b101 | Pending Read Notification |
| 3'b110, 3'b111 | Reserved |

**BCR** bits for IBI: **BCR[2] = IBI Payload** (1 = an MDB follows) and **BCR[1] = IBI Request Capable** (Table 5, p.35).

### 5.3 HDR Exit Pattern

Source: §5.2.1.1.1, p.229; Fig 91.

- SDA starts High and SCL starts Low.
- **SDA falls 4 times while SCL stays Low.** Each SDA transition is at least **tDIG_H** apart.
- One extra set-up edge may be added if needed.
- The pattern is **always followed by a STOP**. After the 4th fall, the Controller drives SCL High to prepare the STOP.
- All Targets must detect it, even if they support no HDR mode.

HDR Restart Pattern (§5.2.1.1.2, p.230): SDA goes fall, rise, fall, rise, then SCL rises.

Example detector RTL: §5.2.1.1.3, p.231 onward.

### 5.4 Target Reset Pattern

Source: §5.1.11.3, p.222; Fig 90.

- **Fourteen SDA transitions while SCL is held Low**, then **Sr followed by P**, which triggers the reset. It is based on the HDR Exit Pattern but is distinguishable from it.
- Recommended sequence (§5.1.11.1, p.220): `S → [RSTACT CCCs, optionally ending in Sr] → Target Reset Pattern (with its Sr + P) → leave the bus free for the reset time → S + header`.
- Keep the RSTACT and the pattern in one frame. A Target cancels a pending RSTACT on an SCL fall after an S. (§5.1.11.1 Note, p.220)
- **UNVERIFIED:** the spacing between the 14 transitions. The extracted text gives none, and the figure was not readable as text. By analogy with the HDR Exit Pattern it is probably at least tDIG_H, but that is an inference.

---

## 6. BCR and DCR

**BCR** (Table 5, §5.1.1.2.1, p.35)

| Bit | Field | Values |
|---|---|---|
| [7:6] | Device Role | 2'b00 = Target; 2'b01 = Controller-capable; 2'b10 and 2'b11 reserved |
| [5] | Advanced Capabilities | 1 = supported; use GETCAPS |
| [4] | Virtual Target Support | 1 = is a Virtual Target or exposes downstream devices |
| [3] | Offline Capable | 1 = will not always respond |
| [2] | IBI Payload | 1 = an MDB (plus optional bytes) follows an accepted IBI |
| [1] | IBI Request Capable | 1 = capable |
| [0] | Max Data Speed Limitation | 1 = limited; use GETMXDS |

The Primary Controller's static address is 7'h7E (Table 5 Note 1).

**DCR** (Table 6, §5.1.1.2.2, p.36): DCR[7:0] = Device ID, the device type. 8'h00 = generic. The MIPI registry holds the assigned codes.

**LVR**, for legacy I2C devices (Table 7, p.36):
- [7:5] = I2C index (Table 4)
- [4] = I2C mode: 0 = Fm+, 1 = Fm
- [3:0] = reserved

---

## 7. Electrical

| Fact | Citation |
|---|---|
| Nominal VDD is 1.2 V, 1.8 V or 3.3 V (not characterized for 5 V). An optional 1.0 V / 100 pF profile exists, for example for DDR5 SPD. | §6.1, p.359, l.7319-7326 |
| VDD ranges: **1.10/1.20/1.30 V**, **1.65/1.80/1.95 V**, **2.97/3.30/3.63 V** | **Table 82**, p.360 |
| Low-voltage profile: 0.95/1.0/1.25 V | Table 83, p.362 |
| Input levels: VIL max 0.3·VDD, VIH min 0.7·VDD; hysteresis ≥ 0.1·VDD | Table 82 |
| Output Low: VOL ≤ 0.18 V at 2 mA (VDD < 1.4 V); ≤ 0.27 V at 3 mA (VDD ≥ 1.4 V) | Table 82 |
| Output High (Push-Pull): VOH ≥ VDD − 0.18 V or VDD − 0.27 V at the same currents | Table 82 |
| Input current Ii: ±10 µA (≥ 1.8 V) or ±5 µA (< 1.8 V) | Table 82 |
| Pin capacitance Ci: ≤ 5 pF (< 1.8 V) or ≤ 10 pF (≥ 1.8 V) | Table 82 |
| Open Drain pull-up Rp: min = (VDD − VOL)/3 mA (VDD ≥ 1.4 V) or /2 mA (VDD < 1.4 V); max = tr/(0.8473·Cb), which is **2833 Ω** for tr = 120 ns and Cb = 50 pF | Table 82 and notes, p.360-361 |
| The Open Drain pull-up **shall be switched off during Push-Pull**. | Table 82 Note 4, p.361 |
| **Open Drain never occurs on SCL.** | Table 82 Note 6, p.361 |
| A separate weak pull-up or **High-Keeper** is required. | Table 82 Note 7, p.361 |
| The Controller shall provide an **active Open Drain class pull-up** whenever the bus is in Open Drain. It may be a resistor from VDD, a current source, or any balanced method that meets trDA without overpowering a minimum-IOL Target. | §5.1.3.1, p.64, l.1344-1352 |
| The bus **shall have High-Keepers on SDA and SCL**. They must be strong enough to beat leakage and weak enough for a minimum-IOL Target to pull Low within tDIG_L. | §5.1.3.1, p.64, l.1353-1359, 1371-1379 |
| The Controller may provide the High-Keeper but must be able to disable it. Otherwise the system designer adds one externally. | §5.1.3.1, p.64, l.1360-1375 |
| The Controller should switch SCL and SDA independently between three pull-up states: High-Z, High-Keeper, and Open Drain pull-up. | §5.1.3.1, p.64, l.1366-1370 |
| The SDA driver must switch dynamically between Push-Pull and Open Drain. | §6.2, p.366, l.7341-7342 |
| Legacy I2C devices on an I3C bus:<br>• Fm is required<br>• Fm+ is desirable<br>• a 50 ns spike filter is desirable<br>• **clock stretching by the Target is not allowed** | Table 84, p.363 |

### Notes for the RTL signal generator

These follow from the cited rules:
- SCL is always Push-Pull.
- SDA needs three states: drive-0, drive-1 (Push-Pull) and release. Release covers both Open Drain '1' and handing the line to the Target. The FPGA pin also needs a pull-up control or an external High-Keeper.
- The SDA output enable switches at the SCL falling edge after the address ACK (OD to PP) and stays released during the read T-bit SCL High.
- On the KV260, use an external pull-up or keeper and a tri-state IOBUF. This is a board-level suggestion, not from the spec.
