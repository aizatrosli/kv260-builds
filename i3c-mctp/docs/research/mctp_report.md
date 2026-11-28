# MCTP-over-I3C: facts extracted from DMTF specifications

Every fact below comes from the PDFs listed in section 0, read as text with `pdftotext -layout`. Citation format: `[doc vX.Y.Z §clause / Table|Figure N]`. Items marked **UNVERIFIED** are not stated in the downloaded documents. Items marked **(computed)** are my own derivations and do not come from any spec text.

## 0. Sources

The `https://www.dmtf.org/dsp/DSPxxxx` landing pages and `https://www.dmtf.org/standards/pmci` returned HTTP 403 (a Cloudflare challenge). The direct document URLs worked, so I probed version numbers to find the latest published release of each. Download date: 2026-11-28.

| Doc | Version | Date (cover) | Status | URL downloaded | Local file |
|---|---|---|---|---|---|
| DSP0236 MCTP Base | **1.4.0** (supersedes 1.3.3) | 2026-07-31 | Published | https://www.dmtf.org/sites/default/files/standards/documents/DSP0236_1.4.0.pdf | pdf/DSP0236_1.4.0.pdf |
| DSP0236 MCTP Base (for comparison) | 1.3.3 | 2024-02-13 | Published | https://www.dmtf.org/sites/default/files/standards/documents/DSP0236_1.3.3.pdf | pdf/DSP0236_1.3.3.pdf |
| DSP0233 MCTP I3C Binding | **1.0.1** (supersedes 1.0.0) | 2024-03-25 | Published | https://www.dmtf.org/sites/default/files/standards/documents/DSP0233_1.0.1.pdf | pdf/DSP0233_1.0.1.pdf |
| DSP0239 MCTP IDs and Codes | **1.12.0** (supersedes 1.11.1) | 2026-02-24 | Published | https://www.dmtf.org/sites/default/files/standards/documents/DSP0239_1.12.0.pdf | pdf/DSP0239_1.12.0.pdf |
| DSP0235 NVMe-MI over MCTP (checked only for the IC/CRC question) | 1.0.1 | 2018-08-03 | Published | https://www.dmtf.org/sites/default/files/standards/documents/DSP0235_1.0.1.pdf | pdf/DSP0235_1.0.1.pdf |
| SMBus Specification (cited normatively by DSP0233 for the PEC) | 2.0 | 2000-08-03 | – | http://www.smbus.org/specs/smbus20.pdf | pdf/smbus/smbus20.pdf |

Versions probed that returned 404: DSP0236 1.3.4, 1.3.5, 1.4.1, 1.4.2, 1.5.0, 2.0.x; DSP0233 1.0.2–1.2.0; DSP0239 1.12.1, 1.13.x, 1.14.x, 1.15.0. Work-in-progress drafts were not searched.

SHA-256 of the PDFs:
- DSP0236_1.4.0 = ed7bef69…775e
- DSP0236_1.3.3 = 9ddcda90…b6de
- DSP0233_1.0.1 = 3f46205d…4d94e
- DSP0239_1.12.0 = c173cdb3…267d
- DSP0235_1.0.1 = 012c6fdc…3bc3f

**Clause numbering differs between versions.** In DSP0236 1.4.0 the control protocol is §12 and the control messages are §13 (for example, Set EID is §13.4). In 1.3.3 the same material is §11 and §12 (Set EID is §12.4). Table numbers 2 and 12–19 are the same in both versions. DSP0233 1.0.1 references "DSP0236 1.3" and "DSP0239 1.8" [DSP0233 v1.0.1 §2]. It sends readers to "section 8.3" of DSP0236 for the baseline transmission unit, but in both 1.3.3 and 1.4.0 that topic is §8.4.2 (§8.3 is Special EIDs).

---

## 1. MCTP packet: transport header and message header (DSP0236 + DSP0233)

DSP0236 lets each binding position the transport-header sub-fields ("The positioning of the sub-fields … is allowed to vary based on the physical medium binding") [DSP0236 v1.4.0 §8.2 Table 1]. **DSP0233 fixes the I3C byte layout** [DSP0233 v1.0.1 §5.2.1.1 Table 1 (Controller→Target write), §5.2.2.1 Table 3 (Target→Controller read), Figures 4 and 6]:

| I3C byte | Bits | Field | Value / rule | Source |
|---|---|---|---|---|
| 0 (address phase) | [7:1] | I3C Target (dynamic) address | address of the target on the local bus | DSP0233 Tables 1, 3 |
| 0 | [0] | RnW# | 0b for a write (Controller→Target), 1b for a read (Target→Controller) | DSP0233 Tables 1, 3 |
| 1 | [7:4] | MCTP RSVD | reserved: write 0, ignore on read | DSP0236 §8.2 Table 1; §5.3 |
| 1 | [3:0] | Hdr version | **0001b**; all other values reserved | DSP0236 §8.2 Table 1; DSP0233 Tables 1, 3 |
| 2 | [7:0] | Destination EID | see section 2 for special values | DSP0236 §8.2 Table 1 |
| 3 | [7:0] | Source EID | | DSP0236 §8.2 Table 1 |
| 4 | [7] | SOM | 1b on the first packet of a message | DSP0236 §8.2 Table 1 |
| 4 | [6] | EOM | 1b on the last packet of a message | same |
| 4 | [5:4] | Pkt Seq # | increments modulo 4 | same |
| 4 | [3] | TO (Tag Owner) | 1b means the source of the message originated the tag | same |
| 4 | [2:0] | Msg tag | | same |
| 5 | [7] | IC | 0b = no message integrity check, 1b = present. Present **only in the first packet** of a message. | DSP0236 §8.2 Table 1; DSP0233 §5.2.1.1 |
| 5 | [6:0] | Message type | first packet only (see DSP0239, section 8 of this report) | same |
| 6..N-1 | | message-type header and data (plus the MIC in the last bytes if IC=1) | | DSP0236 §8.2 Table 1 |
| N | [7:0] | **PEC** (medium-specific trailer) | mandatory on every MCTP I3C transfer | DSP0233 Tables 1, 3; §5.3.1 |

DSP0236 describes the transport header as 32 bits [DSP0236 v1.4.0 §8.2 Table 1]. DSP0233 adds no medium-specific header bytes beyond the I3C address/RnW phase. Its only trailer is the 1-byte PEC.

**Byte and bit order:**
- Multi-byte numeric fields and bit fields are **big-endian**: the lowest offset holds the most significant byte [DSP0236 v1.4.0 §5.2; DSP0233 v1.0.1 §4.2; DSP0239 v1.12.0 §2.1.4].
- Bit offsets are zero-based, with LSB = bit 0. `[7:5]` is written MSB on the left [DSP0233 Annex A; DSP0236 Annex A].
- Reserved fields are written as 0 and ignored when read [DSP0236 §5.3; DSP0233 §4.1].
- On-wire bit order is MSB first for SMBus [SMBus 2.0 §4.2 / Fig 4-3 text, "Bytes are transferred with the most significant bit (MSB) first"]. **I3C SDR on-wire bit order is defined by MIPI I3C Basic and is UNVERIFIED here.**

Control messages lay out the first message-body byte the same way (IC + type); see section 4.

---

## 2. Special EIDs, transmission unit, multi-packet assembly (DSP0236 v1.4.0)

**Special EIDs** [§8.3 Table 2]:

| EID | Meaning |
|---|---|
| Destination 0x00 | Null Destination EID. Routing uses physical addressing only; the message is not bridged between buses. |
| Source 0x00 | Null Source EID. The endpoint uses physical addressing only (for example, it has no EID yet); not bridged. |
| 0x01–0x07 | Reserved for future definition |
| 0xFF | Broadcast EID, valid on the local bus only. Network-wide broadcast is not supported. Primarily for control messages. |
| All others (0x08–0xFE) | Available for assignment |

Set Endpoint ID rejects 0x00 and 0xFF as illegal values [§13.4 Table 14].

**Transmission unit** [§8.4.1–8.4.2]:
- The transmission unit is the size of the packet payload, meaning the bytes after the 4-byte transport header [§8.4.1].
- The baseline (minimum) transmission unit is **64 bytes**. Every terminus that supports control messages shall accept packets with a transmission unit of 64 bytes or less [§8.4.2].
- All packets of a message shall have the same transmission unit, which shall be at least the baseline, except the EOM packet. The EOM packet's transmission unit is less than or equal to that of the other packets [§8.4.2].
- Larger units require negotiation. That negotiation is message-type and medium specific; path discovery is described in §9.5 [§8.4.2].
- Every message type shall be deliverable with a transmission unit of 64 bytes or less [§8.4.2].
- MCTP control messages always fit in **one packet** with a payload of 64 bytes or less [§12.6].

**Sequence number, tag and assembly:**
- **Pkt Seq #:** the SOM packet may carry any value 0–3. It is recommended to use the previous EOM packet's value + 1 mod 4. After SOM, the value shall increment mod 4 on every packet through EOM [§8.2 Table 1].
- **TO / Msg tag:** both stay the same from SOM to EOM. A source may interleave messages to the same destination if each message has a unique tag. When a request has TO=1, the responder should return the same tag with TO=0. Tags are tracked independently for each TO value [§8.2 Table 1].
- **Assembly key:** (Msg Tag, TO, Source EID). When the Source EID is 0, the key is (Msg Tag, TO, source physical address). Only one assembly may be in progress per message terminus [§8.6].
- **Single-packet message:** SOM=1 and EOM=1 [§8.8]. An assembly starts on a packet with SOM=1 and EOM=0 [§8.8].
- **Packets that are dropped silently, before assembly** [§8.7]:
  - a middle or end packet with no preceding start packet;
  - a bad physical-layer integrity check (the PEC on I3C);
  - a TO=0 packet whose tag the receiver did not originate or no longer expects;
  - an un-routable EID (at a bridge);
  - an unsupported Hdr version;
  - an unsupported transmission unit.
  - The following conditions *should* cause a drop: the destination EID does not match the receiver's valid EID; the message type is unsupported.
- **Assembly terminated or the message dropped** [§8.9]:
  - normal completion on EOM;
  - a new SOM arriving for the same terminus (the old assembly is dropped and the new one starts);
  - an inter-packet timeout, whose value is set by the binding (for I3C see MT3a in section 7);
  - a sequence number that is not the previous value + 1 mod 4;
  - optionally, a middle packet whose payload size differs from the SOM packet's;
  - a bad message integrity check.

---

## 3. Message integrity check (IC=1)

- **DSP0236 does not define the algorithm.** The MIC is "a message type-specific integrity check … The particular message type definition will specify whether this is required, optional, or not to be used, the field size, and what algorithm is to be used". When present, it is carried in the **last bytes of the message body** [DSP0236 v1.4.0 §8.2 Table 1, "Msg integrity check" row].
- MCTP control messages (type 0x00) shall always have **IC = 0b** [DSP0236 v1.4.0 §12.4.2; §12.5 Table 10].
- DSP0233 and DSP0239 define no CRC-32. DSP0233's only CRC is the CRC-8 PEC (section 5). DSP0235 v1.0.1 §7.6 says the NVMe-MI message size includes "Message Integrity check fields, as required by NVMe-MI" and points to the NVMe-MI spec. That spec is not a DMTF document.
- **UNVERIFIED (not found in any downloaded document):** which CRC-32 is used, its polynomial, init value, reflection, final XOR, and the bytes it covers. You need the specific message-type spec, for example NVM Express Management Interface for type 0x04. Do not take these parameters from DSP0236.

---

## 4. MCTP control messages (type 0x00) (DSP0236 v1.4.0 §12–13)

**Format** [§12.4 Figure 23; §12.5 Table 10]. Byte numbers count from the start of the MCTP message body, which is I3C byte 5:

| Body byte | Bits | Field |
|---|---|---|
| 1 | [7] IC = 0; [6:0] Message type = 0x00 | |
| 2 | [7] **Rq**; [6] **D**; [5] rsvd; [4:0] **Instance ID** | |
| 3 | [7:0] Command Code | |
| 4 | [7:0] Completion Code (**responses only**) | |
| … | Message data | |

The request/response tables in §13 number "byte 1" as the first byte after the Command Code. In a response, byte 1 is the Completion Code [§13.1].

**Rq / D / TO usage** [§12.7 Table 11]:

| Class | Dest EID | TO | Rq | D |
|---|---|---|---|---|
| Request | target | 1 | 1 | 0 |
| Response | target | 0 | 0 | 0 |
| Broadcast Request | 0xFF | 1 | 1 | 0 |
| Datagram | target | 1 | 1 | 1 |
| Broadcast Datagram | 0xFF | 1 | 1 | 1 |

A receiver shall silently discard messages with unexpected flag values [§12.7]. A Response uses the request's source EID and source physical address as its destination [§12.3.1 Table 9]. The Command Code is echoed in the Response [§12.5].

**Command codes** [§13.2 Table 12, v1.4.0]:

| Code | Command | Clause | Notes |
|---|---|---|---|
| 0x00 | Reserved | – | |
| 0x01 | Set Endpoint ID | 13.4 | |
| 0x02 | Get Endpoint ID | 13.5 | |
| 0x03 | Get Endpoint UUID | 13.6 | |
| 0x04 | Get MCTP Version Support | 13.7 | |
| 0x05 | Get Message Type Support | 13.8 | |
| 0x06 | Get Vendor Defined Message Support | 13.9 | |
| 0x07 | Resolve Endpoint ID | 13.10 | |
| 0x08 | Allocate Endpoint IDs | 13.11 | |
| 0x09 | Routing Information Update | 13.12 | |
| 0x0A | Get Routing Table Entries | 13.13 | |
| 0x0B | Prepare for Endpoint Discovery | 13.14 | |
| 0x0C | Endpoint Discovery | 13.15 | |
| 0x0D | Discovery Notify | 13.16 | |
| 0x0E | Get Network ID | 13.17 | |
| 0x0F | Query Hop | 13.18 | |
| 0x10 | Resolve UUID | 13.19 | |
| 0x11 | Query rate limit | 13.20 | |
| 0x12 | Request TX rate limit | 13.21 | |
| 0x13 | Update rate limit | 13.22 | |
| 0x14 | Query Supported Interfaces | 13.23 | |
| 0x15 | Query Endpoint Advanced Capabilities | 13.24 | new in 1.4.0 |
| 0x1A | Request EID pool | 13.25 | new in 1.4.0 |
| 0x1B | Release EID pool | 13.26 | new in 1.4.0 |
| 0x1C | Query Supported Recovery Actions | 13.27 | new in 1.4.0 |
| 0x1D | Recovery Request | 13.28 | new in 1.4.0 |
| 0x1E | Query MCTP Device Type | 13.29 | new in 1.4.0 |
| 0xF0–0xFF | Transport Specific (defined by bindings) | 13.30 | |

DSP0236 1.3.3 Table 12 ends at 0x14 plus 0xF0–0xFF. The 1.4.0 change log lists the additions marked above.

For a simple endpoint (column "E"), these are mandatory to accept (Ma): 0x01 Set EID, 0x02 Get EID, 0x04 Get Version, 0x05 Get Msg Type Support, and 0x15 Query Endpoint Advanced Capabilities. 0x03 Get UUID is conditional and is required for hot-plug and add-in devices (note 2). 0x0B, 0x0C and 0x0D depend on the binding (note 3) [§13.2 Table 12 and key]. DSP0233 uses **Discovery Notify** and does **not** use Prepare for Endpoint Discovery or Endpoint Discovery [DSP0233 v1.0.1 §5.4.1].

**Completion codes** [§13.3 Table 13]:

| Value | Name |
|---|---|
| 0x00 | SUCCESS |
| 0x01 | ERROR |
| 0x02 | ERROR_INVALID_DATA |
| 0x03 | ERROR_INVALID_LENGTH |
| 0x04 | ERROR_NOT_READY |
| 0x05 | ERROR_UNSUPPORTED_CMD |
| 0x06 | DELAYED_RESPONSE (new in 1.4.0; wait at least 100 ms before retrieving) |
| 0x80–0xFF | COMMAND_SPECIFIC |
| others | Reserved |

On an error code, no further parameters are returned unless the command says otherwise [§13.3].

**Set Endpoint ID (0x01)** [§13.4 Table 14]:

Request:
- **Byte 1:** [7:2] reserved; [1:0] operation. 00b = Set EID, 01b = Force EID, 10b = Reset EID (optional; static EIDs only; byte 2 ignored), 11b = Set Discovered Flag (byte 2 ignored; returns ERROR_INVALID_DATA if the binding has no Discovered flag).
- **Byte 2:** EID. 0x00 and 0xFF are illegal; returning ERROR_INVALID_DATA is recommended.

Response:
- **Byte 1:** Completion code.
- **Byte 2:**
  - [7:6] reserved;
  - [5:4] EID assignment status: 00 = accepted, 01 = rejected (already assigned by another bus owner, not forced), 10 = rejected (static EID that cannot be changed), 11 = reserved;
  - [3:2] reserved;
  - [1:0] EID allocation status: 00 = no pool, 01 = requires a pool, 10 = pool already allocated, 11 = reserved.
- **Byte 3:** EID setting (the present EID).
- **Byte 4:** EID Pool Size (0x00 = no dynamic pool).

The request is typically sent to the null destination EID. The endpoint may capture the bus owner's physical address and EID from this message [§13.4 text].

**Get Endpoint ID (0x02)** [§13.5 Table 15]:

Request: no data.

Response:
- **Byte 1:** CC.
- **Byte 2:** EID (0x00 = not yet assigned).
- **Byte 3:**
  - [7:6] reserved;
  - [5:4] Endpoint Type: 00 = simple endpoint, 01 = bus owner/bridge;
  - [3:2] reserved;
  - [1:0] EID Type: 00 = dynamic, 01 = static supported, 10 = static and the present EID matches it, 11 = static and the present EID does not match it.
- **Byte 4:** Medium-specific information. It is 0x00 unless the binding specifies otherwise. **For I3C it is [7:0] reserved** [DSP0233 v1.0.1 §5.7 Table 6].

**Get MCTP Version Support (0x04)** [§13.7 Table 18, §13.7.2–13.7.4]:

Request:
- **Byte 1:** Message Type Number. 0xFF = base spec version; 0x00 = control protocol version; 0x01 = DSP0241; 0x02/0x03 = DSP0261; 0x7E/0x7F = vendor-specific.

Response:
- **Byte 1:** CC. 0x80 = message type not supported.
- **Byte 2:** Version Number Entry count (one-based).
- **Bytes 3:6:** entry 1. Further 4-byte entries follow, oldest to newest.
  - Each entry is [31:24] major, [23:16] minor, [15:8] update, [7:0] alpha.
  - The major, minor and update bytes are BCD. A high nibble of 0xF means a single digit. Update = 0xFF means "ignore". Alpha 0x00 means unused.
  - Examples: 1.1.0 → 0xF1F1F000; 1.0a → 0xF1F0FF61.

A DSP0236 1.4.0 implementation returns 5 entries for both 0xFF and 0x00: **0xF1F0FF00, 0xF1F1FF00, 0xF1F2FF00, 0xF1F3FF00, 0xF1F4F000** [§13.7.3, §13.7.4]. Entries are 32-bit values and are sent MSB first per §5.2.

**Get Message Type Support (0x05)** [§13.8 Table 19]:

Request: no data.

Response:
- **Byte 1:** CC.
- **Byte 2:** count of supported message types, **excluding** control (0x00).
- **Bytes (3:N):** one byte per message-type number.

**Other commands, briefly:**
- **Get Endpoint UUID (0x03):** the response is CC followed by 16 UUID bytes in RFC4122 byte order [§13.6 Table 16/17].
- **Get Vendor Defined Message Support (0x06):** the request is a set selector. The response is CC, the next selector (0xFF = no more sets), the Vendor ID (format byte 0x00 = PCI with 2 data bytes, 0x01 = IANA with 4 data bytes), then a 16-bit value [§13.9 Table 20/21].
- **Resolve Endpoint ID (0x07):** the request is the target EID. The response is CC, the bridge EID, then a physical address whose format the binding defines (for I3C, 1 byte; see section 7) [§13.10 Table 22].
- **Discovery Notify (0x0D):** the request has no data; the response is CC [§13.16 Table 30].
- **Vendor-defined message bodies:** 0x7E = 2-byte PCI Vendor ID (MSB first) followed by the body; 0x7F = 4-byte IANA Enterprise ID (MSB first) followed by the body [§14.1 Table 48, §14.2 Table 49].

---

## 5. Carrying MCTP on I3C (DSP0233 v1.0.1)

**Transfers:**
- **Controller → Target:** an I3C **private write** to the target's dynamically assigned address, with RnW=0. The controller ends the transfer with Sr or P, and it may send several packets back to back separated by Sr. The transfer may be preceded by the optional broadcast address 7'h7E. The T-bit is parity on writes [§5.2.1.1, Fig 4, Table 1].
- **Target → Controller:** an I3C **private read** with RnW=1, normally after an IBI (see section 6). There are three read-start forms:
  - (a) Sr plus the target address right after the IBI;
  - (b) S plus the target address;
  - (c) S plus 7'h7E/W, then Sr plus the target address/R.
  
  The target ends the data with **T-bit = 0** (End-of-Data). **One MCTP packet per read transfer** [§5.2.2.1 Fig 6, Table 3; §5.2.2.2].
- A target supports MCTP only at its own unique dynamic address, and that address carries only MCTP [§5.2.1.1, §5.4.1]. The I3C Controller has no address. Targets send only to the controller, and target-to-target traffic is bridged by the controller using the destination EID [§5.1.2, §5.2.1.1].
- A multi-packet message need not be contiguous on the bus [§5.2].

**PEC (CRC-8)** [§5.3.1]:

| Parameter | Value | Source |
|---|---|---|
| Polynomial | C(x) = x^8 + x^2 + x^1 + 1 (0x07), "same as SMBus PEC" | DSP0233 §5.3.1 |
| Init | 0x00 | DSP0233 §5.3.1 |
| Final XOR | 0x00 | DSP0233 §5.3.1 |
| Reflection | DSP0233 does not state it; it says "using the same method as for SMBus PEC". SMBus requires the CRC "be calculated in the order of the bits as received" and bytes go MSB first, which means **no reflection, MSB-first** | SMBus 2.0 §4.2 and §5.4.1.3 (my reading of those two clauses) |
| **Covered bytes** | "the I3C source/destination address field, RnW# bit, and all MCTP packet bytes of an I3C private read or write transfer". **The address byte (7-bit address << 1 \| RnW) IS included.** | DSP0233 §5.3.1 |
| Excluded | Start, Sr, Stop, the **7'h7E broadcast address**, T-bits, ACK/NACK. The PEC is not used for CCCs, IBIs or other preamble traffic. | DSP0233 §5.3.1 |
| Placement | the last byte of every MCTP packet, before Sr/P. The PEC is computed independently for each packet. | DSP0233 §5.3.1, Tables 1/3 |
| Receiver | shall verify the PEC; should discard the packet on a mismatch | DSP0233 §5.3.1 |
| Bridging | the bridge recomputes the PEC and physical address for the destination bus | DSP0233 §5.1.2 |

**Test vectors (computed by me with CRC-8 poly 0x07, init 0, no reflection, no XOR; check("123456789") = 0xF4; these are not from the spec):**
- Write to dynamic address 0x08, Get EID request. Bytes `10 01 00 08 C8 00 80 02` give **PEC = 0x53**. Without the address byte the result would be 0x64.
- Read from 0x08, Get EID response. Bytes `11 01 08 1D C0 00 00 02 00 1D 00 00` give **PEC = 0x8A**.

**MTU and SETMWL/GETMRL** [§5.4.2]:
- The baseline is 64-byte payload + 4-byte header + 1 PEC = **69-byte I3C transfer**. It is the default for both reads and writes and "cannot be negotiated smaller".
- Larger sizes are optional and must be discovered and negotiated. For Controller→Target: SETMWL (desired length), then GETMWL (the target's current maximum). For Target→Controller: SETMRL, then GETMRL. The two directions can differ. If the CCC pair is unused or unsupported, the baseline applies.
- I3C devices that implement MCTP bridging and units larger than the baseline **shall** use these CCCs to answer Query Hop.
- **The values in the CCCs include the MCTP header and the PEC, and exclude the I3C address byte.** A footnote says the I3C spec is unclear here and this is the interpretation agreed with MIPI.
- SETMRL/GETMRL also carry the IBI payload size.
- Shorter packets need no padding. Writes end with Sr/P; reads end with T=0.
- An endpoint's own transfer sizes are discovered through a message-type-specific mechanism; path sizes through Query Hop [§5.4.2, referencing DSP0236 §8.3.1 / §9.5].

**CCCs named by DSP0233:**
- GETSTATUS: presence check and polling. Targets shall always answer it, even when their buffers are full [§5.2.1.2, §5.2.2.4].
- ENEC/DISEC [Table 4].
- ENTDAA, SETNEWDA, RSTDAA, GETPID, GETBCR, GETDCR [§5.4.1, Table 4].
- SETMWL/GETMWL and SETMRL/GETMRL [§5.4.2].
- ENTAS0 [Table 4].
- Dynamic Address Assignment is **required** ("shall support"). Hot-Join *should* be used [§5.4.1].
- DSP0233 contains no single "required CCC list". Which CCCs are mandatory is set by MIPI I3C Basic: **UNVERIFIED**.

**Robustness recommendations** [§5.3.2 Table 4]:
- Re-read GET CCCs until two consecutive values match.
- Read back SETMWL/SETMRL with the matching GET.
- Issue GETPID before and after SETNEWDA.
- Issue RSTDAA twice.
- Terminate each CCC with Stop.
- Space dynamic addresses for maximum Hamming distance.

**Stuck-SDA recovery** [§5.3.3]: the controller drives 8 clocks and watches SDA, then holds SCL for 150 µs. The target must have an SCL-stall detector of 100 µs or more that releases SDA. The controller then drives SCL low for at least 35 ms (for devices with the SMBus timeout).

---

## 6. Target-initiated transfer: IBI (DSP0233 v1.0.1 §5.2.2)

- A target that has a packet ready **shall** raise an IBI with **MDB = 0xAE**, the "MCTP Pending Read ID notification" from the MIPI MDB Values Table. IBI byte 0 is the target address with RnW=1, and byte 1 is MDB 0xAE. The IBI carries no MCTP data [§5.2.2.1 Table 2; §5.2.2.8].
- **Controller flow:**
  1. ACK the IBI and read the MDB.
  2. Read the packet with a private read, either immediately via Sr or later, up to the PT timeout.
  3. The controller may queue IBIs from several targets and service them in any order [§5.2.2.2].
- **Target obligations:**
  - Make sure the MDB was read and the data is ready for the next private read [§5.2.2.2].
  - **If the IBI is NACKed**, retry at the next opportunity, per I3C Basic §5.1.6.2 and DSP0233 §5.8 [§5.2.2.2].
  - **If the IBI is ACKed but no read follows**, start a timer when the IBI is ACKed and re-send the IBI. Wait at least one Tidle between retransmits [§5.2.2.3].
- **Multiple packets** [§5.2.2.5]:
  - A target may raise one IBI per packet only if it can service back-to-back Sr-separated reads. Otherwise it must hold the next IBI until the previous packet has been read.
  - The controller may read without an IBI, for example up to EOM, or until NACK.
  - A target shall not send IBIs for packets that were already read.
- **No packet ready:** the target **NACKs** its address on a read [§5.2.2.6].
- **Early-terminated or over-long read:** the controller treats the last byte it received as the PEC and discards the packet. The target retransmits the whole packet on the next read and re-raises the IBI if IBIs are enabled [§5.2.2.7].
- **Polling mode** [§5.2.2.4]:
  - The target must support both IBI mode and polling mode [§5.2.2].
  - With IBIs disabled, the GETSTATUS pending-interrupt field must be **non-zero** when a packet is ready. The range is 1 (lowest priority) to 15 (highest), and the default **should be 7**.
  - The target shall still answer private reads.
- **Device interrupts shall be enabled by default** after power-on or reset [§5.4.1].
- **Identification:** **DCR = 0xCC** identifies an MCTP device (MIPI DCR registry) [§5.4.1].
- **After ENTDAA**, the controller sends Get MCTP Version Support. The target replies through IBI and read, then sends Discovery Notify by IBI. The bus owner answers and then sends Set EID [§5.4.1, Figure 7 (informative flow)]. The endpoint shall keep sending Discovery Notify until it has an EID, using the MN1/MT4 retry rules [§5.4.1].
- **A target acting as MCTP Bus Owner cannot hot-join** [§5.1.3].
- **BCR requirements: DSP0233 states no explicit BCR bit values.** What it does say:
  - BCR[4] is the I3C *bridge* flag, which is a different concept from an MCTP bridge [§5.1.2].
  - "an I3C Target that supports MCTP shall support IBIs" and must report an IBI payload size [§5.4.2].
  - **UNVERIFIED:** that this translates to BCR[1] = IBI Request Capable and BCR[2] = IBI Payload (MDB follows). The bit positions come from MIPI I3C Basic, which I did not download.

---

## 7. Physical address, timing, other constraints (DSP0233 v1.0.1)

**Physical address format** [§5.6 Table 5]: 1 byte, with [7:1] the I3C address and [0] = 0b. The value 7'h00 denotes the I3C Controller. It is used by Resolve Endpoint ID, Routing Information Update and Get Routing Table Entries.

**Get EID medium-specific byte** [§5.7 Table 6]: reserved (0x00).

**Media and binding IDs** [§5.5 refers to DSP0239]: see section 8.

**Data-packet timing** [§5.8 Table 7]:

| Symbol | Min | Meaning |
|---|---|---|
| PT (endpoint packet-level timeout) | 100 ms | The minimum time an endpoint keeps retrying: the target retries a NACKed or ACKed-but-unread IBI, and the controller retries a NACKed write. Use at least 8 attempts spread across PT; arbitration losses do not count. The target should pause PT while IBIs are disabled. After PT the packet may be silently discarded. |

The controller may confirm the target is present with GETSTATUS and retry indefinitely or stop after PT [§5.8]. Address NACK on a write means the target's buffer cannot hold a maximum-length packet, so retry later [§5.2.1.2]. Before declaring a device absent, the controller retries GETSTATUS per I3C Basic §5.1.9.2.3 [§5.2.1.2].

**Control-message timing** [§5.8 Table 8]:

| Symbol | Min | Max | Meaning |
|---|---|---|---|
| TRECLAIM | 5 s | – | the wait before the bus owner reclaims the EID of a non-responsive hot-plug endpoint (no ACK to repeated GETSTATUS) |
| MN1 | 2 retries | (all retries within MT4 max) | request retries, so at least 3 tries total |
| MT1 | – | 100 ms | request-to-response time at the responder, measured to the start of the IBI or the write |
| MT2 | MT1max + 2·MT3max | MT4 min | requester timeout before a retry |
| MT3 | – | 100 ms | transmission delay |
| **MT3a** | – | **100 ms** | inter-packet delay within a multi-packet message, measured at the transmitter |
| MT4 | 5 s | 6 s | Instance ID expiry |

**Other MCTP-specific I3C rules:**
- Packets travel in SDR. IBIs are SDR-only, and DSP0233 leaves HDR reads for the future [§5.2.2.8].
- The "Pending Read" notification is the MDB 0xAE IBI [Table 2].
- Prepare for Endpoint Discovery and Endpoint Discovery are not used [§5.4.1].
- DSP0233 overrides I3C Basic wherever the two differ [§5].

**Places where DSP0233 defers to MIPI or other specs:**
- the electrical layer and all I3C requirements [§5];
- private read/write transfers [§5.2.1.1];
- the optional 7'h7E broadcast and the T-bit (parity on writes, End-of-Data on reads, I3C Basic §5.1.2.3.4) [§5.2.1.1, §5.2.2.1];
- read-start forms using HCI auto-command (MIPI I3C HCI v1.0) [§5.2.2.1];
- enabling and disabling IBIs [§5.2.2];
- IBI retry (I3C Basic §5.1.6.2) [§5.2.2.2];
- Tidle [§5.2.2.3];
- GETSTATUS retry (§5.1.9.2.3) [§5.2.1.2];
- error types S1, S2, S6 and M1 [§5.3.1, Table 4];
- reserved dynamic addresses [§5.3.2];
- BCR, DCR and Hot-Join definitions [§3];
- the MDB 0xAE registry (MIPI MDB Values Table) [§2, Table 2];
- the DCR 0xCC registry (MIPI DCR Assignments) [§5.4.1];
- the meaning of the SETMWL/SETMRL length field (with the MIPI-agreed interpretation in footnote 2) [§5.4.2];
- the secondary controller, which is out of scope [§5.1.1];
- minimum retry intervals for NACKed IBIs and writes [Table 7];
- the PEC method (SMBus 2.0) [§5.3.1].

---

## 8. DSP0239 v1.12.0 identifiers

**Message types** [§7 Table 1]:

| Code | Type |
|---|---|
| 0x00 | MCTP Control (DSP0236) |
| 0x01 | PLDM (DSP0241) |
| 0x02 | NC-SI over MCTP (DSP0261) |
| 0x03 | Ethernet over MCTP (DSP0261) |
| 0x04 | NVMe-MI over MCTP (DSP0235) |
| 0x05 | SPDM over MCTP (DSP0275) |
| 0x06 | Secured Messages using SPDM over MCTP (DSP0276) |
| 0x07 | CXL FM API over MCTP (DSP0234) |
| 0x08 | CXL CCI over MCTP (DSP0281) |
| 0x09 | PCIe-MI over MCTP (DSP0291) |
| 0x7E | Vendor Defined – PCI |
| 0x7F | Vendor Defined – IANA |
| all other | Reserved |

DSP0236 v1.4.0 §8.12 Table 3 lists only 0x00, 0x7E and 0x7F and defers the rest to DSP0239.

**I3C identifiers:**

| Item | Value | Source |
|---|---|---|
| Physical Medium Identifier, "I3C Basic compatible" | **0x30** (0x31–0x3F reserved) | DSP0239 v1.12.0 §8 Table 2 |
| Physical Transport Binding Identifier, "MCTP over I3C (DSP0233)" | **0x06** | DSP0239 v1.12.0 §9 Table 3 |
| MCTP Host Interface Type Identifier, I3C (SMBIOS-related; not needed for the bus) | 0x0A | DSP0239 v1.12.0 §10 Table 4 |

---

## 9. UNVERIFIED items

1. The CRC-32 parameters for IC=1 are not defined in DSP0236, DSP0233 or DSP0239. They are message-type specific; for NVMe-MI, check the NVMe-MI spec.
2. BCR bit positions and values required for MCTP targets. DSP0233 only implies IBI support with a payload; the bit positions come from MIPI.
3. I3C SDR on-wire bit order, parity, the exact CCC codes (for example the numeric values of SETMWL and GETMRL) and their length-field formats. All of these are MIPI I3C Basic; not downloaded.
4. The set of CCCs that MIPI makes mandatory for a target. DSP0233 makes Dynamic Address Assignment and GETSTATUS responses mandatory, and SETMWL/GETMWL and SETMRL/GETMRL conditional.
5. PEC bit reflection. It is inferred from SMBus 2.0 ("order of the bits as received", MSB first); DSP0233 does not state it explicitly.
