# How trading firms use FPGAs (and what we can learn from it on a KV260)

> **Educational only. Not financial advice.** This repo is a learning project: a
> *paper-trading* bot for BTC/MYR on Luno, with a Xilinx KV260 (Zynq UltraScale+
> XCK26) as the accelerator. It never places live orders. Nothing here is a
> recommendation to trade anything, and nothing here will make money.

This document explains what high-frequency trading (HFT) and market-making firms
actually put inside FPGAs, the hardware design tricks they use, why almost none of
that speed advantage carries over to a retail crypto exchange reached over the public
internet, and how each piece of this repo maps onto the "real" version so you know
what you are actually learning. Figures quoted from vendors are vendor figures; read
them as "order of magnitude", not as guarantees.

---

## 1. What HFT firms put in FPGAs

### The tick-to-trade pipeline

"Tick-to-trade" is the time from the moment a market data packet (a "tick") arrives
on the wire to the moment the resulting order leaves on the wire. An FPGA trading
system is a hardware pipeline that does the whole loop without ever touching a CPU:

```
  exchange multicast feed (UDP)                         exchange order port (TCP)
            |                                                     ^
            v                                                     |
 +--------------------+   +----------------+   +-------------------------------+
 | PHY / MAC          |   | UDP/IP decode  |   | Order entry (OUCH, iLink/SBE, |
 | 10/25G SerDes,     |-->| + session layer|   | FIX): pre-built packet, TCP   |
 | Ethernet framing   |   | (MoldUDP64 seq,|   | seq/checksum patched in, MAC  |
 +--------------------+   |  gap detection)|   +-------------------------------+
                          +-------+--------+                  ^
                                  v                           |
                          +----------------+          +---------------+
                          | Feed handler   |          | Pre-trade     |
                          | ITCH / MDP3 SBE|          | risk gate     |
                          | fixed offsets  |          | (15c3-5: size,|
                          +-------+--------+          |  price, credit|
                                  v                   |  kill switch) |
                          +----------------+          +-------+-------+
                          | Book builder   |                  ^
                          | per-symbol L2/ |   +-----------+  |
                          | L3, top-N regs |-->| Signal /  |--+
                          +----------------+   | trigger   |
                                               | (simple!) |
                                               +-----------+
        ^                                            ^
        |  slow path (microseconds to hours): CPU sets parameters, thresholds,
        +--- risk limits, symbol lists; strategy research happens offline
```

Stage by stage:

- **PHY/MAC.** The 10/25 GbE transceivers (SerDes) and Ethernet MAC. Vendors fight
  over nanoseconds here. AMD's [Alveo UL3524](https://www.amd.com/en/products/accelerators/alveo/ul3524.html)
  (custom Virtex UltraScale+, 4x QSFP-DD, launched Sept 2023) advertises
  [<3 ns transceiver latency](https://ir.amd.com/news-events/press-releases/detail/1158/amd-unveils-purpose-built-fpga-based-accelerator-for),
  but AMD's own footnotes say that is a near-end loopback "RAW mode" figure that
  excludes protocol framing and logic ([EE Journal](https://www.eejournal.com/article/amd-develops-virtex-ultrascale-fpga-with-low-latency-ethernet-for-its-alveo-ul3524-fintech-accelerator-card)).
- **UDP/IP and session layer.** Exchanges publish market data as UDP multicast. Nasdaq
  wraps its feed in [MoldUDP64](https://m.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/moldudp64.pdf),
  a thin header with a session id and a 64-bit sequence number so you can detect
  gaps and request retransmission. Firms typically subscribe to two redundant feeds
  (A/B) and arbitrate between them in hardware, taking whichever copy arrives first.
- **Feed handler.** Decodes binary exchange protocols. Nasdaq
  [TotalView-ITCH 5.0](https://nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
  is a stream of fixed-length binary messages (add order, execute, cancel, delete,
  replace...) keyed by order reference number. CME's
  [MDP 3.0](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457418645/MDP+3.0+-+Simple+Binary+Encoding)
  uses Simple Binary Encoding (SBE), which deliberately prefers fixed positions and
  fixed-length fields so a decoder can pull a field straight from a byte offset.
  That is exactly what hardware likes: no parsing loop, just wires from byte lanes
  to registers. Order entry uses a matching binary protocol (OUCH at Nasdaq, iLink 3
  with SBE at CME).
- **Book builder.** Maintains the order book per instrument. For a trigger you often
  only need the top few price levels, kept in registers; deeper books live in on-chip
  BRAM/URAM or external memory with order-id hash tables.
- **Signal / trigger.** Usually *very* simple in the hardware fast path: "if the best
  offer on instrument X moves through price P, send order Y", a small linear model, or
  a lookup table. The cleverness is in choosing P and Y, which software does.
- **Pre-trade risk gate.** Every order must pass checks before it leaves (see 2.5).
- **Order entry.** Builds the outgoing TCP packet. Because TCP state (sequence
  numbers, checksums) is known in advance, the packet is mostly pre-built.

### Published latency figures

| Figure | What it measures | Source |
|---|---|---|
| 13.9 ns min (507-byte frames), 14.1 ns (68-byte) | STAC-T0 "actionable latency": last bit of the inbound data needed for a decision to first bit of the outbound order. Exegy nxFramework on AMD UL3524, June 2024 | [STAC report](https://stacresearch.com/node/48577), [AMD blog](https://www.amd.com/en/blogs/2024/amd-sets-stac-benchmark-world-record-for-fastest-e.html) |
| 24.2 ns min | Previous STAC-T0 record, Xilinx/LDA, 2020 | [STAC report](https://stacresearch.com/node/45733) |
| 98 ns max (68-byte, 1 Gbps) | Solarflare/LDA FPGA, 2017 | [STAC report](https://stacresearch.com/node/43535) |
| 389 ns / 624 ns | Exegy nxAccess hardware tick-to-trade with / without its pattern matcher, measured SOP-to-SOP on the switch for CME; software path about 1.5 us | [Exegy nxAccess](https://www.exegy.com/products/nxaccess/) |
| ~1.0 us vs ~5.1 us | AMD's Onload vs kernel TCP ping-pong (4-byte payload), CPU kernel-bypass for comparison | [AMD whitepaper](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/white-papers/amd-low-latency-solutions-amd-epyc-9005-amd-alveo-x3.pdf) |

Two lessons from this table. First, STAC-T0 is a *network I/O* benchmark: the
13.9 ns number is a trigger with almost no logic behind it, essentially "packet in,
pre-built packet out". Real strategies with a full book and risk checks run in
hundreds of nanoseconds, like the nxAccess numbers. Second, all of these are
vendor-commissioned measurements; treat them as best case.

### Colocation, cut-through and determinism

None of these numbers matter unless the box is physically next to the matching
engine. Firms rent rack space in the exchange's data centre (**colocation**), and
exchanges equalise cable lengths so every colocated customer is the same distance
away. Light in fibre covers about 20 cm per nanosecond, so a few metres of extra cable
costs more than the entire FPGA pipeline.

**Cut-through processing** means not waiting for a whole packet. A store-and-forward
NIC receives the full frame, checks the CRC, then hands it up. An FPGA starts decoding
as soon as the bytes carrying the price arrive, and can start transmitting the order
before the inbound frame has even finished (and before its CRC is checked, which is
one reason risk checks must be conservative).

**Determinism** is the underrated reason firms use FPGAs. A CPU path has a nice
median, but the tail (99.9th percentile) is ruined by cache misses, interrupts, page
faults, and the OS scheduler. A pipelined FPGA design takes the *same number of
clock cycles every time*. In a race where you are either first or not filled, a
consistent 500 ns beats an average of 300 ns that sometimes takes 20 us, and
deterministic latency also makes risk and queue-position modelling simpler.

### What stays in software

Most of a trading firm is not in the FPGA:

- **Strategy research**: data analysis, backtesting, model fitting. Hours to months.
- **Parameter setting**: the CPU computes fair values, thresholds and order templates
  and writes them into FPGA registers continuously (microseconds to seconds).
- **Slow alpha**: anything that benefits from big models, lots of data or cross-asset
  context. If the edge lasts seconds, a nanosecond head start does not matter.
- **Position management, hedging, reconciliation, compliance reporting, monitoring.**

The FPGA is the reflex; the software is the brain that tells it which reflexes to have.

---

## 2. Hardware design techniques

### 2.1 Pipelining

Split the work into stages separated by registers so each stage fits in one clock
cycle, and a new message can enter every cycle (initiation interval of 1). Latency is
"number of stages x clock period"; throughput is "one message per clock". At 250 MHz
(4 ns/cycle), a 20-stage pipeline is 80 ns. Designers spend their time removing stages,
for example by computing a decode for every possible message type in parallel and
selecting the right one at the end.

### 2.2 Fixed-point instead of floating point

Prices on an exchange are already integers in disguise: ITCH prices are integers with
four implied decimal places. Hardware works in integers or fixed-point (an integer
with an implied binary point, like our `imb_q15`, which is the imbalance times 2^15).
Integer adders and comparators are small and finish in one cycle; floating-point units
are large, multi-cycle and hard to make bit-exact against a software model. Divides
are avoided or replaced by multiplying by a precomputed reciprocal, because a full
divider is either slow (one bit per cycle) or huge (fully pipelined).

### 2.3 Parallel compare and shift registers for the book

A software book keeps a sorted array and searches it. Hardware keeps the top N levels
in N registers per side and compares the incoming price against *all N at once*. Each
comparator produces one bit ("is the new price better than level i?", "is it equal to
level i?"). From those bit vectors, every register decides in the same cycle whether
to keep its value, take the new one, take its neighbour's value (shift down to make
room on insert), or take the other neighbour's value (shift up on delete). Insert,
update and delete all take one or two cycles regardless of position. This is exactly
the structure of our N=8 top-of-book engine.

### 2.4 Speculative, pre-armed orders

The order packet is built *before* the trigger fires. Software writes a template
(instrument, side, size, price) into FPGA memory; the hardware only fills in a few
fields (sequence number, maybe the price) and recomputes checksums incrementally. Some
designs go further and start transmitting the Ethernet preamble and headers while the
decision is still being computed, then either finish the order or deliberately corrupt
the frame (so the switch drops it) if the trigger does not fire. This is part of how
sub-20 ns STAC-T0 numbers are possible.

### 2.5 Hardware risk gates

In the US, [SEC Rule 15c3-5](https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access)
(the Market Access Rule, 2010) requires broker-dealers with market access to have
controls "reasonably designed to prevent the entry of orders that exceed appropriate
pre-set credit or capital thresholds, or that appear to be erroneous", applied
pre-trade, to every order including market-maker quotes, with annual CEO
certification ([Federal Register](https://www.federalregister.gov/documents/2010/11/15/2010-28303/risk-management-controls-for-brokers-or-dealers-with-market-access)).
FINRA's exam findings keep citing firms whose thresholds are unreasonable or
undocumented ([FINRA 2026 report](https://finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/market-access-rule)).
Because the check sits on the critical path, it is implemented in hardware next to the
order generator: max order size, max notional, price collars around a reference
price, position and credit limits, message-rate throttles, duplicate-order detection,
and a kill switch that blocks everything. The 2012 Knight Capital incident (about
USD 440 million lost in 45 minutes from a faulty deployment) is the standard
illustration of why these gates exist.

### 2.6 FPGA vs ASIC vs CPU with kernel bypass

| Approach | Latency | Flexibility | Cost / effort | Typical use |
|---|---|---|---|---|
| CPU, kernel TCP/UDP | ~5 us+ per hop, poor tail | Total | Lowest | Everything slow |
| CPU + kernel bypass ([Onload](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/white-papers/amd-low-latency-solutions-amd-epyc-9005-amd-alveo-x3.pdf), ef_vi, TCPDirect, DPDK) | ~1 us-scale, busy-polling cores | Total | Moderate | Most "low latency" C++ strategies |
| FPGA (Alveo X3/UL3524, vendor frameworks such as [Exegy nxFramework](https://www.exegy.com/products/nxframework/), ex-Enyx) | tens to hundreds of ns, deterministic | Reprogram in hours; logic is limited | High (HDL engineers, long build/verify cycles) | Feed handlers, triggers, risk gates |
| ASIC | lowest, fixed | None after tape-out | Very high, many months | Only the very largest firms, stable logic |

Kernel bypass means the NIC's queues are mapped into user space, so the application
polls packets without a system call or interrupt. AMD's
[Alveo X3522](https://www.amd.com/en/products/accelerators/alveo/x3/a-x3522-p08g-pq-g.html)
is a hybrid: a kernel-bypass NIC (Onload/TCPDirect/ef_vi) with optional FPGA logic for
offload. Many firms run exactly this mix: CPU strategies on bypass NICs, plus FPGAs
for the handful of reactions that are pure races.

---

## 3. Why none of this translates to Luno

Luno's interface, from its [API documentation](https://www.luno.com/en/developers/api):

- **REST over HTTPS** at `https://api.luno.com`, JSON bodies, prices and volumes as
  decimal *strings*. Market data endpoints such as `/api/1/orderbook_top` (top 100
  bids and asks) and `/api/1/ticker` work without authentication (we checked:
  `orderbook_top?pair=XBTMYR` returns HTTP 200 with no key). Private calls use HTTP
  basic auth with an API key id and secret.
- **WebSocket streaming** at `wss://ws.luno.com/api/1/stream/:pair`. The docs say
  "The client must start by sending API key credentials" as the first message
  (`{"api_key_id": ..., "api_key_secret": ...}`), so **the market stream does require
  an API key**, even though it only carries public data. Use a key with the minimum
  permissions and never one with `Perm_W_Orders` or withdrawal rights for this
  project. The server then sends a full order book snapshot with a `sequence`
  number, followed by update messages each carrying the next sequence number and
  zero or more trade updates, at most one create, one delete and one status update.
  It is an **order-level (L3)** feed: creates and deletes carry `order_id`, trades
  carry `maker_order_id`, and orders at the same price are not aggregated. If an
  update arrives out of order, or a keep-alive (an empty message) is missed, the
  client must reconnect and rebuild from a fresh snapshot.
- **Rate limits**: "APIs are rate limited to 300 calls per minute" (HTTP 429 beyond
  that) and "The streaming API is limited to 50 sessions open simultaneously".
- **No colocation** offering that we could find, and no binary protocol. You reach
  Luno over the public internet, through TLS, through whatever cloud and CDN front
  end it uses, from wherever you happen to be.

So the latency budget looks like this:

| Stage | Order of magnitude |
|---|---|
| Our FPGA book + features (Section 5) | ~0.1 us |
| Python feature computation per event | ~1-10 us |
| JSON parse + TLS decrypt of one WebSocket message in Python | ~10-100 us |
| Exchange internal processing + publication | unknown, likely ms |
| Network round trip to `api.luno.com` from Malaysia | tens of ms (measure it, Section 5) |
| REST order placement under a 300/min budget | one call every 200 ms on average |

The network is four to six orders of magnitude bigger than the FPGA stage. Shaving
100 ns off a 30 ms path changes the total by 0.0003%. And jitter on an internet path
(several ms) swamps everything. **A latency race on Luno is unwinnable and
pointless**; anyone who is "faster" there is faster because of where their server is,
not what chip it has.

What an FPGA *can* legitimately do in this setting:

1. **Run a model cheaply and deterministically.** A tiny INT8 predictor on the
   KV260's TPU uses a few watts and gives the same answer in the same number of
   cycles every time. That is useful for a small always-on box, not for speed.
2. **Be bit-exact against a golden model.** Integer features defined in `SPEC.md`
   mean Python, RTL simulation and (later) the board must agree to the last bit. That
   discipline is the real professional skill.
3. **Be a learning vehicle.** You get to build the same *shape* of system as an HFT
   shop: binary wire format, streaming interface, register-based book, fixed-point
   features, risk gate, all on a USD 250-ish board in simulation first.

---

## 4. Mapping: HFT component -> this repo -> what you learn

| HFT component | What we build in this repo | Where | What you learn |
|---|---|---|---|
| Market data feed (ITCH/MDP3 over UDP multicast) | Luno collector: REST snapshots + WebSocket stream, normalised to `clear`/`level`/`trade` JSONL events. Turns Luno's L3 order-level stream into aggregate price levels | `fpga-trading/ftrade/marketdata` | Sequence numbers, gap detection and resync, snapshot+delta books, why exchanges publish binary |
| Feed handler (binary decode at fixed offsets) | `ftrade.wire`: fixed 16-byte little-endian messages, streamed as 4 x 32-bit AXI4-Stream beats with `tlast` on beat 3 | `fpga-trading/ftrade/wire.py`, `SPEC.md` | Designing a wire format for hardware, AXI-Stream handshakes (`tvalid`/`tready`), endianness |
| Book builder | N=8 top-of-book engine per side, parallel compare + shift registers, golden model `ftrade.features.TopBook` | `fpga-trading/rtl` + `fpga-trading/tb` (Verilator/iverilog) | Parallel comparators, insert/delete shifts, bit-exact verification against Python |
| Signal | 8 int32 features (best bid/ask, spread, 5-level volumes, Q15 imbalance, trade-flow EMA via shift, microprice offset) feeding a tiny predictor quantised with `tpuquant` and compiled with `tpucc` for the TPU | `fpga-trading/rtl` (features), `fpga-trading/ftrade/predictor` | Fixed-point arithmetic, EMA by shift instead of multiply, integer division cost, INT8 quantisation, hardware/software split |
| Pre-trade risk gate | Paper-engine limits: max position, max order size, price collar around mid, daily loss stop, order-rate throttle, kill switch | `fpga-trading/ftrade/paper` | What 15c3-5-style checks are and why they come *before* the order, not after |
| Order entry | Simulated fills against the recorded book (queue position ignored or approximated, fees included), plus a backtester | `fpga-trading/ftrade/paper` | Why backtests lie (fill assumptions, latency, fees), how to measure a strategy honestly |
| Colocation, cut-through, 10G MAC | Not built | n/a | Understood, not simulated: our input is a file or a WebSocket |

A note on the feature choices: `flow = flow + ((s*vol - flow) >>> 4)` is an
exponential moving average with alpha = 1/16, implemented with a subtract, an
arithmetic shift and an add, no multiplier. `imb_q15` needs a real division, which is
the most expensive operation in the engine and a good place to compare a sequential
divider against a pipelined one. Those are the same trade-offs HFT designers make.

---

## 5. Measuring latency in our setup

There are three very different clocks to measure. Be explicit about which one each
number comes from.

**RTL cycles (simulation).** Count clock cycles in the testbench from the first beat
of a message (`tvalid && tready` on beat 0) to the cycle the feature vector is
valid. Then convert at an assumed clock. The XCK26 programmable logic comfortably
runs simple designs at 200-300 MHz; assume 200 MHz (5 ns/cycle) until a real
place-and-route timing report says otherwise. A plausible budget for our engine:

| Step | Cycles (estimate) | At 200 MHz |
|---|---|---|
| Receive 4 beats (32-bit stream) | 4 | 20 ns |
| Parallel compare + shift (book update) | 1-2 | 5-10 ns |
| Sums, spread, flow | 1-2 | 5-10 ns |
| Imbalance + microprice divides | 1-40 (pipelined vs bit-serial) | 5-200 ns |
| **Total** | **~8-50** | **~40-250 ns** |

These are design targets; replace them with the counts the testbench actually
prints. Widening the stream to 128 bits (one beat per message) would save 3 cycles,
which is the kind of decision that matters in HFT and does not matter here.

**Python.** Time `TopBook.apply()` over a recorded or `ftrade.synth` event file with
`time.perf_counter_ns()` and report median and 99th percentile per event. Expect
microseconds. Also time the JSON decode of real Luno messages; that usually costs
more than the book logic.

**Network.** Measure what actually dominates:

```bash
# TCP connect, TLS handshake and total time for one public REST call
curl -s -o /dev/null -w 'connect=%{time_connect}s tls=%{time_appconnect}s total=%{time_total}s\n' \
  "https://api.luno.com/api/1/ticker?pair=XBTMYR"
```

Run it many times and look at the spread, not just the minimum. For the stream, log
both the exchange `timestamp` in each update and your local receive time (`rx_ts` in
our events), and plot the difference: it includes clock offset between your PC and
Luno, so trust the *variation* more than the absolute value. If you are behind a VPN
or proxy, the number reflects that path, not your raw connection.

**Honest takeaways.**

1. The FPGA engine will be roughly 10-100x faster than Python per event and roughly
   100,000x faster than the network. Neither speedup is a trading edge on Luno.
2. The valuable result is *bit-exactness and determinism*: identical features from
   Python, Verilator and the board, with a fixed cycle count per message.
3. For a strategy on a venue like this, what matters is the signal itself, fees and
   the spread you pay, and honest fill modelling. Expect most simple signals to lose
   money after fees in paper trading; that is a correct and useful result.
4. Any claim of "FPGA speed advantage" for a retail internet-connected bot should be
   treated with suspicion, including claims you are tempted to make yourself.

---

## 6. Further reading

- STAC-T0 reports: [Exegy/AMD 2024](https://stacresearch.com/node/48577) and
  [LDA/Xilinx 2020](https://stacresearch.com/node/45733); read the methodology, not just the headline.
- [AMD Alveo UL3524 product page](https://www.amd.com/en/products/accelerators/alveo/ul3524.html)
  and the [EE Journal analysis](https://www.eejournal.com/article/amd-develops-virtex-ultrascale-fpga-with-low-latency-ethernet-for-its-alveo-ul3524-fintech-accelerator-card) of its latency claims.
- [Nasdaq TotalView-ITCH 5.0 spec](https://nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf)
  and [MoldUDP64 spec](https://m.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/moldudp64.pdf):
  short, readable, and the best example of a hardware-friendly protocol.
- [CME MDP 3.0 / SBE documentation](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457418645/MDP+3.0+-+Simple+Binary+Encoding).
- [Columbia CSEE W4840 MDP tickerplant project](https://www.cs.columbia.edu/~sedwards/classes/2014/4840/reports/MDP-tickerplant.pdf):
  a student FPGA book builder, close in spirit to this repo.
- [SEC Rule 15c3-5 final rule](https://www.sec.gov/files/rules/final/2010/34-63241.pdf)
  and [FINRA's market access findings](https://finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/market-access-rule).
- [AMD low-latency whitepaper (EPYC + Alveo X3, Onload)](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/white-papers/amd-low-latency-solutions-amd-epyc-9005-amd-alveo-x3.pdf).
- [Luno API documentation](https://www.luno.com/en/developers/api), especially the
  Streaming API and Rate Limiting sections.
- Michael Lewis, *Flash Boys* (2014), for the colocation and microwave-link history;
  read it as journalism with a point of view.
- Larry Harris, *Trading and Exchanges* (2003), for market microstructure: why spreads,
  queues and adverse selection matter more than speed for almost everyone.
