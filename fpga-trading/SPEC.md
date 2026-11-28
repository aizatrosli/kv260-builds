# fpga-trading shared contract

Everything in `fpga-trading/` builds against this file. Paper trading only: nothing here places live orders.

## Units
- **price**: integer MYR (Luno XBTMYR tick is 1 MYR). uint32.
- **volume**: integer units of 1e-6 BTC ("ubtc"). uint32. `vol = round(float_btc * 1e6)`.
- **ts**: integer milliseconds since epoch (exchange timestamp when available, else local receive time).

## Normalized events (`ftrade/events.py`, JSONL, one object per line)
| kind | fields | meaning |
|---|---|---|
| `clear` | ts | empty the book (sent before a full snapshot) |
| `level` | ts, side (`bid`/`ask`), price, volume | absolute new aggregate volume at that price level; volume 0 removes the level |
| `trade` | ts, side (`buy`/`sell` = taker side), price, volume | a trade printed |

Full depth books are expressed as `clear` followed by `level` events. Every event also carries `seq` (int, monotonic per file) and optionally `rx_ts` (local receive ms).

## Wire format (`ftrade/wire.py`) - what the FPGA sees
Fixed 16-byte little-endian messages, streamed to RTL as an AXI4-Stream of 32-bit words, 4 beats/message, `tlast` on beat 3.

| bytes | field |
|---|---|
| 0 | type: `0x43` 'C' clear, `0x4C` 'L' level, `0x54` 'T' trade |
| 1 | side: 0 = bid/buy, 1 = ask/sell |
| 2-3 | reserved (0) |
| 4-7 | price uint32 |
| 8-11 | volume uint32 |
| 12-15 | ts low 32 bits |

Word 0 = bytes 0-3 (type in bits 7:0, side in bits 15:8), word 1 = price, word 2 = volume, word 3 = ts.

## Top-of-book engine semantics (`ftrade/features.py` is the golden model; RTL must match bit-exactly)
- Each side keeps at most **N = 8** levels, bids sorted descending, asks ascending.
- `clear`: both sides emptied, trade-flow register reset to 0.
- `level` with volume > 0: if price exists, overwrite volume; else insert in sorted position if it is better than the worst kept level or the side has fewer than N levels; the level pushed past N is dropped. Otherwise ignored.
- `level` with volume 0: remove the level if present (side shrinks), else ignored.
- `trade`: flow = flow + ((s*vol - flow) >>> 4) where s = +1 for buy, -1 for sell, arithmetic shift right (floor), signed 32-bit, values held in int64 then wrapped to int32. Book unchanged.
- After **every** message the engine emits one feature vector of 8 x int32:

| idx | name | definition (b1/a1 = best bid/ask price, bv_i/av_i = volume at level i, missing levels count as 0) |
|---|---|---|
| F0 | best_bid | b1 or 0 |
| F1 | best_ask | a1 or 0 |
| F2 | spread | a1 - b1 if both sides non-empty else 0 |
| F3 | bid_vol5 | sum bv_1..bv_5 (mod 2^32, reinterpreted as int32) |
| F4 | ask_vol5 | sum av_1..av_5 |
| F5 | imb_q15 | trunc_toward_zero(((B - A) << 15) / (B + A)) with B=F3, A=F4 as unsigned, 0 if B+A == 0. Range [-32768, 32768]. Computed in int64. |
| F6 | flow | trade-flow register |
| F7 | micro_off | floor(F2 * bv_1 / (bv_1 + av_1)) in unsigned 64-bit, 0 if F2 == 0 or bv_1+av_1 == 0 (microprice minus best bid) |

## Python API
- `ftrade.events`: `Event` dataclass, `read_jsonl(path)`, `write_jsonl(path, events)`.
- `ftrade.wire`: `encode(event) -> bytes`, `decode(bytes) -> Event`, `to_words(event) -> [4 x uint32]`.
- `ftrade.features`: `TopBook` with `.apply(event) -> list[int]` (8 features), `.bids`, `.asks`, `FEATURE_NAMES`.
- `ftrade.synth`: `generate(n_events, seed, start_price=250000) -> list[Event]`, a random-walk market with a weak, learnable order-flow signal.
