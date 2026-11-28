# ftrade.marketdata: Luno XBTMYR market-data recorder

This package collects **public, read-only** Luno market data and turns it into the normalized
`Event` stream from `SPEC.md`. Events are `clear`, `level` and `trade`. Prices are integer MYR
and volumes are integer 1e-6 BTC. It writes JSONL(.gz) through `ftrade.events`, so the output
feeds `ftrade.wire` (the FPGA's 16-byte messages) and `ftrade.features` (the golden model)
directly. No code here places, cancels or reads orders.

Only the standard library is needed. `websockets` is optional and only used by the stream mode.

## Run

From `fpga-trading/` (so that `ftrade` is importable):

```sh
python -m ftrade.marketdata record --pair XBTMYR --seconds 600 --out data/xbtmyr_%Y%m%d.jsonl.gz
python -m ftrade.marketdata record --mode rest --interval 1.0 --depth 8 --seconds 3600
python -m ftrade.marketdata latency --n 20 --out data/luno_rtt.json
python -m ftrade.marketdata info data/xbtmyr_sample.jsonl.gz
python -m ftrade.marketdata depth data/xbtmyr_sample.jsonl.gz data/xbtmyr_sample_top8.jsonl.gz --depth 8
```

* `record` writes the events file and a sidecar `<name>.meta.json`. The sidecar has the
  settings, the event counts, and one row per HTTP request (endpoint, status, RTT in ms, bytes),
  plus an RTT summary per endpoint. Ctrl-C stops cleanly and leaves a valid `.gz` file.
* Useful `record` flags:
  * `--book top|full`: `orderbook_top` returns the top 100 price levels, already aggregated.
    `orderbook` returns the whole book, one entry per order, and is aggregated here. It is
    about 70 kB per call.
  * `--resync-every K`: emit `clear` plus a full snapshot every K polls. A resync also happens
    automatically after any request error.
  * `--trades-every k`, `--http keepalive|urllib`.
* `--depth 8`, or the `depth` subcommand afterwards, keeps only the best 8 levels per side. Use
  it for the N=8 `TopBook`/RTL. With a deeper book, a level ranked 9th or lower that moves into
  the top 8 is never re-sent, because it did not change. The 8-level engine would then show
  fewer than 8 levels until the next resync. The `depth` subcommand replays the full book, so
  the 8-level engine always sees the exact top 8.

Python API (for the paper-trading loop and other consumers):

```python
from ftrade.marketdata import iter_events, RestCollector, limit_depth
for ev in iter_events("XBTMYR", mode="rest", interval=1.0, depth=8):   # runs until you break
    ...
```

### Windows / Proxmox Linux VM

* **Windows:** install Python 3.11 or later from python.org, then run
  `py -m ftrade.marketdata record ...` from `fpga-trading\`. Nothing else is needed. If you are
  behind a corporate proxy, set `HTTPS_PROXY`; both HTTP transports honour it.
* **Proxmox Linux VM:** run the same command. To record over a long period, start it from a
  systemd timer or from `tmux`, with `--seconds 3600 --out data/xbtmyr_%Y%m%d_%H%M.jsonl.gz`.
  That gives one file per hour; a full day at 1 Hz with depth 100 is roughly 5 to 20 MB gzipped.
* **Clock:** keep the VM clock synced (chrony or NTP). `rx_ts` is local time and is compared
  with Luno's `timestamp`.

### KV260 ARM cores (later)

The Kria Ubuntu image ships Python 3.10 or later; everything here is pure standard library.
Run the recorder on the PS (Cortex-A53) side and feed the events to the PL through
`ftrade.wire.encode` (DMA or AXI-Stream FIFO). At 1 Hz polling the CPU load is negligible.
Network RTT and Luno's caching set the latency floor, not the board (see below).

## REST polling vs the websocket stream

| | REST (`--mode rest`, default) | Stream (`--mode stream`) |
|---|---|---|
| Auth | none (public endpoints) | an API key is required as the first message (use a read-only key) |
| Book | top 100 aggregated levels per side, polled at most about 1 Hz | full order-level book, every create/delete/trade |
| Freshness | Luno caches market data for up to 1 s, plus one RTT | push, per update (one-way latency) |
| Intra-second | invisible: several changes within one poll collapse into one diff, and short-lived orders are missed | every change, in sequence |
| Trades | `/trades?since=` every poll, deduplicated by sequence, exchange timestamp | in-band `trade_updates`, applied to the maker order |
| Rate limits | 300 calls/min per client (HTTP 429 above that). 1 Hz book plus 1 Hz trades is 120/min | 50 concurrent sessions |
| Integrity | each poll is a full snapshot, so it self-heals | sequence numbers; a gap means reconnect and rebuild (handled automatically) |

What the REST collector does:

* It polls `orderbook_top` (or `orderbook`), aggregates levels per integer price, and diffs
  against the previous snapshot. A diff emits removals first, as volume 0, then inserts and
  changes, best price first.
* It refuses settings that would exceed about 90% of the rate budget, and intervals below 0.5 s.
* On 429, 5xx or network errors it backs off exponentially with jitter, honouring
  `Retry-After`. After an error the next snapshot is a full `clear` resync. Other 4xx errors
  (for example a bad pair) are fatal.
* Trades are fetched with `since` set to the newest trade timestamp seen, paging while a page
  is full (100 trades). Duplicates are dropped by Luno trade `sequence`. The taker side is
  `buy` if `is_buy` is true, otherwise `sell`.
* Event `ts` is Luno's `timestamp`: the snapshot time for levels, the execution time for
  trades. `rx_ts` is the local time when the response arrived. Within a file, `seq` increases
  by one per event. `ts` is *not* guaranteed to be monotonic, because a trade printed before
  the snapshot is fetched after it.
* Volumes are rounded half-even to 1e-6 BTC. A positive volume below 0.5e-6 BTC is clamped to
  1, so that a level that exists is never sent as 0 ("removed").

Stream mode (`stream.py`):

* It needs `pip install websockets` plus the `LUNO_API_KEY_ID` and `LUNO_API_KEY_SECRET`
  environment variables. Create a key with view-only permissions. The secret is only sent to
  `wss://ws.luno.com`, and it is never logged or written to files.
* It keeps the order-level book (`create_update`, `delete_update`, and `trade_updates`, which
  reduce the maker order). It aggregates to price levels and emits only levels whose ubtc
  value changed. The taker side is the opposite of the maker order's side. The trade price is
  `counter / base` rounded to MYR.
* It checks sequence numbers. On a gap or error it reconnects with backoff and starts again
  from `clear` plus a snapshot. It sends a keep-alive every 20 s.
* Without the package or the key, the CLI says so and exits with code 2. Importing the module
  is always safe.
* Observed 2026-11-28: the server closes the socket with
  `{"error_code":"auth_timeout","error":"you took too long to provide credentials"}` if no
  credentials message is sent. Send real, read-only credentials; do not rely on any lenience
  in the server's key validation.

## Measured latency (this repo's sample)

See `data/luno_rtt.json`, from `python -m ftrade.marketdata latency --n 20`. These numbers were
measured from a cloud container through an HTTPS proxy, not from Malaysia. Expect different
figures from a home or VM connection in MY: rerun the command there and commit the result.
For a trading loop, the end-to-end "signal age" is roughly Luno's cache (up to 1 s) plus half
the poll interval plus the RTT. That is why the REST path suits paper trading and model
training, not latency-sensitive strategies.

## Tests

`pytest tests/test_marketdata.py tests/test_marketdata_stream.py`. They run fully offline:
urllib is mocked, the transport is faked, and the websocket messages are canned.
