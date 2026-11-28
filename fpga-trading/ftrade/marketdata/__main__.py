"""CLI: python -m ftrade.marketdata {record,latency,depth,info}

  record   --pair XBTMYR --seconds 600 --out data/xbtmyr_%Y%m%d.jsonl.gz
           [--mode rest|stream] [--interval 1.0] [--depth 0] [--book top|full]
  latency  --n 20 [--out data/luno_rtt.json]   HTTPS RTT to api.luno.com
  depth    IN OUT --depth 8                    rewrite a recording as exact top-N
  info     IN                                  event counts / time span of a recording

Run from fpga-trading/ so that `ftrade` is importable.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

from ..events import read_jsonl, write_jsonl
from .transport import HttpError, KeepAliveTransport, UrllibTransport, _proxy_for
from .rest import RequestLog, RestCollector, rtt_stats


def _default_out(pair: str) -> str:
    return os.path.join("data", f"{pair.lower()}_{datetime.now():%Y%m%d}.jsonl.gz")


def _meta_path(out: str) -> str:
    for ext in (".jsonl.gz", ".jsonl"):
        if out.endswith(ext):
            return out[: -len(ext)] + ".meta.json"
    return out + ".meta.json"


def cmd_record(a) -> int:
    out = datetime.now().strftime(a.out) if a.out else _default_out(a.pair)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log = RequestLog()
    t0 = time.time()
    if a.mode == "rest":
        col = RestCollector(a.pair, interval=a.interval, trades_every=a.trades_every,
                            resync_every=a.resync_every, depth=a.depth, book=a.book,
                            http=a.http, log=log, verbose=not a.quiet)
        gen = col.events(seconds=a.seconds)
    else:
        from .stream import StreamCollector, available
        ok, why = available()
        if not ok:
            print(f"stream mode unavailable: {why}. Use --mode rest.", file=sys.stderr)
            return 2
        col = StreamCollector(a.pair, verbose=not a.quiet)
        gen = col.events(seconds=a.seconds)
    if not a.quiet:
        print(f"recording {a.pair} ({a.mode}) for {a.seconds}s -> {out}", file=sys.stderr)
    first_last = [None, None]

    def track(g):
        for e in g:
            if first_last[0] is None:
                first_last[0] = e.ts
            first_last[1] = e.ts
            yield e

    try:
        n = write_jsonl(out, track(gen))
    except HttpError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 1
    meta = {
        "pair": a.pair, "mode": a.mode, "out": out, "events": n, "counts": col.counts,
        "wall_start": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 1),
        "first_ts": first_last[0], "last_ts": first_last[1],
        "units": {"price": "MYR (int)", "volume": "1e-6 BTC (int)", "ts": "ms epoch"},
    }
    if a.mode == "rest":
        meta.update({"interval_s": a.interval, "trades_every": a.trades_every,
                     "resync_every": a.resync_every, "depth": a.depth, "book": a.book,
                     "http": a.http, "polls": col.polls, "request_errors": col.errors,
                     "trade_duplicates_dropped": col.trades.duplicates if col.trades else 0,
                     "rtt_summary": log.summary(), "requests": log.rows})
    else:
        meta["reconnects"] = col.reconnects
    mp = _meta_path(out)
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("out", "events", "counts", "first_ts", "last_ts")}))
    print(f"meta: {mp}", file=sys.stderr)
    return 0


def _measure(transport, path, params, n, pause):
    rtts, errors = [], 0
    for _ in range(n):
        try:
            rtts.append(transport.get(path, params).rtt_ms)
        except HttpError as e:
            errors += 1
            print(f"  {e}", file=sys.stderr)
        time.sleep(pause)
    transport.close()
    return {**rtt_stats(rtts), "errors": errors, "samples_ms": [round(x, 2) for x in rtts]}


def cmd_latency(a) -> int:
    path, params = "/api/1/ticker", {"pair": a.pair}
    pause = max(a.pause, 0.25)          # stay far below 300 calls/min
    ka = KeepAliveTransport()
    try:
        ka.connect()                     # exclude the one-off TCP/TLS/tunnel setup
        ka.get(path, params)
    except Exception as e:
        print(f"warm-up failed: {e}", file=sys.stderr)
    res = {
        "endpoint": "https://api.luno.com" + path + f"?pair={a.pair}",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "via_proxy": _proxy_for("api.luno.com") is not None,
        "note": "rtt = request sent -> full response body read, wall clock. 'warm' reuses one "
                "keep-alive HTTPS connection (what the REST collector does); 'cold' opens a new "
                "TCP+TLS connection per request (urllib). p99 is nearest-rank (== max for n<100).",
        "n": a.n,
        "warm_keepalive": _measure(ka, path, params, a.n, pause),
        "cold_new_connection": _measure(UrllibTransport(), path, params, a.n, pause),
    }
    txt = json.dumps(res, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    for k in ("warm_keepalive", "cold_new_connection"):
        s = res[k]
        if s.get("n"):
            print(f"{k:20s} n={s['n']} min={s['min_ms']:.1f} median={s['median_ms']:.1f} "
                  f"p99={s['p99_ms']:.1f} ms  errors={s['errors']}")
        else:
            print(f"{k:20s} no successful samples (errors={s['errors']})")
    if a.out:
        print(f"wrote {a.out}", file=sys.stderr)
    return 0


def summarize(path) -> dict:
    counts = {"clear": 0, "level": 0, "trade": 0}
    ts0 = ts1 = None
    last_seq, mono = 0, True
    for e in read_jsonl(path):
        counts[e.kind] += 1
        ts0 = e.ts if ts0 is None else min(ts0, e.ts)
        ts1 = e.ts if ts1 is None else max(ts1, e.ts)
        mono &= e.seq > last_seq
        last_seq = e.seq
    iso = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).isoformat() if t else None
    return {"path": str(path), "events": sum(counts.values()), "counts": counts,
            "first_ts": ts0, "last_ts": ts1, "first_utc": iso(ts0), "last_utc": iso(ts1),
            "span_s": round((ts1 - ts0) / 1000, 1) if ts0 else 0, "seq_monotonic": mono}


def cmd_info(a) -> int:
    print(json.dumps(summarize(a.path), indent=1))
    return 0


def cmd_depth(a) -> int:
    from .book import limit_depth
    n = write_jsonl(a.out, limit_depth(read_jsonl(a.inp), a.depth))
    print(f"wrote {n} events to {a.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m ftrade.marketdata",
                                description="Luno public market-data recorder (read-only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record normalized events to JSONL(.gz)")
    r.add_argument("--pair", default="XBTMYR")
    r.add_argument("--seconds", type=float, default=600)
    r.add_argument("--out", default=None, help="output path; strftime codes allowed "
                   "(default data/<pair>_YYYYmmdd.jsonl.gz)")
    r.add_argument("--mode", choices=("rest", "stream"), default="rest")
    r.add_argument("--interval", type=float, default=1.0, help="book poll interval, s (rest)")
    r.add_argument("--trades-every", type=int, default=1, help="poll /trades every k polls")
    r.add_argument("--resync-every", type=int, default=300, help="clear+snapshot every K polls")
    r.add_argument("--depth", type=int, default=0, help="levels per side to keep (0 = all)")
    r.add_argument("--book", choices=("top", "full"), default="top")
    r.add_argument("--http", choices=("keepalive", "urllib"), default="keepalive")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_record)
    l = sub.add_parser("latency", help="measure HTTPS RTT to api.luno.com")
    l.add_argument("--n", type=int, default=20)
    l.add_argument("--pair", default="XBTMYR")
    l.add_argument("--pause", type=float, default=0.3)
    l.add_argument("--out", default=None, help="write JSON here (e.g. data/luno_rtt.json)")
    l.set_defaults(fn=cmd_latency)
    d = sub.add_parser("depth", help="rewrite a recording as an exact top-N book stream")
    d.add_argument("inp")
    d.add_argument("out")
    d.add_argument("--depth", type=int, default=8)
    d.set_defaults(fn=cmd_depth)
    i = sub.add_parser("info", help="summarize a recording")
    i.add_argument("path")
    i.set_defaults(fn=cmd_info)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
