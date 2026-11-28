"""Normalized market events. See SPEC.md."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from typing import Iterable, Iterator, Optional

KINDS = ("clear", "level", "trade")


@dataclass
class Event:
    kind: str
    ts: int
    side: str = ""        # bid/ask for level, buy/sell for trade, "" for clear
    price: int = 0
    volume: int = 0
    seq: int = 0
    rx_ts: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["rx_ts"] is None:
            del d["rx_ts"]
        if self.kind == "clear":
            for k in ("side", "price", "volume"):
                d.pop(k)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(kind=d["kind"], ts=int(d["ts"]), side=d.get("side", ""),
                     price=int(d.get("price", 0)), volume=int(d.get("volume", 0)),
                     seq=int(d.get("seq", 0)), rx_ts=d.get("rx_ts"))


def btc_to_units(x) -> int:
    return int(round(float(x) * 1_000_000))


def read_jsonl(path) -> Iterator[Event]:
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Event.from_dict(json.loads(line))


def write_jsonl(path, events: Iterable[Event]) -> int:
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    n = 0
    with op(path, "wt") as f:
        for e in events:
            f.write(json.dumps(e.to_dict(), separators=(",", ":")) + "\n")
            n += 1
    return n
