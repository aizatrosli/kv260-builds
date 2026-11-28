"""Luno public market-data collection -> normalized ftrade.events.Event streams.

Entry points:
  iter_events(pair, mode="rest"|"stream", seconds=None, **kw) -> Iterator[Event]
  RestCollector(pair, interval=1.0, ...).events(seconds)   (no API key needed)
  StreamCollector(pair).events(seconds)  (needs `websockets` + LUNO_API_KEY_ID/SECRET)
  limit_depth(events, depth=8)           re-express a deep book as an exact top-N one

Read-only: no order placement anywhere in this package.
"""
from __future__ import annotations
from typing import Iterator, Optional

from ..events import Event
from .book import BookDiffer, aggregate, limit_depth, price_to_int, vol_to_units
from .rest import RestCollector, TradeTracker, poll


def iter_events(pair: str = "XBTMYR", mode: str = "rest", seconds: Optional[float] = None,
                **kw) -> Iterator[Event]:
    """Live normalized events: clear + full snapshot, then level diffs and trades.

    mode="rest": kwargs go to RestCollector (interval, trades_every, resync_every,
    depth, book, http, verbose). mode="stream": kwargs go to StreamCollector.
    seconds=None runs until the consumer stops iterating (or Ctrl-C).
    """
    if mode == "rest":
        return RestCollector(pair, **kw).events(seconds)
    if mode == "stream":
        from .stream import StreamCollector
        return StreamCollector(pair, **kw).events(seconds)
    raise ValueError(f"mode must be 'rest' or 'stream', not {mode!r}")


__all__ = ["iter_events", "poll", "RestCollector", "TradeTracker", "BookDiffer",
           "aggregate", "limit_depth", "price_to_int", "vol_to_units"]
