"""Offline tests for ftrade.marketdata (REST collector, diffing, units). No network."""
import io
import json
import os
import sys
import urllib.error
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ftrade.events import Event, read_jsonl, write_jsonl
from ftrade import wire
from ftrade.features import TopBook
from ftrade.marketdata import book as bk
from ftrade.marketdata.book import BookDiffer, aggregate, limit_depth, price_to_int, vol_to_units
from ftrade.marketdata.rest import RestCollector, TradeTracker, rtt_stats, trade_event
from ftrade.marketdata.transport import Backoff, HttpError, Response, UrllibTransport


# ---------------------------------------------------------------- units
def test_price_and_volume_units():
    assert price_to_int("248001.00") == 248001
    assert price_to_int("255182.000000000000000000") == 255182
    assert vol_to_units("0.0685") == 68500
    assert vol_to_units("0.004206000000000000") == 4206
    assert vol_to_units("0.03678464") == 36785          # round half even at 1e-6
    assert vol_to_units("0.0000005") == 1                # positive never becomes 0 (= removal)
    assert vol_to_units("0") == 0
    assert vol_to_units("100000") == 0xFFFFFFFF          # clamped to uint32
    assert vol_to_units("1.1") == 1_100_000              # exact (float would also do, Decimal is safe)


def test_aggregate_order_level_book():
    # /api/1/orderbook lists every order; several at one price must be summed first
    lv = [{"price": "248001", "volume": "0.217521"}, {"price": "248001", "volume": "0.0685"},
          {"price": "248000", "volume": "0.0000004"}, {"price": "247990", "volume": "0"}]
    assert aggregate(lv) == {248001: 286021, 248000: 1}


# ---------------------------------------------------------------- diffing
def _ev(evs):
    return [(e.kind, e.side, e.price, e.volume) for e in evs]


def test_snapshot_then_diff():
    d = BookDiffer()
    ev = d.update({100: 5, 99: 7}, {101: 3, 103: 4}, ts=1, rx_ts=2)
    assert _ev(ev) == [("clear", "", 0, 0), ("level", "bid", 100, 5), ("level", "bid", 99, 7),
                       ("level", "ask", 101, 3), ("level", "ask", 103, 4)]
    assert all(e.rx_ts == 2 and e.ts == 1 for e in ev)
    # bid 100 removed, 99 changed, 98 new; ask 101 unchanged, 103 removed, 102 new
    ev = d.update({99: 8, 98: 1}, {101: 3, 102: 9}, ts=5)
    assert _ev(ev) == [("level", "bid", 100, 0), ("level", "bid", 99, 8), ("level", "bid", 98, 1),
                       ("level", "ask", 103, 0), ("level", "ask", 102, 9)]
    assert d.update({99: 8, 98: 1}, {101: 3, 102: 9}, ts=6) == []      # no change, no events
    ev = d.update({99: 8, 98: 1}, {101: 3, 102: 9}, ts=7, resync=True)
    assert ev[0].kind == "clear" and len(ev) == 5


def test_diff_reproduces_book_in_topbook():
    snaps = [({100: 5, 99: 7, 97: 1}, {101: 3, 103: 4}),
             ({99: 8, 98: 1}, {101: 3, 102: 9, 104: 2}),
             ({105: 2, 99: 8}, {106: 1}),
             ({}, {106: 1}),
             ({90: 1}, {})]
    d, tb = BookDiffer(), TopBook()
    for i, (b, a) in enumerate(snaps):
        for e in d.update(b, a, ts=i):
            tb.apply(e)
        assert tb.bids == [[p, b[p]] for p in sorted(b, reverse=True)][:8]
        assert tb.asks == [[p, a[p]] for p in sorted(a)][:8]


def test_depth_limited_differ_keeps_topbook_exact():
    # with depth=8, a level deeper than 8 that moves into view is re-sent
    bids = {1000 - i: 10 + i for i in range(12)}
    asks = {1001 + i: 10 + i for i in range(12)}
    d, tb = BookDiffer(depth=8), TopBook()
    for e in d.update(bids, asks, 0):
        tb.apply(e)
    for k in range(4):                     # eat the best bid level 4 times
        bids.pop(max(bids))
        for e in d.update(dict(bids), asks, k + 1):
            tb.apply(e)
        assert tb.bids == [[p, bids[p]] for p in sorted(bids, reverse=True)[:8]]


def test_limit_depth_replay():
    bids = {1000 - i: 10 + i for i in range(12)}
    asks = {1001 + i: 10 + i for i in range(12)}
    raw = BookDiffer().update(bids, asks, 0)
    raw += [Event("level", 1, "bid", 1000, 0), Event("level", 1, "bid", 999, 0),
            Event("trade", 2, "buy", 1001, 5), Event("level", 3, "ask", 1001, 0)]
    out = list(limit_depth(raw, 8))
    assert [e.seq for e in out] == list(range(1, len(out) + 1))
    tb = TopBook()
    for e in out:
        tb.apply(e)
    assert [p for p, _ in tb.bids] == list(range(998, 990, -1))
    assert [p for p, _ in tb.asks] == list(range(1002, 1010))
    assert sum(1 for e in out if e.kind == "trade") == 1


# ---------------------------------------------------------------- trades
def _t(seq, ts, price="248000.00", vol="0.001", is_buy=True):
    return {"sequence": seq, "timestamp": ts, "price": price, "volume": vol, "is_buy": is_buy}


def test_trade_event_side_and_units():
    e = trade_event(_t(1, 1000, "247724.000000000000000000", "0.000863000000000000", False), rx_ts=5)
    assert (e.kind, e.side, e.price, e.volume, e.ts, e.rx_ts) == ("trade", "sell", 247724, 863, 1000, 5)
    assert trade_event(_t(2, 1000, is_buy=True)).side == "buy"


def test_trade_dedup_and_since():
    tt = TradeTracker(since_ms=1000)
    # Luno returns newest -> oldest; trades before start are dropped
    ev = tt.ingest([_t(12, 1020), _t(11, 1010), _t(10, 999)])
    assert [e.ts for e in ev] == [1010, 1020] and tt.since == 1020
    # next poll re-delivers the trade at since (same ms) plus new ones
    ev = tt.ingest([_t(14, 1030), _t(13, 1020), _t(12, 1020)])
    assert [(e.ts) for e in ev] == [1020, 1030] and tt.duplicates == 1
    assert tt.ingest([_t(14, 1030)]) == [] and tt.since == 1030
    assert tt.ingest(None) == []


def test_trade_full_page_same_ms_advances():
    tt = TradeTracker(since_ms=0)
    page = [_t(s, 5000) for s in range(100, 0, -1)]
    assert len(tt.ingest(page)) == 100 and tt.since == 5000
    assert tt.ingest(page) == [] and tt.since == 5001      # no infinite paging loop


# ---------------------------------------------------------------- collector with a fake transport
class FakeTransport:
    """Serves canned Luno responses; can inject HTTP errors."""

    def __init__(self, books, trades, errors=None):
        self.books, self.trades = list(books), list(trades)
        self.errors = dict(errors or {})       # call index -> HttpError
        self.calls = []

    def get(self, path, params=None):
        i = len(self.calls)
        self.calls.append((path, dict(params or {})))
        if i in self.errors:
            raise self.errors[i]
        if path.endswith("/trades"):
            body = {"trades": self.trades.pop(0) if self.trades else None}
        else:
            body = self.books.pop(0) if len(self.books) > 1 else self.books[0]
        return Response(200, {}, json.dumps(body).encode(), 12.5, 1_795_000_000_999)

    def close(self):
        pass


def _book(ts, bids, asks):
    f = lambda d: [{"price": f"{p}.00", "volume": v} for p, v in d]
    return {"timestamp": ts, "bids": f(bids), "asks": f(asks)}


class Clock:
    def __init__(self):
        self.t = 1_795_000_000.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def _collector(ft, **kw):
    c = Clock()
    col = RestCollector("XBTMYR", transport=ft, clock=c, sleep=c.sleep, **kw)
    return col, c


def test_collector_end_to_end():
    books = [_book(1, [(100, "0.5"), (99, "1")], [(101, "0.25")]),
             _book(2, [(100, "0.5"), (99, "2")], [(101, "0.25"), (102, "0.1")]),
             _book(3, [(99, "2")], [(102, "0.1")])]
    t0 = 1_795_000_000_000
    trades = [[_t(2, t0 + 5, "100.00", "0.01", False), _t(1, t0 + 1, "101.00", "0.02", True)],
              [_t(2, t0 + 5, "100.00", "0.01", False)],
              [_t(3, t0 + 9, "101", "0.000001", True)]]
    col, clk = _collector(FakeTransport(books, trades))
    ev = list(col.events(max_polls=3))
    assert [e.seq for e in ev] == list(range(1, len(ev) + 1))
    assert _ev(ev) == [
        ("clear", "", 0, 0), ("level", "bid", 100, 500000), ("level", "bid", 99, 1000000),
        ("level", "ask", 101, 250000),
        ("trade", "buy", 101, 20000), ("trade", "sell", 100, 10000),
        ("level", "bid", 99, 2000000), ("level", "ask", 102, 100000),
        ("level", "bid", 100, 0), ("level", "ask", 101, 0),
        ("trade", "buy", 101, 1)]
    assert col.trades.duplicates == 1
    assert col.counts == {"clear": 1, "level": 7, "trade": 3}
    assert abs(clk.t - 1_795_000_003.0) < 1e-6          # 3 polls at 1.0 s
    assert all(r["rtt_ms"] == 12.5 for r in col.log.rows)
    # /trades is called with the advancing since cursor
    since = [p["since"] for path, p in col.transport.calls if path.endswith("/trades")]
    assert since == [t0, t0 + 5, t0 + 5]


def test_collector_backoff_on_429_and_resync_after_gap():
    books = [_book(1, [(100, "1")], [(101, "1")]),
             _book(2, [(100, "1")], [(101, "2")])]
    err = HttpError(429, "Too Many Requests", retry_after=7.0)
    ft = FakeTransport(books, [], errors={2: err})       # 3rd call = 2nd book poll
    col, clk = _collector(ft)
    ev = list(col.events(max_polls=3))
    kinds = [e.kind for e in ev]
    assert kinds.count("clear") == 2                      # initial + resync after the gap
    assert col.errors == 1 and col.log.rows[2]["status"] == 429
    assert clk.t >= 1_795_000_000.0 + 7.0                 # honoured Retry-After


def test_collector_periodic_resync():
    books = [_book(i, [(100, "1")], [(101, "1")]) for i in range(7)]
    col, _ = _collector(FakeTransport(books, []), resync_every=3)
    ev = list(col.events(max_polls=7))
    assert [e.kind for e in ev].count("clear") == 3       # polls 0, 3, 6


def test_collector_fatal_4xx():
    ft = FakeTransport([_book(1, [], [])], [], errors={0: HttpError(400, "ErrInvalidMarketPair")})
    col, _ = _collector(ft)
    with pytest.raises(HttpError):
        list(col.events(max_polls=1))


def test_rate_budget_enforced():
    with pytest.raises(ValueError):
        RestCollector(interval=0.2, transport=FakeTransport([{}], []))
    RestCollector(interval=0.5, trades_every=1, transport=FakeTransport([{}], []))   # 240/min: ok
    RestCollector(interval=1.0, trades_every=1, transport=FakeTransport([{}], []))


def test_backoff_grows_and_honours_retry_after():
    b = Backoff(base=1, cap=8)
    ds = [b.next() for _ in range(6)]
    assert all(0.5 <= d <= 8 for d in ds) and ds[-1] >= 4
    assert b.next(retry_after=30) == 30
    b.reset()
    assert b.next() <= 1


def test_rtt_stats():
    s = rtt_stats([5, 1, 3, 2, 4])
    assert (s["min_ms"], s["median_ms"], s["max_ms"], s["p99_ms"]) == (1, 3, 5, 5)
    assert rtt_stats([]) == {"n": 0}


# ---------------------------------------------------------------- urllib transport (mocked)
class _FakeResp(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_urllib_transport_ok_and_gzip():
    import gzip
    body = json.dumps({"timestamp": 1, "bids": [], "asks": []}).encode()
    with mock.patch("urllib.request.urlopen", return_value=_FakeResp(body)) as m:
        r = UrllibTransport().get("/api/1/orderbook_top", {"pair": "XBTMYR"})
    assert r.json()["timestamp"] == 1 and r.rtt_ms >= 0
    assert m.call_args[0][0].full_url == "https://api.luno.com/api/1/orderbook_top?pair=XBTMYR"
    gz = _FakeResp(gzip.compress(body), headers={"Content-Encoding": "gzip"})
    with mock.patch("urllib.request.urlopen", return_value=gz):
        assert UrllibTransport().get("/x").json()["timestamp"] == 1


def test_urllib_transport_429_and_network_error():
    e429 = urllib.error.HTTPError("https://api.luno.com/x", 429, "Too Many Requests",
                                  {"Retry-After": "3"}, io.BytesIO(b'{"error":"rate"}'))
    with mock.patch("urllib.request.urlopen", side_effect=e429):
        with pytest.raises(HttpError) as ei:
            UrllibTransport().get("/x")
    assert ei.value.status == 429 and ei.value.retry_after == 3.0 and ei.value.retryable
    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns")):
        with pytest.raises(HttpError) as ei:
            UrllibTransport().get("/x")
    assert ei.value.status == 0 and ei.value.retryable


def test_collector_over_mocked_urllib():
    """Full path: urllib (mocked) -> RestCollector -> events."""
    book = json.dumps(_book(1795000000123, [(248001, "0.0685")], [(248002, "0.1")])).encode()
    trades = json.dumps({"trades": [_t(7, 1_795_000_000_500, "248002", "0.001", True)]}).encode()

    def fake_urlopen(req, timeout=None):
        return _FakeResp(trades if "/trades" in req.full_url else book)

    c = Clock()
    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        col = RestCollector("XBTMYR", http="urllib", clock=c, sleep=c.sleep)
        ev = list(col.events(max_polls=2))
    assert _ev(ev) == [("clear", "", 0, 0), ("level", "bid", 248001, 68500),
                       ("level", "ask", 248002, 100000), ("trade", "buy", 248002, 1000)]
    assert ev[1].ts == 1795000000123


# ---------------------------------------------------------------- round trip
def test_recorded_events_roundtrip_wire_and_features(tmp_path):
    books = [_book(1795000000000 + i * 1000,
                   [(248001 - i, "0.0685"), (247990, "1.5"), (247000 + i, "0.3")],
                   [(248002 + i, "0.006405"), (248067, "0.400494")]) for i in range(5)]
    t0 = 1_795_000_000_000
    trades = [[_t(10 + i, t0 + 100 * i + 1, f"{248001 + i}", "0.0012", i % 2 == 0)] for i in range(5)]
    col, _ = _collector(FakeTransport(books, trades))
    p = tmp_path / "rec.jsonl.gz"
    n = write_jsonl(p, col.events(max_polls=5))
    back = list(read_jsonl(p))
    assert len(back) == n > 0
    tb = TopBook()
    for e in back:
        dec = wire.decode(wire.encode(e))
        assert (dec.kind, dec.side, dec.price, dec.volume, dec.ts) == \
            (e.kind, e.side, e.price, e.volume, e.ts & 0xFFFFFFFF)
        assert len(wire.to_words(e)) == 4
        f = tb.apply(e)
        assert len(f) == 8
    assert tb.bids[0][0] == 248001 - 4 and tb.asks[0][0] == 248002 + 4


SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "xbtmyr_sample.jsonl.gz")


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="no recorded sample")
def test_real_sample_roundtrip():
    evs = list(read_jsonl(SAMPLE))
    assert evs and evs[0].kind == "clear"
    assert [e.seq for e in evs] == list(range(1, len(evs) + 1))
    tb = TopBook()
    for e in evs:
        assert wire.decode(wire.encode(e)).kind == e.kind
        assert 0 <= e.price <= 0xFFFFFFFF and 0 <= e.volume <= 0xFFFFFFFF
        tb.apply(e)
    # depth-limited replay keeps the engine's book equal to the true top 8
    tb8, full = TopBook(), {"bid": {}, "ask": {}}
    for e in limit_depth(evs, 8):
        tb8.apply(e)
    for e in evs:
        if e.kind == "clear":
            full = {"bid": {}, "ask": {}}
        elif e.kind == "level":
            full[e.side].pop(e.price, None) if e.volume == 0 else full[e.side].__setitem__(e.price, e.volume)
    assert tb8.bids == [[p, full["bid"][p]] for p in sorted(full["bid"], reverse=True)[:8]]
    assert tb8.asks == [[p, full["ask"][p]] for p in sorted(full["ask"])[:8]]
