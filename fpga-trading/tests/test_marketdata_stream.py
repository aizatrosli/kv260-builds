"""Offline tests for the Luno websocket book aggregation (canned messages, no network)."""
import os
import sys

import pytest
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ftrade.features import TopBook
from ftrade import wire
from ftrade.marketdata import stream as st
from ftrade.marketdata.stream import SequenceGap, StreamBook

SNAP = {
    "sequence": "24352",
    "asks": [{"id": "A1", "price": "248002.00", "volume": "0.0685"},
             {"id": "A2", "price": "248002.00", "volume": "0.1"},
             {"id": "A3", "price": "248010.00", "volume": "0.5"}],
    "bids": [{"id": "B1", "price": "248001.00", "volume": "0.217521"},
             {"id": "B2", "price": "248001.00", "volume": "0.0685"},
             {"id": "B3", "price": "247990.00", "volume": "1.0"}],
    "status": "ACTIVE",
    "timestamp": 1795847331360,
}


def upd(seq, trades=None, create=None, delete=None, status=None, ts=1795847332000):
    return {"sequence": str(seq), "trade_updates": trades, "create_update": create,
            "delete_update": delete, "status_update": status, "timestamp": ts}


def lv(evs):
    return [(e.kind, e.side, e.price, e.volume) for e in evs]


def test_snapshot_aggregates_orders_to_levels():
    b = StreamBook()
    ev = b.on_snapshot(SNAP, rx_ts=1)
    assert lv(ev) == [("clear", "", 0, 0),
                      ("level", "bid", 248001, 286021), ("level", "bid", 247990, 1000000),
                      ("level", "ask", 248002, 168500), ("level", "ask", 248010, 500000)]
    assert ev[0].ts == 1795847331360 and b.seq == 24352 and b.status == "ACTIVE"


def test_create_delete_and_level_aggregation():
    b = StreamBook()
    b.on_snapshot(SNAP, 0)
    # new bid at an existing price -> level volume grows
    ev = b.on_update(upd(24353, create={"order_id": "B4", "type": "BID", "price": "248001.00",
                                        "volume": "0.01"}), 5)
    assert lv(ev) == [("level", "bid", 248001, 296021)]
    assert ev[0].rx_ts == 5 and ev[0].ts == 1795847332000
    # new ask at a new price
    ev = b.on_update(upd(24354, create={"order_id": "A9", "type": "ASK", "price": "248005",
                                        "volume": "0.2"}))
    assert lv(ev) == [("level", "ask", 248005, 200000)]
    # deleting one of two orders at a price -> reduced level; deleting the last -> volume 0
    assert lv(b.on_update(upd(24355, delete={"order_id": "A1"}))) == [("level", "ask", 248002, 100000)]
    assert lv(b.on_update(upd(24356, delete={"order_id": "A2"}))) == [("level", "ask", 248002, 0)]
    # deleting an unknown order is a no-op
    assert b.on_update(upd(24357, delete={"order_id": "nope"})) == []


def test_trade_updates_reduce_maker_and_set_taker_side():
    b = StreamBook()
    b.on_snapshot(SNAP, 0)
    # market buy lifts A1 fully and A2 partially; taker = buy
    trades = [{"sequence": 1, "base": "0.0685", "counter": "16988.137", "maker_order_id": "A1",
               "taker_order_id": "T1"},
              {"sequence": 2, "base": "0.05", "counter": "12400.1", "maker_order_id": "A2",
               "taker_order_id": "T1"}]
    ev = b.on_update(upd(24353, trades=trades))
    assert lv(ev) == [("trade", "buy", 248002, 68500), ("trade", "buy", 248002, 50000),
                      ("level", "ask", 248002, 50000)]
    assert "A1" not in b.orders and b.orders["A2"][2] == Decimal("0.05")
    # partial fill of a sell into bids + remainder rests as a new ask (docs example)
    trades = [{"sequence": 3, "base": "1.0", "counter": "247990", "maker_order_id": "B3",
               "taker_order_id": "T2"}]
    ev = b.on_update(upd(24354, trades=trades,
                         create={"order_id": "T2", "type": "ASK", "price": "247990", "volume": "0.3"}))
    assert lv(ev) == [("trade", "sell", 247990, 1000000),
                      ("level", "bid", 247990, 0), ("level", "ask", 247990, 300000)]


def test_keepalive_and_sequence_gap():
    b = StreamBook()
    with pytest.raises(SequenceGap):
        b.on_update(upd(1))                         # before snapshot
    b.on_snapshot(SNAP, 0)
    assert b.on_update("") == [] and b.on_update({}) == [] and b.on_update(None) == []
    b.on_update(upd(24353, status={"status": "POSTONLY"}))
    assert b.status == "POSTONLY"
    with pytest.raises(SequenceGap):
        b.on_update(upd(24355))                     # skipped 24354
    with pytest.raises(SequenceGap):
        b.on_update(upd(24353))                     # replay / out of order


def test_resync_after_gap_rebuilds_book():
    b = StreamBook()
    b.on_snapshot(SNAP, 0)
    b.on_update(upd(24353, create={"order_id": "X", "type": "BID", "price": "1", "volume": "1"}))
    snap2 = dict(SNAP, sequence="99", bids=[{"id": "Z", "price": "248000", "volume": "0.5"}])
    ev = b.on_snapshot(snap2, 0)
    assert ev[0].kind == "clear" and "X" not in b.orders
    assert [e for e in lv(ev) if e[1] == "bid"] == [("level", "bid", 248000, 500000)]
    assert lv(b.on_update(upd(100, delete={"order_id": "Z"}))) == [("level", "bid", 248000, 0)]


def test_stream_events_drive_topbook_and_wire():
    b = StreamBook()
    evs = b.on_snapshot(SNAP, 0)
    evs += b.on_update(upd(24353, create={"order_id": "B9", "type": "BID", "price": "248001.5",
                                          "volume": "0.000000001"}))      # 248001.5 -> 248002 (half-even)
    evs += b.on_update(upd(24354, delete={"order_id": "B1"}))
    tb = TopBook()
    for e in evs:
        assert wire.decode(wire.encode(e)).kind == e.kind
        tb.apply(e)
    # sub-ubtc positive volume still yields a present level (1 ubtc)
    assert [248002, 1] in tb.bids
    expect_b1 = sum(v for oid, (s, p, v) in b.orders.items() if s == "bid" and p == 248001)
    assert [248001, round(float(expect_b1) * 1e6)] in tb.bids


def test_available_without_key_or_lib(monkeypatch):
    monkeypatch.delenv(st.ENV_KEY_ID, raising=False)
    monkeypatch.delenv(st.ENV_KEY_SECRET, raising=False)
    ok, why = st.available()
    assert not ok and why
    with pytest.raises(RuntimeError):
        st.StreamCollector("XBTMYR")
    assert st.credentials() is None
    monkeypatch.setenv(st.ENV_KEY_ID, "id")
    monkeypatch.setenv(st.ENV_KEY_SECRET, "secret")
    assert st.credentials() == ("id", "secret")
