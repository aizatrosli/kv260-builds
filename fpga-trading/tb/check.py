#!/usr/bin/env python3
"""Compare tb_tob.cpp output against the golden model ftrade.features.TopBook.

  check.py <dir with msgs.txt> <out.txt> [--json summary.json]
Every expected message is decoded with ftrade.wire.decode and applied to TopBook; each RTL
vector must match all 8 features bit-exactly. Also checks the RTL latency register against the
testbench's own measurement and reports latency min/max per message type.
"""
from __future__ import annotations
import argparse, json, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ftrade import wire                      # noqa: E402
from ftrade.features import TopBook, FEATURE_NAMES   # noqa: E402


def s32(x):
    return x - (1 << 32) if x & 0x80000000 else x


def check(msgs_path, out_path, max_report=10):
    msgs = [[int(t, 16) for t in ln.split()] for ln in open(msgs_path) if ln.strip()]
    outs = [ln.split() for ln in open(out_path) if ln.strip()]
    tb = TopBook()
    mism, lat_bad, report = 0, 0, []
    lat, cov = {}, {}

    def hit(k):
        cov[k] = cov.get(k, 0) + 1

    for i, words in enumerate(msgs):
        ev = wire.decode(struct.pack("<IIII", *words))
        hit(ev.kind)
        if ev.kind == "level":                       # classify the book operation (coverage)
            lv = tb.bids if ev.side == "bid" else tb.asks
            found = any(p == ev.price for p, _ in lv)
            if ev.volume == 0:
                hit("delete" if found else "delete_missing")
            elif found:
                hit("overwrite")
            else:
                worse = all((ev.price < p) if ev.side == "bid" else (ev.price > p) for p, _ in lv)
                if len(lv) < tb.n or not worse:
                    hit("insert_full_drop" if len(lv) == tb.n else "insert")
                else:
                    hit("insert_ignored_full")
        exp = tb.apply(ev)
        if tb.bids and tb.asks and tb.bids[0][0] >= tb.asks[0][0]:
            hit("crossed_or_locked")
        if sum(v for _, v in tb.bids[:5]) > 0xFFFFFFFF or sum(v for _, v in tb.asks[:5]) > 0xFFFFFFFF:
            hit("vol5_wrap")
        if exp[3] < 0 or exp[4] < 0:
            hit("vol5_bit31")
        if abs(exp[5]) == 32768:
            hit("imb_full_scale")
        if exp[6] < 0:
            hit("flow_negative")
        if exp[7] < 0:
            hit("micro_bit31")
        if i >= len(outs):
            continue
        got = [s32(int(t, 16)) for t in outs[i][:8]]
        lr, lt = int(outs[i][8]), int(outs[i][9])
        if lr != lt:
            lat_bad += 1
        d = lat.setdefault(ev.kind, [10 ** 9, -1, 0, 0])
        d[0] = min(d[0], lr); d[1] = max(d[1], lr); d[2] += lr; d[3] += 1
        if got != exp:
            mism += 1
            if len(report) < max_report:
                diff = {FEATURE_NAMES[k]: (got[k], exp[k]) for k in range(8) if got[k] != exp[k]}
                report.append(f"msg {i} {ev.kind} {ev.side} p={ev.price} v={ev.volume}: rtl!=gold {diff}")
    res = dict(messages=len(msgs), vectors=len(outs), mismatches=mism,
               missing=max(0, len(msgs) - len(outs)), extra=max(0, len(outs) - len(msgs)),
               latency_register_mismatches=lat_bad,
               latency={k: dict(min=v[0], max=v[1], mean=round(v[2] / v[3], 2), n=v[3])
                        for k, v in sorted(lat.items())},
               coverage=dict(sorted(cov.items())), report=report)
    res["ok"] = (mism == 0 and res["missing"] == 0 and res["extra"] == 0 and lat_bad == 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out"); ap.add_argument("--json")
    a = ap.parse_args()
    r = check(os.path.join(a.dir, "msgs.txt"), a.out)
    for ln in r["report"]:
        print("  " + ln)
    print(f"check: messages={r['messages']} vectors={r['vectors']} mismatches={r['mismatches']} "
          f"missing={r['missing']} extra={r['extra']} lat_reg_mismatch={r['latency_register_mismatches']} "
          f"latency={json.dumps(r['latency'])} -> {'PASS' if r['ok'] else 'FAIL'}")
    print(f"coverage: {json.dumps(r['coverage'])}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
