#!/usr/bin/env python3
"""Generate AXIS stimulus for tb_tob.cpp from ftrade events.

Writes <out>/stim.txt  one input beat per line: "<word hex> <tlast> <expect>"
       <out>/msgs.txt  the 4 wire words of every message that must produce a feature vector
                       (in order), which tb/check.py decodes and replays through TopBook.

Suites: synth (ftrade.synth.generate), fuzz (dense random book abuse), edge (hand-written
corner cases), framing (malformed AXIS frames that must be dropped), all = edge+framing+synth+fuzz.
"""
from __future__ import annotations
import argparse, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ftrade.events import Event          # noqa: E402
from ftrade import wire, synth           # noqa: E402

M32 = 0xFFFFFFFF


def C():
    return Event(kind="clear", ts=0)


def L(side, price, vol, ts=0):
    return Event(kind="level", ts=ts, side=side, price=price & M32, volume=vol & M32)


def T(side, vol, price=0, ts=0):
    return Event(kind="trade", ts=ts, side=side, price=price & M32, volume=vol & M32)


def edge_events() -> list[Event]:
    e: list[Event] = []
    # empty book: every feature 0 except what trades move
    e += [C(), T("buy", 100), T("sell", 100), L("bid", 100, 0), L("ask", 100, 0), C()]
    # one side only (spread / micro must stay 0)
    e += [L("bid", 100, 5), L("bid", 99, 7), T("sell", 3), C(), L("ask", 101, 5), C()]
    # simple book, overwrite, delete present / missing
    e += [L("bid", 100, 10), L("ask", 102, 30), L("bid", 100, 40), L("ask", 102, 1),
          L("bid", 101, 0), L("ask", 999, 0), L("bid", 100, 0), L("ask", 102, 0)]
    # full book overflow: 12 bids ascending (each new one is best), 12 asks descending
    e += [C()] + [L("bid", 1000 + i, i + 1) for i in range(12)]
    e += [L("ask", 2000 - i, i + 1) for i in range(12)]
    # insert worse than the worst kept level when full -> ignored; equal to worst -> overwrite
    e += [L("bid", 1000, 77), L("bid", 1004, 55), L("ask", 2000, 77), L("ask", 1996, 66)]
    # insert in the middle when full, delete in the middle, refill
    e += [L("bid", 1007, 0), L("bid", 1100, 9), L("bid", 1008, 0), L("ask", 1990, 0),
          L("ask", 1985, 3), L("ask", 1989, 0), L("bid", 1001, 0), L("bid", 1000, 2)]
    # insert at the worst position when not full
    e += [C(), L("bid", 500, 1), L("bid", 400, 1), L("bid", 300, 1), L("ask", 600, 1),
          L("ask", 700, 1), L("ask", 800, 1)]
    # delete best repeatedly until empty
    e += [L("bid", 500, 0), L("bid", 400, 0), L("bid", 300, 0), L("ask", 600, 0),
          L("ask", 700, 0), L("ask", 800, 0)]
    # crossed books: spread negative, micro on the unsigned spread
    e += [C(), L("bid", 105, 1), L("ask", 100, 1), L("bid", 110, 3), L("ask", 90, 1000)]
    # price extremes: 0 and 0xFFFFFFFF are legal prices
    e += [C(), L("bid", 0, 5), L("ask", M32, 5), L("bid", M32, 2), L("ask", 0, 9), C()]
    e += [L("bid", 0x80000000, 1), L("ask", 0x7FFFFFFF, 1), L("ask", 0x80000001, 3), C()]
    # huge volumes: top-5 sums wrap mod 2^32, B+A needs 33 bits, product needs 64
    e += [C()] + [L("bid", 100 - i, M32 - i) for i in range(6)]
    e += [L("ask", 101 + i, 0x80000000 + i) for i in range(6)]
    e += [L("ask", 101, M32), L("bid", 100, 1), L("ask", 101, 1), L("bid", 100, M32)]
    e += [C(), L("bid", 1, M32), L("ask", M32, M32), L("ask", M32, 1), L("bid", 1, 1)]
    e += [C(), L("bid", 10, 0x40000000), L("bid", 9, 0x40000000), L("bid", 8, 0x40000000),
          L("bid", 7, 0x40000000), L("ask", 11, 1)]          # B wraps to 0 -> imb = -32768
    # trade flow: negative, huge (int32 wrap), back to positive, clear resets
    e += [C()] + [T("sell", 1000) for _ in range(5)] + [T("buy", 7) for _ in range(3)]
    e += [T("sell", M32) for _ in range(40)] + [T("buy", M32) for _ in range(40)]
    e += [T("sell", 0x80000000), T("buy", 0), T("sell", 1), T("sell", 15), T("sell", 16), C()]
    e += [T("buy", 0x7FFFFFFF), T("buy", 0x80000000), T("sell", 0), C()]
    # clear mid-stream with a full book
    e += [L("bid", 50 + i, 10 * i + 1) for i in range(10)]
    e += [L("ask", 70 + i, 10 * i + 1) for i in range(10)] + [T("buy", 500), C(), T("buy", 16)]
    return e


def fuzz_events(n: int, seed: int) -> list[Event]:
    rng = random.Random(seed)
    out: list[Event] = [C()]
    base = rng.randint(0, M32)

    def vol():
        r = rng.random()
        if r < 0.20: return 0
        if r < 0.55: return rng.randint(1, 100)
        if r < 0.75: return rng.randint(1, 1 << 20)
        if r < 0.90: return rng.randint(1, M32)
        return M32 - rng.randint(0, 3)

    while len(out) < n:
        r = rng.random()
        if r < 0.005:
            out.append(C())
            if rng.random() < 0.3:
                base = rng.choice([0, M32 - 30, 1 << 31, rng.randint(0, M32)])
        elif r < 0.15:
            out.append(T(rng.choice(["buy", "sell"]), vol()))
        else:
            side = rng.choice(["bid", "ask"])
            spread = rng.choice([4, 12, 30])
            p = (base + rng.randint(-spread, spread)) & M32
            if rng.random() < 0.01:
                p = rng.choice([0, M32, 1 << 31, (1 << 31) - 1])
            out.append(L(side, p, vol()))
    return out[:n]


def write(out_dir: str, items):
    """items: list of (beats [(word, last)], expect_output: bool, msg_words or None)."""
    os.makedirs(out_dir, exist_ok=True)
    n_beats = n_msgs = 0
    with open(os.path.join(out_dir, "stim.txt"), "w") as fs, \
         open(os.path.join(out_dir, "msgs.txt"), "w") as fm:
        for beats, expect, words in items:
            for i, (w, last) in enumerate(beats):
                fs.write(f"{w:08x} {int(last)} {int(expect and i == len(beats) - 1)}\n")
                n_beats += 1
            if expect:
                fm.write(" ".join(f"{w:08x}" for w in words) + "\n")
                n_msgs += 1
    return n_beats, n_msgs


def msg_items(events):
    for ev in events:
        w = wire.to_words(ev)
        yield [(w[0], 0), (w[1], 0), (w[2], 0), (w[3], 1)], True, w


def framing_items():
    """Malformed frames interleaved with good ones. Each bad one must be dropped."""
    good = [L("bid", 100, 5), L("ask", 103, 7), T("buy", 20)]
    it = []
    it += list(msg_items([C()] + good))
    w = wire.to_words(L("bid", 101, 9))
    it.append(([(w[0], 0), (w[1], 1)], False, None))                  # runt (tlast on beat 1)
    it.append(([(w[0], 1)], False, None))                              # runt (tlast on beat 0)
    it += list(msg_items([L("bid", 99, 3)]))
    it.append(([(w[0], 0), (w[1], 0), (w[2], 0), (w[3], 0),            # no tlast on beat 3
                (0x11111111, 0), (0x22222222, 1)], False, None))       # junk until tlast
    it += list(msg_items([L("ask", 104, 1)]))
    bad = list(w); bad[0] = (bad[0] & ~0xFF) | 0x58                    # unknown type 'X'
    it.append(([(bad[0], 0), (bad[1], 0), (bad[2], 0), (bad[3], 1)], False, None))
    it += list(msg_items([T("sell", 5)]))
    weird = wire.to_words(L("ask", 102, 4)); weird[0] |= 0x7F00 | 0xABCD0000  # side byte 0x7F, junk reserved
    it.append(([(weird[0], 0), (weird[1], 0), (weird[2], 0), (weird[3], 1)], True, weird))
    it += list(msg_items(good))
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["synth", "fuzz", "edge", "framing", "all"])
    ap.add_argument("--n-synth", type=int, default=50000)
    ap.add_argument("--n-fuzz", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = []
    if a.suite in ("edge", "all"):
        items += list(msg_items(edge_events()))
    if a.suite in ("framing", "all"):
        items += framing_items()
    if a.suite in ("synth", "all"):
        items += list(msg_items(synth.generate(a.n_synth, seed=a.seed)))
    if a.suite in ("fuzz", "all"):
        items += list(msg_items(fuzz_events(a.n_fuzz, a.seed + 1000)))
    nb, nm = write(a.out, items)
    print(f"gen_stim: suite={a.suite} beats={nb} messages_expected={nm} -> {a.out}")


if __name__ == "__main__":
    main()
