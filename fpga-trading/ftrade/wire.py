"""16-byte binary wire format fed to the FPGA. See SPEC.md."""
from __future__ import annotations
import struct
from .events import Event

T_CLEAR, T_LEVEL, T_TRADE = 0x43, 0x4C, 0x54
_KIND2T = {"clear": T_CLEAR, "level": T_LEVEL, "trade": T_TRADE}
_T2KIND = {v: k for k, v in _KIND2T.items()}
_FMT = "<BBHIII"


def _side_bit(e: Event) -> int:
    return 1 if e.side in ("ask", "sell") else 0


def encode(e: Event) -> bytes:
    return struct.pack(_FMT, _KIND2T[e.kind], _side_bit(e), 0,
                       e.price & 0xFFFFFFFF, e.volume & 0xFFFFFFFF, e.ts & 0xFFFFFFFF)


def decode(b: bytes) -> Event:
    t, s, _, price, vol, ts = struct.unpack(_FMT, b)
    kind = _T2KIND[t]
    side = "" if kind == "clear" else (("ask" if s else "bid") if kind == "level" else ("sell" if s else "buy"))
    return Event(kind=kind, ts=ts, side=side, price=price, volume=vol)


def to_words(e: Event) -> list[int]:
    return list(struct.unpack("<IIII", encode(e)))
