"""Order intents, simulated orders, fills and the paper portfolio.

Everything here is simulated bookkeeping. Nothing talks to an exchange.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

UBTC = 1_000_000          # ubtc per BTC


@dataclass
class OrderIntent:
    """What a strategy (or the risk kill switch) wants to do.

    action: 'new' | 'cancel' | 'cancel_all'
    side:   'buy' | 'sell'                      (new only)
    type:   'market' | 'limit'                  (new only)
    price:  limit price, integer MYR            (limit only)
    volume: ubtc                                (new only)
    """
    action: str = "new"
    side: str = ""
    type: str = "limit"
    price: int = 0
    volume: int = 0
    post_only: bool = False
    order_id: Optional[int] = None    # for cancel
    tag: str = ""                     # free text, e.g. 'mm-bid', 'kill'

    @staticmethod
    def market(side: str, volume: int, tag: str = "") -> "OrderIntent":
        return OrderIntent("new", side, "market", 0, int(volume), False, None, tag)

    @staticmethod
    def limit(side: str, price: int, volume: int, post_only: bool = False,
              tag: str = "") -> "OrderIntent":
        return OrderIntent("new", side, "limit", int(price), int(volume), post_only, None, tag)

    @staticmethod
    def cancel(order_id: int, tag: str = "") -> "OrderIntent":
        return OrderIntent("cancel", order_id=order_id, tag=tag)

    @staticmethod
    def cancel_all(tag: str = "") -> "OrderIntent":
        return OrderIntent("cancel_all", tag=tag)

    def to_dict(self) -> dict:
        d = {"action": self.action, "tag": self.tag}
        if self.action == "new":
            d.update(side=self.side, type=self.type, volume=self.volume)
            if self.type == "limit":
                d.update(price=self.price, post_only=self.post_only)
        elif self.action == "cancel":
            d["order_id"] = self.order_id
        return d


@dataclass
class Fill:
    ts: int
    order_id: int
    side: str
    price: int
    volume: int
    fee_myr: float
    liquidity: str        # 'taker' | 'maker'
    tag: str = ""

    @property
    def notional_myr(self) -> float:
        return self.price * self.volume / UBTC


@dataclass
class Order:
    id: int
    side: str
    type: str
    price: int
    volume: int
    post_only: bool
    tag: str
    decision_ts: int
    arrival_ts: int
    status: str = "pending"     # pending -> resting | filled | cancelled | rejected
    filled: int = 0
    queue_ahead: int = 0        # ubtc displayed ahead of us at our price
    cancel_requested: bool = False
    reject_reason: str = ""

    @property
    def remaining(self) -> int:
        return self.volume - self.filled

    @property
    def is_working(self) -> bool:
        return self.status in ("pending", "resting")


@dataclass
class Portfolio:
    """Spot balances (Luno is spot: no borrowing, no shorting below zero).

    `base_btc` is the starting inventory; `net` = btc - base_btc is the
    trading position strategies and risk reason about. Trading PnL is
    measured against simply holding the starting balances.
    """
    cash: float = 50_000.0          # MYR
    btc: int = 50_000               # ubtc held (0.05 BTC)
    base_cash: float = field(default=None)   # type: ignore[assignment]
    base_btc: int = field(default=None)      # type: ignore[assignment]
    fees_paid: float = 0.0
    traded_notional: float = 0.0
    traded_volume: int = 0
    n_fills: int = 0
    working: dict = field(default_factory=dict)   # id -> Order (pending or resting)

    def __post_init__(self) -> None:
        if self.base_cash is None:
            self.base_cash = self.cash
        if self.base_btc is None:
            self.base_btc = self.btc

    @property
    def net(self) -> int:
        return self.btc - self.base_btc

    def equity(self, mark: float) -> float:
        return self.cash + self.btc * mark / UBTC

    def trading_pnl(self, mark: float) -> float:
        """Net of fees, relative to holding the starting balances."""
        return self.equity(mark) - (self.base_cash + self.base_btc * mark / UBTC)

    def gross_pnl(self, mark: float) -> float:
        return self.trading_pnl(mark) + self.fees_paid

    def apply_fill(self, f: Fill) -> None:
        n = f.price * f.volume / UBTC
        if f.side == "buy":
            self.cash -= n
            self.btc += f.volume
        else:
            self.cash += n
            self.btc -= f.volume
        self.cash -= f.fee_myr
        self.fees_paid += f.fee_myr
        self.traded_notional += n
        self.traded_volume += f.volume
        self.n_fills += 1

    # -- views used by strategies / risk
    def working_orders(self, side: Optional[str] = None, tag: Optional[str] = None) -> list[Order]:
        return [o for o in self.working.values()
                if (side is None or o.side == side) and (tag is None or o.tag == tag)]

    def has_working(self) -> bool:
        return bool(self.working)

    def working_volume(self, side: str) -> int:
        return sum(o.remaining for o in self.working.values() if o.side == side)
