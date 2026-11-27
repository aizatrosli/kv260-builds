"""Tiling for the 16x16 weight-stationary array, with an analytic cost model.

Array model (docs/architecture_diagrams.md of tpuv1): a weight tile shifts in over 16 cycles
into shadow registers, so it is hidden if the previous MATMUL streamed >= 16 rows; a MATMUL
streams one input row per cycle; results appear 16 cycles after the last row plus 16 cycles of
input skew, so every SYNC on the matrix unit costs a ~32-cycle drain. ACTIVATE (1 row/cycle)
runs concurrently with the array except across a SYNC.

Loop orders considered for C[M,N] = X[M,K] W[K,N] (KT = K/16, NT = N/16 tiles):

* ``nk``  - for each M block: for nt: for kt: MATMUL (B = M block rows), ACTIVATE the tile.
  Each weight tile is loaded once per M block and reused for B rows (weight stationary).
  Needs one ACC slot of B rows per in-flight tile (>= 2 slots for MATMUL/ACTIVATE overlap).
* ``kn``  - for each M block: for kt: for nt: MATMUL into NT live ACC slots, ACTIVATE all at
  the end. Same weight traffic, but needs NT*B ACC rows and serialises all ACTIVATEs after the
  array (no overlap): only useful if per-tile syncs dominated, which they do not here.

The M block B is the largest that leaves two ACC slots (B <= ACC_ROWS/2); a larger M is split
into blocks and every weight tile is re-streamed once per block (the K-blocking alternative,
keeping K partial sums resident across blocks, would need KT x more ACC than exists).
"""
from dataclasses import dataclass, field

from .env import ACC_ROWS

W_LOAD = 16      # cycles to shift a weight tile in
DRAIN = 32       # skew + result latency, paid at each SYNC of the matrix unit


@dataclass
class MatmulPlan:
    M: int
    K: int
    N: int
    KT: int
    NT: int
    order: str
    mblocks: list                     # [(m0, rows)]
    weight_tile_loads: int
    est_cycles: int
    alternatives: dict = field(default_factory=dict)

    @property
    def weight_bytes(self):
        return self.weight_tile_loads * 256


def _cost(order, M, KT, NT, mb):
    nmb = -(-M // mb)
    stream = M * KT * NT                                   # rows streamed in total
    exposed = 0 if mb >= W_LOAD else (W_LOAD - mb) * KT * NT * nmb
    if order == "nk":
        syncs = NT * nmb                                   # one RAW sync per output tile
        act_exposed = mb                                   # only the last tile's ACTIVATE
    else:
        syncs = nmb
        act_exposed = M * NT                               # every ACTIVATE after the array
    return stream + exposed + syncs * (DRAIN + W_LOAD) + act_exposed


def plan_matmul(M, K, N, acc_rows=ACC_ROWS):
    KT, NT = -(-K // 16), -(-N // 16)
    mb = min(M, acc_rows // 2)
    alts = {}
    for order in ("nk", "kn"):
        if order == "kn" and NT * mb > acc_rows:
            alts[order] = None                             # does not fit the accumulators
            continue
        alts[order] = _cost(order, M, KT, NT, mb)
    best = min((o for o in alts if alts[o] is not None), key=lambda o: (alts[o], o != "nk"))
    nmb = -(-M // mb)
    blocks = [(m0, min(mb, M - m0)) for m0 in range(0, M, mb)]
    return MatmulPlan(M, K, N, KT, NT, best, blocks, KT * NT * nmb, alts[best], alts)


@dataclass
class ConvPlan:
    OH: int
    OW: int
    taps: int
    CT: int
    NT: int
    windows: list                     # [(oh0, oh_cnt)]
    weight_tile_loads: int


def plan_conv(OH, OW, k, CT, NT, pool=None, acc_rows=ACC_ROWS):
    """Output-row windows so that a window (oh_cnt*OW rows) fits one ACC slot of acc_rows/2;
    pooling needs even window starts/heights. Weight tiles are re-streamed per window."""
    cap = acc_rows // 2
    rows = max(1, cap // OW)
    if pool:
        rows -= rows % 2
        if rows == 0:
            raise ValueError("conv row too wide for a pooled window")
    windows = [(o, min(rows, OH - o)) for o in range(0, OH, rows)]
    return ConvPlan(OH, OW, k * k, CT, NT, windows, k * k * CT * NT * len(windows))
