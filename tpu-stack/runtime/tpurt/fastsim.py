"""Numpy-accelerated subclass of tpuv1's golden ISA simulator (model/isa/sim.py: CoreSim).

Inherits the decoder, the cfg_err legality rules, SYNC/INTERRUPT_HOST/HALT, SET_CONFIG and the
hazard bookkeeping protocol; replaces the per-element Python loops of MATMUL / ACTIVATE / DMA with
numpy (UB, ACC and weight tiles become arrays; hazard sets become interval lists).  Bit-exactness
against the unmodified CoreSim is checked in tests/test_fastsim.py on random programs.

Also implements the vector-unit extension ops X_VEC / X_LUT / X_VCFG (tpurt/vpu.py) when vpu=True.

DDR access: either `mem` (an object with read(addr, n) -> uint8 array and write(addr, data)) or a
tpuv1-style byte dict `ddr` (what tpu_mock passes), so it can stand in for CoreSim inside tpu_mock.
CONVOLVE and pooled ACTIVATE are not accelerated (raise NotImplementedError).
"""
import numpy as np

from . import _paths, intmath, vpu
from .hostops import NumpyOps

_paths.need_tpuv1_model()
import sim as isasim            # noqa: E402  (tpuv1 model/isa/sim.py)
import asm                      # noqa: E402

N = 16
_ops = NumpyOps()


def _first_false(mask):
    return int(np.flatnonzero(~mask)[0])


def _hit(intervals, lo, hi):
    return any(a < hi and lo < b for a, b in intervals)


class DictMem:
    """tpuv1 CoreSim DDR convention: dict address -> byte."""

    def __init__(self, d):
        self.d = d

    def read(self, addr, n):
        d = self.d
        return np.frombuffer(bytes(d[a] for a in range(addr, addr + n)), np.uint8)

    def write(self, addr, data):
        self.d.update(zip(range(addr, addr + len(data)), bytes(data)))


class FastCoreSim(isasim.CoreSim):
    def __init__(self, ub_init=None, wwords=None, check_hazards=True, enable_dma=False, ddr=None, mem=None,
                 vpu_enabled=False, lut_rom=None):
        super().__init__(None, wwords, check_hazards, enable_dma, ddr)
        self.ubA = np.zeros((isasim.UB_ROWS, N), np.int8)
        self.ubV = np.zeros(isasim.UB_ROWS, bool)
        for r, row in enumerate(ub_init or []):
            self.ubA[r] = row
            self.ubV[r] = True
        self.accA = np.zeros((isasim.ACC_ROWS, N), np.int64)
        self.accV = np.zeros(isasim.ACC_ROWS, bool)
        self.tiles = []                                   # numpy weight tiles W[k][n], FIFO order
        for i in range(0, len(self.wq) - len(self.wq) % N, N):
            self.tiles.append(self._tile_from_words(self.wq[i:i + N]))
        self.wq = []
        self.WA = np.zeros((N, N), np.int64)
        self.mem = mem if mem is not None else DictMem(self.ddr)
        self.vpu = vpu_enabled
        self.vpu_mult = (0, 0)
        self.vpu_sq = vpu.SQ_DEFAULT
        self.lut_exp = np.zeros(256, np.int64)
        self.lut_gelu = np.zeros(256, np.int64)
        self.lut_rom = lut_rom or {}
        self.mat_ub_rd, self.mat_acc_wr, self.act_acc_rd, self.act_ub_wr = [], [], [], []
        self.cycles = dict(matmul=0, w_load=0, activate=0, dma=0, vector=0, other=0)

    @property
    def ub(self):                                       # tpuv1-compatible read view (row -> list)
        return {int(r): [int(v) for v in self.ubA[r]] for r in np.flatnonzero(self.ubV)}

    @ub.setter
    def ub(self, value):                                # base __init__ assigns {}
        pass

    @property
    def acc(self):
        return {int(r): [int(v) for v in self.accA[r]] for r in np.flatnonzero(self.accV)}

    @acc.setter
    def acc(self, value):
        pass

    @staticmethod
    def _tile_from_words(words):
        b = np.frombuffer(b"".join((w & ((1 << 128) - 1)).to_bytes(16, "little") for w in words), np.int8)
        return b.reshape(N, N)[::-1].astype(np.int64)    # word t = W[15 - t]

    # ------------------------------------------------------------------ decode extensions (VPU)
    def step(self, word):
        if self.vpu and vpu.is_ext(word) and not (self.halted or self.cfg_err):
            op, f = vpu.xdec(word)
            bad = self._ext_invalid(word, op, f)
            if bad:
                self.cfg_err, self.cfg_err_op, self.cfg_err_why = True, op, bad
                return False
            self.instr_count += 1
            if op == vpu.X_VCFG:
                self.vpu_mult, self.vpu_sq = (f["ma"], f["mb"]), f["sq"]
                return True
            self._sync(asm.U_ALL)                         # sequencer waits for the core to go idle
            if op == vpu.X_LUT:
                t = vpu.rom_table(f["sel"], self.lut_rom[f["idx"]])
                if f["sel"] == vpu.LUT_EXP:
                    self.lut_exp = t
                else:
                    self.lut_gelu = t
                self.cycles["vector"] += 256
            else:
                self._vector(f)
            return True
        return super().step(word)

    # ------------------------------------------------------------------ DMA
    def _dma(self, op, f):
        self.dma_count += 1
        address = f["ddr_addr4"] * 16
        if op == asm.READ_WEIGHTS:
            raw = self.mem.read(address, f["tiles"] * 256).view(np.int8).reshape(-1, N, N)
            self.tiles.extend(t[::-1].astype(np.int64) for t in raw)
            self.cycles["w_load"] += f["tiles"] * N
            return
        self._sync(asm.U_ACT | (asm.U_MATRIX if op == asm.READ_HOST_MEM else 0))
        rl, reps, stride = f["row_len"], f["reps"], f["ddr_stride"] * 16
        u0 = f["ub_row"]
        self.cycles["dma"] += rl * reps + 32
        for rep in range(reps):
            a = address + rep * stride
            u = u0 + rep * rl
            if op == asm.READ_HOST_MEM:
                self.ubA[u:u + rl] = self.mem.read(a, rl * 16).view(np.int8).reshape(rl, N)
                self.ubV[u:u + rl] = True
            else:
                if not self.ubV[u:u + rl].all():
                    raise RuntimeError(f"WRITE_HOST_MEM reads uninitialised UB rows {u}..{u + rl - 1}")
                self.mem.write(a, self.ubA[u:u + rl].tobytes())

    # ------------------------------------------------------------------ compute
    def _next_tile(self):
        if self.wpos // N >= len(self.tiles):
            raise RuntimeError("Weight FIFO underflow (program needs more weight words)")
        self.WA = self.tiles[self.wpos // N]
        self.tiles[self.wpos // N] = None
        self.wpos += N

    def _matmul(self, ub_row, acc_row, B, acc, new_weights):
        lo, hi, alo, ahi = ub_row, ub_row + B, acc_row, acc_row + B
        self._hz(_hit(self.act_ub_wr, lo, hi), "MATMUL reads UB rows an unsynced ACTIVATE writes (RAW)")
        self._hz(_hit(self.act_acc_rd, alo, ahi), "MATMUL writes ACC rows an unsynced ACTIVATE reads (WAR)")
        if new_weights:
            self._next_tile()
        if not self.ubV[lo:hi].all():
            raise RuntimeError(f"MATMUL reads uninitialised UB row {lo + _first_false(self.ubV[lo:hi])}")
        y = self.ubA[lo:hi].astype(np.int64) @ self.WA
        if acc:
            if not self.accV[alo:ahi].all():
                raise RuntimeError(f"MATMUL accumulates into uninitialised ACC row {alo + _first_false(self.accV[alo:ahi])}")
            y = self.accA[alo:ahi] + y
        self.accA[alo:ahi] = intmath.wrap32(y)
        self.accV[alo:ahi] = True
        self.mat_ub_rd.append((lo, hi))
        self.mat_acc_wr.append((alo, ahi))
        self.cycles["matmul"] += B + (N if new_weights else 0)

    def _convolve(self, *a, **k):
        raise NotImplementedError("FastCoreSim does not accelerate CONVOLVE; use tpuv1 CoreSim")

    def _activate(self, acc_row, ub_row, rows, q, pool=0):
        if pool:
            raise NotImplementedError("FastCoreSim does not accelerate pooled ACTIVATE")
        self._hz(_hit(self.mat_acc_wr, acc_row, acc_row + rows), "ACTIVATE reads ACC rows an unsynced MATMUL writes (RAW)")
        self._hz(_hit(self.mat_ub_rd, ub_row, ub_row + rows), "ACTIVATE writes UB rows an unsynced MATMUL reads (WAR)")
        if not self.accV[acc_row:acc_row + rows].all():
            raise RuntimeError(f"ACTIVATE reads uninitialised ACC row {acc_row + _first_false(self.accV[acc_row:acc_row + rows])}")
        p = self.params[q]
        y = intmath.requant(self.accA[acc_row:acc_row + rows], np.array(p["bias"]), np.array(p["m0"]),
                            np.array(p["shift"]), p["zp_out"], p["act_min"], p["act_max"])
        self.ubA[ub_row:ub_row + rows] = y
        self.ubV[ub_row:ub_row + rows] = True
        self.act_acc_rd.append((acc_row, acc_row + rows))
        self.act_ub_wr.append((ub_row, ub_row + rows))
        self.act_qsets.add(q)
        self.cycles["activate"] += rows

    # ------------------------------------------------------------------ VPU (jepa_top extension)
    def _ext_invalid(self, word, op, f):
        if op not in (vpu.X_VEC, vpu.X_LUT, vpu.X_VCFG):
            return f"{vpu.XNAMES[op]} is not supported by the runtime simulator"
        if word & ~vpu.used_bits(op):
            return f"{vpu.XNAMES[op]} reserved bits set"
        if op == vpu.X_LUT:
            return None if f["idx"] in self.lut_rom else f"X_LUT idx {f['idx']} not in LUT ROM"
        if op == vpu.X_VEC:
            if f["count"] == 0 or f["rows"] == 0:
                return "X_VEC count/rows == 0"
            span = (f["rows"] - 1) * f["rstride"] + (f["count"] - 1) * f["vstride"] + 1
            regs = ("a", "o", "b") if f["op"] == vpu.V_ADD else ("a", "o")
            if any(f[r] + span > isasim.UB_ROWS for r in regs):
                return "X_VEC UB range overflow"
        return None

    def _rows(self, base, f):
        return base + np.arange(f["rows"])[:, None] * f["rstride"] + np.arange(f["count"])[None, :] * f["vstride"]

    def _gather(self, base, f):
        idx = self._rows(base, f)
        if not self.ubV[idx].all():
            raise RuntimeError("X_VEC reads uninitialised UB rows")
        return self.ubA[idx], idx                         # [rows, count, 16]

    def _vector(self, f):
        fn = f["op"]
        a, _ = self._gather(f["a"], f)
        if fn == vpu.V_ADD:
            b, _ = self._gather(f["b"], f)
            y = _ops.ew_add(a, b, *self.vpu_mult)
        elif fn == vpu.V_LUT:
            y = _ops.lut8(a, self.lut_gelu)
        elif fn == vpu.V_LN:
            y = _ops.layernorm(np.ascontiguousarray(a))
        else:
            y = _ops.softmax(np.ascontiguousarray(a)[None], self.lut_exp, q=self.vpu_sq)[0]
        oidx = self._rows(f["o"], f)
        self.ubA[oidx] = y
        self.ubV[oidx] = True
        self.cycles["vector"] += f["count"] * f["rows"] * (2 if fn in (vpu.V_LN, vpu.V_SM) else 1)
