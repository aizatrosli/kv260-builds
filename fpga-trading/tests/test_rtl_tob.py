"""RTL top-of-book engine (rtl/tob_engine.v) vs the golden model ftrade.features.TopBook.

Builds the Verilator testbench tb/tb_tob.cpp for both output widths, drives it with stimulus
from tb/gen_stim.py (ftrade.synth + dense fuzz + hand-written edge cases + malformed frames)
under random AXIS stalls, and checks every feature vector bit-exactly with tb/check.py.
Skipped when verilator is not installed.
"""
from __future__ import annotations
import importlib.util, os, shutil, subprocess, sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TB = os.path.join(ROOT, "tb")
RTL = os.path.join(ROOT, "rtl")
SRC = [os.path.join(RTL, f) for f in
       ("tob_engine.v", "feed_rx.v", "book_side.v", "features.v", "div_r4_pipe.v", "delay_sr.v")]
VL = os.environ.get("VL", "verilator")

pytestmark = pytest.mark.skipif(shutil.which(VL) is None, reason="verilator not installed")
_spec = importlib.util.spec_from_file_location("tob_tb_check", os.path.join(TB, "check.py"))
tbcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tbcheck)

N_SYNTH = int(os.environ.get("RTL_N_SYNTH", "50000"))
N_FUZZ = int(os.environ.get("RTL_N_FUZZ", "20000"))


def _run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert p.returncode == 0, f"{' '.join(map(str, cmd))}\n{p.stdout}\n{p.stderr}"
    return p.stdout


@pytest.fixture(scope="session")
def build(tmp_path_factory):
    d = tmp_path_factory.mktemp("rtl_tob")
    sims = {}
    for w in (32, 256):
        _run([VL, "--cc", "--exe", "--build", "-j", "4", "-O3", "--top-module", "tob_engine",
              f"-GOUT_W={w}", "-Mdir", str(d / f"obj_{w}"), "-CFLAGS", f"-O2 -DOUT_W={w}",
              *SRC, os.path.join(TB, "tb_tob.cpp")])
        sims[w] = d / f"obj_{w}" / "Vtob_engine"
    return d, sims


def _stim(d, suite, seed=1, n_synth=N_SYNTH, n_fuzz=N_FUZZ):
    out = d / f"stim_{suite}_{seed}_{n_synth}_{n_fuzz}"
    if not (out / "stim.txt").exists():
        _run([sys.executable, os.path.join(TB, "gen_stim.py"), "--suite", suite, "--seed", str(seed),
              "--n-synth", str(n_synth), "--n-fuzz", str(n_fuzz), "--out", str(out)])
    return out


def _sim(sim, stim, out, seed=1, in_stall=0, out_stall=0, gap=0):
    log = _run([str(sim), f"stim={stim / 'stim.txt'}", f"out={out}", f"seed={seed}",
                f"in_stall={in_stall}", f"out_stall={out_stall}", f"gap={gap}"])
    tbl = dict(kv.split("=", 1) for kv in log.split("TB:")[1].split() if "=" in kv)
    res = tbcheck.check(str(stim / "msgs.txt"), str(out))
    assert res["ok"], res
    return tbl, res


@pytest.mark.parametrize("w", [32, 256])
def test_lint_wall_clean(w):
    _run([VL, "--lint-only", "-Wall", "--top-module", "tob_engine", f"-GOUT_W={w}", *SRC])


@pytest.mark.parametrize("w", [32, 256])
@pytest.mark.parametrize("stall", [(0, 0), (50, 50), (0, 90)])
def test_edge_and_framing(build, w, stall):
    d, sims = build
    stim = _stim(d, "edge")
    tbl, res = _sim(sims[w], stim, d / f"edge_{w}_{stall[0]}_{stall[1]}.txt", 3, *stall)
    assert res["mismatches"] == 0 and res["messages"] > 200
    st = _stim(d, "framing")
    tbl, res = _sim(sims[w], st, d / f"fr_{w}_{stall[0]}_{stall[1]}.txt", 4, *stall)
    assert int(tbl["rx_err"]) == 4          # 2 runts, 1 missing tlast, 1 unknown type


@pytest.mark.parametrize("w", [32, 256])
@pytest.mark.parametrize("seed,in_stall,out_stall", [(1, 0, 0), (2, 30, 40), (3, 70, 10)])
def test_synth_and_fuzz(build, w, seed, in_stall, out_stall):
    d, sims = build
    stim = _stim(d, "all", seed=seed)
    tbl, res = _sim(sims[w], stim, d / f"all_{w}_{seed}.txt", seed, in_stall, out_stall)
    assert res["messages"] >= N_SYNTH + N_FUZZ and res["mismatches"] == 0
    cov = res["coverage"]
    for k in ("insert", "insert_full_drop", "insert_ignored_full", "overwrite", "delete",
              "delete_missing", "crossed_or_locked", "vol5_wrap", "imb_full_scale",
              "flow_negative", "micro_bit31", "clear"):
        assert cov.get(k, 0) > 0, k


@pytest.mark.parametrize("w,mpc", [(32, 0.125), (256, 0.25)])
def test_latency_and_throughput(build, w, mpc):
    d, sims = build
    stim = _stim(d, "all", seed=5, n_synth=5000, n_fuzz=5000)
    tbl, res = _sim(sims[w], stim, d / f"lat_{w}.txt", 5, gap=40)     # unloaded
    for k, v in res["latency"].items():
        assert v["min"] == v["max"] == 25, (k, v)
    tbl, res = _sim(sims[w], stim, d / f"thr_{w}.txt", 5)              # back-to-back input
    assert abs(float(tbl["msgs/cycle"]) - mpc) < 0.002


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_iverilog_compiles(tmp_path):
    _run(["iverilog", "-g2005", "-Wall", "-o", str(tmp_path / "x.vvp"), "-s", "tob_engine", *SRC])
