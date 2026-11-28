import os

from tpuperf import cli, dse
from tpuperf import workloads as W


def test_small_sweep_flags_infeasible():
    grid = dict(array=[(16, 16), (64, 64)], clock_mhz=[100], dsp_packing=[True],
                weight_buffering=["double"], weight_source=["ddr_prefetch"], weight_bits=[8],
                control=["streamlined"])
    rows = dse.sweep([W.jepa_tiny()], grid)
    assert len(rows) == 2
    st = {(r["rows"], r["cols"]): r["status"] for r in rows}
    assert not st[(16, 16)].startswith("INFEASIBLE")
    assert st[(64, 64)].startswith("INFEASIBLE")
    assert dse.pareto(rows, "jepa_tiny")


def test_roadmap_cumulative_monotone():
    rows = dse.roadmap(W.jepa_tiny())
    sp = [r["speedup_cum"] for r in rows]
    assert all(b >= a - 0.01 for a, b in zip(sp, sp[1:]))
    assert sp[-1] > 5
    assert all("loss_if_dropped" in r for r in rows)


def test_cli_commands(tmp_path, capsys):
    out = str(tmp_path)
    assert cli.main(["--out", out, "calibrate"]) == 0
    assert os.path.exists(os.path.join(out, "calibration.csv"))
    assert cli.main(["--out", out, "run", "jepa_tiny", "--preset", "recommended",
                     "--set", "clock_mhz=150", "--layers"]) == 0
    assert cli.main(["--out", out, "resources", "--preset", "proposed"]) == 0
    assert cli.main(["--out", out, "roadmap", "--workload", "mlp3"]) == 0
    text = capsys.readouterr().out
    assert "worst |error|" in text and "jepa_tiny" in text


def test_cli_report_quick(tmp_path):
    md = tmp_path / "R.md"
    assert cli.main(["--out", str(tmp_path), "report", "--quick", "--md", str(md)]) == 0
    s = md.read_text()
    assert "Top hardware changes" in s and "Calibration" in s
    assert (tmp_path / "dse.csv").exists()
