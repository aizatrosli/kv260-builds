import json
import subprocess
import sys

from conftest import ON_BOARD, RUNTIME


def _run(*args, timeout=600):
    return subprocess.run([sys.executable, *args], cwd=RUNTIME, capture_output=True, text=True, timeout=timeout)


def test_bench_cli_sim(tmp_path):
    out = tmp_path / "b.json"
    r = _run("-m", "tpurt.bench", "--backend", "sim", "--images", "2", "--batch", "2", "--pipeline", "on",
             "--vector", "both", "--json", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads(out.read_text())
    assert d["dma"]["loopback_ok"] and d["matmul"]["bit_exact"]
    assert {j["vector"] for j in d["jepa"]} == {"host", "vpu"} and all(j["bit_exact"] for j in d["jepa"])
    assert d["host_ops"]["numpy"]["layernorm"] > 0
    assert "ESTIMATED" in r.stdout


def test_bench_host_only():
    r = _run("-m", "tpurt.bench", "--host-only")
    assert r.returncode == 0 and "host ops (numpy" in r.stdout


def test_board_smoke_on_sim():
    r = _run("board_smoke.py", "--backend", "sim", "--images", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    for st in ("driver", "regs", "dma", "matmul", "mlp", "jepa"):
        assert f"[PASS] {st}" in r.stdout
    assert "BOARD_SMOKE_PASS" in r.stdout


def test_board_smoke_off_board_is_clear():
    if ON_BOARD:
        return
    r = _run("board_smoke.py")
    assert r.returncode == 2 and "[FAIL] driver" in r.stdout and "BOARD_SMOKE_FAIL" in r.stdout
