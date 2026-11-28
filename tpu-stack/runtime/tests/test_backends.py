import os
import subprocess
import sys

import pytest

from tpurt import get_backend, BackendUnavailable, selftest, isa
from tpurt.backends.raw import FakeFabric, FakeRegion, RawBackend
from conftest import ON_BOARD, RUNTIME


def _fake_raw():
    reg = FakeRegion()
    return RawBackend(regs=FakeFabric(reg), region=reg)


BACKENDS = {"sim": lambda: get_backend("sim"), "mock": lambda: get_backend("mock"),
            "sim-golden": lambda: get_backend("sim", golden=True), "raw-fake": _fake_raw}


@pytest.mark.parametrize("name", list(BACKENDS))
def test_selftests(name):
    be = BACKENDS[name]()
    for fn in (selftest.dma_loopback, selftest.matmul16, selftest.mlp):
        if name == "sim-golden" and fn is selftest.dma_loopback:
            continue                                     # pure-Python golden sim: keep it short
        r = fn(be, batch=4) if (fn is selftest.mlp and name == "sim-golden") else fn(be)
        assert r["ok"], (name, fn.__name__, r)
    be.close()


def test_cfg_err_and_status_reporting():
    for be in (get_backend("sim"), get_backend("mock"), _fake_raw()):
        res = be.run(isa.to_bytes([isa.matmul(0, 2000, 100), isa.interrupt_host(1), isa.halt()]), [])
        assert res.status_name == "CFG_ERR" and res.cfg_err_op == 0x10, be.name
        ok = be.run(isa.to_bytes([isa.interrupt_host(0xABC), isa.halt()]), [])
        assert ok.ok and ok.irq_tag == 0xABC, be.name


def test_board_backend_imports_cleanly_and_explains_off_board():
    r = subprocess.run([sys.executable, "-c", "import tpurt.backends.tpudev, tpurt.backends.raw; print('ok')"],
                       cwd=RUNTIME, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout
    if ON_BOARD:
        pytest.skip("on a board")
    for name in ("board", "raw"):
        with pytest.raises(BackendUnavailable) as e:
            get_backend(name)
        assert "sim" in str(e.value) or "load" in str(e.value)
    assert get_backend("auto").name == "sim"


def test_raw_backend_pynq_flavor_message():
    try:
        import pynq  # noqa: F401
        pytest.skip("pynq installed")
    except ImportError:
        pass
    with pytest.raises(BackendUnavailable) as e:
        get_backend("pynq")
    assert "pynq" in str(e.value)


@pytest.mark.board
def test_board_selftests():
    be = get_backend("board")
    print(be.info())
    for fn in (selftest.dma_loopback, selftest.matmul16, selftest.mlp):
        r = fn(be)
        assert r["ok"], r
    be.close()
