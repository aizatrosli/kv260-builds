from tpuperf import calibrate as K


def test_every_benchmark_within_15_percent():
    rows = K.calibration_table()
    assert len(rows) >= 9
    bad = [r for r in rows if abs(r["error_pct"]) >= 15.0]
    assert not bad, bad


def test_board_fit_recovers_defaults_region():
    err, params = K.fit_board(lat_ns=range(200, 451, 50), host_us=(2, 4, 6), eff=(0.7, 0.8))
    assert err < 0.10
    assert 200 <= params["dma_latency_ns"] <= 450


def test_axi_push_fit_is_close_to_default():
    bw = K.fit_axi_push(iters=12)
    assert 2.0 < bw < 2.8
