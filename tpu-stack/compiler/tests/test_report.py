from tpucc import report


def test_report(P, tmp_path):
    R = report.build_report(P)
    p = R["program"]
    assert p["dedupe"]["bytes"] < p["no_dedupe"]["bytes"]
    assert p["max_kernel_instrs"] <= 2048
    assert len(R["layers"]) == 73
    assert sum(l["weight_bytes"] for l in R["layers"]) == R["totals"]["weight_bytes"]
    assert 0.5 < R["totals"]["util"] <= 1.0
    assert R["totals"]["array_cycles"] >= R["totals"]["ideal_cycles"]
    assert R["memory"]["ub_peak_bytes"] < R["memory"]["ub_no_reuse_bytes"]
    t = R["partition"]["table"]
    assert t["layernorm"]["current"] == "host" and t["layernorm"]["future"] == "vpu"
    md = report.to_markdown(R)
    for s in ("## Program size", "## Per-layer schedule", "## Kernels", "## Partition plan", "qkv0", "fc25"):
        assert s in md
