"""Scheduling report: per-layer instruction counts, weight traffic, estimated array cycles,
memory, fusion, program size vs the 404 KiB baseline, and the host / VPU partition plan.

    python -m tpucc.report [--out reports/]      (JEPA encoder from jepa-tpu/qmodel.npy)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import env, frontends, passes, codegen, encoding
from .baseline import baseline_size

CLOCK_HZ = 100e6
KIB = 1024


def compile_jepa(P=None, **kw):
    P = P or env.load_qmodel()
    g = frontends.from_jepa(P)
    lg = passes.lower(g)
    return g, lg, codegen.compile_graph(lg, **kw)


def fusion_stats(g_unfused, g_lowered):
    fused = [o for o in g_lowered.ops if o.kind in ("tpu_matmul", "tpu_conv")]
    acc_bytes = sum(4 * g_lowered.T(o.outputs[0]).shape[0] * g_lowered.T(o.outputs[0]).shape[1] for o in fused)
    return dict(fused_ops=len(fused), ops_before=len(g_unfused.ops), ops_after=len(g_lowered.ops),
                int32_intermediate_bytes_kept_in_acc=acc_bytes,
                host_roundtrip_bytes_avoided=acc_bytes + acc_bytes // 4)


def partition_table(g):
    cur, fut = passes.device_plan(g, "current"), passes.device_plan(g, "future")
    rows = defaultdict(lambda: dict(count=0, elems=0))
    for o in g.ops:
        r = rows[o.kind]
        r["count"] += 1
        r["elems"] += g.T(o.outputs[0]).shape[0] * g.T(o.outputs[0]).shape[1]
        r["current"], r["future"] = cur[o.name], fut[o.name]
    return dict(rows)


def build_report(P=None):
    P = P or env.load_qmodel()
    g, lg, prog = compile_jepa(P)
    prog_nd = codegen.compile_graph(lg, dedupe=False)
    prog_nr = codegen.compile_graph(lg, reuse=False)
    prog_rc = codegen.compile_graph(lg, reset_config_per_kernel=True)
    base = baseline_size(lg)
    enc = encoding.estimate(prog)
    layers = []
    for name, s in prog.stats.items():
        if name.startswith("<"):
            continue
        ideal = s.macs / 256
        layers.append(dict(op=name, kind=s.kind, shape=s.shape, plan=s.plan, kernel=s.kernel,
                           instrs=s.instrs, set_config=s.counts["SET_CONFIG"], matmul=s.counts["MATMUL"] + s.counts["CONVOLVE"],
                           activate=s.counts["ACTIVATE"], sync=s.counts["SYNC"], read_weights=s.counts["READ_WEIGHTS"],
                           weight_bytes=s.weight_bytes, macs=s.macs, array_cycles=s.array_cycles,
                           util=ideal / s.array_cycles if s.array_cycles else 0.0))
    kernels = []
    for kind, k in prog.steps:
        if kind != "kernel":
            continue
        st = prog.stats.get(f"<kernel{k.step}>")
        c = Counter(codegen.U.NAMES[w & 0xFF] for w in k.words)
        kernels.append(dict(kernel=k.index, step=k.step, instrs=len(k.words), ops=list(k.ops),
                            rhm=c["READ_HOST_MEM"], whm=c["WRITE_HOST_MEM"]))
    host_steps = [ops for kind, ops in prog.steps if kind == "host"]
    dma_bytes = sum(s.dma_bytes for s in prog.stats.values())
    io_bytes = sum(lg.T(t).ub_rows * 16 for t in lg.inputs + lg.outputs)
    cycles = sum(l["array_cycles"] for l in layers)
    macs = sum(l["macs"] for l in layers)
    mem = prog.mem
    R = dict(
        graph=dict(ops_unfused=len(g.ops), ops_lowered=len(lg.ops),
                   by_class=dict(Counter(o.cls for o in lg.ops))),
        program=dict(
            baseline=dict(programs=base["programs"], instrs=base["instrs"], bytes=base["bytes"], set_config=base["set_config"]),
            no_dedupe=dict(instrs=prog_nd.n_instrs, bytes=prog_nd.n_bytes, set_config=prog_nd.opcode_counts()["SET_CONFIG"]),
            dedupe=dict(instrs=prog.n_instrs, bytes=prog.n_bytes, set_config=prog.opcode_counts()["SET_CONFIG"]),
            dedupe_reset_per_kernel=dict(instrs=prog_rc.n_instrs, bytes=prog_rc.n_bytes,
                                         set_config=prog_rc.opcode_counts()["SET_CONFIG"]),
            opcodes=dict(prog.opcode_counts()), kernels=len(prog.kernels),
            max_kernel_instrs=max(len(k.words) for k in prog.kernels), host_steps=len(host_steps)),
        encoding=enc,
        memory=dict(ub_peak_rows=mem.peak_rows, ub_peak_bytes=mem.peak_rows * 16,
                    ub_live_lower_bound_bytes=mem.live_peak_rows * 16,
                    ub_no_reuse_bytes=mem.noreuse_rows * 16, ub_no_reuse_peak_check=prog_nr.mem.peak_rows * 16,
                    buffers=len(mem.ub_base), ddr_staging_bytes=mem.ddr_bytes),
        fusion=fusion_stats(g, lg),
        totals=dict(macs=macs, array_cycles=cycles, ideal_cycles=macs / 256, util=macs / 256 / cycles,
                    array_ms=cycles / CLOCK_HZ * 1e3,
                    weight_bytes=sum(l["weight_bytes"] for l in layers), dma_act_bytes=dma_bytes,
                    dma_ms=dma_bytes / 16 / CLOCK_HZ * 1e3),
        partition=dict(table=partition_table(lg), rationale=passes.RATIONALE,
                       current=dict(host_steps=len(host_steps), kernels=len(prog.kernels), dma_bytes=dma_bytes),
                       future=dict(host_steps=0, kernels=1, dma_bytes=io_bytes,
                                   instrs_with_compact_encoding=enc["table_plus_loop"]["instrs"])),
        layers=layers, kernels=kernels,
    )
    return R


def _kib(b):
    return f"{b / KIB:,.1f} KiB"


def to_markdown(R):
    p, e, m, t, f = R["program"], R["encoding"], R["memory"], R["totals"], R["fusion"]
    L = []
    w = L.append
    w("# JEPA encoder on tpuv1 - schedule report\n")
    w("Generated by `python -m tpucc.report` from `jepa-tpu/qmodel.npy` (dim 128, 6 blocks, 4 heads, MLP 512, 64 tokens).")
    w("All 73 matmuls (+requant) of this schedule run bit-exact in tpuv1's ISA simulator (`tests/test_jepa.py`).\n")
    w("## Program size\n")
    w("| program | instructions | bytes | SET_CONFIG |")
    w("|---|---:|---:|---:|")
    b = p["baseline"]
    w(f"| baseline: 73 per-matmul programs (tpuv1 compile_mlp, as jepa-tpu rtl_check) | {b['instrs']:,} | {_kib(b['bytes'])} | {b['set_config']:,} |")
    w(f"| tpucc, one schedule, no SET_CONFIG dedupe | {p['no_dedupe']['instrs']:,} | {_kib(p['no_dedupe']['bytes'])} | {p['no_dedupe']['set_config']:,} |")
    w(f"| **tpucc, SET_CONFIG dedupe + parameter-set caching** | **{p['dedupe']['instrs']:,}** | **{_kib(p['dedupe']['bytes'])}** | {p['dedupe']['set_config']:,} |")
    rc = p["dedupe_reset_per_kernel"]
    w(f"| tpucc, dedupe, registers assumed lost at every kernel launch | {rc['instrs']:,} | {_kib(rc['bytes'])} | {rc['set_config']:,} |")
    q, lp = e["qparam_table"], e["table_plus_loop"]
    w(f"| proposal A: LOAD_QPARAMS + parameter table (estimate) | {q['instrs']:,} | {_kib(q['program_bytes'])} + table {_kib(q['table_bytes'])} | 0 |")
    w(f"| proposal B: A + LOOP (estimate) | {lp['instrs']:,} | {_kib(lp['program_bytes'])} + table {_kib(lp['table_bytes'])} | 0 |")
    w("")
    w(f"tpucc program: {p['kernels']} kernels (largest {p['max_kernel_instrs']} instructions; limit 2048), "
      f"{p['host_steps']} host steps. Opcodes: " + ", ".join(f"{k} {v:,}" for k, v in sorted(p["opcodes"].items(), key=lambda kv: -kv[1])) + ".")
    w("The baseline's DMA-free programs and ours are not identical in content: ours also carries the "
      "READ_HOST_MEM / WRITE_HOST_MEM / READ_WEIGHTS of the host partition. Dedupe removes the "
      "repeated requant registers of Q.K^T / P.V tiles (one scalar multiplier per layer) and the shared zp/min/max word; "
      "per-channel M0/bias of the linear layers change every tile, so they remain - the parameter table proposal removes them.\n")
    w("## Totals\n")
    w(f"- MACs {t['macs'] / 1e6:.1f} M; estimated array cycles {t['array_cycles']:,} (ideal {t['ideal_cycles']:,.0f}, "
      f"utilisation {100 * t['util']:.0f}%) = {t['array_ms']:.2f} ms at 100 MHz (RTL measured 4.1 ms matmul-busy with ROM weights).")
    w(f"- Weight bytes streamed {_kib(t['weight_bytes'])} (static + runtime-packed K^T/V); activation DMA {_kib(t['dma_act_bytes'])} "
      f"(~{t['dma_ms']:.2f} ms at 16 B/cycle).")
    w(f"- UB: peak {_kib(m['ub_peak_bytes'])} with liveness-based reuse (live-set lower bound {_kib(m['ub_live_lower_bound_bytes'])}), "
      f"vs {_kib(m['ub_no_reuse_bytes'])} without reuse; {m['buffers']} UB buffers; DDR staging {_kib(m['ddr_staging_bytes'])}.")
    w(f"- Fusion: {f['fused_ops']} matmul+requant(+act) ops; {f['ops_before']} IR ops -> {f['ops_after']}. "
      f"{_kib(f['int32_intermediate_bytes_kept_in_acc'])} of int32 accumulators never leave ACC; a host-side requant would "
      f"have cost {_kib(f['host_roundtrip_bytes_avoided'])} of DMA.\n")
    w("Cycle model: MATMUL streams 1 row/cycle; weight tile load 16 cycles, hidden behind a previous MATMUL of >= 16 rows; "
      "32-cycle drain at each SYNC on the matrix unit; ACTIVATE overlaps the array. DMA not included.\n")
    w("## Per-layer schedule\n")
    w("| op | kind | MxKxN | plan | kernel | instrs | SET_CONFIG | MATMUL | ACT | SYNC | RD_W | weight KiB | MACs | est. cycles | util |")
    w("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for l in R["layers"]:
        w(f"| {l['op']} | {l['kind']} | {l['shape']} | {l['plan']} | {l['kernel']} | {l['instrs']} | {l['set_config']} | {l['matmul']} | "
          f"{l['activate']} | {l['sync']} | {l['read_weights']} | {l['weight_bytes'] / KIB:.1f} | {l['macs']:,} | {l['array_cycles']:,} | {100 * l['util']:.0f}% |")
    w("\nPlan: `nk` = output-tile outer, K inner (weight-stationary: each tile loaded once and reused for all B rows), "
      "`B=64x1` = one M block of 64 rows. `kn` (K outer, NT live accumulators) was costed for every layer and never wins.\n")
    w("## Kernels\n")
    w("| kernel | instrs | RHM | WHM | ops |")
    w("|---:|---:|---:|---:|---|")
    for k in R["kernels"]:
        w(f"| {k['kernel']} | {k['instrs']} | {k['rhm']} | {k['whm']} | {', '.join(k['ops'])} |")
    w("\n## Partition plan\n")
    w("| op kind | count | elements | current tpuv1 | future (VECTOR opcode + UB->weight path) |")
    w("|---|---:|---:|---|---|")
    rat = R["partition"]["rationale"]
    for kind, r in sorted(R["partition"]["table"].items(), key=lambda kv: kv[1]["current"]):
        dflt = {"tpu": "systolic array + activation unit (fused requant)", "view": "zero-cost UB view (aliased buffer)"}
        cur, fut = rat[kind] if kind in rat else (dflt[r["current"]], dflt[r["future"]])
        w(f"| {kind} | {r['count']} | {r['elems']:,} | {r['current']}: {cur.split(': ', 1)[-1]} | {r['future']}: {fut.split(': ', 1)[-1]} |")
    pc, pf = R["partition"]["current"], R["partition"]["future"]
    w("")
    w(f"- Current: {pc['kernels']} TPU kernels separated by {pc['host_steps']} host steps; {_kib(pc['dma_bytes'])} of activation DMA "
      "(LN/softmax/GELU/residual inputs and outputs, K/V heads out for packing).")
    w(f"- Future: one program, no host steps, {_kib(pf['dma_bytes'])} DMA (image in, features out). GELU moves into the fc1 ACTIVATE "
      "(fc1->fc2 in one kernel), the residual add into ACTIVATE.eltwise or the VPU, LN/softmax into the VPU, and K^T/V come straight "
      f"from the UB. A single program only fits the 2048-entry instruction FIFO with the compact encoding (~{pf['instrs_with_compact_encoding']} instrs).")
    w("- P.V weights need no transpose: a V tile's weight word t is UB row (token 15-t) of V, so the UB->weight path only "
      "needs a reversed row order; K^T needs a 16x16 transpose (or a column-read mode).")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports"))
    a = ap.parse_args()
    R = build_report()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "jepa_schedule.md").write_text(to_markdown(R))
    (out / "jepa_schedule.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    p = R["program"]
    print(f"baseline {p['baseline']['bytes'] / KIB:.1f} KiB -> tpucc {p['dedupe']['bytes'] / KIB:.1f} KiB "
          f"({p['dedupe']['instrs']} instrs, {p['kernels']} kernels); UB peak {R['memory']['ub_peak_bytes'] / KIB:.1f} KiB; "
          f"report -> {out / 'jepa_schedule.md'}")


if __name__ == "__main__":
    main()
