"""tpucc - graph compiler / scheduler for the tpuv1 16x16 INT8 TPU (KV260).

Pipeline::

    g    = frontends.from_jepa(P)          # or from_mlp / from_cnn: unfused IR graph
    lg   = passes.lower(g)                 # fuse -> legalize (weight packing) -> list schedule
    prog = codegen.compile_graph(lg)       # tiling, UB allocation, tpuv1 ISA via asm.py
    run  = runtime.run_program(prog, {"img": x})   # tpuv1 sim.CoreSim + host steps
    ref  = refexec.run(g, {"img": x})      # numpy / jepa intops reference

``python -m tpucc.report`` writes reports/jepa_schedule.md (program size, per-layer schedule,
memory, partition plan). docs/compact_encoding.md proposes LOAD_QPARAMS + LOOP.
"""
