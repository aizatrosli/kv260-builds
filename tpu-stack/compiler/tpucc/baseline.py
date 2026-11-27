"""The 404 KiB baseline: jepa-tpu rtl_check.py builds one tpuv1 program per encoder matmul with
tpuv1's own MLP compiler (model/mlp/compile_mlp.emit_program), which writes all 33 requant
registers for every 16-column output tile. We regenerate exactly those programs here."""
import numpy as np

from . import env           # noqa: F401  (puts tpuv1 model/ on sys.path)
import compile_mlp as CM   # noqa: E402  tpuv1 model/mlp/compile_mlp.py


class _OneLayer:
    """Same duck-typed single-layer net as jepa-tpu rtl_check.OneLayer."""

    def __init__(self, K, Nout, bias, m0, shift):
        bc = lambda v: np.broadcast_to(np.asarray(v, np.int64), (Nout,))
        self.dims, self.acts = [K, Nout], ["none"]
        self.layers = [dict(wq=np.zeros((K, Nout), np.int64), bq=bc(bias), bhw=bc(bias),
                            m0=[int(v) for v in bc(m0)], shift=[int(v) for v in bc(shift)],
                            zp_out=0, act_min=-128, act_max=127)]


def baseline_programs(g):
    """g: any graph containing tpu_matmul ops (fused). Returns {op name: instruction words}."""
    out = {}
    for o in g.ops:
        if o.kind != "tpu_matmul":
            continue
        x, y = g.T(o.inputs[0]), g.T(o.outputs[0])
        net = _OneLayer(x.shape[1], y.shape[1], o.attrs["bias"], o.attrs["m0"], o.attrs["shift"])
        prog, *_ = CM.emit_program(net, x.shape[0])
        out[o.name] = prog
    return out


def baseline_size(g):
    progs = baseline_programs(g)
    n = sum(len(p) for p in progs.values())
    return dict(programs=len(progs), instrs=n, bytes=16 * n,
                set_config=sum(1 for p in progs.values() for w in p if w & 0xFF == 0x30),
                per_op={k: len(v) for k, v in progs.items()})
