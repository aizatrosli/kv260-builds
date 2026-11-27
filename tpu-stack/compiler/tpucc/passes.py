"""Graph passes: fusion, legalisation for tpuv1, partitioning and list scheduling."""
import copy

from .ir import Graph, Op

ACT_FUNC = {None: 0, "none": 0, "relu": 1, "relu6": 2}


def _single_consumer(g, t, kind):
    cs = g.consumers(t)
    if len(cs) == 1 and cs[0].kind == kind and t not in g.outputs:
        return cs[0]
    return None


def fuse(g):
    """matmul|conv2d -> requant -> [clamp] -> [pool2]  ==>  tpu_matmul|tpu_conv.

    The int32 accumulator never leaves the ACC memory and the int8 result is written straight
    into the UB by ACTIVATE, so no intermediate tensor needs a DMA round trip.
    Returns a new graph; the input graph is unchanged."""
    g = copy.deepcopy(g)
    out_ops = []
    dead = set()
    for op in g.ops:
        if op.name in dead:
            continue
        if op.kind not in ("matmul", "conv2d"):
            out_ops.append(op)
            continue
        rq = _single_consumer(g, op.outputs[0], "requant")
        if rq is None:
            out_ops.append(op)
            continue
        a = dict(op.attrs)
        a.update({k: rq.attrs[k] for k in ("bias", "m0", "shift", "zp_out", "act_min", "act_max")})
        a["fused"] = [op.name, rq.name]
        a["act"] = None
        dead.add(rq.name)
        drop = [op.outputs[0]]
        last = rq
        cl = _single_consumer(g, last.outputs[0], "clamp")
        if cl is not None:
            a["act_min"] = max(a["act_min"], cl.attrs["lo"])
            a["act_max"] = min(a["act_max"], cl.attrs["hi"])
            a["act"] = cl.attrs.get("act", "clamp")
            a["fused"].append(cl.name)
            dead.add(cl.name)
            drop.append(last.outputs[0])
            last = cl
        if op.kind == "conv2d":
            a["out_hw"] = _conv_out_hw(a)
            pl = _single_consumer(g, last.outputs[0], "pool2")
            if pl is not None:
                a["pool"] = pl.attrs["mode"]
                a["fused"].append(pl.name)
                dead.add(pl.name)
                drop.append(last.outputs[0])
                last = pl
        fused = Op(op.name, "tpu_matmul" if op.kind == "matmul" else "tpu_conv", list(op.inputs),
                   list(last.outputs), a)
        for t in drop:
            del g.tensors[t]
        out_ops.append(fused)
    g.ops = out_ops
    g.validate()
    return g


def _conv_out_hw(a):
    H, W = a["hw"]
    k, s, p = a["k"], a["stride"], a["pad"]
    return (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1


def legalize(g):
    """Make the graph executable on the current tpuv1:
    * every matmul/conv must be fused with a requant (tpuv1 has no int32 UB output in Phase A);
    * an activation x activation matmul (weights not a constant, e.g. Q.K^T, P.V) gets a host
      ``pack_weights`` op that turns the runtime tensor into a weight-tile stream in DDR
      (with the transpose for K^T). tpuv1 can only load weights through the Weight FIFO."""
    g = copy.deepcopy(g)
    ops = []
    for op in g.ops:
        if op.kind in ("matmul", "conv2d", "requant", "clamp", "pool2"):
            raise ValueError(f"{op.name}: unfused {op.kind} cannot run on tpuv1 (run passes.fuse first)")
        if op.kind == "tpu_matmul" and not g.T(op.inputs[1]).is_const:
            w = g.T(op.inputs[1])
            tr = bool(op.attrs.get("transpose_b"))
            shape = (w.shape[1], w.shape[0]) if tr else w.shape
            pk = g.tensor(f"{w.name}.wpack", shape, "int8", kind="weight_stream")
            ops.append(Op(f"pack_{op.name}", "pack_weights", [w.name], [pk], dict(transpose=tr)))
            op.inputs[1] = pk
            op.attrs["transpose_b"] = False
            op.attrs["dynamic_weights"] = True
        ops.append(op)
    g.ops = ops
    g.validate()
    return g


# ---------------------------------------------------------------------------------------------
# partitioning
# ---------------------------------------------------------------------------------------------
PLAN_CURRENT = {"tpu_matmul": "tpu", "tpu_conv": "tpu", "layernorm": "host", "softmax": "host",
                "lut": "host", "add": "host", "pack_weights": "host",
                "slice": "view", "concat": "view", "flatten_planar": "view"}
PLAN_FUTURE = {"tpu_matmul": "tpu", "tpu_conv": "tpu", "layernorm": "vpu", "softmax": "vpu",
               "lut": "act_lut", "add": "vpu", "pack_weights": "ub2w",
               "slice": "view", "concat": "view", "flatten_planar": "view"}
RATIONALE = {
    "layernorm": ("host: needs row sum/sum-of-squares, isqrt, reciprocal",
                  "VPU (VECTOR opcode): 1 pass sum/sumsq + isqrt + recip + 49-bit multiply per lane"),
    "softmax": ("host: row max, exp LUT, row sum, reciprocal",
                "VPU: max-reduce, 256-entry exp LUT, sum, reciprocal; output 0..127 int8"),
    "lut": ("host: int8->int8 table (GELU)",
            "activation_unit LUT func (reserved sigmoid/tanh codes): fuses into the fc1 ACTIVATE, no extra pass"),
    "add": ("host: sat8((a*Ma + b*Mb + 2^15) >> 16)",
            "VPU 2-operand mode or ACTIVATE.eltwise (reserved field) with ub_operand_row"),
    "pack_weights": ("host: UB->DDR, (transpose), tile pack, READ_WEIGHTS back",
                     "UB->weight-loader path (+ transpose for K^T), MATMUL flag selects it; no DDR trip"),
}


def device_plan(g, target="current"):
    table = PLAN_CURRENT if target == "current" else PLAN_FUTURE
    return {op.name: table[op.kind] for op in g.ops}


# ---------------------------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------------------------
def schedule(g, target="current"):
    """List scheduling that keeps same-device ops together (fewer TPU kernels / host trips).

    Ready views are placed immediately (zero cost). Otherwise stay on the current device while
    it has ready work; on a switch pick the device of the earliest ready op in program order.
    Returns a new graph whose op list is the schedule."""
    g = copy.deepcopy(g)
    dev = device_plan(g, target)
    avail = {n for n, t in g.tensors.items() if t.is_const} | set(g.inputs)
    pending = list(g.ops)
    order, cur = [], None
    while pending:
        ready = [o for o in pending if all(i in avail for i in o.inputs)]
        if not ready:
            raise ValueError("cycle in graph")
        views = [o for o in ready if dev[o.name] == "view"]
        if views:
            pick = views[0]
        else:
            same = [o for o in ready if dev[o.name] == cur]
            pick = same[0] if same else ready[0]
            cur = dev[pick.name]
        order.append(pick)
        pending.remove(pick)
        avail |= set(pick.outputs)
    g.ops = order
    g.validate()
    return g


def lower(g):
    """fuse -> legalize -> schedule (the standard pipeline for tpuv1)."""
    return schedule(legalize(fuse(g)))
