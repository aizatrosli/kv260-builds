"""Stable export format: <dir>/manifest.json + <dir>/weights.npz.  Schema: README.md."""
import json
import os
import numpy as np

from .graph import Graph, Node, TensorInfo
from . import intops as I
from .int4 import pack_int4, pack_uint4

FORMAT = "tpuquant-graph"
VERSION = 1


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def save(graph, path, extra=None):
    os.makedirs(path, exist_ok=True)
    arrays = {}

    def put(key, arr, dtype=None):
        arr = np.asarray(arr)
        if dtype:
            arr = arr.astype(dtype)
        arrays[key] = arr
        return dict(key=key, dtype=str(arr.dtype), shape=list(arr.shape))

    tensors = {}
    for name, t in graph.tensors.items():
        e = dict(shape=t.shape, dtype=t.dtype, zero_point=int(t.zp), range=[t.act_min, t.act_max])
        e["scale"] = float(t.scale) if t.uniform else put(f"tensor/{name}/scale", t.scale, np.float64)
        tensors[name] = e
    nodes = []
    for n in graph.nodes:
        params = {k: put(f"{n.name}/{k}", v) for k, v in n.params.items()}
        if "w4" in n.params:                         # packed nibbles along the output-channel axis
            params["w4_packed"] = put(f"{n.name}/w4_packed", pack_int4(n.params["w4"], axis=-1))
        if "gmul" in n.params:
            params["gmul_packed"] = put(f"{n.name}/gmul_packed", pack_uint4(n.params["gmul"].astype(np.int64) - 1, -1))
        nodes.append(dict(name=n.name, op=n.op, inputs=list(n.inputs), output=n.output,
                          attrs=_jsonable(n.attrs), params=params))
    outputs = []
    for o in graph.outputs:
        post = graph.output_post.get(o)
        outputs.append(dict(name=o, post=None if post is None else
                            dict(scale=put(f"output/{o}/post_scale", post["scale"], np.float64),
                                 bias=put(f"output/{o}/post_bias", post["bias"], np.float64))))
    man = dict(format=FORMAT, version=VERSION, name=graph.name,
               arith=dict(requant="tflite_double_rounding (tpuv1 tflite_int8.py)",
                          input_rounding="round_half_to_even", constants=I.CONSTANTS),
               inputs=[dict(name=i, shape=graph.tensors[i].shape, dtype="float32") for i in graph.inputs],
               outputs=outputs, tensors=tensors, nodes=nodes,
               stats=dict(weight_bytes=graph.weight_bytes(), n_nodes=len(graph.nodes)),
               meta=_jsonable({k: v for k, v in graph.meta.items()}))
    if extra:
        man["extra"] = _jsonable(extra)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    np.savez_compressed(os.path.join(path, "weights.npz"), **arrays)
    return man


def load(path):
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    if man.get("format") != FORMAT or man.get("version") != VERSION:
        raise ValueError(f"unsupported export {man.get('format')} v{man.get('version')}")
    z = np.load(os.path.join(path, "weights.npz"))
    get = lambda ref: z[ref["key"]]
    g = Graph(man["name"])
    for name, e in man["tensors"].items():
        sc = e["scale"] if isinstance(e["scale"], float) else get(e["scale"])
        g.add_tensor(TensorInfo(name, e["shape"], sc, e["zero_point"], e["dtype"], *e["range"]))
    for nd in man["nodes"]:
        params = {k: get(r) for k, r in nd["params"].items() if k not in ("w4_packed", "gmul_packed")}
        g.add_node(Node(nd["name"], nd["op"], nd["inputs"], nd["output"], nd["attrs"], params))
    g.inputs = [i["name"] for i in man["inputs"]]
    for o in man["outputs"]:
        g.outputs.append(o["name"])
        if o["post"]:
            g.output_post[o["name"]] = dict(scale=get(o["post"]["scale"]), bias=get(o["post"]["bias"]))
    g.meta = man.get("meta", {})
    return g
