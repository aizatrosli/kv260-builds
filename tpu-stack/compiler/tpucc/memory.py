"""Kernel segmentation, view aliasing, liveness and Unified-Buffer / DDR allocation."""
from dataclasses import dataclass, field

from .env import UB_ROWS
from .passes import device_plan

DDR_ACT = 0x8_1000_0000     # host-visible activations (DMA staging), host constants
DDR_W = 0x8_2000_0000       # static weight-tile images (could equally be a ROM)
DDR_WDYN = 0x8_4000_0000    # weight tiles packed at run time (K^T, V)


@dataclass
class Step:
    kind: str                     # "kernel" | "host"
    ops: list                     # op names
    index: int = 0
    first_pos: int = 0
    last_pos: int = 0


def segment(g):
    """Split the scheduled op list into alternating TPU kernels and host steps.
    View ops never break a kernel (they cost nothing). Returns (steps, pos, step_of)."""
    dev = device_plan(g)
    pos = {o.name: i for i, o in enumerate(g.ops)}
    steps, step_of = [], {}
    for o in g.ops:
        d = dev[o.name]
        if d == "view":
            continue
        kind = "kernel" if d == "tpu" else "host"
        if not steps or steps[-1].kind != kind:
            steps.append(Step(kind, [], len(steps), pos[o.name], pos[o.name]))
        st = steps[-1]
        st.ops.append(o.name)
        st.last_pos = pos[o.name]
        step_of[o.name] = st.index
    return steps, pos, step_of


@dataclass
class MemPlan:
    alias: dict                   # tensor -> (root, row offset)
    ub_base: dict                 # root -> base UB row
    intervals: dict               # root -> (start pos, end pos)
    rows: dict                    # root -> UB rows
    peak_rows: int                # allocator high-water mark
    live_peak_rows: int           # max over time of live rows (lower bound for any allocator)
    noreuse_rows: int             # sum of all buffers (no reuse)
    rhm: dict = field(default_factory=dict)    # step index -> [root] loaded at kernel start
    whm: dict = field(default_factory=dict)    # step index -> [tensor] written back at kernel end
    ddr: dict = field(default_factory=dict)    # tensor -> DDR byte address (host-visible copies)
    ddr_bytes: int = 0

    def ub_row(self, g, t):
        root, off = self.alias[t]
        return self.ub_base[root] + off


def resolve_aliases(g):
    parent = {}
    for o in g.ops:
        if o.kind == "slice":
            x = g.T(o.inputs[0])
            parent[o.outputs[0]] = (o.inputs[0], (o.attrs["c0"] // 16) * x.shape[0])
        elif o.kind == "flatten_planar":
            parent[o.outputs[0]] = (o.inputs[0], 0)
        elif o.kind == "concat":
            off = 0
            for i in o.inputs:
                if i in parent:
                    raise ValueError(f"{o.name}: concat input {i} is already a view")
                parent[i] = (o.outputs[0], off)
                off += g.T(i).ktiles * g.T(i).shape[0]
    alias = {}
    for t in g.tensors:
        root, off = t, 0
        while root in parent:
            root, d = parent[root]
            off += d
        alias[t] = (root, off)
    return alias


def plan_memory(g, steps, pos, step_of, reuse=True):
    dev = device_plan(g)
    alias = resolve_aliases(g)
    members = {}
    for t, (r, _) in alias.items():
        if not g.T(t).is_const:
            members.setdefault(r, []).append(t)
    prod = {}
    for o in g.ops:
        for t in o.outputs:
            prod[t] = o
    intervals, rows, rhm, whm, need_ddr = {}, {}, {}, {}, set(g.inputs) | set(g.outputs)
    for o in g.ops:
        if dev[o.name] == "host":
            need_ddr |= set(o.inputs) | set(o.outputs)
    for root, ts in members.items():
        writers = [prod[t] for t in ts if t in prod and dev[prod[t].name] == "tpu"]
        tpu_uses = [o for o in g.ops if dev[o.name] == "tpu" and o.inputs[0] in ts]
        if not writers and not tpu_uses:
            continue                                   # host-only tensor: lives in DDR only
        end_cands = [pos[o.name] for o in tpu_uses]
        if writers:
            start = min(pos[w.name] for w in writers)
            k_last = max(step_of[w.name] for w in writers)
            host_read = [t for t in ts if t in need_ddr]
            if host_read:
                end_cands.append(steps[k_last].last_pos)
                whm.setdefault(k_last, []).extend(host_read)
        else:
            k_first = min(step_of[o.name] for o in tpu_uses)
            start = steps[k_first].first_pos
            rhm.setdefault(k_first, []).append(root)
        intervals[root] = (start, max(end_cands + [start]))
        rows[root] = g.T(root).ub_rows
    # first-fit allocation in order of start position (offline linear scan)
    ub_base, active, peak = {}, [], 0
    for root in sorted(intervals, key=lambda r: (intervals[r][0], -rows[r])):
        s, e = intervals[root]
        if reuse:
            active = [a for a in active if intervals[a][1] >= s]
        cur = 0
        for a in sorted(active, key=lambda a: ub_base[a]):
            if ub_base[a] - cur >= rows[root]:
                break
            cur = max(cur, ub_base[a] + rows[a])
        ub_base[root] = cur
        active.append(root)
        peak = max(peak, cur + rows[root])
    if peak > UB_ROWS:
        raise MemoryError(f"UB overflow: {peak} rows > {UB_ROWS}")
    live = 0
    if intervals:
        T = max(e for _, e in intervals.values()) + 1
        live = max(sum(rows[r] for r, (s, e) in intervals.items() if s <= p <= e) for p in range(T))
    # DDR: host-visible activation copies (bump allocation, 16-byte rows)
    ddr, a = {}, DDR_ACT
    for t in sorted(need_ddr, key=lambda t: list(g.tensors).index(t)):
        if g.T(t).attrs.get("kind") == "weight_stream":   # lives in the weight-tile region
            continue
        ddr[t] = a
        a += g.T(t).ub_rows * 16
        a = (a + 255) // 256 * 256
    return MemPlan(alias, ub_base, intervals, rows, peak, live, sum(rows.values()), rhm, whm, ddr, a - DDR_ACT)
