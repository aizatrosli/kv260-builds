"""Per-op timing with a host / device split.

Every host step and every device job is recorded as (op, category, wall seconds).  Device jobs also
carry the kernel-measured RUN->IRQ time (wall_ns) and the hardware perf counters, so the ioctl /
driver overhead = round trip - kernel wall is visible.  report() aggregates per op.
"""
import time
from collections import OrderedDict
from contextlib import contextmanager

PERF_KEYS = ("total", "array_active", "w_stall", "w_shift", "non_matrix", "sync", "act", "retired")


class Profiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.ops = OrderedDict()     # op -> dict(cat, n, t, dev_ns, perf{})
        self.images = 0
        self.wall = 0.0

    def _rec(self, op, cat):
        r = self.ops.get(op)
        if r is None:
            r = self.ops[op] = dict(cat=cat, n=0, t=0.0, dev_ns=0, instr=0, perf={k: 0 for k in PERF_KEYS})
        return r

    @contextmanager
    def span(self, op, cat="host"):
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            r = self._rec(op, cat)
            r["n"] += 1
            r["t"] += time.perf_counter() - t

    def device(self, op, seconds, result, n_instr=0):
        if not self.enabled:
            return
        r = self._rec(op, "device")
        r["n"] += 1
        r["t"] += seconds
        r["dev_ns"] += result.wall_ns
        r["instr"] += n_instr
        for k, v in (result.perf or {}).items():
            if k in r["perf"]:
                r["perf"][k] += v

    def summary(self):
        host = sum(r["t"] for r in self.ops.values() if r["cat"] == "host")
        xfer = sum(r["t"] for r in self.ops.values() if r["cat"] == "xfer")
        dev_rt = sum(r["t"] for r in self.ops.values() if r["cat"] == "device")
        dev_k = sum(r["dev_ns"] for r in self.ops.values() if r["cat"] == "device") / 1e9
        runs = sum(r["n"] for r in self.ops.values() if r["cat"] == "device")
        n = max(self.images, 1)
        perf = {k: sum(r["perf"][k] for r in self.ops.values() if r["cat"] == "device") for k in PERF_KEYS}
        return dict(images=self.images, wall_s=self.wall, ms_per_image=1e3 * self.wall / n,
                    img_per_s=self.images / self.wall if self.wall else 0.0,
                    host_ms_per_image=1e3 * host / n, xfer_ms_per_image=1e3 * xfer / n,
                    device_roundtrip_ms_per_image=1e3 * dev_rt / n, device_kernel_ms_per_image=1e3 * dev_k / n,
                    driver_overhead_ms_per_image=1e3 * (dev_rt - dev_k) / n, runs_per_image=runs / n,
                    overlap=(host + xfer + dev_rt) / self.wall if self.wall else 0.0,
                    perf_per_image={k: v / n for k, v in perf.items()})

    def report(self, title="", estimated=False):
        n = max(self.images, 1)
        tot = sum(r["t"] for r in self.ops.values()) or 1e-12
        lines = [title] if title else []
        lines.append(f"{'op':28s} {'cat':6s} {'calls/img':>9s} {'ms/img':>9s} {'%':>6s} {'kernel ms/img':>13s} "
                     f"{'instr/run':>9s} {'array_util':>10s}")
        for op, r in self.ops.items():
            util = (f"{r['perf']['array_active'] / r['perf']['total']:.2f}" if r["perf"]["total"] else "")
            kern = f"{r['dev_ns'] / 1e6 / n:13.3f}" if r["cat"] == "device" else " " * 13
            ipr = f"{r['instr'] / r['n']:9.0f}" if r["cat"] == "device" and r["n"] else " " * 9
            lines.append(f"{op:28s} {r['cat']:6s} {r['n'] / n:9.2f} {1e3 * r['t'] / n:9.3f} {100 * r['t'] / tot:6.1f} "
                         f"{kern} {ipr} {util:>10s}")
        s = self.summary()
        est = " (device time ESTIMATED by the simulator's cycle model)" if estimated else ""
        lines.append(f"images={s['images']} wall={s['wall_s']:.3f}s -> {s['ms_per_image']:.3f} ms/img, "
                     f"{s['img_per_s']:.1f} img/s")
        lines.append(f"host ops {s['host_ms_per_image']:.3f} ms/img | host<->DMA copies {s['xfer_ms_per_image']:.3f} | "
                     f"device round trip {s['device_roundtrip_ms_per_image']:.3f} (kernel {s['device_kernel_ms_per_image']:.3f}, "
                     f"driver/ioctl overhead {s['driver_overhead_ms_per_image']:.3f}) | runs/img {s['runs_per_image']:.1f} | "
                     f"overlap x{s['overlap']:.2f}{est}")
        return "\n".join(lines)
