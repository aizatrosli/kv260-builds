"""Generic executor for a step graph (tpurt.program): owns the DMA pool, the resident constants,
the compiled schedule and N double-buffer contexts; runs batches synchronously or pipelined.

Subclasses (JepaExecutor, the smoke-test MLP) build `steps` and implement set_input(ctx, item);
a final host step stores ctx.result.
"""
import queue
import threading
import time
from collections import deque

import numpy as np

from .backends.base import TpuRunError
from .buffers import DmaPool, FirstFit, WeightStore
from .hostops import get_ops
from .profiler import Profiler
from .program import schedule


class Context:
    """One in-flight batch: its own DMA blocks / host arrays and prepared jobs (double buffering)."""

    def __init__(self, ex, idx):
        self.ex, self.idx = ex, idx
        self.blocks, self.host = {}, {}
        # Liveness-based reuse: a tensor's block is returned to the pool after its last step, so
        # later tensors of the same size (the next layer's) reuse the memory.  Steps run in order
        # within a context, so disjoint [first, last] step ranges never alias in time.
        life = {}
        for i, st in enumerate(ex.steps):
            for t in st.reads() + st.writes():
                life[t] = (life.get(t, (i, i))[0], i)
        need = sorted((t for t in ex.dev_tensors if t.kind != "const"), key=lambda t: (life[t][0], t.name))
        layout, live = FirstFit(1 << 40, align=256), []
        offs = {}
        for t in need:
            start = life[t][0]
            for u in [u for u in live if life[u][1] < start and u.kind != "out"]:
                layout.free(offs[u])
                live.remove(u)
            offs[t] = layout.alloc(t.nbytes, 256)
            live.append(t)
        # one pool block per context, carved at the planned offsets (contexts never share memory)
        self.region = ex.pool.alloc(max(layout.hwm, 256), align=4096, name=f"ctx{idx}") if need else None
        for t in need:
            self.blocks[t] = self.region.sub(offs[t], t.nbytes, name=f"c{idx}:{t.name}")
        self.planned_bytes = layout.hwm
        self.unshared_bytes = sum(t.nbytes for t in need)
        for t in ex.host_tensors:
            shape = getattr(t, "shape", (t.rows, 16))
            self.host[t] = np.asarray(t.data).reshape(shape) if t.kind == "const" else np.zeros(shape, np.int8)
        self.jobs = []
        for prog in ex.programs:
            rel = []
            for i, t, off in prog.relocs:
                blk = ex.consts.get(t.name) if t.kind == "const" else self.blocks[t]
                rel.append((i, *blk.reloc(off)))
            self.jobs.append(ex.backend.prepare(prog.to_bytes(), rel, label=prog.label()))
        self.input, self.result, self.n_valid = None, None, 0

    def read(self, t):
        """Cached (host-memory) copy of a tensor: one sequential copy out of the DMA block (the
        driver's buffers are non-cacheable; never compute on them element by element)."""
        if t in self.host:
            return self.host[t]
        blk = self.blocks[t]
        with self.ex.prof.span("dma_read", "xfer"):
            blk.invalidate()
            return blk.np(np.int8, count=t.nbytes).reshape(getattr(t, "shape", (t.rows, 16))).copy()

    def out(self, t):
        """Writable array for a tensor: a host array or a zero-copy view of its DMA block."""
        if t in self.host:
            return self.host[t]
        return self.blocks[t].np(np.int8, count=t.nbytes).reshape(getattr(t, "shape", (t.rows, 16)))

    def done(self, t):
        """Host finished writing t: cache flush hook (no-op on coherent memory)."""
        if t in self.blocks:
            self.blocks[t].flush()

    def free(self):
        if self.region is not None:
            self.region.free()


class GraphExecutor:
    def __init__(self, backend, steps, contexts=2, arena_size=None, profiler=None, rw_group=False,
                 limit=2048, timeout_ms=0, ops="auto"):
        self.backend = backend
        self.prof = profiler or Profiler()
        self.timeout_ms = timeout_ms
        if not hasattr(self, "ops"):
            self.ops = get_ops(ops)
        self.pool = DmaPool(backend, arena_size=arena_size or backend.preferred_arena)
        self.consts = WeightStore(self.pool)
        self.steps = steps
        with self.prof.span("compile", "setup"):
            self.items = schedule(steps, limit=limit, rw_group=rw_group)
        self.programs = [it for k, it in self.items if k == "dev"]
        in_prog = set()
        for p in self.programs:
            in_prog |= {t for _, t, _ in p.relocs}
        every = {t for st in steps for t in st.reads() + st.writes()}
        self.dev_tensors = sorted((t for t in every if t in in_prog or t.kind == "out"), key=lambda t: t.name)
        self.host_tensors = sorted((t for t in every if t not in self.dev_tensors), key=lambda t: t.name)
        with self.prof.span("weights_load", "setup"):
            for t in self.dev_tensors:
                if t.kind == "const":
                    self.consts.put(t.name, t.data)          # resident: written once, shared by all
        self.contexts = [Context(self, i) for i in range(contexts)]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------- run
    def set_input(self, ctx, item):
        ctx.input, ctx.n_valid = item, len(item)

    def _gen(self, ctx):
        k = 0
        for kind, it in self.items:
            if kind == "host":
                it.fn(ctx)
            else:
                yield k
                k += 1

    def _run_dev(self, ctx, k):
        job = ctx.jobs[k]
        t = time.perf_counter()
        res = self.backend.run_job(job, self.timeout_ms)
        dt = time.perf_counter() - t
        with self._lock:
            self.prof.device(job.label, dt, res, job.n_instr)
        if not res.ok:
            raise TpuRunError(res, f"{job.label} (program {k}, {job.n_instr} instructions)")
        for t_ in self.programs[k].stores:
            if t_ in ctx.blocks:
                ctx.blocks[t_].invalidate()                 # device wrote: cache invalidate hook
        return res

    def run_one(self, item, n_items=1):
        ctx = self.contexts[0]
        t = time.perf_counter()
        self.set_input(ctx, item)
        for k in self._gen(ctx):
            self._run_dev(ctx, k)
        self.prof.wall += time.perf_counter() - t
        self.prof.images += n_items
        return ctx.result

    def run_pipelined(self, items, counts):
        """Run `items` (one per batch) over the double-buffer contexts: host steps of one batch run on
        this thread while the TPU job of another runs on one worker thread (ioctl and the C kernels
        release the GIL).  Plain SimpleQueues keep the hand-off cheap (~10 us)."""
        t0 = time.perf_counter()
        results = [None] * len(items)
        todo, done = queue.SimpleQueue(), queue.SimpleQueue()

        def worker():
            while True:
                job = todo.get()
                if job is None:
                    return
                ctx, k = job
                try:
                    self._run_dev(ctx, k)
                    done.put((ctx, None))
                except BaseException as e:            # propagate to the main thread
                    done.put((ctx, e))

        th = threading.Thread(target=worker, name="tpu-jobs", daemon=True)
        th.start()
        free, active, nxt, pending = deque(self.contexts), {}, 0, 0
        try:
            while nxt < len(items) or active:
                while free and nxt < len(items):
                    ctx = free.popleft()
                    self.set_input(ctx, items[nxt])
                    active[ctx] = [self._gen(ctx), nxt, False]
                    nxt += 1
                for ctx, st in list(active.items()):
                    if not st[2]:
                        try:
                            todo.put((ctx, next(st[0])))
                            st[2] = True
                            pending += 1
                        except StopIteration:
                            results[st[1]] = ctx.result
                            del active[ctx]
                            free.append(ctx)
                if pending:
                    ctx, err = done.get()
                    pending -= 1
                    if err is not None:
                        raise err
                    active[ctx][2] = False
        finally:
            while pending:
                done.get()
                pending -= 1
            todo.put(None)
            th.join()
        self.prof.wall += time.perf_counter() - t0
        self.prof.images += sum(counts)
        return results

    def describe(self):
        host = [it for k, it in self.items if k == "host"]
        return (f"  {len(self.programs)} TPU programs/batch ({sum(p.n_instr for p in self.programs)} instructions, max "
                f"{max(p.n_instr for p in self.programs)}), {len(host)} host steps/batch\n"
                f"  resident constants {self.consts.bytes / 1024:.0f} KiB in {len(self.consts.blocks)} blocks "
                f"(loaded once); activations {self.contexts[0].planned_bytes / 1024:.0f} KiB/context with "
                f"liveness reuse (vs {self.contexts[0].unshared_bytes / 1024:.0f} KiB); pool {self.pool.stats()}")

    def close(self):
        for c in self.contexts:
            c.free()
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
