"""End-to-end executor for the tiny I-JEPA ViT encoder (jepa-tpu: dim 128, 6 blocks, 4 heads, 64 tokens).

    be = tpurt.get_backend("board")            # or "sim" / "mock"
    ex = JepaExecutor(be, batch=4)             # weights packed + DMA'd once (resident)
    y = ex.forward_q(img_q)                    # int8 [n, 64, 128], bit-exact with quantize.int_forward
    print(ex.prof.report())

Matmuls (embed, qkv, Q.K^T, P.V, proj, fc1, fc2) run on the TPU with tpuv1's ACTIVATE requant.
The attention matmuls use activations as weights: today the host repacks K^T / V into weight-tile
order (pack_kv) and the TPU streams them with READ_WEIGHTS (no UB->weight path yet).

vector="host": LayerNorm / softmax / GELU / residual run on the host (C kernels or numpy, bit-exact
with intops) between TPU programs (5 programs + 6 host calls per layer).
vector="vpu":  they become VECTOR instructions (jepa_top sequencer extension X_VEC/X_LUT/X_VCFG, tpurt/vpu.py) fused into the TPU
programs; only pack_kv stays on the host.  Runs on the sim/mock backends today and on a bitstream
with the VPU once its encoding is confirmed.

Batching: `batch` images share one program per matmul layer (M = 64*batch rows, <= 1024); weights
are streamed once per batch.  Double buffering: `contexts` (default 2) independent sets of DMA
buffers; run_many(pipeline=True) overlaps host ops of one batch with the TPU job of the other.
"""
import numpy as np

from . import vpu
from .executor import GraphExecutor
from .data import load_qmodel, quant_img
from .hostops import get_ops, tile, untile
from .profiler import Profiler
from .program import Gemm, Host, Tensor, Vec, pack_weight_tiles, split_gemm_nt

TOK = 64


class JepaExecutor(GraphExecutor):
    def __init__(self, backend, P=None, batch=1, vector="host", ops="auto", contexts=2, arena_size=None,
                 profiler=None, rw_group=False, limit=2048, timeout_ms=0):
        if not 1 <= batch <= 16:
            raise ValueError("batch must be 1..16 (M = 64*batch rows must fit one ACC half)")
        if vector not in ("host", "vpu"):
            raise ValueError("vector must be 'host' or 'vpu'")
        if vector == "vpu" and not backend.has_vpu:
            raise RuntimeError(f"backend {backend.name} has no VPU (VECTOR opcode); use vector='host'")
        self.P = P if P is not None else load_qmodel()
        c = self.P["cfg"]
        self.D, self.H, self.L, self.F = c["dim"], c["heads"], c["depth"], c["mlp"]
        if vector == "vpu" and hasattr(backend, "lut_rom"):
            # Simulated backends: supply the LUT ROM. On a bitstream it is baked in (lrom.hex) and must
            # hold the same tables in compile_jepa's layout (vpu.lrom_hex(vpu.jepa_lut_rom(P, L))).
            backend.lut_rom = vpu.jepa_lut_rom(self.P, self.L)
        self.dh, self.B, self.M = self.D // self.H, batch, TOK * batch
        self.vector, self.ops = vector, get_ops(ops)
        self.prof = profiler or Profiler()
        with self.prof.span("graph", "setup"):
            steps = self._graph()
        super().__init__(backend, steps, contexts=contexts, arena_size=arena_size, profiler=self.prof,
                         rw_group=rw_group, limit=limit, timeout_ms=timeout_ms)

    # ------------------------------------------------------------------------------------- graph
    def _graph(self):
        P, D, H, L, F, dh, B, M = self.P, self.D, self.H, self.L, self.F, self.dh, self.B, self.M
        T, TF, KTd = D // 16, F // 16, self.dh // 16
        vp = self.vector == "vpu"
        steps = []

        def act(name, tiles, kind="act"):
            t = Tensor(name, tiles * M, kind)
            t.shape = (tiles, M, 16)
            return t

        def const(name, data, shape):
            raw = np.ascontiguousarray(data)
            t = Tensor(name, raw.nbytes // 16, "const", raw)
            t.shape = shape
            return t

        def weights(name, W):
            return const(name, np.frombuffer(pack_weight_tiles(W), np.int8), (W.shape[0] * W.shape[1] // 16, 16))

        def lin(name, op, x, KT, out, NT, w, p):
            for g in split_gemm_nt(Gemm(name, x, 0, M, KT, M, out, 0, M, NT, w, 0, np.asarray(p["m0"]),
                                        np.asarray(p["shift"]), np.asarray(p["bias"]), op=op), 24):
                steps.append(g)

        ops = self.ops
        # ---- vector-op emitters (host step or VECTOR instruction) ----------------------------
        def residual(name, a, b, out, mult, then_ln=None, ln_out=None):
            if vp:
                steps.append(Vec(name, vpu.V_ADD, a, 0, out, 0, M, T, M, b=b, mult=mult, op="residual"))
                if then_ln:
                    steps.append(Vec(name + ".ln", vpu.V_LN, out, 0, ln_out, 0, M, T, M, op="layernorm"))
                return

            def fn(ctx, a=a, b=b, out=out, ln_out=ln_out, mult=mult):
                av, bv = ctx.read(a), ctx.read(b)
                with ctx.ex.prof.span("residual"):
                    xo = ops.ew_add(av, bv, *mult, out=ctx.out(out))
                ctx.done(out)
                if ln_out is not None:
                    with ctx.ex.prof.span("layernorm"):
                        ops.layernorm(xo, out=ctx.out(ln_out))
                    ctx.done(ln_out)
            steps.append(Host(name, fn, [a, b], [out] + ([ln_out] if ln_out is not None else []), op="residual"))

        # ---- tensors / constants ----------------------------------------------------------------
        img = act("img", 1)
        w_embed = weights("w_embed", P["embed"]["w"])
        pos = const("pos", np.tile(tile(P["pos"].astype(np.int8)), (1, B, 1)), (T, M, 16))

        def load_input(ctx):
            with ctx.ex.prof.span("input_pack"):
                o = ctx.out(img)
                o[0] = ctx.input.reshape(M, 16)
            ctx.done(img)
        steps.append(Host("input", load_input, [], [img], op="input"))
        e = act("e", T)
        lin("embed", "embed", img, 1, e, T, w_embed, P["embed"])
        x = act("x0", T)
        h = act("h0", T)
        residual("add0", e, pos, x, P["add0"], then_ln=True, ln_out=h)

        for l in range(L):
            qkv = act(f"qkv{l}", 3 * T)
            lin(f"qkv{l}", "qkv", h, T, qkv, 3 * T, weights(f"w_qkv{l}", P[f"qkv{l}"]["w"]), P[f"qkv{l}"])
            kT = Tensor(f"kT{l}", B * H * 4 * KTd * 16)
            kT.shape = (B * H, 4, KTd, 16, 16)
            vw = Tensor(f"vw{l}", B * H * KTd * 4 * 16)
            vw.shape = (B * H, KTd, 4, 16, 16)

            def pack(ctx, qkv=qkv, kT=kT, vw=vw):
                q = ctx.read(qkv)
                with ctx.ex.prof.span("pack_kv"):
                    ops.pack_kv(q, B, H, kT_out=ctx.out(kT), v_out=ctx.out(vw))
                ctx.done(kT)
                ctx.done(vw)
            steps.append(Host(f"pack_kv{l}", pack, [qkv], [kT, vw], op="pack_kv"))
            s = Tensor(f"s{l}", B * H * 4 * TOK)
            s.shape = (B * H, 4, TOK, 16)
            p = Tensor(f"p{l}", B * H * 4 * TOK)
            p.shape = s.shape
            m0, sh = (int(np.ravel(v)[0]) for v in P[f"qk{l}"])
            for b in range(B):
                for hh in range(H):
                    g = b * H + hh
                    steps.append(Gemm(f"qk{l}.{b}.{hh}", qkv, 2 * hh * M + b * TOK, M, KTd, TOK, s, g * 4 * TOK, TOK, 4,
                                      kT, g * 4 * KTd * 256, np.full(64, m0), np.full(64, sh), np.zeros(64, int),
                                      x_load=(0, T * M), op="qk"))
            sq = int(P.get(f"smq{l}", vpu.SQ_DEFAULT))     # per-layer softmax numerator (1/q scale)
            if vp:
                lut_e = vpu.lut_idx(vpu.LUT_EXP, l, L)
                for g in range(B * H):
                    steps.append(Vec(f"sm{l}.{g}", vpu.V_SM, s, g * 4 * TOK, p, g * 4 * TOK, TOK, 4, TOK,
                                     lut_idx=lut_e, sq=sq, op="softmax"))
            else:
                def smx(ctx, s=s, p=p, lut=P[f"exp{l}"], sq=sq):
                    sv = ctx.read(s)
                    with ctx.ex.prof.span("softmax"):
                        ops.softmax(sv, lut, out=ctx.out(p), q=sq)
                    ctx.done(p)
                steps.append(Host(f"softmax{l}", smx, [s], [p], op="softmax"))
            o = act(f"o{l}", T)
            m0, sh = (int(np.ravel(v)[0]) for v in P[f"pv{l}"])
            for b in range(B):
                for hh in range(H):
                    g = b * H + hh
                    steps.append(Gemm(f"pv{l}.{b}.{hh}", p, g * 4 * TOK, TOK, 4, TOK, o, 2 * hh * M + b * TOK, M, KTd,
                                      vw, g * KTd * 4 * 256, np.full(32, m0), np.full(32, sh), np.zeros(32, int),
                                      op="pv"))
            a = act(f"a{l}", T)
            lin(f"proj{l}", "proj", o, T, a, T, weights(f"w_proj{l}", P[f"proj{l}"]["w"]), P[f"proj{l}"])
            xa, h2 = act(f"xa{l}", T), act(f"h2_{l}", T)
            residual(f"adda{l}", x, a, xa, P[f"adda{l}"], then_ln=True, ln_out=h2)
            h1 = act(f"h1_{l}", TF)
            gg = h1 if vp else act(f"g{l}", TF)          # VPU: GELU in place (UB holds 64K rows)
            lin(f"fc1{l}", "fc1", h2, T, h1, TF, weights(f"w_fc1{l}", P[f"fc1{l}"]["w"]), P[f"fc1{l}"])
            if vp:
                steps.append(Vec(f"gelu{l}", vpu.V_LUT, h1, 0, h1, 0, M, TF, M,
                                 lut_idx=vpu.lut_idx(vpu.LUT_GELU, l, L), op="gelu"))
            else:
                def gelu(ctx, h1=h1, gg=gg, lut=P[f"gelu{l}"]):
                    hv = ctx.read(h1)
                    with ctx.ex.prof.span("gelu"):
                        ops.lut8(hv, lut, out=ctx.out(gg))
                    ctx.done(gg)
                steps.append(Host(f"gelu{l}", gelu, [h1], [gg], op="gelu"))
            f2 = act(f"f{l}", T)
            lin(f"fc2{l}", "fc2", gg, TF, f2, T, weights(f"w_fc2{l}", P[f"fc2{l}"]["w"]), P[f"fc2{l}"])
            last = l == L - 1
            xn = act(f"x{l + 1}", T)
            hn = act("y" if last else f"h{l + 1}", T, kind="out" if (last and vp) else "act")
            residual(f"addb{l}", xa, f2, xn, P[f"addb{l}"], then_ln=True, ln_out=hn)
            x, h = xn, hn
        self.y = h

        def collect(ctx, y=h):
            yv = ctx.read(y)
            with ctx.ex.prof.span("output_unpack"):
                ctx.result = np.stack([untile(yv[:, b * TOK:(b + 1) * TOK]) for b in range(ctx.n_valid)])
        steps.append(Host("output", collect, [h], [], op="output"))
        return steps

    # ------------------------------------------------------------------------------------- run
    def set_input(self, ctx, img_q):
        buf = np.zeros((self.B, TOK, 16), np.int8)
        buf[:len(img_q)] = img_q
        ctx.input, ctx.n_valid = buf, len(img_q)

    @staticmethod
    def _check(img_q):
        img_q = np.asarray(img_q)
        if img_q.ndim == 2:
            img_q = img_q[None]
        if img_q.ndim != 3 or img_q.shape[1:] != (TOK, 16):
            raise ValueError(f"expected int8 tokens [n, {TOK}, 16], got {img_q.shape}")
        if img_q.size and (img_q.min() < -128 or img_q.max() > 127):
            raise ValueError("input tokens must be int8")
        return img_q.astype(np.int8)

    def forward_q(self, img_q):
        """int8 tokens [n, 64, 16] -> int8 encoder output [n, 64, 128], synchronous."""
        img_q = self._check(img_q)
        return np.concatenate([self.run_one(img_q[i:i + self.B], len(img_q[i:i + self.B]))
                               for i in range(0, len(img_q), self.B)])

    def run_many(self, img_q, pipeline="auto"):
        """Any number of images in batches; pipeline=True overlaps host work and TPU jobs across
        the double-buffer contexts.  "auto" pipelines from batch 4 up: below that the per-job thread
        hand-off (~40 us x86, several times more on the A53) costs more than the overlap gains."""
        img_q = self._check(img_q)
        batches = [img_q[i:i + self.B] for i in range(0, len(img_q), self.B)]
        if pipeline == "auto":
            pipeline = self.B >= 4
        if not pipeline or len(self.contexts) < 2 or len(batches) < 2:
            return self.forward_q(img_q)
        return np.concatenate(self.run_pipelined(batches, [len(b) for b in batches]))

    def encode(self, images):
        """float images [n, 32, 32] (normalised like jepa-tpu/train.load_mnist) -> int8 [n, 64, 128]."""
        return self.run_many(quant_img(self.P, images))

    # ------------------------------------------------------------------------------------- info
    def describe(self):
        return (f"JepaExecutor backend={self.backend.name} vector={self.vector} batch={self.B} M={self.M} "
                f"host_ops={self.ops.name} contexts={len(self.contexts)}\n" + super().describe())
