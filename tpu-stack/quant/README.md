# tpuquant: quantisation and export toolkit for the TPU-KV260

`tpuquant` turns a float PyTorch model into a **bit-exact integer graph** for the tpuv1 systolic
array (INT8 x INT8 -> INT32, TFLite per-channel requant) plus the jepa-tpu vector unit (integer
LayerNorm, LUT softmax, LUT GELU, saturating residual add). It then writes the graph to a stable
`manifest.json` + `weights.npz` that a compiler can read.

```
tpuquant/
  intops.py     bit-exact integer ops (the arithmetic contract; == tpuv1 tflite_int8.py and jepa-tpu intops.py)
  observers.py  calibration: minmax, percentile, MSE-optimal clipping (2-pass histogram)
  qscheme.py    activation qparams (sym / asym), INT8 + INT4 weight quantisation
  ptq.py        torch.fx trace -> calibrate -> lower to the integer Graph (QConfig)
  graph.py      the IR, a bit-exact integer interpreter, a float / fake-quant interpreter
  analysis.py   per-layer SQNR / cosine, sensitivity, calibration comparison, mixed precision
  int4.py       nibble packing, group multipliers, INT4 hardware cost report
  export.py     save / load (schema below)
  jepa.py       adapters for ../../jepa-tpu (read-only) + MNIST loader
scripts/jepa_report.py   full study -> results/jepa_report.{md,json}, results/export_jepa_*/
tests/                   pytest (40 tests, ~30 s)
```

## Quick start

```python
from tpuquant import Calibrator, QConfig, export
cal = Calibrator(model, calib_batch)                 # fx trace + activation statistics (once)
g = cal.build(QConfig(act_method="mse", softmax_mult="mse"))   # lower; cheap, rebuild freely
y = g.run(x)[0]                                      # integer execution, dequantised output
export.save(g, "out/model")                          # out/model/manifest.json + weights.npz
g2 = export.load("out/model"); g2.run_int(x)         # same integers, bit for bit
```
For jepa-tpu: `cal = tpuquant.jepa.calibrate_encoder(tpuquant.jepa.load_encoder(), xtr[:256])`.
`python3 scripts/jepa_report.py` reproduces every number below (~8 min on 4 CPUs). `pytest` runs the tests.
MNIST is downloaded with torchvision into `data/` the first time.

## How a model is lowered

1. `torch.fx.symbolic_trace` (pass `concrete_args` for optional arguments). Shapes are probed at
   batch 2 and batch 3, so the batch dimension becomes the symbol `N` (or `N*k`).
2. Calibration runs two passes per tensor. Pass 1 records min and max. Pass 2 fills a 4096-bin
   histogram. Any observer can then be evaluated from the same statistics, so trying a new
   calibration method needs no new run.
3. Lowering, with these fusions:
   - **Linear / Conv2d:** per-output-channel symmetric INT8 weights in [-127, 127]. Bias is
     `round(b/(s_in*s_w))`, and the input zero-point term is folded into it as
     `bias_hw = bias - zp_in*sum_k w`, the same way tpuv1 does it. The multiplier is
     `s_in*s_w/s_out`, converted with TFLite `QuantizeMultiplier`. A following ReLU/ReLU6 becomes
     a clamp (`act_min/act_max`, TFLite rule). A following scalar mul/div (such as `1/sqrt(d)`) is
     folded into the multiplier.
   - **LayerNorm:** jepa integer LN, with output at scale 1/32. gamma/beta are folded into the
     consuming linears. If the LN feeds the graph output, the affine is applied on the host after
     dequant (`outputs[].post`). Otherwise it becomes an `affine` node, which is a per-channel
     requant.
   - **Softmax (last dim):** a 256-entry exp LUT built from the logit scale. The output is
     0..127 at scale `1/out_mult`.
   - **GELU / Sigmoid / Tanh / SiLU / standalone ReLU:** a 256-entry int8 LUT.
   - **Residual add:** `sat8(((a-za)*Ma + (b-zb)*Mb + 2^15) >> 16 + zo)`. With zp = 0 this is
     exactly jepa's `ew_add`. A `+ nn.Parameter` input (position embedding) becomes a `const`
     node.
   - **tensor @ tensor:** a `matmul` node with a scalar multiplier. Both operands are forced
     symmetric (zp 0), because the operand loaded as "weights" has no zero-point path.
   - **mean / AdaptiveAvgPool2d(1):** an int32 sum followed by requant. **MaxPool2d:** exact on
     int8.
   - **view/reshape/flatten/(un)squeeze, permute/transpose, getitem, contiguous/dropout:** layout
     nodes with no arithmetic.
   - **Split scales** (`split_scales=True`): when layout ops split a linear's output (fused QKV
     into q, k and v), each part keeps its own scale. The producer gets a per-channel output
     scale, which costs nothing in tpuv1's per-channel requant. Jepa-tpu does the same.

   Unsupported ops raise `NotImplementedError` naming the fx node. `nn.MultiheadAttention` is a
   black box to fx, so write attention explicitly, as jepa's `model.py` does.

`QConfig` options:
- `act_method`: `minmax`, `percentile[:p]` (default 99.99) or `mse`.
- `act_symmetric`.
- `weight_bits` and `layer_bits={name: bits}`.
- `int4_scheme`, `group_size`.
- `weight_method`: `minmax` or `mse`.
- `softmax_mult`: None means 1/127, the jepa spec. An int or an observer name calibrates it.
- `split_scales`, `fold_ln`, `output_affine`, `method_overrides`.
- `fixed_qparams`: import scales from elsewhere.

### Arithmetic contract (`intops.py`)
- **Requant** is identical to tpuv1 `tflite_int8.requant`: gemmlowp double rounding
  (`SaturatingRoundingDoublingHighMul`, then `RoundingDivideByPOT`), `+zp_out`, then clamp.
  This was checked against the tpuv1 reference on 20k random cases, including corner cases.
  `shift` lies in [-31, 30] and fits the 8-bit SET_CONFIG field.
- **LN / softmax / LUT / add** are identical to jepa-tpu `intops.py`, checked on random tensors.
  Generalising them to non-zero zero points is exact: the zp cancels in LN and in softmax.
- **Float -> int8 input quantisation** uses `np.round`, which rounds half to even, as jepa and
  tpuv1 `compile_mlp` do.
- `softmax(s, lut, out_mult)` with `out_mult=127` is exactly jepa's softmax. Any other value is
  only a different numerator constant in `(out_mult << 22) // Z`, plus a clamp at 127.

## Accuracy findings (jepa-tpu encoder; 256 calibration images, 200 test images)

**Reproduction.** Given jepa-tpu's own scales (`jepa.jepa_fixed_qparams`), the toolkit emits the
**identical program**: every w/bias/m0/shift and every LUT matches, and the outputs are bit-exact
(`tests/test_jepa.py`). On its own calibration it reaches the same accuracy: cos 0.9981 vs 0.9981.

| INT8 config | cos mean | cos min | SQNR dB |
|---|---|---|---|
| jepa-tpu (percentile 99.99, symmetric) | 0.99808 | 0.99661 | 24.1 |
| minmax / percentile / MSE (sym) | 0.99810 / 0.99807 / 0.99810 | 0.9967-0.9968 | 24.1-24.2 |
| MSE, asymmetric | 0.99814 | 0.99679 | 24.3 |
| **MSE + calibrated softmax scale** (sym) | **0.99915** | **0.99878** | **27.7** |
| MSE + calibrated softmax scale, asym | 0.99921 | 0.99885 | 28.0 |

- **The calibration method hardly matters** (±0.0001). The error floor comes from the fixed-format
  tensors, not from where the clipping range is set.
- **The sensitivity analysis names the culprit.** The softmax probabilities (7 bits at 1/127) are
  5 of the 6 most harmful tensors, and the input embedding is the 6th.
- **Fix: calibrate the softmax output scale.** `softmax_mult="mse"` gives `out_mult` ≈ 180-580
  per layer, so probabilities above about 0.2-0.7 saturate. This lifts the cosine from 0.9981 to
  **0.9992** and cuts the error by about 2.3x (+3.5 dB). In the vector unit this is one
  per-layer constant (a 10-bit numerator instead of 127). The datapath grows by about 3 bits.
- **Asymmetric activations** add only +0.1 dB here, because most tensors are roughly symmetric.

## INT4 weights

| scheme (group 32) | integer graph cos | SQNR dB | on tpuv1 |
|---|---|---|---|
| INT8 (reference) | 0.99915 | 27.7 | as is |
| channel, minmax / mse | 0.99703 / 0.99754 | 22.3 / 23.1 | weight loader: sign-extend nibbles |
| group_int (4-bit M in 1..16), minmax / mse | 0.99716 / **0.99764** | 22.5 / 23.3 | + 4x5-bit multiply per lane |
| group_pow2 (M = 2^e), mse | 0.99709 | 22.4 | + shift per lane |
| group_float (true float group scales), mse | 0.99775 (fake-quant only) | - | needs per-group partial-sum rescale |

- **Cost:** INT4 costs about 4.4 dB SQNR and about 0.0015 cosine. It is still better than the
  original jepa INT8 recipe without the softmax fix.
- **Weight clipping:** MSE weight clipping is worth +0.8 dB at INT4. It does nothing at INT8.
- **Mixed precision:** the INT4 weight-sensitivity ranking is embed >> proj(0, 1) > qkv(0..2) > ...
  The embedding layer has only 2 KiB of weights and accounts for +0.5 dB. Keeping the 3 most
  sensitive layers at INT8 gives cos 0.99813 (24.3 dB) for 612 KiB of weights, against
  1154 KiB at INT8. See `results/jepa_report.md` §5 for the k sweep.
- **What INT4 needs in hardware** (`int4.hardware_report`): the MAC array and the requant stay
  unchanged.
  - **Weight loader:** int4 values are expanded to int8 as `w8 = q4 * M_g`, with |w8| <= 112.
    One 16-column weight row becomes a 64-bit word instead of 128 bits.
  - **group_int:** add a 4x5-bit multiplier per lane.
  - **group_pow2:** add a shift per lane. G must be a multiple of the 16-row K tile.
  - **Group multipliers:** these add 1 nibble per (group, column), which is 3% extra.
  - **Memory:** weights drop from 1154 KiB to 594 KiB (+79 KiB of requant/group metadata). This
    frees about 560 KiB of BRAM, which matters because jepa-tpu's hard-coded design sits right at
    the KV260 limit.
  - **True float group scales:** these need either a per-group requant of partial sums or wider
    weights. They are not offered in the integer graph: their integer path is an INT8 projection,
    and `meta.int_exact` is false.

## Export format (schema v1)

An export is a directory containing `manifest.json` and `weights.npz`. Every array lives in the npz.
The manifest refers to an array by `{"key": <npz key>, "dtype": str, "shape": [..]}`.

```jsonc
{
  "format": "tpuquant-graph", "version": 1, "name": "jepa_encoder",
  "arith": {"requant": "tflite_double_rounding (tpuv1 tflite_int8.py)",
            "input_rounding": "round_half_to_even",
            "constants": {"LN_Q": 32, "LN_K": 24, "SM_K": 22, "SM_OUT": 127, "EXP_ONE": 32768, "EW_SH": 16}},
  "inputs":  [{"name": "img", "shape": ["N", 1, 32, 32], "dtype": "float32"}],
  "outputs": [{"name": "ln_norm", "post": {"scale": ArrRef, "bias": ArrRef} | null}],
  "tensors": {"<name>": {"shape": [...], "dtype": "int8"|"float", "scale": float | ArrRef,
                         "zero_point": int, "range": [lo, hi]}},
  "nodes":   [{"name", "op", "inputs": [tensor...], "output": tensor, "attrs": {...}, "params": {"p": ArrRef}}],
  "stats": {"weight_bytes": int, "n_nodes": int}, "meta": {"qconfig": {...}, "int_exact": bool},
  "extra": {...}   // optional, e.g. accuracy metrics
}
```

**Conventions:**
- Nodes are listed in topological order.
- Shapes are PyTorch logical layout (NCHW for conv activations). `"N"` / `"N*k"` mark the batch
  dimension.
- A tensor's real value is `(q - zero_point) * scale`. A non-scalar `scale` is an array that
  broadcasts against the tensor, with batch dim 1. It appears only on a split linear output and
  its layout views. Consumers always see a scalar scale.
- `range` is the int8 range a node actually produces (fused ReLU, softmax 0..127).

**Node table.** In every `requant(...)` below, `bias`, `m0` and `shift` are per output channel,
and the result is `requant(acc, bias, m0, shift, zp_out, act_min, act_max)`, exactly as tpuv1
`tflite_int8.requant`.

| op | inputs | attrs | params | integer semantics |
|---|---|---|---|---|
| `quantize` | float | - | - | `clip(round_half_even(x/scale)+zp, -128, 127)` (host) |
| `const` | - | - | `value` int8 | constant tensor |
| `linear` | x [..,K] | zp_in, zp_out, act_min, act_max, weight_bits, weight_scheme, (group_size) | `w` int8 [K,N], `bias` int32 [N] (zp_in folded), `m0` int32 [N], `shift` int8 [N], `w_scale` f32 [N]; INT4: `w4` int8 [K,N], `w4_packed` u8 [K,N/2], `gmul` u8 [K/G,N], `gmul_packed` u8 | `acc = x @ w` (raw int8 sums), then requant |
| `conv2d` | x [N,C,H,W] | as linear + stride[2], padding[2] | `w` int8 **HWIO** [kh,kw,cin,cout] (tpuv1 compile_cnn layout), bias/m0/shift [cout] | pad with `zp_in`, im2col in (kh,kw,cin) order, `acc = cols @ w`, requant; output NCHW |
| `matmul` | a [..,M,K], b [..,K,N] (zp 0) | m0, shift (scalars), zp_out, act_min, act_max | - | `requant(a @ b, 0, m0, shift, ...)` |
| `layernorm` | x [..,D] | - | - | `intops.layernorm` (sum, sum of squares, isqrt, reciprocal); out scale 1/LN_Q, zp 0 |
| `affine` | x | axis, zp_in, zp_out, act_min, act_max | `w` int8 [C], `bias` int32, m0, shift | `requant(x*w, bias, ...)` (diagonal linear) |
| `softmax` | x [..,T] | axis=-1, out_mult | `lut` int32 [256] | `e = lut[max-x]`, `inv = (out_mult<<SM_K)//sum e`, `min((e*inv + 2^(SM_K-1)) >> SM_K, 127)`; scale 1/out_mult |
| `lut` | x | fn | `lut` int8 [256] | `lut[x+128]` |
| `add` | a, b (broadcast) | ma, mb (Q16), za, zb, zo | - | `clip((((a-za)*ma + (b-zb)*mb + 2^15) >> 16) + zo)` |
| `rescale` | x | za, zo | `m` int (Q16, broadcastable) | add with one input |
| `mean` | x | axes, keepdim, m0, shift, zp_in, zp_out | - | `requant(sum(x-zp_in), 0, m0, shift, zp_out)` |
| `maxpool2d` | x | kernel, stride, padding | - | window max (pad = -128) |
| `reshape` / `permute` / `getitem` | x | shape (symbolic) / perm / index (int, `{"slice":[a,b,c]}`, `{"newaxis":true}`, `"..."`) | - | layout only; no arithmetic |

**Packing:**
- `pack_int4` puts element 2j in bits [3:0] and element 2j+1 in bits [7:4] (two's complement).
  It packs along the output-channel (last) axis, so one packed row holds the 16 columns of one
  tpuv1 weight word.
- `gmul_packed` stores `M-1` as unsigned nibbles.
- `w` always holds the expanded int8 values the array multiplies (`w4 * gmul`), so a compiler can
  ignore INT4 entirely.

Versioning: a breaking change bumps `version`. `export.load` refuses any version it does not know.

## Tests

`pytest` (from this directory): 40 tests, about 30 s.
- **Bit-exactness:** requant vs tpuv1 on 20k random cases; vector ops vs jepa-tpu.
- **Observers:** behaviour of each calibration method.
- **INT4:** all schemes, packing round trip.
- **Generic PTQ** on a CNN (conv/relu/maxpool/avgpool/linear) and a tiny ViT (LN/attention/GELU/
  residual/mean):
  - the float interpreter matches torch;
  - INT8 cos > 0.98 for every method × sym/asym;
  - structural checks: ReLU fusion, LN folding, split scales, symmetric attention operands;
  - export round trip.
- **jepa:**
  - bit-exact reproduction of jepa-tpu's program;
  - INT8 ≥ 0.998;
  - calibrated softmax > 0.999;
  - INT4 > 0.99;
  - jepa export round trip.

## Known limitations
- **Calibration precision.** Percentiles come from a 4096-bin histogram, not exact sorting. Scales
  therefore differ from jepa's by about 0.1%, so a self-calibrated graph is *not* bit-identical
  to jepa-tpu's. Use `fixed_qparams` for that.
- **Softmax** is supported over the last dim only. **LayerNorm** is supported over the last dim
  only. **Conv** supports groups=1 only. **Asymmetric act×act matmuls** are not supported: they
  would need a runtime column sum.
- **The integer LN ignores `eps`**, as jepa-tpu's does.
- **The residual stream is int8**, per the jepa spec. Residual adds show up in the sensitivity
  list right after softmax. An int16 residual would be the next accuracy step.
