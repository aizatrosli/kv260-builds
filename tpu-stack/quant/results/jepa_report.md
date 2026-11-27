# tpuquant study: jepa-tpu encoder

Calibration 256 MNIST train images, evaluation 200 test images. Metric: cosine of the integer encoder output (dequantised) vs float PyTorch, per image; SQNR over all outputs.

## 1. Reproduction of jepa-tpu

With jepa-tpu's own scales the toolkit emits the identical program (bit-exact, see tests). Its accuracy here: cos 0.99808 (min 0.99661).

## 2. Activation calibration (INT8 weights)

`+sm` = softmax output scale calibrated (`softmax_mult='mse'`) instead of the fixed 1/127.

| method | symmetric | cos_mean | cos_min | sqnr_db |
|---|---|---|---|---|
| minmax | True | 0.99810 | 0.99668 | 24.21 |
| percentile | True | 0.99807 | 0.99663 | 24.14 |
| percentile:99.9 | True | 0.99806 | 0.99591 | 24.11 |
| mse | True | 0.99810 | 0.99682 | 24.21 |
| minmax | False | 0.99809 | 0.99658 | 24.17 |
| percentile | False | 0.99815 | 0.99672 | 24.32 |
| percentile:99.9 | False | 0.99795 | 0.99494 | 23.88 |
| mse | False | 0.99814 | 0.99679 | 24.30 |
| percentile +sm | True | 0.99912 | 0.99875 | 27.55 |
| mse +sm | True | 0.99915 | 0.99878 | 27.68 |
| percentile +sm | False | 0.99920 | 0.99878 | 27.94 |
| mse +sm | False | 0.99921 | 0.99885 | 28.00 |

## 3. Sensitivity (32 images, one tensor quantised at a time, rest float)

Activations (10 most harmful):

| name | op | sqnr_db | cos |
|---|---|---|---|
| softmax_5 | softmax | 37.35 | 0.999908 |
| embed | linear | 37.56 | 0.999912 |
| softmax_3 | softmax | 38.52 | 0.999930 |
| softmax_1 | softmax | 38.73 | 0.999933 |
| softmax_2 | softmax | 38.89 | 0.999935 |
| softmax_4 | softmax | 39.85 | 0.999948 |
| add_10 | add | 40.64 | 0.999957 |
| ln_norm | layernorm | 40.90 | 0.999959 |
| add_12 | add | 41.14 | 0.999962 |
| add_9 | add | 41.15 | 0.999962 |

Weights at INT4 group_int (10 most harmful):

| name | params | sqnr_db | cos |
|---|---|---|---|
| embed | 2048 | 30.96 | 0.999599 |
| blocks_0_proj | 16384 | 37.40 | 0.999909 |
| blocks_1_proj | 16384 | 38.01 | 0.999921 |
| blocks_0_qkv | 49152 | 38.01 | 0.999921 |
| blocks_2_qkv | 49152 | 38.44 | 0.999928 |
| blocks_1_qkv | 49152 | 38.45 | 0.999929 |
| blocks_3_proj | 16384 | 38.87 | 0.999935 |
| blocks_0_fc2 | 65536 | 39.52 | 0.999944 |
| blocks_4_proj | 16384 | 39.59 | 0.999945 |
| blocks_5_qkv | 49152 | 39.67 | 0.999946 |

## 4. INT4 weights (activations: best INT8 config, group size 32)

INT8 reference: cos 0.99915, SQNR 27.68 dB. `group_float` has no integer-exact form on tpuv1 (fake-quant only).

| scheme | weight_method | int_cos | int_cos_min | int_sqnr | fakequant_cos |
|---|---|---|---|---|---|
| channel | minmax | 0.99703 | 0.99601 | 22.26 | 0.99706 |
| channel | mse | 0.99754 | 0.99701 | 23.08 | 0.99753 |
| group_int | minmax | 0.99716 | 0.99621 | 22.46 | 0.99717 |
| group_int | mse | 0.99764 | 0.99719 | 23.26 | 0.99765 |
| group_pow2 | minmax | 0.99705 | 0.99604 | 22.30 | 0.99706 |
| group_pow2 | mse | 0.99709 | 0.99618 | 22.35 | 0.99709 |
| group_float | minmax | nan | nan | nan | 0.99717 |
| group_float | mse | nan | nan | nan | 0.99775 |

## 5. Mixed precision: keep the k most INT4-sensitive layers at INT8

| k | weight_kib | cos_mean | cos_min | sqnr_db | int8 |
|---|---|---|---|---|---|
| 0 | 595 | 0.99764 | 0.99719 | 23.26 |  |
| 1 | 596 | 0.99791 | 0.99714 | 23.78 | embed |
| 3 | 612 | 0.99813 | 0.99742 | 24.26 | embed, blocks_0_proj, blocks_1_proj |
| 6 | 681 | 0.99827 | 0.99758 | 24.61 | embed, blocks_0_proj, blocks_1_proj, blocks_0_qkv, blocks_2_qkv, blocks_1_qkv |
| 12 | 798 | 0.99860 | 0.99801 | 25.52 | embed, blocks_0_proj, blocks_1_proj, blocks_0_qkv, blocks_2_qkv, blocks_1_qkv ... |
| 25 | 1154 | 0.99915 | 0.99879 | 27.68 | embed, blocks_0_proj, blocks_1_proj, blocks_0_qkv, blocks_2_qkv, blocks_1_qkv ... |

## 6. Per-layer accumulated error, recommended INT8 config (last block)

| name | op | sqnr_db | cos | scale | sat |
|---|---|---|---|---|---|
| add_10 | add | 28.39 | 0.999281 | 0.05036 | 6.26e-04 |
| blocks_5_ln1_norm | layernorm | 28.21 | 0.999245 | 0.03125 | 2.29e-05 |
| blocks_5_qkv | linear | 30.41 | 0.999545 | 0.02059 | 1.07e-04 |
| truediv_5 | matmul | 35.31 | 0.999855 | 0.03832 | 7.82e-05 |
| softmax_5 | softmax | 26.64 | 0.998916 | 0.005525 | 9.54e-05 |
| matmul_11 | matmul | 21.65 | 0.996577 | 0.01226 | 2.56e-04 |
| blocks_5_proj | linear | 21.59 | 0.996553 | 0.007364 | 2.98e-04 |
| add_11 | add | 27.59 | 0.999133 | 0.04771 | 3.43e-05 |
| blocks_5_ln2_norm | layernorm | 27.45 | 0.999100 | 0.03125 | 3.05e-05 |
| blocks_5_fc1 | linear | 27.37 | 0.999084 | 0.01655 | 3.06e-04 |
| gelu_5 | lut | 26.08 | 0.998768 | 0.01406 | 7.44e-05 |
| blocks_5_fc2 | linear | 27.54 | 0.999118 | 0.006517 | 8.77e-05 |
| add_12 | add | 27.84 | 0.999182 | 0.04967 | 5.34e-05 |
| ln_norm | layernorm | 27.68 | 0.999147 | 0.03125 | 2.67e-05 |

## 7. Memory

- int8_weight_kib: 1154.0
- requant_param_kib: 61.9
- kv260_free_onchip_kib: 1638.4
- int4_weight_kib: 594.0
- int4_meta_kib: 79.4
- saving_kib: 560.0
- ratio: 0.6

Runtime 468 s.
