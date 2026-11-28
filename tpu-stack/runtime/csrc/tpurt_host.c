// SPDX-License-Identifier: MIT
/*
 * tpurt host kernels: the non-matmul ops of the JEPA integer graph (jepa-tpu/intops.py), bit exact,
 * operating directly on the TPU's tiled activation layout so no transpose is ever needed:
 *
 *     x[t][m][l]  (T tiles x M rows x 16 lanes, int8)  ==  X[m][16*t + l]   (token-major)
 *
 * This is the layout READ_HOST_MEM / WRITE_HOST_MEM move, so these kernels read and write the DMA
 * buffers in place.  Integer widths follow intops.py exactly (int64 wherever numpy used int64 and the
 * value can exceed int32).  Built by `make` into tpurt/_lib/libtpurt_host.so and used via ctypes.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#define LN_Q 32
#define LN_K 24
#define SM_K 22
#define EW_SH 16

static inline int8_t sat8(int64_t v) { return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); }
static inline int8_t sat8_32(int32_t v) { return (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v)); }

static inline int64_t isqrt64(int64_t v)
{
	int64_t r = (int64_t)floor(sqrt((double)v));
	while (r > 0 && r * r > v) r--;
	while ((r + 1) * (r + 1) <= v) r++;
	return r;
}

/* LayerNorm over the T*16 values of every row m (intops.layernorm).  Tile-major passes keep the
 * inner loops on 16 contiguous int8 lanes so they vectorise (NEON on the A53, SSE/AVX on x86).
 * Exactness: |S| <= 2^14 and |d| < 2^15 fit int32 for D <= 128, where d*d < 2^30 also fits int32
 * (summed in int64); wider rows square in int64. */
void tpurt_layernorm(const int8_t *x, int8_t *y, int T, int M)
{
	const int32_t D = T * 16;
	int32_t S[M];
	int64_t Q[M], scale[M];
	for (int m = 0; m < M; m++) { S[m] = 0; Q[m] = 0; }
	for (int t = 0; t < T; t++)
		for (int m = 0; m < M; m++) {
			const int8_t *p = x + ((size_t)t * M + m) * 16;
			int32_t acc = 0;
			for (int l = 0; l < 16; l++) acc += p[l];
			S[m] += acc;
		}
	for (int t = 0; t < T; t++)
		for (int m = 0; m < M; m++) {
			const int8_t *p = x + ((size_t)t * M + m) * 16;
			const int32_t s = S[m];
			int64_t acc = 0;
			if (D <= 128) {                 /* |d| < 2^15: d*d < 2^30 fits int32 */
				for (int l = 0; l < 16; l++) {
					int32_t d = D * p[l] - s;
					acc += d * d;
				}
			} else {
				for (int l = 0; l < 16; l++) {
					int64_t d = (int64_t)D * p[l] - s;
					acc += d * d;
				}
			}
			Q[m] += acc;
		}
	for (int m = 0; m < M; m++) {
		int64_t sig = isqrt64(Q[m] / D);
		if (sig < 1) sig = 1;
		scale[m] = (((int64_t)1 << LN_K) / sig) * LN_Q;     /* d*LN_Q*inv == d*(LN_Q*inv), exact */
	}
	/* y = sat8((d*k + 2^23) >> 24) with k = scale < 2^30.  For D <= 128 (|d| < 2^15) it is computed
	 * exactly in int32: k = kh*2^15 + kl, A = d*kh, Bv = d*kl + 2^23  ->  y = (A + (Bv >> 15)) >> 9
	 * (floor((X*2^15 + r) / 2^24) == floor(X / 2^9) for 0 <= r < 2^15). */
	for (int t = 0; t < T; t++)
		for (int m = 0; m < M; m++) {
			const int8_t *p = x + ((size_t)t * M + m) * 16;
			int8_t *o = y + ((size_t)t * M + m) * 16;
			const int32_t s = S[m];
			const int64_t k = scale[m];
			if (D <= 128 && k < ((int64_t)1 << 29)) {
				const int32_t kh = (int32_t)(k >> 15), kl = (int32_t)(k & 0x7FFF);
				for (int l = 0; l < 16; l++) {
					int32_t d = D * p[l] - s;
					int32_t A = d * kh, Bv = d * kl + (1 << (LN_K - 1));
					o[l] = sat8_32((A + (Bv >> 15)) >> (LN_K - 15));
				}
			} else {
				for (int l = 0; l < 16; l++) {
					int64_t v = (int64_t)(D * p[l] - s) * k;
					o[l] = sat8((v + ((int64_t)1 << (LN_K - 1))) >> LN_K);
				}
			}
		}
}

/* Row softmax (intops.softmax) for G groups, each T tiles x M rows: s[g][t][m][l]. lut[256] int32.
   q: per-layer numerator (output scale 1/q, 127..8191), probabilities saturate at 127. */
static inline int8_t sat127(int64_t v) { return (int8_t)(v > 127 ? 127 : v); }

void tpurt_softmax(const int8_t *s, int8_t *p, int G, int T, int M, const int32_t *lut, int q)
{
	int32_t lmax = 0;
	for (int i = 0; i < 256; i++) if (lut[i] > lmax) lmax = lut[i];
	for (int g = 0; g < G; g++) {
		const int8_t *sg = s + (size_t)g * T * M * 16;
		int8_t *pg = p + (size_t)g * T * M * 16;
		for (int m = 0; m < M; m++) {
			int mx = -128;
			for (int t = 0; t < T; t++) {
				const int8_t *r = sg + ((size_t)t * M + m) * 16;
				for (int l = 0; l < 16; l++) if (r[l] > mx) mx = r[l];
			}
			int64_t Z = 0;
			for (int t = 0; t < T; t++) {
				const int8_t *r = sg + ((size_t)t * M + m) * 16;
				for (int l = 0; l < 16; l++) Z += lut[mx - r[l]];
			}
			int64_t inv = ((int64_t)q << SM_K) / Z;
			if ((int64_t)lmax * inv + (1 << (SM_K - 1)) < ((int64_t)1 << 31)) {   /* int32 is exact */
				const int32_t iv = (int32_t)inv;
				for (int t = 0; t < T; t++) {
					const int8_t *r = sg + ((size_t)t * M + m) * 16;
					int8_t *o = pg + ((size_t)t * M + m) * 16;
					for (int l = 0; l < 16; l++)
						o[l] = sat127((lut[mx - r[l]] * iv + (1 << (SM_K - 1))) >> SM_K);
				}
				continue;
			}
			for (int t = 0; t < T; t++) {
				const int8_t *r = sg + ((size_t)t * M + m) * 16;
				int8_t *o = pg + ((size_t)t * M + m) * 16;
				for (int l = 0; l < 16; l++)
					o[l] = sat127(((int64_t)lut[mx - r[l]] * inv + ((int64_t)1 << (SM_K - 1))) >> SM_K);
			}
		}
	}
}

/* y = lut[x + 128] (GELU). lut[256] int8. */
void tpurt_lut8(const int8_t *x, int8_t *y, size_t n, const int8_t *lut)
{
	for (size_t i = 0; i < n; i++) y[i] = lut[x[i] + 128];
}

/* Residual add (intops.ew_add): sat8((a*ma + b*mb + 2^15) >> 16).  int32 fast path (vectorises)
 * when |ma|, |mb| < 2^22 (|a*ma + b*mb| < 2^30, so no overflow: bit-identical to int64). */
void tpurt_ew_add(const int8_t *a, const int8_t *b, int8_t *y, size_t n, int64_t ma, int64_t mb)
{
	if (ma > -(1 << 22) && ma < (1 << 22) && mb > -(1 << 22) && mb < (1 << 22)) {
		const int32_t A = (int32_t)ma, B = (int32_t)mb;
		for (size_t i = 0; i < n; i++)
			y[i] = sat8_32((a[i] * A + b[i] * B + (1 << (EW_SH - 1))) >> EW_SH);
		return;
	}
	for (size_t i = 0; i < n; i++)
		y[i] = sat8((a[i] * ma + b[i] * mb + ((int64_t)1 << (EW_SH - 1))) >> EW_SH);
}

/* Same, with b broadcast with period nb elements (positional embedding for a batch). */
void tpurt_ew_add_bcast(const int8_t *a, const int8_t *b, int8_t *y, size_t n, size_t nb, int64_t ma, int64_t mb)
{
	for (size_t i = 0; i < n; i += nb)
		tpurt_ew_add(a + i, b, y + i, n - i < nb ? n - i : nb, ma, mb);
}

/*
 * Weight-tile packing for the attention matmuls (activations used as weights).  A weight tile is
 * 16 words of 16 bytes, word w = W_tile[15 - w] (tpuv1 FIFO order), byte n = column n.
 *   qkv [24][M][16]  (q | k | v, each 8 tiles of 16 dims), images of 64 tokens, H heads of dh dims.
 *   kT : for (b, h): W = k_bh^T [dh x 64]  -> tiles nt (4, tokens) outer, kt (dh/16) inner
 *   v  : for (b, h): W = v_bh   [64 x dh]  -> tiles nt (dh/16) outer, kt (4, tokens) inner
 */
void tpurt_pack_kv(const int8_t *qkv, int8_t *kT, int8_t *v, int B, int H, int M, int D)
{
	const int dh = D / H, KTd = dh / 16, Tt = 64 / 16, Td = D / 16;
	for (int b = 0; b < B; b++)
		for (int h = 0; h < H; h++) {
			int8_t *kd = kT + (size_t)(b * H + h) * Tt * KTd * 256;
			int8_t *vd = v + (size_t)(b * H + h) * KTd * Tt * 256;
			for (int nt = 0; nt < Tt; nt++)
				for (int kt = 0; kt < KTd; kt++) {
					/* kT tile (kt, nt): W[k][n] = k[token 16nt+n][dim 16kt+k] */
					const int8_t *src = qkv + ((size_t)(Td + h * KTd + kt) * M + b * 64 + 16 * nt) * 16;
					int8_t *dst = kd + (size_t)(nt * KTd + kt) * 256;
					for (int k = 0; k < 16; k++)
						for (int n = 0; n < 16; n++)
							dst[(15 - k) * 16 + n] = src[n * 16 + k];
				}
			for (int nt = 0; nt < KTd; nt++)
				for (int kt = 0; kt < Tt; kt++) {
					/* v tile (kt, nt): W[k][n] = v[token 16kt+k][dim 16nt+n] */
					const int8_t *src = qkv + ((size_t)(2 * Td + h * KTd + nt) * M + b * 64 + 16 * kt) * 16;
					int8_t *dst = vd + (size_t)(nt * Tt + kt) * 256;
					for (int k = 0; k < 16; k++)
						for (int n = 0; n < 16; n++)
							dst[(15 - k) * 16 + n] = src[k * 16 + n];
				}
		}
}
