import numpy as np

from tpuquant import intops as I


def test_sm_numerator_matches_reference_formula():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 64))         # attention-like: no dominant key
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    q = I.sm_numerator(p)
    assert I.SM_OUT <= q <= I.SM_QMAX and (q - I.SM_OUT) % 8 == 0
    err = lambda q: ((np.minimum(np.round(p * q), 127) / q - p) ** 2).mean()
    assert err(q) <= min(err(c) for c in range(127, 8192, 8))
    assert err(q) < err(127)


def test_softmax_large_numerator_saturates():
    lut = I.exp_lut(0.05)
    s = np.array([[100, -100, -100, -100]])
    assert I.softmax(s, lut, 8191).max() == 127
