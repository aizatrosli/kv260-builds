import random

import numpy as np

from tpurt import _paths, intmath

_paths.need_tpuv1_model()
import tflite_int8 as T  # noqa: E402


def test_requant_matches_tpuv1_scalar():
    rnd = random.Random(0)
    cases = []
    for _ in range(4000):
        acc = rnd.choice([rnd.randrange(-2**31, 2**31), rnd.randrange(-5000, 5000), -2**31, 2**31 - 1])
        bias = rnd.choice([0, rnd.randrange(-2**31, 2**31), rnd.randrange(-100, 100)])
        m0 = rnd.choice([rnd.randrange(2**30, 2**31), 0, -2**31, 2**31 - 1])
        sh = rnd.randrange(-31, 31)
        zp, lo = rnd.randrange(-128, 128), rnd.randrange(-128, 0)
        cases.append((acc, bias, m0, sh, zp, lo, rnd.randrange(-128, 128)))   # incl. act_min > act_max
    c = np.array(cases, dtype=np.int64).T
    got = np.array([int(intmath.requant(c[0][i], c[1][i], c[2][i], c[3][i], c[4][i], c[5][i], c[6][i]))
                    for i in range(0, len(cases), 37)])
    vec = intmath.requant(c[0], c[1], c[2], c[3], c[4], c[5], c[6])
    ref = np.array([T.requant(*map(int, k)) for k in cases])
    assert (vec == ref).all()
    assert (got == ref[::37]).all()
