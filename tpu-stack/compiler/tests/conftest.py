import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tpucc import env, frontends, passes, codegen, refexec  # noqa: E402


@pytest.fixture(scope="session")
def P():
    return env.load_qmodel()


@pytest.fixture(scope="session")
def jepa_graph(P):
    return frontends.from_jepa(P)


@pytest.fixture(scope="session")
def jepa_lowered(jepa_graph):
    return passes.lower(jepa_graph)


@pytest.fixture(scope="session")
def jepa_prog(jepa_lowered):
    return codegen.compile_graph(jepa_lowered)


@pytest.fixture(scope="session")
def jepa_img(P):
    return frontends.synthetic_image(P, seed=0)


@pytest.fixture(scope="session")
def jepa_ref(jepa_graph, jepa_img):
    return refexec.run(jepa_graph, {"img": jepa_img})


@pytest.fixture(scope="session")
def jepa_golden(P, jepa_img):
    """jepa-tpu quantize.int_forward: final output and the matmul log (name, x, w, acc)."""
    Q = env.jepa_quantize()
    log = []
    out = Q.int_forward(P, jepa_img, log)
    return out, log
