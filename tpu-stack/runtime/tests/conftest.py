import os
import sys
from pathlib import Path

import numpy as np
import pytest

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))

from tpurt import _paths, data  # noqa: E402

_paths.need_jepa()
ON_BOARD = os.path.exists("/dev/tpu0")


@pytest.fixture(scope="session")
def P():
    return data.load_qmodel()


@pytest.fixture(scope="session")
def images(P):
    imgs, src = data.load_images(8)
    return data.quant_img(P, imgs)


@pytest.fixture(scope="session")
def golden(P, images):
    import quantize                       # jepa-tpu's own spec (needs torch; torch is installed here)
    return [quantize.int_forward(P, images[i]) for i in range(len(images))]


def pytest_collection_modifyitems(config, items):
    if ON_BOARD:
        return
    skip = pytest.mark.skip(reason="/dev/tpu0 absent (board-only test)")
    for it in items:
        if "board" in it.keywords:
            it.add_marker(skip)
