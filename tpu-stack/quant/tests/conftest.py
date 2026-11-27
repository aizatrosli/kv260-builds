import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TPUV1_QUANT = os.environ.get("TPUV1_DIR", "/home/user/tpuv1") + "/model/quantization"
