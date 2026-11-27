#!/usr/bin/env bash
# Reproduce everything. Needs: Verilator >= 5.030 (5.020 miscompiles the PULP xbar address map),
# python3 with torch + numpy, tpuv1 cloned at $TPU, PULP axi + common_cells at the Bender.lock revs in $DEPS.
set -euo pipefail
TPU=${TPU:-$HOME/tpuv1}; DEPS=${DEPS:-$HOME/deps}; VL=${VL:-verilator}; J=$(cd "$(dirname "$0")" && pwd)
VFLAGS="--binary -j 4 --timing -Wno-fatal -Wno-lint -Wno-style -Wno-UNOPTFLAT -Wno-MULTIDRIVEN"
INC="-I$DEPS/axi/include -I$DEPS/common_cells/include -y $DEPS/axi/src -y $DEPS/common_cells/src -y $DEPS/common_cells/src/deprecated \
 -y $TPU/rtl/compute -y $TPU/rtl/activation -y $TPU/rtl/control -y $TPU/rtl/memory -y $TPU/rtl/top \
 $DEPS/common_cells/src/cc_pkg.sv $DEPS/common_cells/src/deprecated/cf_math_pkg.sv $DEPS/axi/src/axi_pkg.sv $TPU/rtl/common/tpu_pkg.sv"
cd "$J"
[ -f ckpt.pt ]     || python3 train.py --epochs 2 --out ckpt.pt                 # ~7 min on 4 CPU cores
[ -f qmodel.npy ]  || python3 quantize.py --ckpt ckpt.pt --out qmodel.npz       # int8 graph + accuracy vs float
python3 rtl_check.py   --qmodel qmodel.npy --img 7 --out "$TPU/build/jepa"      # 73 matmul cases
python3 vpu_vectors.py --qmodel qmodel.npy --img 7 --out build/vpu              # 3,584 VPU jobs
python3 export_rom.py  --cases "$TPU/build/jepa/cases" --out rom
$VL $VFLAGS --top-module tb_vpu -Mdir build/obj_vpu rtl/vpu.sv tb/tb_vpu.sv >/dev/null
./build/obj_vpu/Vtb_vpu +VEC=build/vpu | grep -E "tb_vpu:|VPU_"
$VL $VFLAGS --top-module tb_tpu_core_rom -Mdir build/obj_rom $INC rtl/rom_stream.sv tb/tb_tpu_core_rom.sv >/dev/null
(cd "$TPU" && "$J/build/obj_rom/Vtb_tpu_core_rom" +CASES=build/jepa/cases/cases.txt | grep -E "passed|_PASS|FAIL")
