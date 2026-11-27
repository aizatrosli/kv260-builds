#!/usr/bin/env bash
TPU=${TPU:-$HOME/tpuv1}; DEPS=${DEPS:-$HOME/deps}; VL=${VL:-verilator}; J=$(cd "$(dirname "$0")" && pwd)
$VL --binary -j 4 --timing -Wno-fatal -Wno-lint -Wno-style -Wno-UNOPTFLAT -Wno-MULTIDRIVEN --top-module tb_jepa_top -Mdir $J/build/obj_e2e \
  -I$DEPS/axi/include -I$DEPS/common_cells/include -y $DEPS/axi/src -y $DEPS/common_cells/src -y $DEPS/common_cells/src/deprecated \
  -y $TPU/rtl/compute -y $TPU/rtl/activation -y $TPU/rtl/control -y $TPU/rtl/memory -y $TPU/rtl/top \
  $DEPS/common_cells/src/cc_pkg.sv $DEPS/common_cells/src/deprecated/cf_math_pkg.sv $DEPS/axi/src/axi_pkg.sv $TPU/rtl/common/tpu_pkg.sv \
  $J/rtl/vpu.sv $J/rtl/jepa_top.sv $J/tb/tb_jepa_top.sv
