#!/usr/bin/env bash
# Build and run every simulation check for the I3C/MCTP generator.
# Needs Verilator >= 5.030 (tested with 5.038) and python3; sigrok-cli (optional) for the decoder test. Usage: VL=/path/to/verilator ./run_sim.sh [+VCD]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
VL=${VL:-verilator}
VFLAGS="--binary -j 4 --timing -Wno-fatal -Wno-lint -Wno-style"
RTL="rtl/crc8_pec.sv rtl/sync_fifo.sv rtl/mctp_i3c_tx.sv rtl/i3c_sdr_engine.sv rtl/i3c_mctp_gen.sv"
TB="tb/i3c_target_bfm.sv tb/i3c_timing_mon.sv"
mkdir -p build/mut

echo "== Python reference model";  python3 sw/test_mctp_i3c.py
echo "== RTL lint"
$VL --lint-only -Wall -Wno-UNUSEDSIGNAL -Wno-DECLFILENAME -Wno-PINCONNECTEMPTY $RTL rtl/i3c_mctp_axil.sv rtl/kv260_i3c_top.sv --top-module kv260_i3c_top >build/lint.log 2>&1 || { cat build/lint.log; exit 1; }

echo "== tb_i3c_mctp (protocol scenarios)"
$VL $VFLAGS --top-module tb_i3c_mctp -Mdir build/obj $RTL $TB tb/tb_i3c_mctp.sv >build/verilate.log
./build/obj/Vtb_i3c_mctp +OUT=build +BUSVCD "$@" > build/sim.log
grep -E "^\[.*\] (ok|FAIL)|^timing:|tb_i3c_mctp:" build/sim.log | sed 's/^\[[0-9]*\] //'
grep -q "tb_i3c_mctp: ALL PASS" build/sim.log

echo "== Python cross-check of the bus dump"
python3 sw/check_dump.py build/sim_dump.txt

echo "== tb_axil (register interface)"
$VL $VFLAGS --top-module tb_axil -Mdir build/obj_axil $RTL rtl/i3c_mctp_axil.sv rtl/kv260_i3c_top.sv $TB tb/tb_axil.sv >build/verilate_axil.log
./build/obj_axil/Vtb_axil > build/sim_axil.log
grep -E "^(ok|FAIL)|tb_axil:" build/sim_axil.log
grep -q "tb_axil: ALL PASS" build/sim_axil.log

echo "== timing monitor self-test (each out-of-spec setting must be caught)"
for a in "+T_LOW_OD=15 +BUSVCD:tLOW_OD" "+T_HIGH_INIT=4:tHIGH_INIT" "+T_LOW_PP=3 +T_HIGH_PP=3:PP SCL period" "+T_CAS=0:tCAS" "+T_CBP=0:tCBP"; do
  args=${a%%:*}; want=${a##*:}
  ./build/obj/Vtb_i3c_mctp +OUT=build/mut $args > build/mut/sim.log || true
  if grep -q "TIMING VIOLATION: $want" build/mut/sim.log; then
    echo "caught: $args -> $want"
  else
    echo "MISSED: $args"; exit 1
  fi
done
echo "== sigrok decoders on the simulated bus capture"
if command -v "${SIGROK_CLI:-sigrok-cli}" >/dev/null; then
  python3 sigrok/test_decoders.py build
else
  echo "skipped: sigrok-cli not found (set SIGROK_CLI=/path/to/sigrok-cli)"
fi
echo "ALL CHECKS PASSED"
