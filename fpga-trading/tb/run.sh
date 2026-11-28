#!/usr/bin/env bash
# Build the Verilator testbench for both output widths, generate stimulus from ftrade, run with
# and without random AXIS stalls, and check every vector against ftrade.features.TopBook.
#   tb/run.sh            (env: N_SYNTH=50000 N_FUZZ=20000 SEED=1 VL=verilator)
set -euo pipefail
TB=$(cd "$(dirname "$0")" && pwd); RTL="$TB/../rtl"; B="$TB/build"
VL=${VL:-verilator}; N_SYNTH=${N_SYNTH:-50000}; N_FUZZ=${N_FUZZ:-20000}; SEED=${SEED:-1}
SRC="$RTL/tob_engine.v $RTL/feed_rx.v $RTL/book_side.v $RTL/features.v $RTL/div_r4_pipe.v $RTL/delay_sr.v"
mkdir -p "$B"

echo "== lint"
$VL --lint-only -Wall --top-module tob_engine $SRC
$VL --lint-only -Wall --top-module tob_engine -GOUT_W=256 $SRC

for W in 32 256; do
  echo "== build OUT_W=$W"
  $VL --cc --exe --build -j 4 -O3 --top-module tob_engine -GOUT_W=$W -Mdir "$B/obj_$W" \
      -CFLAGS "-O2 -DOUT_W=$W" $SRC "$TB/tb_tob.cpp" >/dev/null
done

python3 "$TB/gen_stim.py" --suite all --n-synth "$N_SYNTH" --n-fuzz "$N_FUZZ" --seed "$SEED" --out "$B/stim_all"
fail=0
for W in 32 256; do
  # "<in_stall %> <out_stall %> <idle gap after each message>": line rate, random stalls on
  # both sides, heavy output back-pressure, sparse input, and spaced (unloaded latency)
  for cfg in "0 0 0" "30 40 0" "0 85 0" "70 0 0" "0 0 40"; do
    set -- $cfg
    o="$B/out_${W}_$1_$2_$3.txt"
    echo "== run OUT_W=$W in_stall=$1% out_stall=$2% gap=$3"
    "$B/obj_$W/Vtob_engine" stim="$B/stim_all/stim.txt" out="$o" seed="$SEED" in_stall=$1 out_stall=$2 gap=$3 || fail=1
    python3 "$TB/check.py" "$B/stim_all" "$o" --json "${o%.txt}.json" || fail=1
  done
done
[ $fail -eq 0 ] && echo "ALL PASS" || { echo "FAILURES"; exit 1; }
