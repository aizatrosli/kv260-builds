#!/usr/bin/env bash
# host_check.sh - check that the build PC can build the tpuv1 KV260 bitstream.
#
# Run on the HOST PC (not the board):
#   bash host_check.sh                 # quick checks (no Vivado start)
#   bash host_check.sh --vivado-tcl    # also start Vivado in batch mode and ask it whether the
#                                      # XCK26 part and the KV260 board files are available
#                                      # (takes 30-90 s)
#
# Environment overrides:
#   VIVADO_SETTINGS  path to settings64.sh (default: search /opt/AMD, /tools/Xilinx, /opt/Xilinx)
#   TPUV1            path to the tpuv1 checkout (default: $HOME/tpuv1)
#   MIN_FREE_GB      free space wanted on the build disk (default 100)
#
# Exit code: 0 = all required checks pass, 1 = at least one FAIL (WARNs do not fail).
# Nothing is installed or changed.
set -uo pipefail

WANT_VIVADO="2025.2"            # version tpuv1's README and old deploy scripts use
TPUV1="${TPUV1:-$HOME/tpuv1}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
RUN_TCL=0
[[ "${1:-}" == "--vivado-tcl" ]] && RUN_TCL=1

fails=0
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; fails=$((fails + 1)); }
info() { printf '[INFO] %s\n' "$*"; }

echo "== Host OS"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    info "OS: ${PRETTY_NAME:-unknown}"
    # Vivado 2025.2 lists Ubuntu 22.04/24.04 LTS and RHEL-family hosts; other distros often work
    # but are unsupported. Check UG973 for the exact list.
fi
info "Kernel: $(uname -r), arch $(uname -m)"
if [[ "$(uname -m)" == "x86_64" ]]; then pass "x86_64 host"; else fail "Vivado needs an x86_64 host"; fi

echo "== CPU / RAM"
cores=$(nproc 2>/dev/null || echo 1)
mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
info "$cores CPU threads, ${mem_gb} GiB RAM"
# build_kv260.tcl uses maxThreads 2 and -jobs 1 on purpose (tpuv1 lesson L22), so RAM matters
# more than cores. 16 GiB is a comfortable floor for an XCK26 (ZU5-class) implementation.
if (( mem_gb >= 15 )); then pass "RAM ~16 GiB or more"; elif (( mem_gb >= 8 )); then warn "RAM ${mem_gb} GiB: builds may swap"; else fail "RAM ${mem_gb} GiB is too little for implementation"; fi

echo "== Disk"
check_dir="$TPUV1"; [[ -d "$check_dir" ]] || check_dir="$HOME"
free_gb=$(df -P -BG "$check_dir" | awk 'NR==2 {gsub("G","",$4); print $4}')
info "free on $(df -P "$check_dir" | awk 'NR==2 {print $6}'): ${free_gb} GB"
if (( free_gb >= MIN_FREE_GB )); then pass "free space >= ${MIN_FREE_GB} GB"; else warn "only ${free_gb} GB free (Vivado install + one build wants ~${MIN_FREE_GB} GB to be comfortable)"; fi

echo "== Vivado"
settings="${VIVADO_SETTINGS:-}"
if [[ -z "$settings" ]]; then
    for c in /opt/AMD/$WANT_VIVADO/Vivado/settings64.sh /tools/Xilinx/$WANT_VIVADO/Vivado/settings64.sh \
             /tools/Xilinx/Vivado/$WANT_VIVADO/settings64.sh /opt/Xilinx/Vivado/$WANT_VIVADO/settings64.sh; do
        [[ -r "$c" ]] && { settings="$c"; break; }
    done
fi
if [[ -n "$settings" && -r "$settings" ]]; then
    pass "settings file: $settings"
    # settings64.sh references unset variables, so relax -u while sourcing it.
    set +u
    # shellcheck disable=SC1090
    . "$settings" >/dev/null 2>&1
    set -u
else
    warn "no settings64.sh for Vivado $WANT_VIVADO found; relying on PATH"
fi

if command -v vivado >/dev/null 2>&1; then
    ver_line=$(vivado -version 2>/dev/null | head -1)
    info "vivado -version: $ver_line"
    if grep -q "v$WANT_VIVADO" <<<"$ver_line"; then
        pass "Vivado $WANT_VIVADO"
    else
        warn "Vivado is not $WANT_VIVADO; tpuv1 was closed on $WANT_VIVADO (other versions may re-time differently, and 2026.x uses the new tiered licensing)"
    fi
else
    fail "vivado not on PATH (install Vivado $WANT_VIVADO ML Standard; Kria devices are covered by the no-cost edition)"
fi

if command -v bootgen >/dev/null 2>&1; then
    pass "bootgen: $(command -v bootgen) (needed for .bit -> .bit.bin)"
else
    fail "bootgen not on PATH (it ships with Vivado; or use bootgen-xlnx on the board)"
fi

echo "== License"
# Vivado ML Standard (no-cost) needs no license file and includes Kria (XCK26) devices.
# A license is only needed for paid IP or Enterprise-only devices. tpuv1 uses only the PS block
# (zynq_ultra_ps_e) and XPM memories, which need no license.
if [[ -n "${XILINXD_LICENSE_FILE:-}${LM_LICENSE_FILE:-}" ]]; then
    info "license env set: XILINXD_LICENSE_FILE='${XILINXD_LICENSE_FILE:-}' LM_LICENSE_FILE='${LM_LICENSE_FILE:-}'"
else
    info "no license env set (fine for Standard edition)"
fi
ls "$HOME"/.Xilinx/*.lic >/dev/null 2>&1 && info "found license files in ~/.Xilinx"

echo "== Other tools"
for t in git python3 make; do
    if command -v "$t" >/dev/null 2>&1; then pass "$t"; else fail "$t missing"; fi
done
if command -v bender >/dev/null 2>&1; then
    pass "bender $(bender --version 2>/dev/null | awk '{print $2}') (generates build/kv260_sources.tcl)"
else
    fail "bender missing: build_kv260.tcl sources build/kv260_sources.tcl, which only bender can generate (cargo install bender, or a release binary)"
fi
if command -v dtc >/dev/null 2>&1; then pass "dtc (optional on host)"; else warn "dtc missing (optional; you can compile the overlay on the board)"; fi
if python3 -c 'import numpy' 2>/dev/null; then pass "python3 numpy"; else warn "python3 numpy missing (needed for tpuv1 compilers / golden model)"; fi
if command -v ssh >/dev/null 2>&1; then pass "ssh"; else warn "ssh missing (needed to copy files to the board)"; fi
for d in /dev/ttyUSB*; do [[ -e "$d" ]] && info "serial device present: $d"; done

echo "== tpuv1 checkout"
if [[ -f "$TPUV1/soc/vivado/scripts/build_kv260.tcl" ]]; then
    pass "tpuv1 at $TPUV1"
    if [[ -f "$TPUV1/build/kv260_sources.tcl" ]]; then pass "build/kv260_sources.tcl exists"; else warn "build/kv260_sources.tcl not generated yet (see BRINGUP.md day 1, step 1)"; fi
else
    warn "tpuv1 not found at $TPUV1 (set TPUV1=...)"
fi

if (( RUN_TCL )) && command -v vivado >/dev/null 2>&1; then
    echo "== Vivado batch query (part + board files)"
    tmp=$(mktemp -d)
    cat > "$tmp/q.tcl" <<'TCL'
puts "PARTS=[llength [get_parts -quiet xck26-sfvc784-2LV-c]]"
puts "BOARDS=[join [get_board_parts -quiet *kv260*] ,]"
TCL
    out=$(cd "$tmp" && vivado -mode batch -nolog -nojournal -source q.tcl 2>&1)
    parts=$(sed -n 's/^PARTS=//p' <<<"$out")
    boards=$(sed -n 's/^BOARDS=//p' <<<"$out")
    if [[ "$parts" == "1" ]]; then pass "part xck26-sfvc784-2LV-c installed and usable"; else fail "part xck26-sfvc784-2LV-c not available (install Zynq UltraScale+ MPSoC/Kria device support)"; fi
    if [[ -n "$boards" ]]; then pass "KV260 board parts: $boards"; else warn "no KV260 board files (Tools > Vivado Store > Boards); build_kv260.tcl still runs without them"; fi
    rm -rf "$tmp"
fi

echo
if (( fails )); then echo "HOST_CHECK: $fails failure(s)"; exit 1; fi
echo "HOST_CHECK: OK"
