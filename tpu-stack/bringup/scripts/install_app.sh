#!/usr/bin/env bash
# install_app.sh - install a PL image as a dfx-mgr "accelerated app" and (optionally) load it.
#
# Run ON THE BOARD as root.
#
#   sudo bash install_app.sh --bin tpu_kv260.bit.bin --dtso ../overlay/tpu_kv260.dtso --load
#   sudo bash install_app.sh --bin tpu_kv260.bit.bin --dtbo tpu_kv260.dtbo
#   sudo bash install_app.sh --name tpu_kv260 --load-only     # (re)load an installed app
#   sudo bash install_app.sh --name tpu_kv260 --remove        # unload (if active) and delete
#
# What dfx-mgr expects (AMD "On-target Utilities and Firmware"):
#   /lib/firmware/xilinx/<name>/<name>.bit.bin   bitstream converted by bootgen (-process_bitstream bin)
#   /lib/firmware/xilinx/<name>/<name>.dtbo      overlay; its fpga_full firmware-name must be <name>.bit.bin
#   /lib/firmware/xilinx/<name>/shell.json       {"shell_type": "XRT_FLAT", "num_slots": "1"}
# No .xclbin is needed: tpuv1 is a plain Vivado design without XRT.
#
# Safety: tpuv1 lesson L5 - a driver bound to a PL that is being unloaded can hang the CPU.
# This script refuses to unload/replace while tpu_kv260 is bound; run tpu-unload.sh first.
set -euo pipefail

NAME="tpu_kv260"
BIN=""
DTBO=""
DTSO=""
SHELL_JSON=""
DO_LOAD=0
LOAD_ONLY=0
REMOVE=0
FW_ROOT="/lib/firmware/xilinx"
DEV="a0000000.fabric"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,20p' "$0"; exit "${1:-0}"; }
die()   { echo "ERROR: $*" >&2; exit 1; }

while (( $# )); do
    case "$1" in
        --name)       NAME="$2"; shift 2;;
        --bin)        BIN="$2"; shift 2;;
        --dtbo)       DTBO="$2"; shift 2;;
        --dtso|--dts) DTSO="$2"; shift 2;;
        --shell-json) SHELL_JSON="$2"; shift 2;;
        --load)       DO_LOAD=1; shift;;
        --load-only)  LOAD_ONLY=1; DO_LOAD=1; shift;;
        --remove)     REMOVE=1; shift;;
        -h|--help)    usage 0;;
        *)            echo "unknown argument: $1" >&2; usage 1;;
    esac
done

[[ $EUID -eq 0 ]] || die "run as root (sudo)"
command -v xmutil >/dev/null 2>&1 || die "xmutil not found; this must run on Kria Ubuntu"
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "app name must be [A-Za-z0-9_-]"
DST="$FW_ROOT/$NAME"

tpu_bound() { [[ -e /sys/bus/platform/drivers/tpu_kv260/$DEV ]]; }
guard_unbound() {
    if tpu_bound; then
        die "tpu_kv260 driver is bound to $DEV. Run tpuv1 software/driver/tpu-unload.sh first (L5)."
    fi
}
active_app() {
    # xmutil listapps marks the active app's slot with a number in the "Active_slot" column; -1 = inactive.
    xmutil listapps 2>/dev/null | awk -v n="$1" '$1==n && $NF!="-1" {found=1} END {exit !found}'
}

load_app() {
    guard_unbound
    echo "-> xmutil unloadapp (whatever is loaded now, usually k26-starter-kits)"
    xmutil unloadapp >/dev/null 2>&1 || true
    echo "-> xmutil loadapp $NAME"
    if ! xmutil loadapp "$NAME"; then
        echo "loadapp failed. Last kernel messages:" >&2
        dmesg | tail -20 >&2
        exit 1
    fi
    state=$(cat /sys/class/fpga_manager/fpga0/state)
    echo "fpga0 state: $state"
    [[ "$state" == operating ]] || die "PL not operating after loadapp"
    if [[ -e /sys/bus/platform/devices/$DEV ]]; then
        echo "OK: platform device $DEV present"
    else
        echo "WARN: $DEV not present - the dtbo did not create the fabric node" >&2
    fi
    dmesg | grep -Ei 'fpga|overlay|afi|fabric|uio' | tail -8 || true
    echo "Next: build/load the driver (build_module.sh, then tpuv1 software/driver/tpu-load.sh)"
}

if (( REMOVE )); then
    guard_unbound
    if active_app "$NAME"; then
        echo "-> unloading active app $NAME"
        xmutil unloadapp || true
    fi
    [[ -d "$DST" ]] && rm -rf -- "$DST" && echo "removed $DST"
    echo "Restore the default with: sudo xmutil loadapp k26-starter-kits"
    exit 0
fi

if (( LOAD_ONLY )); then
    [[ -d "$DST" ]] || die "$DST does not exist; install first"
    load_app
    exit 0
fi

# ---- install --------------------------------------------------------------------------------
[[ -n "$BIN" && -f "$BIN" ]] || die "--bin <file.bit.bin> is required"
case "$BIN" in
    *.bit) die "$BIN is a raw .bit; convert it first: bootgen -image bit_bin.bif -arch zynqmp -process_bitstream bin";;
esac
[[ -n "$DTBO" || -n "$DTSO" ]] || DTSO="$HERE/../overlay/$NAME.dtso"
command -v dtc >/dev/null 2>&1 || die "dtc missing: sudo apt install device-tree-compiler"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [[ -n "$DTSO" ]]; then
    [[ -f "$DTSO" ]] || die "overlay source $DTSO not found"
    echo "-> compiling $DTSO"
    dtc -q -@ -I dts -O dtb -o "$tmp/$NAME.dtbo" "$DTSO"
    DTBO="$tmp/$NAME.dtbo"
fi
[[ -f "$DTBO" ]] || die "overlay $DTBO not found"

# The overlay must point at <name>.bit.bin, or dfx-mgr/fpga_manager loads nothing (or the wrong file).
fw=$(dtc -q -I dtb -O dts "$DTBO" | sed -n 's/.*firmware-name = "\(.*\)";.*/\1/p' | head -1)
[[ "$(basename "$fw")" == "$NAME.bit.bin" ]] || die "dtbo firmware-name is '$fw', expected '$NAME.bit.bin'"

# The board image must not already be in use while we overwrite it.
if active_app "$NAME"; then
    guard_unbound
    echo "-> $NAME is active; unloading before replacing files"
    xmutil unloadapp || true
fi

mkdir -p "$DST"
install -m 0644 "$BIN"  "$DST/$NAME.bit.bin"
install -m 0644 "$DTBO" "$DST/$NAME.dtbo"
if [[ -n "$SHELL_JSON" ]]; then
    install -m 0644 "$SHELL_JSON" "$DST/shell.json"
else
    printf '{\n    "shell_type" : "XRT_FLAT",\n    "num_slots": "1"\n}\n' > "$DST/shell.json"
fi
echo "installed:"
ls -l "$DST"
sha256sum "$DST/$NAME.bit.bin" | sed 's/^/    /'

# dfx-mgr watches /lib/firmware/xilinx with inotify; give it a moment, restart if it missed it.
sleep 1
if ! xmutil listapps 2>/dev/null | awk -v n="$NAME" '$1==n {f=1} END {exit !f}'; then
    echo "-> $NAME not listed yet; restarting dfx-mgr"
    systemctl restart dfx-mgr || true
    sleep 2
fi
xmutil listapps

(( DO_LOAD )) && load_app
exit 0
