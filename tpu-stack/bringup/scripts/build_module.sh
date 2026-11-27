#!/usr/bin/env bash
# build_module.sh - build tpuv1's kernel module (tpu_kv260.ko) natively on the KV260,
# optionally also the u-dma-buf module that the m0..m5 board tests use.
#
# Run ON THE BOARD (as a normal user; it uses sudo only for apt):
#   bash build_module.sh                                  # build ~/tpuv1/software/driver/kernel
#   bash build_module.sh --install-headers                # apt install headers + toolchain first
#   bash build_module.sh --udmabuf                        # also build ikwzm/u-dma-buf
#   TPUV1=/path/to/tpuv1 bash build_module.sh
#
# Why native: the Kria Ubuntu kernel is a vendor flavour (22.04: *-xilinx-zynqmp, 5.15;
# 24.04: *-xilinx, 6.8). A module only loads into the exact kernel it was built against
# ("version magic" / "disagrees about version of symbol" errors otherwise). Building on the
# board against /lib/modules/$(uname -r)/build guarantees the match.
# After any apt upgrade that installs a new kernel + reboot, run this script again.
#
# Loading is NOT done here: load the PL first (install_app.sh --load), then run
# tpuv1's software/driver/tpu-load.sh, which insmods the .ko and binds it to a0000000.fabric.
set -euo pipefail

TPUV1="${TPUV1:-$HOME/tpuv1}"
UDMABUF_DIR="${UDMABUF_DIR:-$HOME/udmabuf}"
UDMABUF_REPO="https://github.com/ikwzm/udmabuf.git"
INSTALL_HEADERS=0
WITH_UDMABUF=0

for a in "$@"; do
    case "$a" in
        --install-headers) INSTALL_HEADERS=1;;
        --udmabuf)         WITH_UDMABUF=1;;
        -h|--help)         sed -n '2,20p' "$0"; exit 0;;
        *) echo "unknown argument: $a" >&2; exit 1;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
KREL="$(uname -r)"
KDIR="/lib/modules/$KREL/build"
echo "kernel: $KREL ($(uname -m))"

if (( INSTALL_HEADERS )); then
    sudo apt-get update
    # linux-headers-$(uname -r) is the per-ABI headers package for the running kernel
    # (24.04: linux-headers-6.8.0-NNNN-xilinx, which pulls linux-xilinx-headers-6.8.0-NNNN).
    if ! sudo apt-get install -y build-essential device-tree-compiler "linux-headers-$KREL"; then
        cat >&2 <<EOF
Could not install linux-headers-$KREL. Usual cause: the archive has moved on to a newer
kernel ABI and the running kernel's headers are no longer in the index. Fix:
    sudo apt full-upgrade && sudo reboot      # boots the newest kernel
    bash $0 --install-headers                  # then retry
(Answer "keep the local version" if flash-kernel asks about a modified file.)
EOF
        exit 1
    fi
fi

[[ -f "$KDIR/Makefile" ]] || die "no kernel build tree at $KDIR (run with --install-headers)"
if ! command -v make >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then die "make/gcc missing (run with --install-headers)"; fi

# ---- tpu_kv260.ko -----------------------------------------------------------------------------
SRC="$TPUV1/software/driver/kernel"
[[ -f "$SRC/tpu_kv260.c" && -f "$SRC/Makefile" ]] || die "tpuv1 driver source not found at $SRC (set TPUV1=...)"
# Build in a copy so the tpuv1 checkout stays clean (object files, Module.symvers ...).
BUILD="${BUILD_DIR:-$HOME/tpu_build/driver}"
rm -rf "$BUILD"
mkdir -p "$BUILD/kernel"
cp -a "$SRC/." "$BUILD/kernel/"
cp -a "$TPUV1/software/driver/include" "$BUILD/"
echo "-> building tpu_kv260.ko in $BUILD/kernel"
make -C "$BUILD/kernel" KDIR="$KDIR" -j"$(nproc)"
KO="$BUILD/kernel/tpu_kv260.ko"
[[ -f "$KO" ]] || die "build produced no $KO"

vm=$(modinfo -F vermagic "$KO")
echo "vermagic: $vm"
[[ "${vm%% *}" == "$KREL" ]] || die "vermagic '$vm' does not match running kernel $KREL"
echo "OK: $KO matches the running kernel"

# ---- u-dma-buf (optional, for the m0..m5 UIO board tests) --------------------------------------
if (( WITH_UDMABUF )); then
    if [[ ! -d "$UDMABUF_DIR/.git" ]]; then
        git clone --depth 1 "$UDMABUF_REPO" "$UDMABUF_DIR"
    fi
    echo "-> building u-dma-buf.ko in $UDMABUF_DIR"
    make -C "$UDMABUF_DIR" KERNEL_SRC="$KDIR" all
    UKO="$UDMABUF_DIR/u-dma-buf.ko"
    [[ -f "$UKO" ]] || die "u-dma-buf build produced no $UKO"
    echo "load with: sudo insmod $UKO udmabuf0=16777216     # 16 MiB, as tpuv1's deploy script used"
fi

cat <<EOF

Next steps (as root, PL image already loaded with install_app.sh --load):
    sudo KO=$KO sh $TPUV1/software/driver/tpu-load.sh
    ls -l /dev/tpu0 && dmesg | tail
Before 'xmutil unloadapp' ALWAYS run:
    sudo sh $TPUV1/software/driver/tpu-unload.sh
EOF
