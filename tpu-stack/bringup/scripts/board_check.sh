#!/usr/bin/env bash
# board_check.sh - read-only health report for a KV260 running Kria Ubuntu.
#
# Run ON THE BOARD, preferably as root so xmutil, dmesg and debugfs work:
#   sudo bash board_check.sh            # full report
#   sudo bash board_check.sh --no-stats # skip the (slow, ~2 s) xmutil platformstats call
#
# It changes nothing: no apps are loaded or unloaded, no modules inserted.
# Save the output from day 0 as your baseline:  sudo bash board_check.sh | tee ~/baseline.txt
set -uo pipefail

APP="${APP:-tpu_kv260}"
DEV="a0000000.fabric"
NO_STATS=0
[[ "${1:-}" == "--no-stats" ]] && NO_STATS=1

sec()  { printf '\n== %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
ok()   { printf '[ OK ] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

[[ $EUID -eq 0 ]] || warn "not root: xmutil/dmesg/debugfs sections may be empty (re-run with sudo)"

sec "OS and kernel"
# shellcheck disable=SC1091
[[ -r /etc/os-release ]] && . /etc/os-release && info "OS: ${PRETTY_NAME:-?}"
kr=$(uname -r)
info "kernel: $kr"
case "$kr" in
    *-xilinx-zynqmp) info "Ubuntu 22.04 style kernel flavour (-xilinx-zynqmp)";;
    *-xilinx)        info "Ubuntu 24.04 style kernel flavour (-xilinx)";;
    *)               warn "unexpected kernel flavour; is this the Kria Ubuntu image?";;
esac
info "cmdline: $(cat /proc/cmdline)"
if grep -q 'uio_pdrv_genirq.of_id=generic-uio' /proc/cmdline; then
    ok "generic-uio enabled on the kernel command line"
elif [[ -r /sys/module/uio_pdrv_genirq/parameters/of_id ]] && grep -q generic-uio /sys/module/uio_pdrv_genirq/parameters/of_id; then
    ok "uio_pdrv_genirq of_id=generic-uio (module parameter)"
else
    warn "generic-uio not enabled: /dev/uioN for the fabric node will not appear (only needed by m0..m5 board tests)"
fi
if [[ -e "/lib/modules/$kr/build/Makefile" ]]; then ok "kernel headers for $kr installed"; else warn "no kernel headers for $kr (build_module.sh --install-headers)"; fi
info "uptime: $(uptime -p 2>/dev/null || uptime)"
info "memory: $(awk '/MemTotal|CmaTotal|CmaFree/ {printf "%s %d MiB  ", $1, $2/1024}' /proc/meminfo)"

sec "Boot firmware (QSPI A/B)"
if have xmutil; then
    xmutil bootfw_status 2>&1 | sed 's/^/    /'
else
    warn "xmutil not found (Kria Ubuntu ships it; on 20.04 it is 'xlnx-config --xmutil')"
fi

sec "Board identity"
if have xmutil; then xmutil boardid 2>&1 | grep -Ei 'product|revision|serial|name' | head -8 | sed 's/^/    /'; fi

sec "Accelerated apps (dfx-mgr)"
if have systemctl; then info "dfx-mgr.service: $(systemctl is-active dfx-mgr 2>/dev/null || echo unknown)"; fi
if have xmutil; then xmutil listapps 2>&1 | sed 's/^/    /'; fi
if [[ -d /lib/firmware/xilinx/$APP ]]; then
    info "/lib/firmware/xilinx/$APP:"
    find "/lib/firmware/xilinx/$APP" -maxdepth 1 -type f -printf '    %M %s %TY-%Tm-%Td %TH:%TM %f\n'
else
    info "/lib/firmware/xilinx/$APP not installed yet"
fi

sec "FPGA manager / PL"
for f in name state flags; do
    [[ -r /sys/class/fpga_manager/fpga0/$f ]] && info "fpga0 $f: $(cat /sys/class/fpga_manager/fpga0/$f)"
done
if [[ -r /sys/kernel/debug/clk/pl0_ref/clk_rate ]]; then
    rate=$(cat /sys/kernel/debug/clk/pl0_ref/clk_rate)
    info "pl0_ref (pl_clk0) rate: $rate Hz, enable_count $(cat /sys/kernel/debug/clk/pl0_ref/clk_enable_count 2>/dev/null || echo ?)"
    if (( rate >= 99000000 && rate <= 101000000 )); then ok "pl_clk0 ~100 MHz"; else warn "pl_clk0 is not ~100 MHz (tpuv1 is timed for 100 MHz)"; fi
else
    info "pl0_ref clock not visible (mount debugfs, run as root)"
fi
if ls -d /sys/firmware/devicetree/base/__symbols__ >/dev/null 2>&1; then ok "base DT has __symbols__ (overlays can resolve labels)"; else warn "base DT has no __symbols__: overlays using &amba/&fpga_full will fail to apply"; fi
find /sys/kernel/config/device-tree/overlays/ -mindepth 1 -maxdepth 1 -printf '    overlay: %f\n' 2>/dev/null

sec "TPU device"
if [[ -e /sys/bus/platform/devices/$DEV ]]; then
    ok "platform device $DEV present"
    drv=$(basename "$(readlink -f /sys/bus/platform/devices/$DEV/driver 2>/dev/null)" 2>/dev/null)
    info "bound driver: ${drv:-none}"
else
    info "platform device $DEV absent (app not loaded, or the dtbo lacks the fabric node)"
fi
for u in /sys/class/uio/uio*; do
    [[ -e "$u/name" ]] || continue
    info "$(basename "$u"): name=$(cat "$u/name") addr=$(cat "$u/maps/map0/addr" 2>/dev/null)"
done
have lsmod && lsmod | grep -E '^(tpu_kv260|u_dma_buf|uio_pdrv_genirq|zocl)' | sed 's/^/    module: /'
if [[ -e /dev/tpu0 ]]; then ok "/dev/tpu0 present"; else info "/dev/tpu0 absent"; fi
[[ -e /dev/udmabuf0 ]] && info "/dev/udmabuf0 present ($(cat /sys/class/u-dma-buf/udmabuf0/size 2>/dev/null) bytes)"
if [[ -r /sys/bus/platform/devices/$DEV/jobs_ok ]]; then
    for a in jobs_ok jobs_failed irq_count irq_missed buf_bytes; do
        [[ -r /sys/bus/platform/devices/$DEV/$a ]] && info "$a: $(cat /sys/bus/platform/devices/$DEV/$a)"
    done
fi

sec "Kernel log (FPGA, overlay, TPU, AXI, DMA errors)"
if dmesg >/dev/null 2>&1; then
    dmesg | grep -Ei 'fpga|overlay|tpu|fabric|uio|afi|dfx|u-dma-buf|udmabuf|SError|Internal error|asynchronous|vermagic|disagrees|taint|firmware|thermal|throttl' \
        | tail -40 | sed 's/^/    /'
else
    warn "cannot read dmesg (run as root)"
fi

sec "Thermal / fan"
for z in /sys/class/thermal/thermal_zone*; do
    [[ -r "$z/temp" ]] && info "$(cat "$z/type"): $(( $(cat "$z/temp") / 1000 )) C"
done
for h in /sys/class/hwmon/hwmon*; do
    n=$(cat "$h/name" 2>/dev/null)
    case "$n" in
        *fan*) info "fan hwmon ($n): pwm1=$(cat "$h/pwm1" 2>/dev/null) (0..255)";;
        ina260*)      info "INA260 ($n): power1_input=$(cat "$h/power1_input" 2>/dev/null) uW, curr1=$(cat "$h/curr1_input" 2>/dev/null) mA, in1/2=$(cat "$h/in1_input" "$h/in2_input" 2>/dev/null | tr '\n' ' ')mV";;
    esac
done

if (( ! NO_STATS )) && have xmutil; then
    sec "xmutil platformstats -p (power + temperatures)"
    timeout 20 xmutil platformstats -p 2>&1 | sed 's/^/    /'
fi

sec "Network"
ip -brief addr 2>/dev/null | sed 's/^/    /'
if ping -c1 -W2 8.8.8.8 >/dev/null 2>&1; then ok "internet reachable"; else warn "no internet (apt will fail)"; fi

sec "Storage"
df -h / /boot/firmware 2>/dev/null | sed 's/^/    /'
echo
echo "BOARD_CHECK: done"
