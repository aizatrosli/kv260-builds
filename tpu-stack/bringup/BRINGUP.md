# KV260 bring-up guide for the tpuv1 INT8 TPU

Written 2026-11-27 for a new Kria KV260 Vision AI Starter Kit. The goal: get from an unopened box to tpuv1 (github.com/ilmiahmad/tpuv1) running in the PL, with `/dev/tpu0` working and the repo's board tests passing. After that, the JEPA encoder work in `../../jepa-tpu` can move onto the board.

Sources are numbered `[S#]` and listed at the end. Anything marked **(inferred)** comes from reading the tpuv1 sources, not from AMD documentation. Check those items on the board before relying on them.

Files next to this guide:

| Path | Purpose |
|---|---|
| `scripts/host_check.sh` | Checks the build PC: Vivado 2025.2, bootgen, bender, disk, RAM, license, and (with `--vivado-tcl`) whether the part and board files are installed |
| `scripts/board_check.sh` | Read-only board report: OS, kernel, boot firmware, `xmutil listapps`, FPGA state, PL clock, TPU device, dmesg, thermal, `platformstats` |
| `scripts/install_app.sh` | Installs `.bit.bin` + `.dtbo` + `shell.json` in `/lib/firmware/xilinx/<name>/`, then runs `xmutil loadapp`. Also does reload and remove |
| `scripts/build_module.sh` | Builds `tpu_kv260.ko` on the board against the running kernel and checks vermagic. Optionally builds `u-dma-buf` |
| `overlay/tpu_kv260.dtso` | Device-tree overlay source for the TPU image (tpuv1 doesn't ship one, see §5.2) |
| `overlay/shell.json` | dfx-mgr descriptor (`XRT_FLAT`, 1 slot) |
| `TIMELINE.md` | Order of work for days 0–3, including a "first 2 hours" checklist |

---

## 0. Key facts (as of 2026-11-27)

| Topic | Fact | Source |
|---|---|---|
| SOM silicon | XCK26-SFVC784-2LV-C (custom Zynq UltraScale+ MPSoC, ZU5EV class). 256,200 system logic cells, 117,120 LUTs, 234,240 FFs, 1,248 DSP48E2, 144 BRAM36 (5.1 Mb), 64 URAM (18 Mb), 3.5 Mb distributed RAM. 4 GB 64-bit DDR4 (non-ECC), 512 Mb QSPI, 16 GB eMMC, TPM 2.0. Quad A53, dual R5F, VCU | [S14] DS987 v1.2 Table 5 |
| tpuv1 footprint | 32.2k LUT (27%), 40k FF (17%), 328 DSP (26%), 32 URAM (50%), 44 BRAM36 (31%) at 100 MHz | tpuv1 README |
| Ubuntu images | **24.04 LTS Server** (`iot-limerick-kria-classic-server-2404-classic-24.04-x07-20250423.img.xz`), **22.04 LTS Desktop** (`iot-limerick-kria-classic-desktop-2204-20240304-165.img.xz`), plus Ubuntu Core 24/22 developer previews | [S3] |
| Which Ubuntu | AMD recommends 24.04 for new users. The AMD example apps (smartcam etc.) are **not supported on 24.04**, but tpuv1 doesn't need them | [S4] |
| Kernel | 24.04: `linux-xilinx` 6.8.0-10xx (`-xilinx` flavour; newest in the archive is 6.8.0-1043, 2026-11-18). 22.04: 5.15 `-xilinx-zynqmp` | [S16], [S17] |
| Boot firmware | QSPI A/B images, updated with `xmutil bootfw_update`. Latest K26 release is **1.08 (KV260 build, 2026.6 base)**. Ubuntu 24.04 needs **≥ 1.02**. Avoid 1.04 if you use DisplayPort with 2025.1 EDF images | [S1], [S2] |
| Vivado | tpuv1 targets **Vivado 2025.2**. 2025.2 ML Standard is no-cost, needs **no license file**, and covers Kria devices. 2026.1 brought tiered licensing (reports say the free tier changes on Linux; unconfirmed), so stay on 2025.2 | [S11], [S12] |
| Vitis / PetaLinux | Not needed. tpuv1 is a plain Vivado design (no XRT, no xclbin), loaded by dfx-mgr | [S6] |
| Kria-PYNQ | README covers KV260/KR260/KD240 on **Ubuntu 22.04** only (recommends 2022.1 boot FW). No 24.04 statement and no tagged releases. Treat it as optional and 22.04-only. tpuv1 doesn't use PYNQ | [S5] |
| Power | 12 V, 3 A barrel supply on J12. Not in the kit (the AMD Basic Accessory Pack includes one). There is no power switch: the board boots when power is applied | [S4] |
| microSD | 16 GB UHS-I recommended (32 GB gives room for builds). Goes in J11 | [S4] |
| Console | Micro-USB J4 → FTDI bridge, 4 interfaces. The UART is the **second** port (e.g. `/dev/ttyUSB1`, sometimes `ttyUSB2`), 115200 8N1, no flow control | [S4] uart.html |
| Thermal | Active fansink in the kit. The kit dissipates up to 15 W. Ambient 0–35 °C operating | [S13] |
| Ethernet | 1 GbE RJ45 **J10**. This is also the only port the boot image recovery tool works on | [S1] |
| SD lock-in (24.04) | An SD card is bound to the **first** kit it boots on, because the default bitstream is picked from the EEPROM once | [S4] known_issues |

---

## 1. Shopping / hardware checklist

Order these with the kit, since the kit box has only the SOM, carrier, and fansink:

- [ ] **12 V / 3 A (36 W) DC supply** with the barrel plug that fits J12. The easiest route is the AMD **Kria Basic Accessory Pack**, which has the PSU, USB cable, Ethernet cable, and SD card. If you buy a third-party supply, check plug size and polarity against UG1089 before applying power. Community posts disagree on whether it's 2.1 mm or 2.5 mm, so measure it or buy the AMD one [S4][S18].
- [ ] **2× microSD cards, 16–32 GB, UHS-I, A1/A2 class** from a known brand. One is the working card and one holds a known-good image for rollback (§9).
- [ ] **USB-A to micro-B cable** with data lines (for J4: UART and JTAG).
- [ ] **Ethernet cable**, plus a free port on a router with DHCP and internet access.
- [ ] Optional: HDMI or DP monitor and a USB keyboard. Not needed if you use serial and SSH.
- [ ] Optional: **USB power meter or bench supply** that shows 12 V current. It's a second opinion next to the on-SOM INA260 in §8.
- [ ] Optional: a small USB fan or open-air placement for long runs. Keep the fansink unobstructed.
- [ ] Anti-static mat or strap. The carrier PCB is bare.

Don't remove the SOM from the carrier, and don't take off the fansink.

---

## 2. Host PC setup

**Requirements.** x86_64 Linux (Ubuntu 22.04/24.04 LTS is the usual choice, see UG973 for the list) or Windows. tpuv1's old deploy scripts assume Linux paths (`/opt/AMD/2025.2/Vivado/settings64.sh`). Plan on roughly 16 GB RAM, about 100 GB free disk, and an SD card reader.

1. **Install Vivado 2025.2 ML Standard Edition.** Use the AMD Unified Installer 2025.2. Select *Vivado* and, under devices, at least *Zynq UltraScale+ MPSoC* / *Kria SOMs*. Vitis isn't required.
   - **License:** none needed. UG973 2025.2 lists all Kria devices under ML Standard Edition, and ML Standard has needed no license since 2016.x [S11]. tpuv1 uses only `zynq_ultra_ps_e` and XPM memories, so no licensed IP is involved.
   - **Download 2025.2 now and keep the installer.** Since 2026.1, AMD uses a tiered licensing model. AMD has said 2025.2 can be used "forever" and gets fixes until 2026.3, but reports say the free tier changes for Linux. It isn't clear how long the 2025.2 download will stay available [S12].
   - Install the KV260 board files: *Tools → Vivado Store → Boards → Kria KV260*. `build_kv260.tcl` still runs without them, because it applies the board preset only if `get_board_parts *kv260_som*` finds something.
2. **Other tools:** `git`, `python3` + `numpy`, `make`, `dtc` (`device-tree-compiler`), `ssh`/`scp`, and a serial terminal (`picocom`, `minicom`, or `screen`). On Windows, install the FTDI VCP driver.
3. **Bender** (PULP dependency manager). `build_kv260.tcl` sources `build/kv260_sources.tcl`, and nothing in the repo creates that file. Bender generates it (§5.1). Install it with `cargo install bender` or a release binary.
4. **Serial access on Linux:** `sudo usermod -aG dialout $USER`, then log out and back in.
5. Run the check:
   ```bash
   TPUV1=~/tpuv1 bash scripts/host_check.sh --vivado-tcl
   ```

---

## 3. Day 0: flash, firmware, first boot

### 3.1 Flash the SD card
1. Download **Ubuntu 24.04 LTS Server for Kria** from https://ubuntu.com/download/amd (file `iot-limerick-kria-classic-server-2404-classic-24.04-x07-20250423.img.xz`) [S3].
   - Pick 22.04 Desktop instead only if you need Kria-PYNQ or the AMD example apps [S4][S5]. Everything below works on both. Kernel flavours and header package names differ (§5.4).
2. Write it with Raspberry Pi Imager ("Use custom" image) or with `xzcat image.img.xz | sudo dd of=/dev/sdX bs=4M conv=fsync status=progress` [S4].
3. Optional: also write the same image to the second card and label it "known good".

### 3.2 Connect
Follow this order [S4 connections]: SD in **J11**. Micro-USB in **J4**. Ethernet in **J10**. 12 V plug into **J12**, *then* the PSU into mains. The board powers on as soon as mains is applied.

Open the console **before** applying power, so you see the boot firmware and U-Boot output:
```bash
dmesg | grep ttyUSB          # 4 ports appear; the UART is the 2nd (often /dev/ttyUSB1)
picocom -b 115200 /dev/ttyUSB1
```

What a healthy power-on looks like: the fan spins (it's loud until Linux takes PWM control), the console shows Xilinx Zynq MP First Stage Boot Loader / PMU FW / U-Boot, and Ubuntu boots. The first boot resizes the filesystem and runs cloud-init, so it takes a few minutes.

### 3.3 First login
- Log in as `ubuntu` / `ubuntu`. You must change the password [S4 boot]. If `snap` says "device not yet seeded", wait a few minutes [S4 known issues].
- Run `ip -brief addr` and note the IP address. Then `ping -c3 8.8.8.8`.
- Optional: `sudo hostnamectl set-hostname kv260`, and add your SSH key (`ssh-copy-id ubuntu@<ip>`). Put a `Host kv260` entry in `~/.ssh/config` on the host.
- **Baseline:** copy `scripts/` to the board and run `sudo bash board_check.sh | tee ~/day0_baseline.txt`.

### 3.4 Boot firmware: check, then update *before* `apt upgrade`
The KV260 boots from QSPI firmware (two slots, A and B), and Linux comes from the SD card. A mismatched firmware is the most common reason Ubuntu doesn't boot [S5].

```bash
sudo xmutil bootfw_status                 # shows Image A/B versions, active/requested image
```
- If Ubuntu boots at all, the firmware is compatible enough to update from Linux. 24.04 needs **≥ 1.02** [S2].
- Update the firmware **before** you run `apt upgrade`. The Kria wiki warns that `image_update` and `xmutil bootfw_*` may stop working after apt update/upgrade (this is a 22.04-era note, but doing it in this order is safe either way) [S2].

Update from the Ubuntu PPA package [S1]:
```bash
sudo add-apt-repository ppa:ubuntu-xilinx/default --yes
sudo apt update
sudo apt install kria-k26-bootfw
ls /usr/lib/firmware/xilinx/bootfw/              # BOOT-<variant>.bin, pick the k26 / kv260 one
sudo xmutil bootfw_update -i /usr/lib/firmware/xilinx/bootfw/<BOOT-k26...>.bin
sudo reboot
# IMMEDIATELY after this reboot, log in and validate, or the next reset falls back to the old slot:
sudo xmutil bootfw_update -v
sudo xmutil bootfw_status
```
The alternative is to download the latest KV260 `BOOT.BIN` (1.08 as of today) from the K26 table on the wiki [S1] and pass it to `-i`. Note that `bootfw_status` keeps showing the *factory* (recovery) version, which never changes [S1].

**If Linux doesn't boot at all**, use the **Boot Image Recovery Tool** [S1][S19]:
1. Connect the host PC directly to **J10** and give it a static IP of 192.168.0.x (not .111).
2. Hold the **FWUEN** (firmware update) button while you apply power. The UART shows the recovery app.
3. Browse to `http://192.168.0.111`, upload the `BOOT.BIN` to slot A or B, mark it requested, and power-cycle.

### 3.5 Packages
Use the AMD-documented PPAs for 24.04 [S4 boot]. Answer **"keep the local version currently installed"** if flash-kernel prompts. If the run stops with flash-kernel "deferring update", run it again [S4 known issues].
```bash
sudo add-apt-repository ppa:xilinx-apps --yes
sudo add-apt-repository ppa:ubuntu-xilinx/default --yes
sudo add-apt-repository ppa:xilinx-apps/xilinx-drivers --yes
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git device-tree-compiler python3-numpy picocom \
                    "linux-headers-$(uname -r)"
sudo reboot              # if the upgrade installed a new kernel
```
After that reboot, `uname -r` may have changed. That's why the kernel module is built on day 1, not day 0. The AMD guide also adds `ppa:lely/ppa`, which is only needed for CANopen and the AMD apps.

**Optional: Kria-PYNQ** (22.04 image only, per its README) [S5]:
`git clone https://github.com/Xilinx/Kria-PYNQ && cd Kria-PYNQ && sudo bash install.sh -b KV260` (about 25 min). JupyterLab is then at `http://<ip>:9090/lab`. PYNQ's `Overlay()` loads a `.bit` + `.hwh` on its own and bypasses dfx-mgr. Don't mix it with `xmutil loadapp` in the same session.

**End of day 0:** `sudo bash board_check.sh | tee ~/day0_after_upgrade.txt`. `xmutil listapps` should list `k26-starter-kits` as active.

---

## 4. What tpuv1 needs on the board (summary)

| Item | Where it comes from | On the board |
|---|---|---|
| `tpu_kv260.bit.bin` | `build_kv260.tcl` → `build/kv260/tpu_kv260.bit` → `bootgen` with `soc/vivado/scripts/bit_bin.bif` | `/lib/firmware/xilinx/tpu_kv260/tpu_kv260.bit.bin` |
| `tpu_kv260.dtbo` | **Not in tpuv1.** Use `overlay/tpu_kv260.dtso` from this folder | `/lib/firmware/xilinx/tpu_kv260/tpu_kv260.dtbo` |
| `shell.json` | `overlay/shell.json`: `{"shell_type":"XRT_FLAT","num_slots":"1"}` [S6] | `/lib/firmware/xilinx/tpu_kv260/shell.json` |
| `tpu_kv260.ko` | `software/driver/kernel/` built on the board (`build_module.sh`) | Loaded by `software/driver/tpu-load.sh` |
| `build_id.txt` | Written by `build_kv260.tcl` (BUILD_ID is baked into the bitstream) | Passed to board tests as `--build-id` |
| `u-dma-buf.ko` (only for m0–m5 UIO tests) | github.com/ikwzm/udmabuf, built on the board | `insmod u-dma-buf.ko udmabuf0=16777216` |

AMD's rules for an app directory [S6]: it lives at `/lib/firmware/xilinx/<app>/`, the `.bit.bin` and `.dtbo` share a base name, and `shell.json` sits next to them. dfx-mgr picks up new folders with inotify. `xmutil listapps`, `loadapp`, and `unloadapp` are front ends to dfx-mgr. Only one flat app can be loaded at a time.

---

## 5. Day 1: build and load the TPU

### 5.1 Build the bitstream (host)
```bash
cd ~/tpuv1
source /opt/AMD/2025.2/Vivado/settings64.sh      # adjust to your install path

# 1. Fetch the pinned PULP deps (Bender.lock / Bender.local are committed)
bender checkout

# 2. Generate the source list build_kv260.tcl expects   (inferred target list, see note)
mkdir -p build
bender script vivado -t rtl -t tpu -t m0_elab -t m3 -t m4 -t kv260 > build/kv260_sources.tcl

# 3. Synthesise + implement (single-threaded on purpose: maxThreads 2, -jobs 1)
vivado -mode batch -source soc/vivado/scripts/build_kv260.tcl -log build/kv260/vivado.log -journal build/kv260/vivado.jou
#   look for:  KV260_BUILD_DONE build_id=0x........ wns=... uram=32 bram=...
grep -E "WNS|Timing constraints are met|VIOLATED" build/kv260/timing.rpt | head

# 4. .bit -> .bit.bin (byte-swapped, header-less format the FPGA manager wants)
cd build/kv260 && bootgen -image ../../soc/vivado/scripts/bit_bin.bif -arch zynqmp -process_bitstream bin -w
ls -l tpu_kv260.bit.bin build_id.txt
```
Notes:
- **The target list is inferred.** The scripts that generated `kv260_sources.tcl` were removed in tpuv1 commit `8d4e0a0` ("Remove scripts/"). The list above covers every module `kv260_top` instantiates: `tpu_fabric` (target `tpu`), `idma_subsystem` + `tpu_m0_pkg` (`m0_elab`), the array (`m3`), the core and activation (`m4`), and the top (`kv260`), plus iDMA's own `rtl` target (the old `m5_dma_prepare.sh` used `-t rtl -t m0_elab`). If synthesis reports a missing module, add the target that owns it.
- Expect WNS of about +1.25 ns (README). A negative WNS means don't load it. Look at `timing.rpt` first.
- Copy to the board: `scp build/kv260/tpu_kv260.bit.bin build/kv260/build_id.txt kv260:~/tpu/`
- `bootgen` also exists on the board (`sudo apt install bootgen-xlnx`) if you'd rather convert there [S9].

### 5.2 The device-tree overlay
tpuv1 doesn't ship a dtbo. Its deleted `scripts/deploy.sh` cloned the dtbo and `shell.json` of a private reference app (`ne_vslam_pja`, described there as "fclk0 100 MHz, afi config-afi (HPM0 128-bit)") and only rewrote `firmware-name`. You won't have that app, so `overlay/tpu_kv260.dtso` reproduces the same content explicitly. It follows the structure of AMD's KV260 overlays [S8] and is sized to tpuv1's block design (`soc/vivado/bd/kv260_ps.tcl`) and driver:

| Overlay item | Value | Why |
|---|---|---|
| `&fpga_full firmware-name` | `"tpu_kv260.bit.bin"` | Must match the file name in the app folder |
| `resets` | `zynqmp_reset 116..119` | pl_resetn0..3, as in AMD overlays [S8] |
| `clocking0` | `zynqmp_clk 71` → 99 999 001 Hz | pl_clk0 = 100 MHz (tpuv1 is timed for it) [S8] |
| `afi0 config-afi` | regid 4/5 (HP0) = 0 → 128-bit. regid 14 = **0xA00** → HPM0 and HPM1 128-bit | BD sets MAXIGP0 and SAXIGP2 (HP0) to 128-bit. Bit meanings from [S10] |
| `fabric@a0000000` | `generic-uio`, reg 0xA000_0000 + 64 KiB, `interrupts = <0 89 4>` | Driver binds to `a0000000.fabric`. pl_ps_irq0 = SPI 89 = IRQ ID 121 (both numbers appear in tpuv1 comments) |

The memory windows above +64 KiB (UB 0xA010_0000, ACC 0xA020_0000, WFIFO 0xA030_0000, IFIFO 0xA040_0000, which together make up the 8 MiB HPM0 aperture) are not in the overlay. The driver doesn't map them, and only the m2/m5 tests read them through `/dev/mem`.

The `generic-uio` compatible only produces `/dev/uioN` if `uio_pdrv_genirq` is told to claim it. Use the kernel argument `uio_pdrv_genirq.of_id=generic-uio`, or `options uio_pdrv_genirq of_id=generic-uio` in `/etc/modprobe.d/` when the driver is built as a module [S17]. Check with `board_check.sh`. `/dev/tpu0` doesn't depend on this, because `tpu-load.sh` binds by `driver_override`. The m0–m5 tests do depend on it.

Compile it (on the host or the board): `dtc -@ -I dts -O dtb -o tpu_kv260.dtbo overlay/tpu_kv260.dtso`. `install_app.sh` does this for you.

### 5.3 Install and load the app (board)
```bash
cd ~/bringup/scripts
sudo bash install_app.sh --bin ~/tpu/tpu_kv260.bit.bin --dtso ../overlay/tpu_kv260.dtso --load
```
This creates `/lib/firmware/xilinx/tpu_kv260/{tpu_kv260.bit.bin,tpu_kv260.dtbo,shell.json}`, checks that the dtbo's `firmware-name` matches, waits for dfx-mgr to list the app, then runs `xmutil unloadapp` (unloads k26-starter-kits) and `xmutil loadapp tpu_kv260`. Finally it checks that `fpga0` is `operating` and that `a0000000.fabric` exists. To check by hand:
```bash
xmutil listapps                                   # tpu_kv260 active
cat /sys/class/fpga_manager/fpga0/state           # operating
sudo cat /sys/kernel/debug/clk/pl0_ref/clk_rate   # ~99999001
ls /sys/bus/platform/devices/ | grep fabric       # a0000000.fabric
```

### 5.4 Build and load the kernel module (board)
Headers:

| Image | `uname -r` looks like | Headers package |
|---|---|---|
| 24.04 | `6.8.0-NNNN-xilinx` | `linux-headers-$(uname -r)` (pulls `linux-xilinx-headers-6.8.0-NNNN`) [S16] |
| 22.04 | `5.15.0-NNNN-xilinx-zynqmp` | `linux-headers-$(uname -r)` |

```bash
git clone https://github.com/ilmiahmad/tpuv1 ~/tpuv1
bash ~/bringup/scripts/build_module.sh --install-headers --udmabuf
sudo KO=$HOME/tpu_build/driver/kernel/tpu_kv260.ko sh ~/tpuv1/software/driver/tpu-load.sh
#   -> "/dev/tpu0 ready"; dmesg: "tpu_kv260 a0000000.fabric: /dev/tpu0 ready (interrupt, irq NN)"
cat /sys/bus/platform/devices/a0000000.fabric/{jobs_ok,irq_count}
```
`build_module.sh` builds out of tree in `~/tpu_build`, so the checkout stays clean, and it refuses a `.ko` whose vermagic doesn't match `uname -r`. The driver has no `of_match_table` on purpose. It binds only through `driver_override`, which `tpu-load.sh` sets. Module parameters: `use_irq` (default 1), `max_buf_mb` (64), `total_buf_mb` (256).

The driver allocates DMA buffers with `dma_alloc_coherent` and doesn't reserve its own memory, so large jobs need CMA. `board_check.sh` prints `CmaTotal`/`CmaFree`. If it's smaller than you need, add `cma=512M` to the kernel command line.

### 5.5 Run tpuv1's tests (board)
Order: cheapest and safest first. Build the test inputs on the **host** and copy them over.

```bash
# host
cd ~/tpuv1
python3 software/runtime/tests/test_host.py                 # runtime vs mock device
python3 model/mlp/compile_phase_b.py                        # -> build/m5/cases  (runs compile_mlp.py first)
python3 model/mlp/export_bundle.py                          # -> build/bundles   (MLP bundles for the driver test)
python3 software/runtime/tests/board_test_driver.py --mock --bundles build/bundles   # same checks on the mock
scp -r build/m5/cases build/bundles kv260:~/tpu/
```
```bash
# board: A) driver path (/dev/tpu0)  -- the main acceptance test
cd ~/tpuv1/software/runtime/tests
sudo python3 board_test_driver.py --bundles ~/tpu/bundles --only basic,negatives,bundles
sudo python3 board_test_driver.py --bundles ~/tpu/bundles            # full: + recovery,timeout,loop,concurrent,signals
#   final line TPU_DRIVER_TEST_PASS; exit 0
sudo python3 ../tools/run_case.py ~/tpu/cases                        # Phase B MLP/2D-DMA/error cases via /dev/tpu0
```
```bash
# board: B) raw UIO path (m5 gate), with the driver UNBOUND
sudo sh ~/tpuv1/software/driver/tpu-unload.sh                         # gives the node back to uio_pdrv_genirq
sudo insmod ~/udmabuf/u-dma-buf.ko udmabuf0=16777216
mkdir -p /tmp/tpu_m5 && cp -r ~/tpu/cases /tmp/tpu_m5/
cd ~/tpuv1/software/linux/idma_runtime
sudo python3 m5_board_test.py --build-id "$(cat ~/tpu/build_id.txt)"   # default cases dir /tmp/tpu_m5/cases
```
How to read the milestone tests: `m0`, `m2`, and `m4` check for **older** VERSION values (0x110, 0x200, 0x300). The current `kv260_top` reports **0x500**. So on today's bitstream only `m5_board_test.py` (which accepts 0x400 or 0x500) and the driver tests are meaningful. m0, m2, and m4 will report a VERSION failure, and that's expected. m0 and m2 also hard-code `/dev/uio4`. m4 and m5 find the node by name, `fabric`.

`BUILD_ID` mismatch means the board is running a stale bitstream (tpuv1 lesson L7). Reinstall and reload the app.

---

## 6. Days 2–3: performance and the JEPA encoder

- Check the README numbers: `mlp_wide` about 354 µs / 37 GOPS, `mlp3` about 67 µs, `digits_b128` about 153 µs, digits accuracy 91.7%. Measure power while you do it (§8).
- The **JEPA encoder** (`../../jepa-tpu`) is verified in simulation only and **cannot run end-to-end on the stock tpuv1 bitstream**. Q·Kᵀ and P·V need a UB-to-weight path, LayerNorm, softmax, GELU, and residual adds need the VPU, and neither is integrated yet (see JEPA README §5). What you *can* do on the board now:
  1. Run the encoder's 73 matmuls one at a time through `/dev/tpu0`. Re-emit them as Phase B cases (prog.bin + relocations.json + hex) with tpuv1's compiler instead of the `tb_tpu_core` format `rtl_check.py` writes today, then run them with `run_case.py` and compare against `quantize.int_forward`.
  2. Do the vector ops on the A53 in NumPy, using `intops.py` (bit-exact spec), to get a working hybrid encoder.
  3. Only after that, rebuild the bitstream with `rom_stream.sv` + `vpu.sv`. Budget check: XCK26 has 64 URAM + 144 BRAM36 (about 23.1 Mb of block memory) [S14]. tpuv1 uses 32 URAM + 44 BRAM36, and the hard-coded model needs about 12.8 Mb more, so shrink the UB or the program first (JEPA README §5.3).

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No console output, fan doesn't spin | No 12 V (PSU, plug, or polarity) | Check the J12 supply with a meter. The board has no power switch |
| Fan spins, nothing on the console | Wrong `ttyUSB` (the UART is the 2nd of 4), wrong baud, or host not in `dialout` | Try each `/dev/ttyUSB*` at 115200 8N1 with no flow control |
| Fan at full speed all the time | Normal until Linux loads the fan driver. If it stays at full speed after boot, the thermal/fan driver didn't bind | Check `board_check.sh` (thermal/fan section) and `dmesg | grep -i fan`. Never block the fan to "fix" noise |
| Boot stops after FSBL/U-Boot, or "Wrong image format", or a kernel panic at an early stage | **Boot firmware too old for the image** (24.04 needs ≥ 1.02) | Update with `xmutil bootfw_update` from a working image, or use the Recovery Tool on J10 (§3.4) |
| Booted the new firmware once, then the old one came back | Didn't run `xmutil bootfw_update -v` right after the first reboot | Update again and validate on the first boot [S1] |
| SD card boots on kit A but not on kit B | 24.04 binds the SD card to the first kit | Reflash the card for the second kit [S4] |
| SD errors, read-only root, corrupted files after power loss | Pulled power without `shutdown`, or a bad card | Always `sudo shutdown -h now` [S4]. Reflash from the known-good card |
| `apt upgrade` stops with flash-kernel "deferring update" | Known issue | Run it again. Keep the local version when prompted [S4] |
| `xmutil listapps` doesn't show `tpu_kv260` | Wrong folder or file names, or dfx-mgr missed the inotify event | Names must be `<app>/<app>.bit.bin`, `<app>.dtbo`, `shell.json`. `sudo systemctl restart dfx-mgr`. For details: `sudo systemctl stop dfx-mgr; sudo dfx-mgrd` (it runs in the foreground with messages) [S6] |
| `loadapp` fails, dmesg says `Direct firmware load ... failed` | `firmware-name` doesn't match the `.bit.bin` file | Check `dtc -I dtb -O dts tpu_kv260.dtbo | grep firmware-name`. `install_app.sh` checks this |
| `loadapp` fails, dmesg says overlay "resolve" / "symbol not found" | A label (`&amba`, `&fpga_full`, `&gic`, `&zynqmp_clk`) is missing from the base DT, or the base DT has no `__symbols__` | `ls /sys/firmware/devicetree/base/__symbols__/`. Use the label names that are present |
| `loadapp` says "already loaded", or the wrong app is active | Another app is loaded | `sudo xmutil unloadapp` first (after `tpu-unload.sh`) |
| fpga0 `operating`, but the **first register read hangs the whole CPU** | AXI hang: PL clock off or wrong frequency, PL held in reset, wrong address, or a stale or failed bitstream | Check `pl0_ref clk_rate` (should be about 100 MHz), the `clocking0` node, the resets, and that `a0000000` is in the HPM0 aperture. Power-cycle to recover. Never read PL addresses while the app is unloaded (lesson L5) |
| VERSION or SCRATCH reads back shifted or garbled | AFI width mismatch (HPM0 isn't 128-bit) | `config-afi <14 0xa00>`. Reload the app (lesson L1) |
| `/dev/tpu0` doesn't appear, "bind failed" | `a0000000.fabric` missing (dtbo), module not loaded, or IRQ request failed | `dmesg | tail`. Try `insmod tpu_kv260.ko use_irq=0` (polling) to isolate the IRQ |
| `insmod: invalid module format`, `version magic ... should be ...`, `disagrees about version of symbol` | Module built for another kernel (often after `apt upgrade` + reboot) | Run `build_module.sh` again. Don't copy `.ko` files between kernels |
| `module verification failed: signature and/or required key missing - tainting kernel` | Unsigned out-of-tree module | Harmless warning |
| Build error `asm/unaligned.h: No such file` | Kernel ≥ 6.12 moved it to `linux/unaligned.h` | Not an issue on 6.8 (24.04). Patch your copy if a newer kernel arrives |
| Job times out, device reported WEDGED | Core waiting on weights that will never arrive, or an AXI error | Read the `STATUS` bits (`dma_rd_err`, `dma_wr_err`, ...) in the test output. `tpu-unload.sh` → `xmutil unloadapp` → `loadapp` → `tpu-load.sh`. The PL can't be reset safely with traffic in flight (driver comment, L5/L30) |
| iDMA copies wrong data, or SIGBUS in the UIO tests | Uncached/`O_SYNC` mapping used with an unaligned memcpy, or missing `sync_for_device`/`sync_for_cpu` | The tests use cached u-dma-buf and explicit sync (m0 docstring). Keep `udmabuf0` loaded, and rebuild it after every kernel update |
| `Missing udmabuf0 for kernel ...` | u-dma-buf not built or not loaded for this kernel | `build_module.sh --udmabuf`, then `insmod ... udmabuf0=16777216` |
| m0/m2/m4 report a VERSION FAIL on a good bitstream | Those are tests from older milestones | Use m5 + `board_test_driver.py` (§5.5) |
| `/dev/mem` mmap fails in m2 | Strict devmem (`CONFIG_IO_STRICT_DEVMEM`) blocks claimed I/O ranges | Use the driver path instead. The UB/ACC windows aren't claimed by the overlay, so this usually works |
| Timing fails after an RTL change | — | Don't load it. A bitstream with negative slack can corrupt data without any error |

Useful one-liners:
```bash
sudo dmesg -w                                     # keep open in a 2nd SSH session while loading
sudo cat /sys/kernel/debug/clk/clk_summary | grep -E "pl0|pl1"
dtc -I fs -O dts /sys/firmware/devicetree/base 2>/dev/null | grep -A8 "fabric@a0000000"
xmutil platformstats -h
```

---

## 8. Power and thermal measurement

`xmutil platformstats` reads the SOM's INA260 (5 V SOM input) and the PS/PL SYSMON. `-p` prints SOM total power, current, and voltage, plus LPD, FPD, and PL temperatures. Other flags: `-a` (all), `-c` (CPU utilisation), `-i <ms>` (poll interval), `-l <file>` (log to file) [S15]. Run `xmutil platformstats -h` on your image to confirm.

Procedure (repeat 3× and average):
```bash
# 1. idle, default app            -> baseline
sudo xmutil platformstats -p | tee p0_idle_default.txt
# 2. tpu_kv260 loaded, driver bound, no jobs   -> PL static + clock tree at 100 MHz
sudo xmutil platformstats -p | tee p1_tpu_idle.txt
# 3. under load: run the loop test in one shell, sample in another
sudo python3 board_test_driver.py --bundles ~/tpu/bundles --only loop --loops 20000 &
for i in $(seq 30); do sudo xmutil platformstats -p | grep -Ei "power|temp"; sleep 1; done | tee p2_tpu_load.txt
```
- **Delta P(load − TPU idle)** is the TPU's dynamic power. Divide GOPS by it to get GOPS/W. These are SOM-input numbers, so they include DDR, the PS, and regulator losses. Check the trend against the Vivado `report_power` estimate.
- Direct sysfs readings work too: `cat /sys/class/hwmon/hwmon*/name`, then the INA260 `power1_input` (µW). `board_check.sh` prints them.
- A USB or bench meter on the 12 V input gives the whole-kit number (carrier, fan, Ethernet).
- **Thermal limits:** the kit is designed for 15 W of dissipation with its active fansink and 0–35 °C ambient [S13]. The commercial K26 silicon is specified to 0–85 °C junction [S14]. Stop and investigate if the PL or FPD temperature goes above about 80 °C during a soak, or keeps rising after 10 minutes at constant load.

---

## 9. Rollback plan

Work through these levels, from least to most disruptive:

1. **Bad TPU job or wedged device:** `sudo sh tpu-unload.sh` → `sudo xmutil unloadapp` → `sudo xmutil loadapp tpu_kv260` → `tpu-load.sh`. If the CPU is hung, power-cycle. Nothing persists across a reboot, because the app isn't loaded at boot.
2. **Bad bitstream or overlay:** keep the previous good files in `~/tpu/known_good/` with their `build_id.txt`. Reinstall them with `install_app.sh --bin known_good/tpu_kv260.bit.bin --dtbo known_good/tpu_kv260.dtbo --load`. To go back to stock: `sudo bash install_app.sh --name tpu_kv260 --remove`, then `sudo xmutil loadapp k26-starter-kits`.
3. **Bad kernel module:** `rmmod tpu_kv260` (or reboot, since the module isn't auto-loaded). Nothing is installed in `/lib/modules`.
4. **Kernel upgrade broke things:** before upgrading, run `sudo apt-mark hold linux-xilinx linux-image-xilinx linux-headers-xilinx` (24.04 meta names, check with `dpkg -l | grep xilinx`) during a test campaign. Unhold when done. Fallback: the known-good SD card.
5. **Broken OS / SD:** swap in the known-good card. Make an image of the working card after day 1: `sudo dd if=/dev/sdX bs=4M status=progress | xz -T0 > kv260_day1.img.xz` on the host.
6. **Bad boot firmware:** the A/B scheme protects you. Unvalidated firmware falls back on the next reset. If both slots are bad, use the **Recovery Tool** (FWUEN button + J10 + `http://192.168.0.111`) to write a known `BOOT.BIN` [S1]. Keep a copy of the `BOOT.BIN` you started with and the one you installed.
7. **JTAG** (J4, same USB cable) is a last resort for debugging (ILA via Vivado Hardware Manager) but not for loading. Programming the PL over JTAG behind Linux's back skips the overlay (clocks, AFI widths, device node), and any bound driver can then hang the CPU.

---

## Sources

- [S1] AMD Kria wiki, *Kria SOM Boot Firmware Update*: https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/3020685316/Kria+SOM+Boot+Firmware+Update
- [S2] AMD Kria wiki, *Kria SOMs & Starter Kits* (Ubuntu 24.04 requires K26 Boot FW ≥ 01.02, fw-before-apt warning): https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/1641152513/Kria+SOMs+Starter+Kits
- [S3] Canonical, *Install Ubuntu on AMD* (Kria images): https://ubuntu.com/download/amd
- [S4] AMD, *Booting Kria Starter Kit Linux on KV260, Ubuntu 24.04*: intro https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_24_04/build/html/docs/intro.html · SD card …/docs/sdcard.html · connections …/docs/connections.html · UART …/docs/uart.html · firmware …/docs/fwupdate.html · boot …/docs/boot.html · known issues …/docs/known_issues.html
- [S5] Xilinx, *Kria-PYNQ* README: https://github.com/Xilinx/Kria-PYNQ
- [S6] AMD, *On-target Utilities and Firmware* (dfx-mgr, xmutil, app folder, shell.json): https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/target.html
- [S7] AMD, *Generating DTSI and DTBO Overlay Files*: https://xilinx.github.io/kria-apps-docs/creating_applications/2022.1/build/html/docs/dtsi_dtbo_generation.html
- [S8] AMD, KV260 benchmark-b4096 overlay (fpga_full / afi0 / clocking0 / IRQ 89 layout): https://raw.githubusercontent.com/Xilinx/kria-apps-firmware/xlnx_rel_v2022.1/boards/kv260/benchmark-b4096/kv260-benchmark-b4096.dtsi
- [S9] Xilinx, *kria-apps-firmware*: https://github.com/Xilinx/kria-apps-firmware
- [S10] AFI register bits (regid 14 = FPD afi_fs, 0x200 = HPM0 128-bit, 0x800 = HPM1 128-bit): https://qiita.com/ikwzm/items/a1ad7e22ed7c44940d88 and https://lore.kernel.org/linux-fpga/20220824035542.706433-5-nava.kishore.manne@amd.com/
- [S11] AMD UG973 (2025.2), *Supported Devices* (Kria in ML Standard; no license since 2016.x): https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Supported-Devices
- [S12] Slashdot, 2026-05-23, Vivado 2026.1 tiered licensing, 2025.2 support until 2026.3: https://hardware.slashdot.org/story/26/05/23/1917255/amd-xilinx-is-excluding-linux-from-the-free-tier-for-its-fpga-dev-tool
- [S13] AMD DS986, KV260 *Thermal* (active cooling, 15 W, 0–35 °C): https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Thermal
- [S14] AMD DS987 v1.2, *Kria K26 SOM Data Sheet* (mirror): https://www.sundance.com/wp-content/uploads/docs/k26-som-Data-Sheet.pdf
- [S15] Xilinx *platformstats*: https://github.com/Xilinx/platformstats · element14, *xmutil: understanding platformstats*: https://community.element14.com/members-area/personalblogs/b/blog/posts/xmutil-understanding-platformstats
- [S16] Ubuntu archive, `linux-xilinx` (noble 6.8.0-10xx headers): https://mirror.seas.harvard.edu/ubuntu/pool/universe/l/linux-xilinx/
- [S17] controlpaths, *Sending PL overlays from Ubuntu in the KR260* (generic-uio, `uio_pdrv_genirq.of_id`): https://controlpaths.pages.dev/2025/08/31/ubuntu-overlays-kria
- [S18] Altium, *Getting Started with the Kria KV260* (accessories not in box; barrel plug note): https://resources.altium.com/p/getting-started-kria-kv260-vision-ai-starter-kit
- [S19] AMD UG1089, KV260 *Firmware Update*: https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit/Firmware-Update
- u-dma-buf: https://github.com/ikwzm/udmabuf
