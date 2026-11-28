# KV260 + tpuv1 timeline (days 0–3)

Details for each step are in `BRINGUP.md` (section numbers in brackets).

## Before the kit arrives
- [ ] Order the 12 V/3 A PSU (or the Basic Accessory Pack), 2× 16–32 GB microSD, a micro-USB data cable, and an Ethernet cable [§1]
- [ ] Install Vivado **2025.2** ML Standard (no license). Keep the installer [§2]
- [ ] Install bender, dtc, and a serial terminal. Clone tpuv1. Run `bash scripts/host_check.sh --vivado-tcl` [§2]
- [ ] **Start the first bitstream build now.** It's the longest step and needs no board [§5.1]
- [ ] Download the Ubuntu 24.04 Kria image [§3.1]

## First 2 hours with the board (day 0)
| min | Step | Done when |
|---|---|---|
| 0–15 | Flash the SD card (and the spare). Connect J11 SD, J4 USB, J10 Ethernet, J12 power last [§3.1–3.2] | Card written |
| 15–20 | Open the serial console on the 2nd ttyUSB at 115200, then apply power | Fan spins, FSBL/U-Boot text appears |
| 20–35 | First boot. Log in `ubuntu`/`ubuntu` and change the password. Note the IP, set up SSH [§3.3] | `ssh kv260` works |
| 35–40 | `sudo bash board_check.sh \| tee day0_baseline.txt` | Baseline saved |
| 40–70 | `xmutil bootfw_status` → install `kria-k26-bootfw` → `bootfw_update -i` → reboot → **`bootfw_update -v` right away** [§3.4] | New version active and validated |
| 70–110 | Add the PPAs, `apt upgrade` (keep local flash-kernel config), install build-essential, dtc, and headers, then reboot [§3.5] | `uname -r` stable, headers installed |
| 110–120 | `board_check.sh` again. `xmutil listapps` shows k26-starter-kits active. `platformstats -p` gives the idle power baseline | Day-0 report saved |

Stop rule: if any step fails, fix it before moving on. Don't load custom PL on a board whose firmware isn't validated.

## Day 1: TPU in the PL
1. Host: check the bitstream (timing met, `KV260_BUILD_DONE`), run bootgen → `tpu_kv260.bit.bin`, and copy it to the board with `build_id.txt` [§5.1]
2. Board: `install_app.sh --bin ... --dtso ../overlay/tpu_kv260.dtso --load`. Confirm `fpga0` is operating, `pl0_ref` is about 100 MHz, and `a0000000.fabric` exists [§5.2–5.3]
3. Board: `build_module.sh --install-headers --udmabuf`, then `tpu-load.sh` → `/dev/tpu0` [§5.4]
4. Host: `test_host.py`, `compile_phase_b.py`, `export_bundle.py`, then the `board_test_driver.py --mock` run. Copy cases and bundles to the board [§5.5]
5. Board: `board_test_driver.py --only basic,negatives,bundles` → `run_case.py` → full `board_test_driver.py` [§5.5]
6. Save known-good copies (`.bit.bin`, `.dtbo`, `build_id.txt`) to `~/tpu/known_good/`. Image the SD card [§9]

## Day 2: confidence and numbers
1. UIO path: `tpu-unload.sh`, enable generic-uio, load u-dma-buf, run `m5_board_test.py --build-id ...` [§5.5]
2. Check the performance numbers in the README (mlp_wide about 37 GOPS, digits 91.7%)
3. Power: idle default / TPU idle / TPU under load with `platformstats -p` (3× each). Record GOPS/W [§8]
4. Thermal soak: 30 min of the `loop` test. Log temperatures. Confirm the fan responds [§8]
5. Practice recovery once on purpose: `tpu-unload` → `unloadapp` → `loadapp` → `tpu-load` [§9]

## Day 3: toward JEPA
1. Re-emit the JEPA encoder's 73 matmuls as Phase B cases and run them with `run_case.py` on `/dev/tpu0`. Compare bit-exact against `quantize.int_forward` [§6]
2. Hybrid encoder: TPU matmuls plus A53 NumPy vector ops (`intops.py`). Measure images/s against the simulation estimate (about 170/s)
3. Plan the ROM/VPU bitstream: memory budget (UB shrink, program compression) and a new Vivado build. Use the same install/test loop as day 1
