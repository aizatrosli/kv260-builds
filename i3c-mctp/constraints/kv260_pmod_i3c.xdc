# KV260 PMOD J2 pin assignment for kv260_i3c_top (I3C/MCTP generator).
#
# LOCs and IOSTANDARD come from AMD's own KV260 constraints:
#   Xilinx/Kria-PYNQ        kv260/base/vivado/constraints/base.xdc          (pmod[0..7])
#   Xilinx/kria-vitis-platforms kv260/platforms/kv260_bist/xdc/pin.xdc       (same LOCs, DRIVE 4, SLEW SLOW)
# pmod[i] -> SOM240_1 HDA11..HDA18 on HDIO bank 45 (DS987). The official files use LVCMOS33,
# i.e. this header is a 3.3 V interface. The mapping of pmod[i] to J2 physical pin numbers is in the
# KV260 carrier schematic (XTP682, AMD login) and is NOT verified here.
#
# HD-bank LVCMOS33 supports DRIVE 4/8/12/16 and SLEW SLOW/FAST (UG571 Table 3-2). 12.5 MHz is far
# below the HD-bank limit (UG571 Table 3-2 notes). DRIVE 8 is a starting point; check edges on a scope.
# No internal PULLUP on SDA: the HD weak pull-up (DS925 IRPU, roughly 17-44 kOhm at 3.3 V) is far
# too weak for the I3C Open-Drain pull-up (Rp <= 2833 Ohm for 120 ns / 50 pF, I3C Basic Table 82).

set_property -dict {PACKAGE_PIN H12 IOSTANDARD LVCMOS33 DRIVE 8 SLEW SLOW} [get_ports i3c_scl]
set_property -dict {PACKAGE_PIN E10 IOSTANDARD LVCMOS33 DRIVE 8 SLEW SLOW} [get_ports i3c_sda]
set_property -dict {PACKAGE_PIN D10 IOSTANDARD LVCMOS33 DRIVE 8 SLEW SLOW} [get_ports i3c_sda_pu]
set_property -dict {PACKAGE_PIN C11 IOSTANDARD LVCMOS33 DRIVE 4 SLEW SLOW} [get_ports trig]

# Pad-to-flop timing is not critical (SDA is resynchronised with two flops); keep the tools quiet.
set_false_path -from [get_ports i3c_sda]
set_false_path -to   [get_ports {i3c_scl i3c_sda i3c_sda_pu trig}]
