##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

'''
MIPI I3C Basic SDR-mode decoder (written from I3C Basic v1.1.1, Public Release Edition).

Decodes START / Repeated START / STOP, address headers (arbitrable after S, Push-Pull after Sr),
ACK/NACK, CCC codes (Table 16), write data with the T-bit odd-parity check (§5.1.2.3.3), read data
with the End-of-Data T-bit (§5.1.2.3.4), ENTDAA rounds (PID/BCR/DCR and DA+PAR, §5.1.4.2), and the
HDR Exit / Target Reset patterns. Optional SCL-low timing checks (Tables 86/87).

An IBI and a Controller private read after START put identical bits on the wire, so both are
shown as "Private read or IBI". Stack 'mctp_i3c' on top for MCTP-over-I3C (DMTF DSP0233).
HDR modes are not decoded.
'''

from .pd import Decoder
