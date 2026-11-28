##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

'''
AMD APML sideband: SB-RMI (Remote Management Interface) and SB-TSI (Temperature Sensor Interface).
Stacks on 'i2c' (SMBus/I2C, APML rev 1.0 parts) or on 'i3c_sdr' (APML-I3C, rev 2.0/2.1 parts).

Sources (public AMD Processor Programming References):
  - PPR 56214 rev 3.05 (Family 19h Model 21h), ch. 5 APML / ch. 6 SB-TSI   (SMBus, SB-RMI rev 10h)
  - PPR 55901 rev 0.47 (Family 19h Model 11h), vol 3 ch. 4/5              (I3C, SB-RMI rev 20h)
  - PPR 57238 rev 0.24 (Family 1Ah Model 02h), vol 4 ch. 5/6              (I3C, SB-RMI rev 21h)

SB-TSI: register reads/writes with names and decoded fields, CPU temperature and thresholds in
0.125 degC steps, the 11-bit two's complement offset, update rate, alert status/config.
SB-RMI: register reads/writes, the soft mailbox sequence (request and response with message names
and error codes), and the processor state access commands 71h/72h/73h (read MSR, read CPUID,
load address / write register, read data/status) with thread number and status code. The SB-RMI
command code is 1 byte (rev 10h/20h) or 2 bytes (rev 21h); option 'rmi_cmd_len' picks it or
'auto' infers it from the traffic. On SMBus the optional PEC bytes are checked.
'''

from .pd import Decoder
