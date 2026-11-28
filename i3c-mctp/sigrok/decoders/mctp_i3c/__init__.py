##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

'''
MCTP over I3C (DMTF DSP0233 v1.0.1). Stacks on 'i3c_sdr'.

Decodes each I3C private write/read as one MCTP packet: header version, destination/source EID,
SOM/EOM/sequence/TO/tag (DSP0236 §8.2), the IC/message-type byte on SOM packets, and checks the
CRC-8 PEC over the address byte (with RnW) and all packet bytes (DSP0233 §5.3.1). A one-byte read
of 0xAE after START is shown as the MCTP Pending Read IBI (DSP0233 Table 2).
Stack 'mctp' on top to reassemble messages.
'''

from .pd import Decoder
