##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

'''
DMTF MCTP messages (DSP0236 v1.4.0, DSP0239 v1.12.0). Stacks on 'mctp_i3c'.

Reassembles packets by (tag, TO, source EID) (DSP0236 §8.6), checks the sequence number and the
transmission-unit rule (§8.4.2, §8.9), names the message type (DSP0239 Table 1), and decodes MCTP
control messages: Rq/D/Instance ID, command (Table 12), completion code (Table 13), and the payloads
of Set Endpoint ID, Get Endpoint ID, Get MCTP Version Support and Get Message Type Support.
'''

from .pd import Decoder
