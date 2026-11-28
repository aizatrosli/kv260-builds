#!/usr/bin/env python3
"""Self-test of the Python reference model (run: python3 sw/test_mctp_i3c.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mctp_i3c as m

# CRC-8/SMBUS check value (poly 0x07, init 0, no reflection, no xorout)
assert m.crc8_pec(b"123456789") == 0xF4
# T-bit examples given in I3C Basic v1.1.1 §5.1.2.3.3 text: 0x00 -> 1, 0xFF -> 1, 0x01 -> 0
assert (m.t_bit(0x00), m.t_bit(0xFF), m.t_bit(0x01)) == (1, 1, 0)
# PAR = ~XOR(DA)
assert m.daa_par(0x0A) == 1 and m.daa_par(0x0B) == 0
# PEC includes the address byte (DSP0233 §5.3.1). Vectors computed with this model, not from the spec.
assert m.crc8_pec(bytes.fromhex("1001000 8C8008002".replace(" ", ""))) == 0x53
assert m.crc8_pec(bytes.fromhex("01000 8C8008002".replace(" ", ""))) == 0x64
# fragmentation / reassembly
msg = bytes([0x7F]) + bytes(range(149))
pk = m.fragment(msg, addr=0x0A, dst=0x1D, src=0x08, to=1, tag=5, seq=2)
assert [len(p.payload) for p in pk] == [64, 64, 22]
assert [(p.som, p.eom, p.seq) for p in pk] == [(1, 0, 2), (0, 0, 3), (0, 1, 0)]
assert m.reassemble([m.parse(p.wire(), 0x0A, 0)[0] for p in pk]) == msg
assert all(m.parse(p.wire(), 0x0A, 0)[1] for p in pk)
# control messages (DSP0236 §12.4, §13.4, §13.5)
assert m.get_eid_request(iid=2) == bytes([0x00, 0x82, 0x02])
assert m.set_eid_request(0x1D) == bytes([0x00, 0x80, 0x01, 0x00, 0x1D])
print("test_mctp_i3c: ALL PASS")
