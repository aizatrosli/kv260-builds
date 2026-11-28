"""MCTP-over-I3C reference model and helpers (pure Python, no dependencies).

Independent of the RTL. Used by the Verilator flow to generate stimulus and to check what the
generator put on the bus, and by kv260_driver.py to build messages on the board.

Spec references:
  DSP0236 v1.4.0  MCTP Base Specification
  DSP0233 v1.0.1  MCTP I3C Transport Binding
  DSP0239 v1.12.0 MCTP IDs and Codes
  MIPI I3C Basic v1.1.1 (Public Release Edition)
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------- constants
HDR_VERSION = 0x1            # DSP0233 Table 1, byte 1 [3:0]; DSP0236 §8.2 Table 1
BASELINE_TU = 64             # DSP0236 §8.4.2
EID_NULL = 0x00              # DSP0236 §8.3 Table 2
EID_BROADCAST = 0xFF
MDB_MCTP_PENDING_READ = 0xAE  # DSP0233 §5.2.2.1 Table 2
DCR_MCTP = 0xCC              # DSP0233 §5.4.1
I3C_BCAST = 0x7E             # I3C Basic §5.1.2

# DSP0239 v1.12.0 §7 Table 1
MSG_TYPE = {
    "control": 0x00, "pldm": 0x01, "ncsi": 0x02, "ethernet": 0x03, "nvme_mi": 0x04,
    "spdm": 0x05, "secured_spdm": 0x06, "cxl_fmapi": 0x07, "cxl_cci": 0x08, "pcie_mi": 0x09,
    "vendor_pci": 0x7E, "vendor_iana": 0x7F,
}

# DSP0236 v1.4.0 §13.2 Table 12
CTRL_CMD = {
    "set_eid": 0x01, "get_eid": 0x02, "get_uuid": 0x03, "get_version": 0x04,
    "get_msg_types": 0x05, "get_vdm_support": 0x06, "resolve_eid": 0x07,
    "discovery_notify": 0x0D,
}

# DSP0236 v1.4.0 §13.3 Table 13
CC = {0x00: "SUCCESS", 0x01: "ERROR", 0x02: "ERROR_INVALID_DATA", 0x03: "ERROR_INVALID_LENGTH",
      0x04: "ERROR_NOT_READY", 0x05: "ERROR_UNSUPPORTED_CMD", 0x06: "DELAYED_RESPONSE"}

# I3C Basic v1.1.1 Table 16 (subset used here)
CCC = {
    "ENEC_B": 0x00, "DISEC_B": 0x01, "RSTDAA": 0x06, "ENTDAA": 0x07, "SETMWL_B": 0x09,
    "SETMRL_B": 0x0A, "RSTACT_B": 0x2A, "ENEC_D": 0x80, "DISEC_D": 0x81, "SETDASA": 0x87,
    "SETNEWDA": 0x88, "SETMWL_D": 0x89, "SETMRL_D": 0x8A, "GETMWL": 0x8B, "GETMRL": 0x8C,
    "GETPID": 0x8D, "GETBCR": 0x8E, "GETDCR": 0x8F, "GETSTATUS": 0x90,
}


# ---------------------------------------------------------------- CRC / parity
def crc8_pec(data: bytes, crc: int = 0) -> int:
    """DSP0233 §5.3.1: x^8+x^2+x+1, init 0, no final XOR, MSB first (SMBus 2.0 §5.4.1.3)."""
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def t_bit(b: int) -> int:
    """I3C Basic §5.1.2.3.3: T = XOR(Data[7:0], 1) (odd parity)."""
    return (bin(b & 0xFF).count("1") & 1) ^ 1


def daa_par(da: int) -> int:
    """I3C Basic §5.1.4.2: PAR = ~XOR(DA[6:0])."""
    return (bin(da & 0x7F).count("1") & 1) ^ 1


# ---------------------------------------------------------------- packets
@dataclass
class Packet:
    addr: int          # 7-bit I3C dynamic address
    rnw: int           # 0 = Controller write, 1 = Target read
    dst: int
    src: int
    som: int
    eom: int
    seq: int
    to: int
    tag: int
    payload: bytes     # bytes after the 4-byte transport header (incl. IC|type in the first)
    hdr_ver: int = HDR_VERSION

    def header(self) -> bytes:
        return bytes([self.hdr_ver & 0x0F, self.dst, self.src,
                      (self.som << 7) | (self.eom << 6) | ((self.seq & 3) << 4) | (self.to << 3) | (self.tag & 7)])

    def wire(self) -> bytes:
        """Bytes after the I3C address phase, including the PEC (DSP0233 Tables 1/3)."""
        body = self.header() + self.payload
        pec = crc8_pec(bytes([(self.addr << 1) | self.rnw]) + body)
        return body + bytes([pec])


def fragment(msg: bytes, *, addr: int, dst: int, src: int, to: int, tag: int, seq: int = 0,
             tu: int = BASELINE_TU, rnw: int = 0) -> list[Packet]:
    """Split one MCTP message (IC|type byte first) into packets (DSP0236 §8.2, §8.4.2)."""
    assert len(msg) >= 1
    pkts, off = [], 0
    while off < len(msg):
        chunk = msg[off:off + tu]
        off += len(chunk)
        pkts.append(Packet(addr, rnw, dst, src, int(not pkts), int(off >= len(msg)), seq, to, tag, chunk))
        seq = (seq + 1) & 3
    return pkts


def parse(wire: bytes, addr: int, rnw: int) -> tuple[Packet, bool]:
    """Parse the bytes of one transfer; return (packet, pec_ok)."""
    assert len(wire) >= 5
    body, pec = wire[:-1], wire[-1]
    ok = crc8_pec(bytes([(addr << 1) | rnw]) + body) == pec
    f = body[3]
    return Packet(addr, rnw, body[1], body[2], f >> 7, (f >> 6) & 1, (f >> 4) & 3, (f >> 3) & 1, f & 7,
                  body[4:], body[0] & 0x0F), ok


def reassemble(pkts: list[Packet]) -> bytes:
    """Reassemble one message and check DSP0236 §8.6-8.9 rules; raises on violation."""
    assert pkts and pkts[0].som == 1, "first packet must have SOM"
    assert pkts[-1].eom == 1, "last packet must have EOM"
    tu = len(pkts[0].payload)
    for i, p in enumerate(pkts):
        assert p.hdr_ver == HDR_VERSION, "header version"
        assert (p.tag, p.to, p.src) == (pkts[0].tag, pkts[0].to, pkts[0].src), "tag/TO/src must be constant"
        assert p.seq == (pkts[0].seq + i) & 3, "sequence number mod 4"
        assert (p.som == 1) == (i == 0) and (p.eom == 1) == (i == len(pkts) - 1), "SOM/EOM placement"
        if i < len(pkts) - 1:
            assert len(p.payload) == tu, "all non-EOM packets have the same TU"
        else:
            assert len(p.payload) <= tu, "EOM packet TU <= others"
    return b"".join(p.payload for p in pkts)


# ---------------------------------------------------------------- control messages
def ctrl_request(cmd: int, data: bytes = b"", iid: int = 0, datagram: bool = False) -> bytes:
    """DSP0236 §12.4 Fig 23 / §12.7 Table 11: IC=0, type 0x00, Rq=1, D, Instance ID, command."""
    return bytes([0x00, 0x80 | (0x40 if datagram else 0) | (iid & 0x1F), cmd]) + data


def ctrl_response(cmd: int, cc: int, data: bytes = b"", iid: int = 0) -> bytes:
    return bytes([0x00, iid & 0x1F, cmd, cc]) + data


def set_eid_request(eid: int, op: int = 0, iid: int = 0) -> bytes:
    """DSP0236 §13.4 Table 14: byte 1 [1:0] operation, byte 2 EID (0x00/0xFF illegal)."""
    return ctrl_request(CTRL_CMD["set_eid"], bytes([op & 3, eid]), iid)


def get_eid_request(iid: int = 0) -> bytes:
    return ctrl_request(CTRL_CMD["get_eid"], b"", iid)


def get_version_request(msg_type: int = 0xFF, iid: int = 0) -> bytes:
    return ctrl_request(CTRL_CMD["get_version"], bytes([msg_type]), iid)


def get_eid_response(eid: int, ep_type: int = 0, eid_type: int = 0, iid: int = 0) -> bytes:
    """DSP0236 §13.5 Table 15; byte 4 is reserved for I3C (DSP0233 §5.7 Table 6)."""
    return ctrl_response(CTRL_CMD["get_eid"], 0x00, bytes([eid, ((ep_type & 3) << 4) | (eid_type & 3), 0x00]), iid)


if __name__ == "__main__":
    assert crc8_pec(b"123456789") == 0xF4  # CRC-8/SMBUS check value
    print("crc8_pec('123456789') = 0x%02X" % crc8_pec(b"123456789"))
