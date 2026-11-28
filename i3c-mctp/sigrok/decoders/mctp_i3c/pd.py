##
## MCTP over I3C (DMTF DSP0233 v1.0.1) protocol decoder for libsigrokdecode (API 3).
## Stacks on 'i3c_sdr'.
##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

import sigrokdecode as srd

'''
OUTPUT_PYTHON format (for 'mctp'):

['PACKET', dict] with keys
  'addr', 'rnw'        I3C dynamic address and direction (0 = Controller->Target write)
  'hdr_ver', 'dst', 'src', 'som', 'eom', 'seq', 'to', 'tag'
  'payload'            list of (byte, ss, es) after the 4-byte transport header (IC|type first on SOM)
  'pec_ok'             PEC matched (DSP0233 §5.3.1)
  'ss', 'es'           sample range of the packet
['PENDING_READ', dict] 'addr': Target address that sent MDB 0xAE (DSP0233 §5.2.2.1 Table 2)
'''


def crc8_pec(data, crc=0):
    # DSP0233 §5.3.1: x^8 + x^2 + x + 1 (0x07), init 0, no final XOR, MSB first like the SMBus PEC.
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


A_HDRVER, A_DST, A_SRC, A_FLAGS, A_MSGHDR, A_PAYLOAD, A_PEC, A_PKT, A_IBI, A_WARN = range(10)


class Decoder(srd.Decoder):
    api_version = 3
    id = 'mctp_i3c'
    name = 'MCTP/I3C'
    longname = 'MCTP over I3C (DSP0233)'
    desc = 'DMTF MCTP I3C transport binding: packet header, PEC, Pending Read IBI.'
    license = 'gplv2+'
    inputs = ['i3c']
    outputs = ['mctp_pkt']
    tags = ['PC']
    options = (
        {'id': 'addrs', 'desc': 'MCTP Target addresses (hex, comma separated; empty = any)',
         'default': ''},
    )
    annotations = (
        ('hdr-ver', 'Header version'),
        ('dst', 'Destination EID'),
        ('src', 'Source EID'),
        ('flags', 'SOM/EOM/Seq/TO/Tag'),
        ('msg-hdr', 'IC / message type'),
        ('payload', 'Payload byte'),
        ('pec', 'PEC'),
        ('packet', 'Packet'),
        ('ibi', 'Pending Read IBI'),
        ('warning', 'Warning'),
    )
    annotation_rows = (
        ('fields', 'Fields', (A_HDRVER, A_DST, A_SRC, A_FLAGS, A_MSGHDR, A_PAYLOAD, A_PEC)),
        ('packets', 'Packets', (A_PKT, A_IBI)),
        ('warnings', 'Warnings', (A_WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.addrs = None

    def start(self):
        self.out_python = self.register(srd.OUTPUT_PYTHON)
        self.out_ann = self.register(srd.OUTPUT_ANN)
        a = self.options['addrs'].strip()
        self.addrs = set(int(x, 16) for x in a.split(',') if x.strip()) if a else None

    def ann(self, ss, es, idx, texts):
        self.put(ss, es, self.out_ann, [idx, texts])

    def decode(self, ss, es, data):
        ptype, p = data
        if ptype != 'XFER' or p['kind'] != 'private' or not p['ack'] or not p['data']:
            return
        if self.addrs is not None and p['addr'] not in self.addrs:
            return
        d = p['data']
        addr, rnw = p['addr'], p['rnw']
        b = [x[0] for x in d]

        # A single-byte read 0xAE after S: the MCTP Pending Read MDB of an IBI (DSP0233 Table 2).
        if rnw and p['arb'] and len(b) == 1 and b[0] == 0xAE:
            self.ann(d[0][2], d[0][3], A_IBI, ['IBI from 0x%02X: MDB 0xAE = MCTP Pending Read' % addr,
                                               'Pending Read', 'IBI'])
            self.put(ss, es, self.out_python, ['PENDING_READ', {'addr': addr}])
            return
        if len(b) < 5:
            self.ann(d[0][2], d[-1][3], A_WARN, ['Transfer too short for an MCTP packet (%d bytes)' % len(b),
                                                 'Short'])
            return

        pec = crc8_pec(bytes([(addr << 1) | rnw] + b[:-1]))
        pec_ok = pec == b[-1]
        hv, dst, src, fl = b[0] & 0x0F, b[1], b[2], b[3]
        som, eom, seq, to, tag = fl >> 7, (fl >> 6) & 1, (fl >> 4) & 3, (fl >> 3) & 1, fl & 7
        self.ann(d[0][2], d[0][3], A_HDRVER, ['Hdr ver %d%s' % (hv, '' if hv == 1 else ' (not 0001b)'),
                                              'Ver %d' % hv])
        self.ann(d[1][2], d[1][3], A_DST, ['Dest EID 0x%02X%s' % (dst, eid_note(dst)), 'D %02X' % dst])
        self.ann(d[2][2], d[2][3], A_SRC, ['Src EID 0x%02X%s' % (src, eid_note(src)), 'S %02X' % src])
        self.ann(d[3][2], d[3][3], A_FLAGS, ['%s%s Seq %d TO %d Tag %d' % ('SOM ' if som else '',
                                             'EOM' if eom else '', seq, to, tag),
                                             'S%dE%d q%d T%d t%d' % (som, eom, seq, to, tag)])
        for i in range(4, len(b) - 1):
            if i == 4 and som:
                ic, mt = b[4] >> 7, b[4] & 0x7F
                self.ann(d[i][2], d[i][3], A_MSGHDR, ['IC %d, type 0x%02X %s' % (ic, mt, MSG_TYPES.get(mt, '')),
                                                      'T %02X' % mt])
            else:
                self.ann(d[i][2], d[i][3], A_PAYLOAD, ['%02X' % b[i]])
        self.ann(d[-1][2], d[-1][3], A_PEC, ['PEC 0x%02X %s' % (b[-1], 'ok' if pec_ok else 'BAD (expect 0x%02X)' % pec),
                                             'PEC ' + ('ok' if pec_ok else 'BAD')])
        txt = 'MCTP pkt %s 0x%02X: EID 0x%02X->0x%02X %sseq %d tag %d%s, %d B payload%s' % (
            'from' if rnw else 'to', addr, src, dst, ('SOM ' if som else '') + ('EOM ' if eom else ''),
            seq, tag, ' TO' if to else '', len(b) - 5, '' if pec_ok else ', PEC BAD')
        self.ann(d[0][2], d[-1][3], A_PKT, [txt, 'MCTP pkt'])
        if not pec_ok:
            self.ann(d[-1][2], d[-1][3], A_WARN, ['PEC mismatch: got 0x%02X, computed 0x%02X (DSP0233 §5.3.1)'
                                                  % (b[-1], pec), 'PEC'])
        if hv != 1:
            self.ann(d[0][2], d[0][3], A_WARN, ['Header version %d: receivers drop it (DSP0236 §8.7)' % hv,
                                                'Hdr ver'])
        self.put(ss, es, self.out_python, ['PACKET', {
            'addr': addr, 'rnw': rnw, 'hdr_ver': hv, 'dst': dst, 'src': src, 'som': som, 'eom': eom,
            'seq': seq, 'to': to, 'tag': tag, 'payload': [(x[0], x[2], x[3]) for x in d[4:-1]],
            'pec_ok': pec_ok, 'ss': d[0][2], 'es': d[-1][3]}])


def eid_note(e):
    # DSP0236 v1.4.0 §8.3 Table 2
    if e == 0x00:
        return ' (null)'
    if e == 0xFF:
        return ' (broadcast)'
    if 0x01 <= e <= 0x07:
        return ' (reserved)'
    return ''


# DSP0239 v1.12.0 §7 Table 1
MSG_TYPES = {0x00: 'MCTP Control', 0x01: 'PLDM', 0x02: 'NC-SI', 0x03: 'Ethernet', 0x04: 'NVMe-MI',
             0x05: 'SPDM', 0x06: 'Secured SPDM', 0x07: 'CXL FM API', 0x08: 'CXL CCI', 0x09: 'PCIe-MI',
             0x7E: 'Vendor (PCI)', 0x7F: 'Vendor (IANA)'}
