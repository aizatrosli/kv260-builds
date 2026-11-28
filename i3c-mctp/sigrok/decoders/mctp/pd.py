##
## MCTP message decoder (DMTF DSP0236 v1.4.0, DSP0239 v1.12.0) for libsigrokdecode (API 3).
## Stacks on a transport-binding decoder that outputs 'mctp_pkt' (here: 'mctp_i3c').
##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

import sigrokdecode as srd

# DSP0239 v1.12.0 §7 Table 1
MSG_TYPES = {0x00: 'MCTP Control', 0x01: 'PLDM', 0x02: 'NC-SI', 0x03: 'Ethernet', 0x04: 'NVMe-MI',
             0x05: 'SPDM', 0x06: 'Secured SPDM', 0x07: 'CXL FM API', 0x08: 'CXL CCI', 0x09: 'PCIe-MI',
             0x7E: 'Vendor Defined - PCI', 0x7F: 'Vendor Defined - IANA'}

# DSP0236 v1.4.0 §13.2 Table 12
CTRL_CMDS = {
    0x01: 'Set Endpoint ID', 0x02: 'Get Endpoint ID', 0x03: 'Get Endpoint UUID',
    0x04: 'Get MCTP Version Support', 0x05: 'Get Message Type Support',
    0x06: 'Get Vendor Defined Message Support', 0x07: 'Resolve Endpoint ID', 0x08: 'Allocate Endpoint IDs',
    0x09: 'Routing Information Update', 0x0A: 'Get Routing Table Entries',
    0x0B: 'Prepare for Endpoint Discovery', 0x0C: 'Endpoint Discovery', 0x0D: 'Discovery Notify',
    0x0E: 'Get Network ID', 0x0F: 'Query Hop', 0x10: 'Resolve UUID', 0x11: 'Query rate limit',
    0x12: 'Request TX rate limit', 0x13: 'Update rate limit', 0x14: 'Query Supported Interfaces',
    0x15: 'Query Endpoint Advanced Capabilities', 0x1A: 'Request EID pool', 0x1B: 'Release EID pool',
    0x1C: 'Query Supported Recovery Actions', 0x1D: 'Recovery Request', 0x1E: 'Query MCTP Device Type',
}

# DSP0236 v1.4.0 §13.3 Table 13
CC = {0x00: 'SUCCESS', 0x01: 'ERROR', 0x02: 'ERROR_INVALID_DATA', 0x03: 'ERROR_INVALID_LENGTH',
      0x04: 'ERROR_NOT_READY', 0x05: 'ERROR_UNSUPPORTED_CMD', 0x06: 'DELAYED_RESPONSE'}

BASELINE_TU = 64   # DSP0236 §8.4.2

A_MSG, A_CTRL, A_FIELD, A_WARN = range(4)


def cmd_name(c):
    if c in CTRL_CMDS:
        return CTRL_CMDS[c]
    if 0xF0 <= c <= 0xFF:
        return 'Transport specific 0x%02X' % c
    return 'Reserved 0x%02X' % c


def cc_name(c):
    if c in CC:
        return CC[c]
    return 'COMMAND_SPECIFIC 0x%02X' % c if c >= 0x80 else 'Reserved 0x%02X' % c


def ver_str(v):
    # §13.7: BCD major.minor.update, 0xF high nibble = single digit, update 0xFF = ignore
    def bcd(x):
        return '%d' % (x & 0x0F) if (x >> 4) == 0xF else '%d%d' % (x >> 4, x & 0x0F)
    s = '%s.%s' % (bcd(v >> 24), bcd((v >> 16) & 0xFF))
    if ((v >> 8) & 0xFF) != 0xFF:
        s += '.' + bcd((v >> 8) & 0xFF)
    if v & 0xFF:
        s += chr(v & 0xFF)
    return s


def decode_ctrl(rq, cmd, body):
    '''Return a short text for known control payloads (body = bytes after the command code).'''
    try:
        if cmd == 0x01:                                         # Table 14
            if rq:
                op = ['Set EID', 'Force EID', 'Reset EID', 'Set Discovered Flag'][body[0] & 3]
                return '%s 0x%02X' % (op, body[1])
            st = ['accepted', 'rejected', 'rejected (static)', 'reserved'][(body[1] >> 4) & 3]
            return '%s, assignment %s, EID 0x%02X, pool %d' % (cc_name(body[0]), st, body[2], body[3])
        if cmd == 0x02:                                         # Table 15
            if rq:
                return ''
            ep = ['simple endpoint', 'bus owner/bridge', 'reserved', 'reserved'][(body[2] >> 4) & 3]
            return '%s, EID 0x%02X, %s' % (cc_name(body[0]), body[1], ep)
        if cmd == 0x04:                                         # Table 18
            if rq:
                return 'for message type 0x%02X' % body[0]
            n = body[1]
            vs = [ver_str(int.from_bytes(bytes(body[2 + 4 * i:6 + 4 * i]), 'big')) for i in range(n)]
            return '%s, versions %s' % (cc_name(body[0]), ', '.join(vs))
        if cmd == 0x05:                                         # Table 19
            if rq:
                return ''
            n = body[1]
            return '%s, types %s' % (cc_name(body[0]), ', '.join('0x%02X' % t for t in body[2:2 + n]))
    except IndexError:
        return 'payload too short'
    if not rq and body:
        return cc_name(body[0])
    return ''


class Decoder(srd.Decoder):
    api_version = 3
    id = 'mctp'
    name = 'MCTP'
    longname = 'Management Component Transport Protocol'
    desc = 'DMTF MCTP messages: reassembly, message types, control commands (DSP0236).'
    license = 'gplv2+'
    inputs = ['mctp_pkt']
    outputs = []
    tags = ['PC']
    annotations = (
        ('message', 'Message'),
        ('control', 'Control command'),
        ('field', 'Field'),
        ('warning', 'Warning'),
    )
    annotation_rows = (
        ('messages', 'Messages', (A_MSG, A_CTRL)),
        ('fields', 'Fields', (A_FIELD,)),
        ('warnings', 'Warnings', (A_WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.asm = {}       # assembly key -> dict

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)

    def ann(self, ss, es, idx, texts):
        self.put(ss, es, self.out_ann, [idx, texts])

    def decode(self, ss, es, data):
        ptype, p = data
        if ptype != 'PACKET':
            return
        if not p['pec_ok'] or p['hdr_ver'] != 1:
            return                                   # dropped before assembly (DSP0236 §8.7)
        # Assembly key (§8.6): (tag, TO, source EID); source physical address when the source EID is 0.
        # On I3C the Controller has physical address 0x00 (DSP0233 §5.6 Table 5).
        phys = p['addr'] if p['rnw'] else 0x00
        key = (p['tag'], p['to'], p['src'] if p['src'] else ('pa', phys))
        a = self.asm.get(key)
        if p['som']:
            if a is not None:
                self.ann(a['ss'], p['ss'], A_WARN, ['New SOM: earlier assembly dropped (§8.9)', 'Dropped'])
            a = {'ss': p['ss'], 'seq': p['seq'], 'tu': len(p['payload']), 'data': list(p['payload']),
                 'npk': 1, 'first': p}
            if not p['eom'] and a['tu'] < BASELINE_TU:
                self.ann(p['ss'], p['es'], A_WARN, ['Transmission unit %d < 64-byte baseline (§8.4.2)' % a['tu'],
                                                    'TU < 64'])
            self.asm[key] = a
        else:
            if a is None:
                self.ann(p['ss'], p['es'], A_WARN, ['Middle/end packet without SOM: dropped (§8.7)', 'No SOM'])
                return
            if p['seq'] != (a['seq'] + 1) & 3:
                self.ann(p['ss'], p['es'], A_WARN, ['Sequence %d, expected %d: message dropped (§8.9)'
                                                    % (p['seq'], (a['seq'] + 1) & 3), 'Seq'])
                del self.asm[key]
                return
            n = len(p['payload'])
            if (not p['eom'] and n != a['tu']) or (p['eom'] and n > a['tu']):
                self.ann(p['ss'], p['es'], A_WARN, ['Payload size %d breaks the TU rule (TU %d, §8.4.2)'
                                                    % (n, a['tu']), 'TU'])
            a['seq'] = p['seq']
            a['data'] += p['payload']
            a['npk'] += 1
        if p['eom']:
            del self.asm[key]
            self.message(a, p)

    def message(self, a, last):
        f = a['first']
        d = a['data']
        ss, es = a['ss'], last['es']
        if not d:
            return
        ic, mt = d[0][0] >> 7, d[0][0] & 0x7F
        b = [x[0] for x in d]
        tname = MSG_TYPES.get(mt, 'Reserved 0x%02X' % mt)
        head = 'MCTP %s, EID 0x%02X->0x%02X, tag %d%s, %d byte(s) in %d packet(s)%s' % (
            tname, f['src'], f['dst'], f['tag'], ' TO' if f['to'] else '', len(b), a['npk'],
            ', IC=1 (integrity check in last bytes, message-type specific)' if ic else '')
        self.ann(d[0][1], d[0][2], A_FIELD, ['IC %d, type 0x%02X (%s)' % (ic, mt, tname), 'Type %02X' % mt])
        if mt == 0x00 and len(b) >= 3:
            # Control message, §12.4 Fig 23: [Rq D rsvd IID] [command] [CC in responses]
            rq, dg, iid, cmd = b[1] >> 7, (b[1] >> 6) & 1, b[1] & 0x1F, b[2]
            kind = 'Datagram' if (rq and dg) else ('Request' if rq else 'Response')
            self.ann(d[1][1], d[1][2], A_FIELD, ['Rq %d D %d IID %d' % (rq, dg, iid), 'IID %d' % iid])
            self.ann(d[2][1], d[2][2], A_FIELD, ['Command 0x%02X %s' % (cmd, cmd_name(cmd)), '%02X' % cmd])
            if not rq and len(d) > 3:
                self.ann(d[3][1], d[3][2], A_FIELD, ['Completion code %s' % cc_name(b[3]), 'CC %02X' % b[3]])
            extra = decode_ctrl(rq, cmd, b[3:])
            # Table 11: Request TO=1 Rq=1 D=0; Response TO=0 Rq=0 D=0; Datagram TO=1 Rq=1 D=1
            if (rq and not f['to']) or (not rq and f['to']) or (not rq and dg):
                self.ann(ss, es, A_WARN, ['TO/Rq/D combination not allowed (§12.7 Table 11)', 'Flags'])
            if ic:
                self.ann(ss, es, A_WARN, ['Control messages must have IC=0 (§12.4.2)', 'IC'])
            txt = 'Control %s: %s (IID %d)%s' % (kind, cmd_name(cmd), iid, (': ' + extra) if extra else '')
            self.ann(ss, es, A_CTRL, [txt + ' | EID 0x%02X->0x%02X' % (f['src'], f['dst']), cmd_name(cmd)])
            return
        if mt in (0x7E, 0x7F):
            n = 2 if mt == 0x7E else 4                   # §14.1 Table 48 / §14.2 Table 49
            vid = int.from_bytes(bytes(b[1:1 + n]), 'big') if len(b) > n else None
            if vid is not None:
                head += ', vendor %s 0x%0*X' % ('PCI' if mt == 0x7E else 'IANA', 2 * n, vid)
        self.ann(ss, es, A_MSG, [head + ': ' + ' '.join('%02X' % x for x in b[1:33]) +
                                 (' ...' if len(b) > 33 else ''), tname])
