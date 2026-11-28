##
## I3C Basic SDR protocol decoder for libsigrokdecode (API 3).
##
## Written from "MIPI Alliance Specification for I3C Basic, v1.1.1 (Public Release Edition)".
## Section/table numbers in comments refer to that document.
##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##

import sigrokdecode as srd

'''
OUTPUT_PYTHON format (for stacked decoders such as 'mctp_i3c'):

[<ptype>, <pdata>]
 - ['START', None] / ['SR', None] / ['STOP', None]
 - ['XFER', dict]   one message, emitted when it ends (at Sr or P):
      'addr'   7-bit address of the header
      'rnw'    0 write, 1 read
      'ack'    True if the header was ACKed
      'arb'    True if the header followed S (arbitrable), False after Sr
      'ccc'    None for a private transfer, else the CCC code this message belongs to
      'kind'   'private', 'ccc-bcast', 'ccc-direct', 'entdaa', 'bcast-hdr'
      'data'   list of (byte, T-bit, ss, es)
      'parity_errors'  number of write bytes whose T-bit was not odd parity (§5.1.2.3.3)
      'end'    'eod' (read ended with T=0), 'abort' (Controller ended a read with T=1),
               'nack', or 'sr'/'p' for writes
 - ['DAA', dict]    one ENTDAA round: 'pid', 'bcr', 'dcr', 'da', 'par_ok', 'ack'
 - ['PATTERN', dict]  'kind' ('hdr-exit' or 'target-reset' or 'unknown'), 'toggles'
'''

# I3C Basic v1.1.1 Table 16 (CCC codes in I3C Basic). (name, defining byte?)
CCC_NAMES = {
    0x00: 'ENEC', 0x01: 'DISEC', 0x02: 'ENTAS0', 0x03: 'ENTAS1', 0x04: 'ENTAS2', 0x05: 'ENTAS3',
    0x06: 'RSTDAA', 0x07: 'ENTDAA', 0x08: 'DEFTGTS', 0x09: 'SETMWL', 0x0A: 'SETMRL', 0x0B: 'ENTTM',
    0x0C: 'SETBUSCON', 0x12: 'ENDXFER', 0x1F: 'HDR-dummy', 0x20: 'ENTHDR0', 0x21: 'ENTHDR1',
    0x22: 'ENTHDR2', 0x23: 'ENTHDR3', 0x24: 'ENTHDR4', 0x25: 'ENTHDR5', 0x26: 'ENTHDR6',
    0x27: 'ENTHDR7', 0x28: 'SETXTIME', 0x29: 'SETAASA', 0x2A: 'RSTACT', 0x2B: 'DEFGRPA',
    0x2C: 'RSTGRPA', 0x2D: 'MLANE',
    0x80: 'ENEC', 0x81: 'DISEC', 0x82: 'ENTAS0', 0x83: 'ENTAS1', 0x84: 'ENTAS2', 0x85: 'ENTAS3',
    0x86: 'RSTDAA(deprecated)', 0x87: 'SETDASA', 0x88: 'SETNEWDA', 0x89: 'SETMWL', 0x8A: 'SETMRL',
    0x8B: 'GETMWL', 0x8C: 'GETMRL', 0x8D: 'GETPID', 0x8E: 'GETBCR', 0x8F: 'GETDCR',
    0x90: 'GETSTATUS', 0x91: 'GETACCCR', 0x92: 'ENDXFER', 0x93: 'SETBRGTGT', 0x94: 'GETMXDS',
    0x95: 'GETCAPS', 0x96: 'SETROUTE', 0x97: 'D2DXFER', 0x98: 'SETXTIME', 0x99: 'GETXTIME',
    0x9A: 'RSTACT', 0x9B: 'SETGRPA', 0x9C: 'RSTGRPA', 0x9D: 'MLANE',
}


def ccc_name(c):
    if c in CCC_NAMES:
        return CCC_NAMES[c]
    if 0x61 <= c <= 0x7F or 0xE0 <= c <= 0xFE:
        return 'Vendor/ext 0x%02X' % c
    return 'Reserved 0x%02X' % c


def odd_parity_ok(byte, t):
    # §5.1.2.3.3: T = XOR(Data[7:0], 1)
    return t == ((bin(byte).count('1') & 1) ^ 1)


# Annotation indices
A_START, A_SR, A_STOP, A_BIT, A_ADDR, A_ACK, A_NACK, A_WDATA, A_RDATA, A_TBIT, A_CCC, \
    A_DAA, A_PATTERN, A_XFER, A_WARN, A_TIMING = range(16)


class Decoder(srd.Decoder):
    api_version = 3
    id = 'i3c_sdr'
    name = 'I3C SDR'
    longname = 'MIPI I3C Basic (SDR mode)'
    desc = 'MIPI I3C Basic SDR: headers, CCCs, ENTDAA, private transfers, T-bits, patterns.'
    license = 'gplv2+'
    inputs = ['logic']
    outputs = ['i3c']
    tags = ['Embedded/industrial']
    channels = (
        {'id': 'scl', 'name': 'SCL', 'desc': 'I3C clock'},
        {'id': 'sda', 'name': 'SDA', 'desc': 'I3C data'},
    )
    options = (
        {'id': 'timing', 'desc': 'Check SDR timing (needs a samplerate)', 'default': 'yes',
         'values': ('yes', 'no')},
    )
    annotations = (
        ('start', 'START'),
        ('repeat-start', 'Repeated START'),
        ('stop', 'STOP'),
        ('bit', 'Bit'),
        ('address', 'Address header'),
        ('ack', 'ACK'),
        ('nack', 'NACK'),
        ('data-write', 'Write data'),
        ('data-read', 'Read data'),
        ('t-bit', 'T-bit'),
        ('ccc', 'CCC'),
        ('daa', 'ENTDAA'),
        ('pattern', 'HDR Exit / Target Reset pattern'),
        ('xfer', 'Message'),
        ('warning', 'Warning'),
        ('timing', 'Timing'),
    )
    annotation_rows = (
        ('bits', 'Bits', (A_BIT,)),
        ('fields', 'Fields', (A_START, A_SR, A_STOP, A_ADDR, A_ACK, A_NACK, A_WDATA, A_RDATA,
                              A_TBIT, A_PATTERN)),
        ('cmds', 'Commands', (A_CCC, A_DAA, A_XFER)),
        ('warnings', 'Warnings', (A_WARN, A_TIMING)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.samplerate = None
        self.state = 'IDLE'
        self.in_frame = False
        self.ccc = None            # CCC of the current frame (until P), §5.1.9
        self.ccc_bcast = False
        self.after_s = False
        self.bits = []             # (value, sample at SCL rise)
        self.msg = None
        self.toggles = 0
        self.falls = 0
        self.tog_ss = None
        self.last_rise = None
        self.last_fall = None
        self.prev_rise = None
        self.od_bit = False

    def metadata(self, key, value):
        if key == srd.SRD_CONF_SAMPLERATE:
            self.samplerate = value

    def start(self):
        self.out_python = self.register(srd.OUTPUT_PYTHON)
        self.out_ann = self.register(srd.OUTPUT_ANN)

    # ------------------------------------------------------------------ helpers
    def ann(self, ss, es, idx, texts):
        self.put(ss, es, self.out_ann, [idx, texts])

    def py(self, ss, es, data):
        self.put(ss, es, self.out_python, data)

    def bitw(self):
        if self.prev_rise is not None and self.last_rise is not None and self.last_rise > self.prev_rise:
            return self.last_rise - self.prev_rise
        return 1

    def ns(self, samples):
        return samples * 1e9 / self.samplerate

    def check_low(self, od):
        # SCL low period just ended (at this rising edge). Tables 86/87.
        if self.options['timing'] != 'yes' or not self.samplerate or self.last_fall is None:
            return
        t = self.ns(self.samplenum - self.last_fall)
        if od and t < 200.0:
            self.ann(self.last_fall, self.samplenum, A_TIMING,
                     ['tLOW_OD %.0f ns < 200 ns (Table 86)' % t, 'tLOW_OD'])
        elif not od and t < 24.0:
            self.ann(self.last_fall, self.samplenum, A_TIMING,
                     ['tLOW %.0f ns < 24 ns (Table 87)' % t, 'tLOW'])

    def new_msg(self, ss):
        self.msg = {'ss': ss, 'addr': None, 'rnw': 0, 'ack': False, 'arb': self.after_s,
                    'ccc': None, 'kind': 'private', 'data': [], 'parity_errors': 0, 'end': None}

    def end_msg(self, es, how):
        m = self.msg
        self.msg = None
        if m is None or m['addr'] is None:
            return
        if m['end'] is None:
            m['end'] = how
        ss = m.pop('ss')
        self.py(ss, es, ['XFER', m])
        a = m['addr']
        d = ' '.join('%02X' % b[0] for b in m['data'])
        if m['kind'] == 'private' and m['rnw'] and m['arb']:
            # Same bits on the wire for an IBI (Target won arbitration, Controller ACKed, §5.1.6.2)
            # and for a Controller private read after S; a logic capture cannot tell them apart.
            what = 'Private read or IBI 0x%02X' % a
        elif m['kind'] == 'private':
            what = 'Private %s 0x%02X' % ('read' if m['rnw'] else 'write', a)
        elif m['kind'] == 'ccc-direct' and a == 0x7E:
            what = 'Direct CCC %s' % ccc_name(m['ccc'])
        elif m['kind'] == 'entdaa' and not m['ack']:
            what = 'ENTDAA: no Target left'
        elif m['kind'] == 'ccc-direct':
            what = '%s -> 0x%02X %s' % (ccc_name(m['ccc']), a, 'R' if m['rnw'] else 'W')
        elif m['kind'] == 'ccc-bcast':
            what = 'Broadcast %s' % ccc_name(m['ccc'])
        elif m['kind'] == 'entdaa':
            what = 'ENTDAA'
        else:
            what = 'Broadcast header 0x7E %s' % ('R' if m['rnw'] else 'W')
        if not m['ack'] and m['kind'] != 'entdaa':
            what += ' (NACK)'
        self.ann(ss, es, A_XFER, ['%s: %s' % (what, d) if d else what, what])

    # ------------------------------------------------------------------ conditions
    def on_start(self):
        s = self.samplenum
        if self.in_frame:
            self.end_msg(s, 'sr')
            self.ann(s, s, A_SR, ['Repeated START', 'Sr'])
            self.py(s, s, ['SR', None])
            self.after_s = False
        else:
            self.ann(s, s, A_START, ['START', 'S'])
            self.py(s, s, ['START', None])
            self.after_s = True
            self.ccc = None
        self.in_frame = True
        self.new_msg(s)
        self.state = 'HDR'
        self.bits = []

    def on_stop(self):
        s = self.samplenum
        self.end_msg(s, 'p')
        self.ann(s, s, A_STOP, ['STOP', 'P'])
        self.py(s, s, ['STOP', None])
        self.in_frame = False
        self.ccc = None
        self.state = 'IDLE'

    def flush_pattern(self):
        if self.toggles > 2:
            # HDR Exit: SDA falls 4 times with SCL Low (§5.2.1.1.1).
            # Target Reset: 14 SDA transitions with SCL Low (§5.1.11.3), i.e. 7 falls from High.
            kind = {4: 'hdr-exit', 7: 'target-reset'}.get(self.falls, 'unknown')
            label = {'hdr-exit': 'HDR Exit Pattern', 'target-reset': 'Target Reset Pattern'}.get(
                kind, 'SDA pattern')
            self.ann(self.tog_ss, self.samplenum, A_PATTERN,
                     ['%s (%d SDA falls with SCL Low)' % (label, self.falls), label])
            self.py(self.tog_ss, self.samplenum, ['PATTERN', {'kind': kind, 'toggles': self.toggles,
                                                              'falls': self.falls}])
            if self.state not in ('IDLE',):
                self.state = 'WAIT_SRP'
        self.toggles = 0
        self.falls = 0
        self.tog_ss = None

    # ------------------------------------------------------------------ bit handling
    def on_rise(self, sda):
        self.prev_rise, self.last_rise = self.last_rise, self.samplenum
        st = self.state
        if st == 'IDLE' or st == 'WAIT_SRP':
            return
        self.bits.append((sda, self.samplenum))
        n = len(self.bits)
        if st == 'HDR':
            self.check_low(self.after_s)            # arbitrable header: Open Drain (§5.1.2.2)
            if n == 9:
                self.header_done()
        elif st in ('WDATA', 'RDATA', 'CCC'):
            self.check_low(False)
            if n == 9:
                self.byte_done()
        elif st == 'DAA_ID':
            self.check_low(True)
            if n == 64:
                self.daa_id_done()
        elif st == 'DAA_DA':
            self.check_low(True)
            if n == 9:
                self.daa_da_done()

    def put_bits(self, bits):
        w = self.bitw()
        for i, (v, s) in enumerate(bits):
            e = bits[i + 1][1] if i + 1 < len(bits) else s + w
            self.ann(s, e, A_BIT, ['%d' % v])

    def header_done(self):
        b = self.bits
        self.put_bits(b)
        val = 0
        for v, _ in b[:8]:
            val = (val << 1) | v
        addr, rnw, ack = val >> 1, val & 1, b[8][0] == 0
        ss, es_hdr, es = b[0][1], b[8][1], b[8][1] + self.bitw()
        m = self.msg
        m.update({'addr': addr, 'rnw': rnw, 'ack': ack, 'arb': self.after_s})
        arb = 'arbitrable ' if self.after_s else ''
        self.ann(ss, es_hdr, A_ADDR, ['%sAddress 0x%02X %s' % (arb, addr, 'Read' if rnw else 'Write'),
                                      '0x%02X %s' % (addr, 'R' if rnw else 'W')])
        self.ann(es_hdr, es, A_ACK if ack else A_NACK, ['ACK', 'A'] if ack else ['NACK', 'N'])
        self.bits = []
        if addr == 0x7E and rnw and self.ccc == 0x07:
            m['kind'] = 'entdaa'
        if not ack:
            m['end'] = 'nack'
            self.state = 'WAIT_SRP'
            return
        if addr == 0x7E and not rnw:
            m['kind'] = 'bcast-hdr'
            self.state = 'CCC'                     # next byte is a CCC code (§5.1.9.1)
        elif addr == 0x7E and rnw and self.ccc == 0x07:
            m['kind'] = 'entdaa'
            self.state = 'DAA_ID'                  # 48-bit PID, BCR, DCR (§5.1.4.2)
        else:
            if self.ccc is not None and self.ccc >= 0x80:
                m['kind'], m['ccc'] = 'ccc-direct', self.ccc
            self.state = 'RDATA' if rnw else 'WDATA'

    def byte_done(self):
        b = self.bits
        self.put_bits(b)
        val = 0
        for v, _ in b[:8]:
            val = (val << 1) | v
        t = b[8][0]
        ss, et, es = b[0][1], b[8][1], b[8][1] + self.bitw()
        m = self.msg
        self.bits = []
        if self.state == 'CCC':
            self.ccc = val
            self.ccc_bcast = val < 0x80                # §5.1.9.2: bit 7 = Direct
            m['ccc'] = val
            m['kind'] = 'ccc-bcast' if self.ccc_bcast else 'ccc-direct'
            ok = odd_parity_ok(val, t)
            self.ann(ss, et, A_CCC, ['CCC %s (0x%02X, %s)' % (ccc_name(val), val,
                                     'broadcast' if self.ccc_bcast else 'direct'), ccc_name(val)])
            self.ann(et, es, A_TBIT, ['T=%d' % t, 'T'])
            if not ok:
                self.ann(ss, es, A_WARN, ['T-bit parity error on CCC 0x%02X' % val, 'Parity'])
                m['parity_errors'] += 1
            self.state = 'WDATA'
            return
        m['data'].append((val, t, ss, es))
        if self.state == 'WDATA':
            ok = odd_parity_ok(val, t)
            self.ann(ss, et, A_WDATA, ['Write 0x%02X' % val, '%02X' % val])
            self.ann(et, es, A_TBIT, ['T=%d %s' % (t, 'parity ok' if ok else 'PARITY ERROR'), 'T'])
            if not ok:
                m['parity_errors'] += 1
                self.ann(ss, es, A_WARN, ['T-bit parity error on 0x%02X (§5.1.2.3.3)' % val, 'Parity'])
        else:
            self.ann(ss, et, A_RDATA, ['Read 0x%02X' % val, '%02X' % val])
            self.ann(et, es, A_TBIT, ['T=%d %s' % (t, 'more data' if t else 'End-of-Data'), 'T'])
            if t == 0:
                m['end'] = 'eod'                       # §5.1.2.3.4
                self.state = 'WAIT_SRP'
            else:
                m['end'] = 'abort'                     # overwritten if another byte follows

    def daa_id_done(self):
        b = self.bits
        val = 0
        for v, _ in b:
            val = (val << 1) | v
        pid, bcr, dcr = val >> 16, (val >> 8) & 0xFF, val & 0xFF
        ss, es = b[0][1], b[-1][1] + self.bitw()
        self.ann(ss, b[47][1], A_DAA, ['PID 0x%012X' % pid, 'PID'])
        self.ann(b[48][1], b[55][1], A_DAA, ['BCR 0x%02X' % bcr, 'BCR'])
        self.ann(b[56][1], es, A_DAA, ['DCR 0x%02X' % dcr, 'DCR'])
        self.daa = {'pid': pid, 'bcr': bcr, 'dcr': dcr, 'ss': ss}
        self.bits = []
        self.state = 'DAA_DA'

    def daa_da_done(self):
        b = self.bits
        self.put_bits(b)
        val = 0
        for v, _ in b[:8]:
            val = (val << 1) | v
        da, par, ack = val >> 1, val & 1, b[8][0] == 0
        par_ok = par == ((bin(da).count('1') & 1) ^ 1)    # PAR = ~XOR(DA) (§5.1.4.2)
        es = b[8][1] + self.bitw()
        self.ann(b[0][1], b[8][1], A_DAA, ['Assign DA 0x%02X, PAR %s' % (da, 'ok' if par_ok else 'ERROR'),
                                           'DA 0x%02X' % da])
        self.ann(b[8][1], es, A_ACK if ack else A_NACK, ['ACK', 'A'] if ack else ['NACK', 'N'])
        if not par_ok:
            self.ann(b[0][1], es, A_WARN, ['ENTDAA PAR error', 'PAR'])
        d = self.daa
        d.update({'da': da, 'par_ok': par_ok, 'ack': ack})
        ss = d.pop('ss')
        self.ann(ss, es, A_XFER, ['ENTDAA: PID 0x%012X BCR 0x%02X DCR 0x%02X -> DA 0x%02X %s' % (
            d['pid'], d['bcr'], d['dcr'], da, 'ACK' if ack else 'NACK'), 'DAA 0x%02X' % da])
        self.py(ss, es, ['DAA', d])
        self.bits = []
        self.state = 'WAIT_SRP'

    # ------------------------------------------------------------------ main loop
    def decode(self):
        while True:
            # 0: SCL rise (sample), 1: START/Sr, 2: STOP, 3: SDA edge with SCL low, 4: SCL fall
            pins = self.wait([{0: 'r'}, {0: 'h', 1: 'f'}, {0: 'h', 1: 'r'}, {0: 'l', 1: 'e'}, {0: 'f'}])
            scl, sda = pins
            if self.matched[3]:
                if self.toggles == 0:
                    self.tog_ss = self.samplenum
                self.toggles += 1
                if sda == 0:
                    self.falls += 1
            if self.matched[4]:
                self.last_fall = self.samplenum
                self.toggles = 0
                self.falls = 0
            if self.matched[0]:
                self.flush_pattern()
                self.on_rise(sda)
            if self.matched[1]:
                self.flush_pattern()
                self.on_start()
            elif self.matched[2]:
                self.flush_pattern()
                self.on_stop()
