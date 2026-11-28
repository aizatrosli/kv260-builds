##
## AMD APML (SB-RMI / SB-TSI) protocol decoder for libsigrokdecode (API 3).
## Stacks on 'i2c' (SMBus) or 'i3c_sdr' (I3C SDR private transfers).
##
## This program is free software; you can redistribute it and/or modify it under the terms of the
## GNU General Public License as published by the Free Software Foundation; either version 2 of the
## License, or (at your option) any later version.
##
## Section and table numbers refer to these AMD PPRs:
##   [56214] PPR Family 19h Model 21h B0, rev 3.05: ch. 5 APML (SMBus), 5.6 SB-RMI regs, ch. 6 SB-TSI
##   [55901] PPR Family 19h Model 11h B2, rev 0.47, vol 3: ch. 4 APML-I3C (SB-RMI rev 20h)
##   [57238] PPR Family 1Ah Model 02h C1, rev 0.24, vol 4: ch. 5 APML-I3C (SB-RMI rev 21h), ch. 6 SB-TSI
##

import sigrokdecode as srd


def crc8(data, crc=0):
    # SMBus PEC: CRC-8, x^8 + x^2 + x + 1, init 0 ([56214] 5.4.1 refers to the SMBus PEC).
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def le(bs):
    return int.from_bytes(bytes(bs), 'little')


def hexs(bs):
    return ' '.join('%02X' % b for b in bs)


# ---------------------------------------------------------------- SB-TSI ([57238] 6.4, [56214] 6.4)
TSI_REGS = {
    0x01: 'CpuTempInt', 0x02: 'Status', 0x03: 'Config', 0x04: 'UpdateRate', 0x07: 'HiTempInt',
    0x08: 'LoTempInt', 0x09: 'ConfigWr', 0x10: 'CpuTempDec', 0x11: 'CpuTempOffInt',
    0x12: 'CpuTempOffDec', 0x13: 'HiTempDec', 0x14: 'LoTempDec', 0x22: 'TimeoutConfig',
    0x32: 'AlertThreshold', 0xBF: 'AlertConfig', 0xFE: 'ManId', 0xFF: 'Revision',
}
# UpdateRate[UpRate] valid values 00h..0Ah
UPRATE = ['0.0625', '0.125', '0.25', '0.5', '1', '2', '4', '8', '16', '32', '64']
# Integer register -> (decimal register, quantity name); decimal -> integer
TSI_PAIRS = {0x01: (0x10, 'CPU temperature'), 0x07: (0x13, 'High threshold'),
             0x08: (0x14, 'Low threshold'), 0x11: (0x12, 'Temperature offset')}
TSI_DEC_OF = {v[0]: k for k, v in TSI_PAIRS.items()}


def bits_str(v, names):
    on = [n for b, n in names if v >> b & 1]
    return ', '.join(on) if on else 'none set'


def tsi_field(reg, v):
    if reg in (0x01, 0x07, 0x08):
        return '%d degC (integer part)' % v
    if reg in (0x10, 0x13, 0x14):
        return '+%.3f degC (decimal part)%s' % ((v >> 5) * 0.125, '' if not v & 0x1F else ', reserved bits set')
    if reg == 0x11:
        return 'offset high byte (11-bit two\'s complement with CpuTempOffDec)'
    if reg == 0x12:
        return 'offset low byte, bits 7:5 = %d' % (v >> 5)
    if reg == 0x02:
        return bits_str(v, [(4, 'TempHighAlert'), (3, 'TempLowAlert')])
    if reg in (0x03, 0x09):
        # bit 1 AraDis on [56214]; bits 3 SBRMIAddrMode / 0 RMISoftReset on [57238]
        return bits_str(v, [(7, 'AlertMask'), (6, 'RunStop'), (5, 'ReadOrder'),
                            (3, 'SBRMIAddrMode (2-byte SB-RMI command)'), (1, 'AraDis'),
                            (0, 'RMISoftReset')])
    if reg == 0x04:
        return '%s Hz' % UPRATE[v] if v < len(UPRATE) else 'reserved value'
    if reg == 0x22:
        return 'TimeoutEn=%d' % (v >> 7)
    if reg == 0x32:
        return '%d consecutive sample(s)' % ((v & 7) + 1)
    if reg == 0xBF:
        return 'AlertCompEn=%d (%s)' % (v & 1, 'comparator' if v & 1 else 'latched, read to clear')
    if reg == 0xFF:
        return 'SB-TSI revision 0x%02X' % v
    return ''


def tsi_value(int_reg, ib, db):
    if int_reg == 0x11:
        raw = (ib << 3) | (db >> 5)                     # 11-bit two's complement ([57238] 6.2.6)
        if raw & 0x400:
            raw -= 0x800
        return raw * 0.125
    return ib + (db >> 5) * 0.125                       # [57238] 6.2.5


# ---------------------------------------------------------------- SB-RMI
def rmi_reg_name(r):
    # [57238] 5.6 (rev 21h) and [56214] 5.6 (rev 10h)
    fixed = {0x00: 'Revision', 0x01: 'Control', 0x02: 'Status', 0x03: 'ReadSize',
             0x40: 'SoftwareInterrupt', 0x41: 'ThreadNumber (rev 10h)', 0x4B: 'Thread128CS',
             0x4C: 'RasStatus', 0x4E: 'ThreadNumberLow', 0x4F: 'ThreadNumberHigh',
             0x71: 'Write Register/Load Address', 0x72: 'Read Data/Status',
             0x73: 'Read CPUID/Read Register', 0x90: 'MP0AlertMask'}
    if r in fixed:
        return fixed[r]
    if 0x30 <= r <= 0x37:
        return 'OutBndMsg%d' % (r - 0x30)
    if 0x38 <= r <= 0x3F:
        return 'InBndMsg%d' % (r - 0x38)
    if 0x80 <= r <= 0x87:
        return 'MP0OutBndMsg%d' % (r - 0x80)
    if r in (0x04, 0x05) or 0x08 <= r <= 0x0D or 0x43 <= r <= 0x4A or 0x91 <= r <= 0x98 or 0xD8 <= r <= 0xDF:
        return 'ThreadEnableStatus'
    if 0x10 <= r <= 0x1F or 0x50 <= r <= 0x5F:
        return 'AlertStatus'
    if 0x20 <= r <= 0x2F or 0xC0 <= r <= 0xCF:
        return 'AlertMask'
    return 'reg 0x%02X' % r


RMI_REV = {0x10: 'APML 1.0, SMBus, 1-byte command', 0x20: 'APML-I3C 2.0, 1-byte command',
           0x21: 'APML-I3C 2.1, 2-byte command', 0x31: '2-byte command'}


def rmi_field(reg, v, rev):
    if reg == 0x00:
        return 'SB-RMI revision 0x%02X%s' % (v, ' (%s)' % RMI_REV[v] if v in RMI_REV else '')
    if reg == 0x01:
        b7 = 'PECEn' if rev == 0x10 else ('HwAlertMask' if rev else 'PECEn(rev10h)/HwAlertMask(rev2x)')
        b1 = 'AraDis' if rev == 0x10 else ('TSISoftReset' if rev else 'AraDis(rev10h)/TSISoftReset(rev2x)')
        return bits_str(v, [(7, b7), (4, 'SwAlertMask'), (3, 'BlkRWEn'), (2, 'TimeoutDis'),
                            (1, b1), (0, 'AlertMask')])
    if reg == 0x02:
        return bits_str(v, [(7, 'HwAlertSts'), (6, 'MP0AlertSts'), (1, 'SwAlertSts'), (0, 'AlertSts')])
    if reg == 0x03:
        return 'RdSize %d byte(s)' % (v & 0x3F)
    if reg == 0x40:
        return 'SwInt (mailbox request)' if v & 1 else ''
    if reg == 0x4C:
        return bits_str(v, [(5, 'pcie_err'), (4, 'dram_cecc_err'), (3, 'mca_err'),
                            (2, 'reset_ctrl_err'), (1, 'fch_err'), (0, 'fatal_err')])
    if reg == 0x3F:
        return 'start bit %s' % ('set' if v & 0x80 else 'CLEAR (must be 80h, [57238] 5.4.2.2.2)')
    return ''


# Soft mailbox messages: [57238] Table 135 (rev 21h); [56214] Table 86 gives the older names of 10h/11h.
MBOX = {
    0x01: 'Read Package Power Consumption', 0x02: 'Write Package Power Limit',
    0x03: 'Read Package Power Limit', 0x04: 'Read Max Package Power Limit', 0x05: 'Read TDP',
    0x06: 'Read Max cTDP', 0x07: 'Read Min cTDP', 0x08: 'Read BIOS Boost Fmax',
    0x09: 'Read APML Boost Limit', 0x0A: 'Write APML Boost Limit',
    0x0B: 'Write APML Boost Limit All Cores', 0x0C: 'Read DRAM Throttle', 0x0D: 'Write DRAM Throttle',
    0x0E: 'Read PROCHOT Status', 0x0F: 'Read PROCHOT Residency',
    0x10: 'Read VDDIOMem Power (56214; reserved in 57238)',
    0x11: 'Read NBIO Error Logging Register (56214; reserved in 57238)',
    0x13: 'Read IOD BIST Result', 0x14: 'Read CCD BIST Result', 0x15: 'Read CCX BIST Result',
    0x16: 'Read CCLK Frequency Limit', 0x17: 'Read Socket C0 Residency',
    0x18: 'Get Max DDR Bandwidth and Utilization', 0x1C: 'Get MP1 Firmware Version',
    0x1F: 'Read PPIN fuse', 0x21: 'Read RTC', 0x31: 'WRITE_FAST_PPT_LIMIT',
    0x34: 'WRITE_THERMCTL_LIMIT', 0x35: 'WRITE_VRM_VDD_CURRENT_LIMIT',
    0x36: 'WRITE_VRM_VDD_MAXIMUM_CURRENT_LIMIT',
    0x40: 'BMC_REPORT_DIMM_POWER_CONSUMPTION', 0x41: 'BMC_REPORT_DIMM_THERMAL_SENSOR',
    0x42: 'BMC_RAS_PCIE_CONFIG_ACCESS', 0x43: 'BMC_RAS_MCA_VALIDITY_CHECK',
    0x44: 'BMC_RAS_MCA_MSR_DUMP', 0x45: 'BMC_RAS_FCH_RESET_REASON',
    0x46: 'GET_DIMM_TEMP_RANGE_AND_REFRESH_RATE', 0x47: 'GET_DIMM_POWER_CONSUMPTION',
    0x48: 'GET_DIMM_THERMAL_SENSOR', 0x49: 'PWR_CURRENT_ACTIVE_FREQ_LIMIT_SOCKET',
    0x4A: 'PWR_CURRENT_ACTIVE_FREQ_LIMIT_CORE', 0x4B: 'PWR_SVI_TELEMETRY_ALL_RAILS',
    0x4C: 'GET_SOCKET_FREQ_RANGE', 0x4D: 'GET_CURRENT_IO_BANDWIDTH',
    0x4E: 'GET_CURRENT_XGMI_BANDWIDTH', 0x4F: 'GMI3_LINK_WIDTH_RANGE', 0x50: 'XGMI_LINK_WIDTH_RANGE',
    0x51: 'APB_DISABLE', 0x52: 'APB_ENABLE', 0x53: 'GET_CURRENT_DFPSTATE_FREQUENCY',
    0x54: 'SET_LCLK_DPM_LEVEL_RANGE', 0x55: 'BMC_GET_RAPL_UNITS', 0x56: 'BMC_GET_RAPL_CORE_LO_COUNTER',
    0x57: 'BMC_GET_RAPL_CORE_HI_COUNTER', 0x58: 'BMC_GET_RAPL_PKG_COUNTER',
    0x59: 'BMC_GET_CPU_BASE_FREQUENCY', 0x5A: 'BMC_CONTROL_PCIE_RATE',
    0x5B: 'BMC_RAS_DBG_LOG_VALIDITY_CHECK', 0x5C: 'BMC_RAS_DBG_LOG_DUMP',
    0x5D: 'PWR_EFFICIENCY_MODE_SELECTION', 0x5E: 'DF_PSTATE_RANGE', 0x5F: 'GET_LCLK_DPM_LEVEL_RANGE',
    0x60: 'GET_UCODE_REVISION', 0x61: 'BMC_RAS_RUNTIME_ERR_VALIDITY_CHECK',
    0x62: 'BMC_RAS_GET_RUNTIME_ERR_INFO', 0x63: 'BMC_RAS_SET_ERR_THRESHOLD',
    0x65: 'BMC_RAS_GET_OOB_CONFIG', 0x68: 'BMC_PCIE_CONFIG_WRITE',
    0x6A: 'BMC_RAS_DELAY_RESET_ON_SYNCFLOOD_OVERRIDE', 0x6B: 'BMC_RAS_RESET_ON_SYNCFLOOD',
    0x70: 'SPD_SB_RD', 0x73: 'XGMI_PSTATE_RANGE', 0x74: 'CPU_RAIL_ISO_FREQ_POLICY',
    0x76: 'DFC_ENABLE', 0x78: 'Read DRAM Throttle (enhanced)',
}
MBOX_UNIT = {0x01: 'mW', 0x02: 'mW', 0x03: 'mW', 0x04: 'mW', 0x05: 'mW', 0x06: 'mW', 0x07: 'mW',
             0x08: 'MHz', 0x09: 'MHz', 0x0C: '%', 0x0D: '%', 0x16: 'MHz'}
# [57238] Table 145 / [56214] Table 87
MBOX_ERR = {0x00: 'No error', 0x01: 'Command Aborted', 0x02: 'Unknown Command', 0x03: 'Invalid Core',
            0x05: 'Command Failed with Error', 0x09: 'Invalid Input Arguments',
            0x0A: 'Invalid OOB RAS Config', 0x0B: 'Data Not Ready'}
# [57238] Table 134 / [56214] Table 85
RMI_STATUS = {0x00: 'Success', 0x11: 'Command Timeout', 0x22: 'Warm reset', 0x40: 'Unknown Command Format',
              0x41: 'Invalid Read Length', 0x42: 'Excessive Data Length', 0x44: 'Invalid thread',
              0x45: 'Unsupported Command', 0x81: 'Command Aborted'}


def mbox_name(c):
    return MBOX.get(c, 'message 0x%02X' % c)


def with_unit(cmd, v):
    u = MBOX_UNIT.get(cmd)
    return '0x%08X (%d%s)' % (v, v, ' ' + u if u else '')


A_TSI, A_RMI, A_MBOX, A_PROC, A_FIELD, A_WARN = range(6)


class Decoder(srd.Decoder):
    api_version = 3
    id = 'amd_apml'
    name = 'AMD APML'
    longname = 'AMD APML sideband: SB-RMI and SB-TSI'
    desc = 'AMD SB-TSI temperature sensor and SB-RMI management interface over SMBus or I3C.'
    license = 'gplv2+'
    inputs = ['i2c', 'i3c']
    outputs = []
    tags = ['PC']
    options = (
        {'id': 'rmi_addrs', 'desc': 'SB-RMI addresses (hex, comma separated)', 'default': '3c,38'},
        {'id': 'tsi_addrs', 'desc': 'SB-TSI addresses (hex, comma separated)', 'default': '4c,48'},
        {'id': 'rmi_cmd_len', 'desc': 'SB-RMI command code length', 'default': 'auto',
         'values': ('auto', '1', '2')},
    )
    annotations = (
        ('tsi', 'SB-TSI access'),
        ('rmi', 'SB-RMI register access'),
        ('mbox', 'SB-RMI mailbox'),
        ('proc', 'SB-RMI processor state access'),
        ('field', 'Field'),
        ('warning', 'Warning'),
    )
    annotation_rows = (
        ('fields', 'Fields', (A_FIELD,)),
        ('tsi-row', 'SB-TSI', (A_TSI,)),
        ('rmi-row', 'SB-RMI', (A_RMI, A_MBOX, A_PROC)),
        ('warnings', 'Warnings', (A_WARN,)),
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.seg = None          # current I2C segment being built
        self.txn = []            # segments since START (for the SMBus PEC)
        self.dev = {}            # per target address state
        self.da_map = {}         # I3C dynamic address -> static address (SETDASA / SETNEWDA)

    def start(self):
        self.out_ann = self.register(srd.OUTPUT_ANN)
        parse = lambda s: set(int(x, 16) for x in s.split(',') if x.strip())
        self.rmi = parse(self.options['rmi_addrs'])
        self.tsi = parse(self.options['tsi_addrs'])
        self.forced_len = None if self.options['rmi_cmd_len'] == 'auto' else int(self.options['rmi_cmd_len'])

    def ann(self, ss, es, idx, texts):
        self.put(ss, es, self.out_ann, [idx, texts])

    def st(self, addr):
        if addr not in self.dev:
            self.dev[addr] = {'ptr': None, 'ptr_ss': None, 'regs': {}, 'cmdlen': None, 'rev': None,
                              'mb_ss': None, 'proc': None, 'pec': None}
        return self.dev[addr]

    # ------------------------------------------------------------ input normalisation
    def decode(self, ss, es, data):
        ptype, pdata = data[0], data[1]
        if ptype == 'XFER':                          # from i3c_sdr
            self.i3c_xfer(ss, es, pdata)
            return
        if ptype in ('START', 'START REPEAT', 'SR', 'STOP'):
            # i2c has no end-of-segment item: a segment ends at Sr or P (or a read at the master's NACK).
            if self.seg is not None and self.seg['ack']:
                self.segment(self.seg)
            self.seg = None
            if ptype in ('START', 'STOP'):
                self.txn = []
        elif ptype in ('ADDRESS READ', 'ADDRESS WRITE'):
            self.seg = {'addr': pdata, 'rnw': 1 if ptype == 'ADDRESS READ' else 0, 'ss': ss,
                        'es': es, 'ack': None, 'data': [], 'bus': 'smbus'}
        elif ptype in ('DATA READ', 'DATA WRITE') and self.seg is not None:
            self.seg['data'].append((pdata, ss, es))
            self.seg['es'] = es
        elif ptype in ('ACK', 'NACK') and self.seg is not None:
            s = self.seg
            s['es'] = es
            if s['ack'] is None:                    # address ACK/NACK
                s['ack'] = ptype == 'ACK'
                if not s['ack']:
                    self.segment(s)
                    self.seg = None
            elif s['rnw'] and ptype == 'NACK':      # the master ends a read with NACK
                self.segment(s)
                self.seg = None

    def i3c_xfer(self, ss, es, m):
        kind = m['kind']
        data = [(b[0], b[2], b[3]) for b in m['data']]
        if kind == 'ccc-direct' and m['addr'] != 0x7E and m['ack'] and data:
            if m['ccc'] == 0x87:                      # SETDASA: static address -> dynamic address
                self.da_map[data[0][0] >> 1] = m['addr']
                self.ann(ss, es, A_FIELD, ['SETDASA: static 0x%02X -> dynamic 0x%02X' % (m['addr'], data[0][0] >> 1),
                                           'SETDASA'])
            elif m['ccc'] == 0x88:                    # SETNEWDA
                old = self.da_map.pop(m['addr'], m['addr'])
                self.da_map[data[0][0] >> 1] = old
            return
        if kind == 'ccc-bcast' and m['ccc'] == 0x06:   # RSTDAA
            self.da_map = {}
            return
        if kind != 'private':
            return
        seg = {'addr': self.da_map.get(m['addr'], m['addr']), 'rnw': m['rnw'], 'ss': ss, 'es': es,
               'ack': m['ack'], 'data': data, 'bus': 'i3c'}
        if m['arb']:
            self.txn = []
        self.segment(seg)

    # ------------------------------------------------------------ dispatch
    def segment(self, s):
        self.txn.append(s)
        if not s['ack']:
            if s['addr'] in self.rmi or s['addr'] in self.tsi:
                self.ann(s['ss'], s['es'], A_WARN, ['0x%02X %s NACKed' % (s['addr'], 'read' if s['rnw'] else 'write'),
                                                    'NACK'])
            return
        if s['addr'] == 0x0C and s['rnw'] and s['bus'] == 'smbus' and s['data']:
            a = s['data'][0][0]
            self.ann(s['ss'], s['es'], A_TSI, ['SMBus Alert Response: alert from 0x%02X' % (a >> 1), 'ARA'])
            return
        if s['addr'] in self.tsi:
            self.tsi_seg(s)
        elif s['addr'] in self.rmi:
            self.rmi_seg(s)

    def pec_check(self, s, payload, segs_before):
        '''If the last byte of a segment is a valid SMBus PEC, annotate and return True.'''
        if s['bus'] != 'smbus' or len(payload) < 1:
            return False
        raw = []
        for p in segs_before:
            raw += [(p['addr'] << 1) | p['rnw']] + [b[0] for b in p['data']]
        raw += [(s['addr'] << 1) | s['rnw']] + [b[0] for b in s['data'][:-1]]
        pec = s['data'][-1]
        ok = crc8(raw) == pec[0]
        if ok:
            self.ann(pec[1], pec[2], A_FIELD, ['PEC 0x%02X ok' % pec[0], 'PEC'])
        return ok

    def prev_write(self, s):
        '''The write segment of the same transaction just before this read, if any.'''
        if len(self.txn) >= 2 and self.txn[-2]['addr'] == s['addr'] and not self.txn[-2]['rnw']:
            return self.txn[-2]
        return None

    # ------------------------------------------------------------ SB-TSI
    def tsi_seg(self, s):
        d = self.st(s['addr'])
        b = [x[0] for x in s['data']]
        dev = 'SB-TSI 0x%02X' % s['addr']
        if not s['rnw']:
            if not b:
                return
            reg = b[0]
            self.ann(s['data'][0][1], s['data'][0][2], A_FIELD, ['Register 0x%02X %s' % (reg, TSI_REGS.get(reg, '?')),
                                                                 '%02X' % reg])
            d['ptr'], d['ptr_ss'] = reg, s['ss']
            if len(b) == 1:
                return                              # Send Byte / command for a following read
            if len(b) > 2:
                self.ann(s['ss'], s['es'], A_WARN, ['SB-TSI supports single byte writes only ([57238] 6.2.2)',
                                                    'Multi-byte'])
            v = b[1]
            d['regs'][reg] = v
            self.ann(s['data'][1][1], s['data'][1][2], A_FIELD, ['0x%02X' % v])
            txt = '%s write %s (0x%02X) = 0x%02X' % (dev, TSI_REGS.get(reg, 'reg'), reg, v)
            f = tsi_field(reg, v)
            self.ann(s['ss'], s['es'], A_TSI, [txt + (': ' + f if f else ''), 'TSI W %02X=%02X' % (reg, v)])
            self.tsi_pair(s, d, reg)
            if reg not in TSI_REGS:
                self.ann(s['ss'], s['es'], A_WARN, ['Write to unimplemented SB-TSI register 0x%02X' % reg, 'Reg?'])
            if reg in (0x01, 0x02, 0x10, 0xFE, 0xFF):
                self.ann(s['ss'], s['es'], A_WARN, ['Register 0x%02X is read-only' % reg, 'RO'])
            return
        if not b:
            return
        reg = d['ptr']
        pw = self.prev_write(s)
        ss = pw['ss'] if pw else s['ss']
        if reg is None:
            self.ann(s['ss'], s['es'], A_TSI, ['%s read (register unknown): 0x%02X' % (dev, b[0]), 'TSI R'])
            return
        v = b[0]
        d['regs'][reg] = v
        self.ann(s['data'][0][1], s['data'][0][2], A_FIELD, ['0x%02X' % v])
        how = '' if pw else ' (Receive Byte, address pointer)'
        f = tsi_field(reg, v)
        self.ann(ss, s['es'], A_TSI, ['%s read %s (0x%02X)%s = 0x%02X%s' % (dev, TSI_REGS.get(reg, 'reg'), reg, how, v,
                                                                       ': ' + f if f else ''),
                                      'TSI R %02X=%02X' % (reg, v)])
        if len(b) > 1:
            self.ann(s['ss'], s['es'], A_WARN, ['SB-TSI returns one byte per read ([57238] 6.2.2)', 'Multi-byte'])
        if reg == 0x03 and v & 0x08:
            for a in self.rmi:
                ra = self.st(a)
                if ra['cmdlen'] != 2:
                    ra['cmdlen'] = 2
            self.ann(ss, s['es'], A_FIELD, ['SBRMIAddrMode=1: SB-RMI uses 2-byte command codes ([57238] 5.4.1.1)',
                                            '2-byte RMI'])
        self.tsi_pair(s, d, reg, ss)

    def tsi_pair(self, s, d, reg, ss=None):
        '''When the second half of an integer/decimal pair is accessed, show the combined value.'''
        if reg in TSI_PAIRS:
            ir, dr = reg, TSI_PAIRS[reg][0]
        elif reg in TSI_DEC_OF:
            ir, dr = TSI_DEC_OF[reg], reg
        else:
            return
        r = d['regs']
        if ir in r and dr in r:
            val = tsi_value(ir, r[ir], r[dr])
            name = TSI_PAIRS[ir][1]
            self.ann(ss if ss is not None else s['ss'], s['es'], A_TSI,
                     ['%s = %.3f degC (0x%02X/0x%02X)' % (name, val, r[ir], r[dr]), '%.3f C' % val])
            if ir == 0x01:
                del r[ir], r[dr]                     # a later pair must be read again

    # ------------------------------------------------------------ SB-RMI
    def cmd_len(self, d, b, s):
        if self.forced_len:
            return self.forced_len
        if d['cmdlen']:
            return d['cmdlen']
        # Infer from the processor state access framing: [73][WrDataLen].. vs [73][00][WrDataLen]..
        n = len(b)
        if n >= 3 and b[0] in (0x71, 0x72, 0x73):
            if b[1] == n - 2 or (s['bus'] == 'smbus' and b[1] == n - 3):
                d['cmdlen'] = 1
            elif b[1] == 0x00 and b[2] == n - 3:
                d['cmdlen'] = 2
        return d['cmdlen'] or 1

    def rmi_seg(self, s):
        d = self.st(s['addr'])
        b = [x[0] for x in s['data']]
        dev = 'SB-RMI 0x%02X' % s['addr']
        if not s['rnw']:
            if not b:
                return
            n = self.cmd_len(d, b, s)
            if len(b) < n:
                self.ann(s['ss'], s['es'], A_WARN, ['Command code is %d bytes, got %d' % (n, len(b)), 'Short'])
                return
            reg = le(b[:n])
            cs, ce = s['data'][0][1], s['data'][n - 1][2]
            self.ann(cs, ce, A_FIELD, ['Command 0x%0*X %s' % (2 * n, reg, rmi_reg_name(reg)), '%02X' % reg])
            d['ptr'], d['ptr_ss'] = reg, s['ss']
            p = s['data'][n:]
            if not p:
                return                              # command code only: a read follows
            if reg in (0x71, 0x72, 0x73):
                self.proc_request(s, d, reg, p, dev)
                return
            pv = [x[0] for x in p]
            if s['bus'] == 'smbus' and len(pv) >= 2:
                if self.pec_check(s, pv, []):
                    pv, p = pv[:-1], p[:-1]
                elif len(pv) == 2 and pv[0] != 1:
                    # Write Byte + PEC whose PEC does not match
                    raw = [s['addr'] << 1] + [x[0] for x in s['data'][:-1]]
                    self.ann(p[1][1], p[1][2], A_WARN, ['PEC 0x%02X BAD (expect 0x%02X)' % (pv[1], crc8(raw)),
                                                        'PEC BAD'])
                    pv, p = pv[:-1], p[:-1]
            if len(pv) == 1:
                vals = [(reg, pv[0], p[0])]
                kind = ''
            elif pv[0] == len(pv) - 1:              # Block Write: byte count then data ([57238] Table 146)
                self.ann(p[0][1], p[0][2], A_FIELD, ['Byte count %d' % pv[0], 'N'])
                vals = [(reg + i, pv[1 + i], p[1 + i]) for i in range(pv[0])]
                kind = 'block '
            else:
                self.ann(s['ss'], s['es'], A_WARN, ['Write of %d bytes is neither single nor block (count mismatch)'
                                                    % len(pv), 'Len?'])
                vals = [(reg + i, pv[i], p[i]) for i in range(len(pv))]
                kind = '? '
            for r, v, x in vals:
                self.ann(x[1], x[2], A_FIELD, ['%s = 0x%02X' % (rmi_reg_name(r), v), '%02X' % v])
            txt = '%s %swrite %s' % (dev, kind, ', '.join('%s (0x%02X) = 0x%02X%s' % (
                rmi_reg_name(r), r, v, (' [' + rmi_field(r, v, d['rev']) + ']') if rmi_field(r, v, d['rev']) else '')
                for r, v, _ in vals))
            self.ann(s['ss'], s['es'], A_RMI, [txt, 'RMI W %02X' % reg])
            for r, v, _ in vals:
                self.rmi_reg_written(s, d, r, v, dev)
            return
        # read
        if not b:
            return
        reg = d['ptr']
        pw = self.prev_write(s)
        ss = pw['ss'] if pw else s['ss']
        if reg is None:
            self.ann(s['ss'], s['es'], A_RMI, ['%s read (register unknown): %s' % (dev, hexs(b)), 'RMI R'])
            return
        if reg in (0x71, 0x72, 0x73):
            self.proc_response(s, d, reg, ss, dev, pw)
            return
        p = list(s['data'])
        pv = b[:]
        if s['bus'] == 'smbus' and len(pv) >= 2 and self.pec_check(s, pv, [pw] if pw else []):
            pv, p = pv[:-1], p[:-1]
        if len(pv) >= 2 and pv[0] == len(pv) - 1:
            self.ann(p[0][1], p[0][2], A_FIELD, ['Byte count %d' % pv[0], 'N'])
            vals = [(reg + i, pv[1 + i], p[1 + i]) for i in range(pv[0])]
            kind = 'block '
        else:
            vals = [(reg + i, pv[i], p[i]) for i in range(len(pv))]
            kind = ''
            if len(pv) > 1:
                self.ann(s['ss'], s['es'], A_WARN, ['%d bytes read without a byte count' % len(pv), 'Len?'])
        for r, v, x in vals:
            self.ann(x[1], x[2], A_FIELD, ['%s = 0x%02X' % (rmi_reg_name(r), v), '%02X' % v])
            d['regs'][r] = v
        txt = '%s %sread %s' % (dev, kind, ', '.join('%s (0x%02X) = 0x%02X%s' % (
            rmi_reg_name(r), r, v, (' [' + rmi_field(r, v, d['rev']) + ']') if rmi_field(r, v, d['rev']) else '')
            for r, v, _ in vals))
        self.ann(ss, s['es'], A_RMI, [txt, 'RMI R %02X' % reg])
        for r, v, _ in vals:
            self.rmi_reg_read(s, d, r, v, ss, dev)

    def rmi_reg_written(self, s, d, r, v, dev):
        d['regs'][r] = v
        if r in (0x3F, 0x38) and d['mb_ss'] is None:
            d['mb_ss'] = s['ss']
        if r == 0x40 and v & 1:
            regs = d['regs']
            cmd = regs.get(0x38)
            din = le([regs.get(0x39, 0), regs.get(0x3A, 0), regs.get(0x3B, 0), regs.get(0x3C, 0)])
            ss = d['mb_ss'] if d['mb_ss'] is not None else s['ss']
            d['mb_ss'] = None
            d['mb_cmd'] = cmd
            if cmd is None:
                self.ann(ss, s['es'], A_MBOX, ['%s mailbox request (command not seen)' % dev, 'MBOX'])
            else:
                self.ann(ss, s['es'], A_MBOX, ['%s mailbox request 0x%02X %s, DataIn 0x%08X' % (dev, cmd, mbox_name(cmd), din),
                                               'MB %02X' % cmd])
            if regs.get(0x3F, 0) & 0x80 == 0:
                self.ann(ss, s['es'], A_WARN, ['InBndMsg7 bit 7 not written as 1 before the request ([57238] 5.4.2.2.2)',
                                               'InBnd7'])
            for r2 in (0x31, 0x32, 0x33, 0x34, 0x37):
                regs.pop(r2 + 0x100, None)
            d['mb_rd_ss'] = None

    def rmi_reg_read(self, s, d, r, v, ss, dev):
        if r == 0x00:
            d['rev'] = v
            if not self.forced_len:
                want = 2 if v in (0x21, 0x31) else 1
                if d['cmdlen'] != want:
                    d['cmdlen'] = want
        if 0x30 <= r <= 0x37:
            regs = d['regs']
            regs[r + 0x100] = v                     # outbound bytes read since the last request
            if d.get('mb_rd_ss') is None:
                d['mb_rd_ss'] = ss
            if all((x + 0x100) in regs for x in (0x31, 0x32, 0x33, 0x34)):
                cmd = regs.get(0x130, d.get('mb_cmd'))
                dout = le([regs[0x131], regs[0x132], regs[0x133], regs[0x134]])
                err = regs.get(0x137)
                e = '' if err is None else ', error %02Xh %s' % (err, MBOX_ERR.get(err, 'reserved'))
                c = d.get('mb_cmd') if cmd is None else cmd
                name = mbox_name(c) if c is not None else 'unknown message'
                self.ann(d['mb_rd_ss'], s['es'], A_MBOX, ['%s mailbox response %s: DataOut %s%s' % (
                    dev, ('0x%02X %s' % (c, name)) if c is not None else name, with_unit(c, dout), e),
                    'MB resp'])
                if err:
                    self.ann(d['mb_rd_ss'], s['es'], A_WARN, ['Mailbox error %02Xh %s' % (err, MBOX_ERR.get(err, 'reserved')),
                                                              'MB err'])
                for x in (0x130, 0x131, 0x132, 0x133, 0x134, 0x137):
                    regs.pop(x, None)
                d['mb_rd_ss'] = None

    # ------------------------------------------------------------ processor state access (71h-73h)
    def proc_request(self, s, d, reg, p, dev):
        pv = [x[0] for x in p]
        wrlen = pv[0]
        self.ann(p[0][1], p[0][2], A_FIELD, ['WrDataLen %d' % wrlen, 'WL'])
        wd = pv[1:1 + wrlen]
        extra = p[1 + wrlen:]
        if len(wd) < wrlen:
            self.ann(s['ss'], s['es'], A_WARN, ['WrDataLen %d but %d data bytes' % (wrlen, len(wd)), 'Len'])
            return
        if extra:
            if len(extra) == 1 and s['bus'] == 'smbus':
                raw = [(s['addr'] << 1)] + [x[0] for x in s['data'][:-1]]
                ok = crc8(raw) == extra[0][0]
                self.ann(extra[0][1], extra[0][2], A_FIELD if ok else A_WARN,
                         ['Intermediate PEC 0x%02X %s' % (extra[0][0], 'ok' if ok else 'BAD (expect 0x%02X)' % crc8(raw)),
                          'PEC'])
            else:
                self.ann(s['ss'], s['es'], A_WARN, ['%d byte(s) after WrData' % len(extra), 'Extra'])
        req = {'reg': reg, 'ss': s['ss']}
        txt = None
        try:
            if reg == 0x73 and wd[1] == 0x86:            # [57238] Table 128 / [56214] Table 80
                nthr = wrlen - 6
                thr = (wd[2] >> 1) | ((wd[3] << 7) if nthr == 2 else 0)
                addr = le(wd[2 + nthr:6 + nthr])
                req.update(kind='Read MSR', n=wd[0])
                txt = 'Read Register MSR 0x%08X, thread %d, %d byte(s)' % (addr, thr, wd[0])
            elif reg == 0x73 and wd[1] == 0x91:          # [57238] Table 131 / [56214] Table 81
                nthr = wrlen - 7
                thr = (wd[2] >> 1) | ((wd[3] << 7) if nthr == 2 else 0)
                fn = le(wd[2 + nthr:6 + nthr])
                ext = wd[6 + nthr]
                half = 'edx:ecx' if ext & 1 else 'ebx:eax'
                req.update(kind='Read CPUID', half=half)
                txt = 'Read CPUID fn 0x%08X ECX %d -> %s, thread %d' % (fn, ext >> 4, half, thr)
            elif reg == 0x72:                            # [56214] Table 82
                req.update(kind='Read Data/Status', n=wd[0])
                txt = 'Read Data/Status, %d byte(s)' % wd[0]
            elif reg == 0x71 and wd[0] == 0x81:          # [56214] Table 83
                thr = wd[1] >> 1
                req.update(kind='Load Address')
                txt = 'Load Address 0x%08X, thread %d' % (le(wd[2:6]), thr)
            elif reg == 0x71 and wd[0] == 0x87:          # [56214] Table 84
                thr = wd[1] >> 1
                req.update(kind='Write Register')
                txt = 'Write Register data 0x%X (%d byte(s)), thread %d' % (le(wd[2:]), wrlen - 2, thr)
        except IndexError:
            txt = None
        if txt is None:
            txt = 'command 0x%02X, WrData %s' % (reg, hexs(wd))
            req['kind'] = 'command 0x%02X' % reg
        d['proc'] = req
        self.ann(s['ss'], s['es'], A_PROC, ['%s %s' % (dev, txt), req['kind']])

    def proc_response(self, s, d, reg, ss, dev, pw):
        p = s['data']
        pv = [x[0] for x in p]
        rdlen = pv[0]
        self.ann(p[0][1], p[0][2], A_FIELD, ['RdDataLen %d' % rdlen, 'RL'])
        body = pv[1:1 + rdlen]
        extra = p[1 + rdlen:]
        if extra:
            if len(extra) == 1 and s['bus'] == 'smbus':
                raw = []
                if pw:
                    pwb = [x[0] for x in pw['data']]
                    # The final PEC excludes the intermediate PEC byte ([56214] 5.4.1).
                    n = self.cmd_len(d, pwb, pw)
                    if len(pwb) > n:
                        wl = pwb[n]
                        pwb = pwb[:n + 1 + wl]
                    raw = [pw['addr'] << 1] + pwb
                raw += [(s['addr'] << 1) | 1] + pv[:-1]
                ok = crc8(raw) == extra[0][0]
                self.ann(extra[0][1], extra[0][2], A_FIELD if ok else A_WARN,
                         ['PEC 0x%02X %s' % (extra[0][0], 'ok' if ok else 'BAD (expect 0x%02X)' % crc8(raw)), 'PEC'])
            else:
                self.ann(s['ss'], s['es'], A_WARN, ['%d byte(s) after RdData' % len(extra), 'Extra'])
        if not body:
            return
        st = body[0]
        self.ann(p[1][1], p[1][2], A_FIELD, ['Status %02Xh %s' % (st, RMI_STATUS.get(st, 'reserved')), 'St'])
        val = le(body[1:])
        req = d['proc'] or {'kind': 'command 0x%02X' % reg}
        if req.get('kind') == 'Read CPUID' and len(body) >= 9:
            lo, hi = le(body[1:5]), le(body[5:9])
            r1, r2 = req['half'].split(':')[::-1]
            vtxt = '%s=0x%08X %s=0x%08X' % (r1, lo, r2, hi)
        elif len(body) > 1:
            vtxt = 'data 0x%0*X' % (2 * (len(body) - 1), val)
        else:
            vtxt = 'no data'
        self.ann(ss, s['es'], A_PROC, ['%s %s result: status %02Xh %s, %s' % (dev, req['kind'], st,
                                       RMI_STATUS.get(st, 'reserved'), vtxt), 'St %02X' % st])
        if st != 0:
            self.ann(ss, s['es'], A_WARN, ['%s status %02Xh %s' % (req['kind'], st, RMI_STATUS.get(st, 'reserved')),
                                           'Status'])
